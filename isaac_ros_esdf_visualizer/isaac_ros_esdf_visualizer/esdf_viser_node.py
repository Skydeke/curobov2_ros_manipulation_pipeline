# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import ast
import os
import re
import tempfile
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import yourdfpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, Pose as RosPose, Vector3
from moveit_msgs.msg import CollisionObject, PlanningScene
from isaac_ros_cumotion_interfaces.srv import ComputeIK, GetEsdf, GetInteractiveTarget
from sensor_msgs.msg import CameraInfo, Image, JointState, PointCloud2
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String
from std_srvs.srv import Trigger
from isaac_ros_cumotion.curobo_server.utils import (
    get_grid_center,
    get_grid_min_corner,
    get_grid_size,
    is_grid_valid,
    load_grid_corners_from_workspace_file,
)
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class _CuPoseCompat:
    """Minimal API-compatible Pose replacement so this file doesn't import curobo.

    Matches ``curobo.types.Pose.from_list()`` and the attribute access pattern
    used by ``ViserVisualizer.add_frame()``.
    """
    position: "torch.Tensor"
    quaternion: "torch.Tensor"

    @staticmethod
    def from_list(values: List[float]) -> "_CuPoseCompat":
        import torch
        return _CuPoseCompat(
            position=torch.tensor(values[:3], dtype=torch.float32),
            quaternion=torch.tensor(values[3:7], dtype=torch.float32),
        )



def _resolve_package_paths_in_urdf(urdf_path: str) -> str:
    with open(urdf_path) as f:
        content = f.read()

    def _replace(match):
        pkg = match.group(1)
        rel = match.group(2)
        try:
            share = get_package_share_directory(pkg)
            return os.path.join(share, rel)
        except Exception:
            return match.group(0)

    resolved = re.sub(r"package://([^/]+)/(.+)", _replace, content)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False)
    tmp.write(resolved)
    tmp.close()
    return tmp.name


def _build_robot_data_dict(xrdf_path: str, urdf_path: str, asset_path: str) -> dict:
    with open(xrdf_path) as f:
        xrdf = yaml.safe_load(f)

    resolved_urdf = _resolve_package_paths_in_urdf(urdf_path)
    urdf = yourdfpy.URDF.load(resolved_urdf, build_scene_graph=True)
    actuated_joint_names = [j.name for j in urdf.actuated_joints]

    mesh_link_names = []
    for link in urdf.robot.links:
        for visual in link.visuals:
            if visual.geometry is not None and hasattr(visual.geometry, "filename"):
                mesh_link_names.append(link.name)
                break

    child_links = {j.child for j in urdf.robot.joints}
    try:
        base_link = next(
            link.name for link in urdf.robot.links if link.name not in child_links
        )
    except StopIteration:
        base_link = urdf.robot.links[0].name if urdf.robot.links else "base_link"

    kin = {}
    coll_geom = xrdf.get("collision", {}).get("geometry", "collision_model")
    spheres = xrdf.get("geometry", {}).get(coll_geom, {}).get("spheres", {})
    kin["collision_spheres"] = spheres
    kin["collision_link_names"] = list(spheres.keys())
    kin["collision_sphere_buffer"] = xrdf.get("collision", {}).get(
        "buffer_distance", 0.0
    )

    sc = xrdf.get("self_collision", {})
    kin["self_collision_ignore"] = sc.get("ignore", {})
    kin["self_collision_buffer"] = sc.get("buffer_distance", {})
    kin["tool_frames"] = xrdf.get("tool_frames", [])

    csp = xrdf.get("cspace", {})
    active_joints = csp.get("joint_names", [])
    default_pos = xrdf.get("default_joint_positions", {})

    active_config = []
    locked_joints = {}
    for j in actuated_joint_names:
        if j in active_joints:
            active_config.append(default_pos.get(j, 0.0))
        else:
            locked_joints[j] = default_pos.get(j, 0.0)

    all_joints = active_joints + list(locked_joints.keys())
    acc_limits = csp.get("acceleration_limits", [10.0])
    jerk_limits = csp.get("jerk_limits", [500.0])
    max_acc = max(acc_limits) if acc_limits else 10.0
    max_jerk = max(jerk_limits) if jerk_limits else 500.0

    kin["cspace"] = {
        "joint_names": all_joints,
        "default_joint_position": active_config + list(locked_joints.values()),
        "null_space_weight": [1.0] * len(all_joints),
        "cspace_distance_weight": [1.0] * len(all_joints),
        "max_acceleration": acc_limits + [max_acc] * len(locked_joints),
        "max_jerk": jerk_limits + [max_jerk] * len(locked_joints),
    }
    kin["lock_joints"] = locked_joints

    extra_links = {}
    for mod in xrdf.get("modifiers", []):
        if "set_base_frame" in mod:
            base_link = mod["set_base_frame"]
        elif "add_frame" in mod:
            fd = mod["add_frame"]
            extra_links[fd["frame_name"]] = {
                "parent_link_name": fd["parent_frame_name"],
                "link_name": fd["frame_name"],
                "joint_name": fd["joint_name"],
                "joint_type": fd["joint_type"],
                "fixed_transform": (
                    fd["fixed_transform"]["position"]
                    + [fd["fixed_transform"]["orientation"]["w"]]
                    + fd["fixed_transform"]["orientation"]["xyz"]
                ),
            }

    kin["extra_links"] = extra_links
    kin["base_link"] = base_link
    kin["urdf_path"] = resolved_urdf
    kin["asset_root_path"] = ""
    kin["mesh_link_names"] = mesh_link_names

    return {"robot_cfg": {"kinematics": kin}}


def _extract_esdf_slice(
    esdf_grid: torch.Tensor,
    origin: torch.Tensor,
    voxel_size: float,
    slice_pose: np.ndarray,
    grid_size_m: np.ndarray,
    slice_resolution: int = 128,
) -> np.ndarray:
    device = esdf_grid.device
    nx, ny, nz = esdf_grid.shape

    half_extent = torch.tensor(
        [(nx - 1) * voxel_size / 2.0,
         (ny - 1) * voxel_size / 2.0,
         (nz - 1) * voxel_size / 2.0],
        device=device,
    )

    half = max(grid_size_m[0], grid_size_m[1]) / 2.0
    u = torch.linspace(-half, half, slice_resolution, device=device)
    v = torch.linspace(-half, half, slice_resolution, device=device)
    uu, vv = torch.meshgrid(u, v, indexing="xy")

    local_points = torch.stack(
        [
            uu.flatten(),
            vv.flatten(),
            torch.zeros(slice_resolution * slice_resolution, device=device),
            torch.ones(slice_resolution * slice_resolution, device=device),
        ],
        dim=1,
    )

    pose_tensor = torch.tensor(slice_pose, dtype=torch.float32, device=device)
    world_points = (pose_tensor @ local_points.T).T[:, :3]

    local_pts = world_points - origin.to(device)
    normalized = local_pts / half_extent

    coords = normalized[:, [2, 1, 0]]
    coords = coords.view(1, 1, slice_resolution, slice_resolution, 3).float()

    esdf_5d = esdf_grid.float().unsqueeze(0).unsqueeze(0)
    sampled = F.grid_sample(
        esdf_5d, coords, mode="bilinear", padding_mode="border", align_corners=True
    )
    values = sampled.squeeze().cpu().numpy()

    max_dist = max(np.max(values), 0.1)
    max_negative_dist = max(np.abs(np.min(values)), 0.05)
    normalized_pos = np.clip(values / max_dist, -1, 1)
    normalized_neg = np.clip(values / max_negative_dist, -1, 1)

    colors = np.zeros((slice_resolution, slice_resolution, 3), dtype=np.uint8)
    neg_mask = normalized_neg < 0
    colors[neg_mask, 0] = ((1 + normalized_neg[neg_mask]) * 255).astype(np.uint8)
    colors[neg_mask, 1] = ((1 + normalized_neg[neg_mask]) * 255).astype(np.uint8)
    colors[neg_mask, 2] = 255
    pos_mask = normalized_pos >= 0
    colors[pos_mask, 0] = 255
    colors[pos_mask, 1] = ((1 - normalized_pos[pos_mask]) * 255).astype(np.uint8)
    colors[pos_mask, 2] = ((1 - normalized_pos[pos_mask]) * 255).astype(np.uint8)

    colors[np.abs(values) < voxel_size * 0.5] = [0, 255, 0]
    return colors


class ESDFViserNode(Node):

    _REACHABILITY_N = 22

    def __init__(self):
        super().__init__("esdf_viser_node")
        self.__obstacle_handles = {}
        self.__attached_object_handles = {}
        self.__attached_object_meshes = []
        self.__esdf_grid_data: Optional[torch.Tensor] = None
        self.__esdf_origin: Optional[np.ndarray] = None
        self.__esdf_grid_center: Optional[np.ndarray] = None
        self.__esdf_voxel_size: float = 0.0
        self.__slice_gizmo = None
        self.__slice_show = None
        self.__slice_image = None
        self.__slice_slider = None
        self.__origin_gizmo = None
        self.__reachability_show = None
        self.__reachability_gizmo = None
        self.__reachability_extent_slider = None
        self.__reachability_image = None
        self.__reachability_image_handle = None
        self.__reachability_bounds_handle = None
        self.__reachability_updating = False
        self.__reachability_pending = False
        self.__reachability_grid = None
        self.__reachability_future = None
        self.__reachability_poll_timer = None
        self.__latest_joint_state = None

        self.declare_parameter("workspace_file_path", "")
        self.declare_parameter("grid_center_m", [0.0, 0.0, 0.0])
        self.declare_parameter("grid_size_m", [2.0, 2.0, 2.0])
        self.declare_parameter("update_esdf_on_request", True)
        self.declare_parameter("use_aabb_on_request", True)
        self.declare_parameter("voxel_size", 0.01)
        self.declare_parameter("max_publish_voxels", 500000)
        self.declare_parameter(
            "esdf_service_name", "/nvblox_node/get_esdf_and_gradient"
        )
        self.declare_parameter("robot_base_frame", "base_link")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("planning_scene_topic", "/planning_scene")
        self.declare_parameter("esdf_service_call_period_secs", 1.0)
        self.declare_parameter("enable_mapper", True)
        self.declare_parameter("viser_host", "0.0.0.0")
        self.declare_parameter("viser_port", 8080)
        self.declare_parameter("viser_content_path", "")
        self.declare_parameter("viser_urdf_path", "")
        self.declare_parameter("viser_asset_path", "")
        self.declare_parameter("viser_add_robot_to_scene", False)
        self.declare_parameter("viser_initialize", True)
        self.declare_parameter("viser_add_control_frames", False)
        self.declare_parameter("viser_visualize_robot_spheres", False)
        self.declare_parameter("viser_visualize_collision_meshes", False)
        self.declare_parameter("visualize_cameras", True)
        self.declare_parameter("rgb_image_topics", ["/kortex_vision/color/image"])
        self.declare_parameter("camera_rgb_info_topics", "['/kortex_vision/color/camera_info']")
        self.__esdf_future = None

        self.__workspace_file_path = (
            self.get_parameter("workspace_file_path").get_parameter_value().string_value
        )
        self.__grid_size_m = (
            self.get_parameter("grid_size_m").get_parameter_value().double_array_value
        )
        self.__update_esdf_on_request = (
            self.get_parameter("update_esdf_on_request")
            .get_parameter_value()
            .bool_value
        )
        self.__use_aabb_on_request = (
            self.get_parameter("use_aabb_on_request").get_parameter_value().bool_value
        )
        self.__grid_center_m = (
            self.get_parameter("grid_center_m").get_parameter_value().double_array_value
        )
        self.__voxel_size = (
            self.get_parameter("voxel_size").get_parameter_value().double_value
        )
        self.__max_publish_voxels = (
            self.get_parameter("max_publish_voxels").get_parameter_value().integer_value
        )
        esdf_service_name = (
            self.get_parameter("esdf_service_name").get_parameter_value().string_value
        )
        self.__robot_base_frame = (
            self.get_parameter("robot_base_frame").get_parameter_value().string_value
        )
        period = (
            self.get_parameter("esdf_service_call_period_secs")
            .get_parameter_value()
            .double_value
        )
        self.__enable_mapper = (
            self.get_parameter("enable_mapper").get_parameter_value().bool_value
        )
        viser_host = self.get_parameter("viser_host").get_parameter_value().string_value
        viser_port = (
            self.get_parameter("viser_port").get_parameter_value().integer_value
        )
        viser_content_path = (
            self.get_parameter("viser_content_path").get_parameter_value().string_value
        )
        viser_urdf_path = (
            self.get_parameter("viser_urdf_path").get_parameter_value().string_value
        )
        viser_asset_path = (
            self.get_parameter("viser_asset_path").get_parameter_value().string_value
        )
        viser_add_robot = (
            self.get_parameter("viser_add_robot_to_scene")
            .get_parameter_value()
            .bool_value
        )
        viser_init = (
            self.get_parameter("viser_initialize").get_parameter_value().bool_value
        )
        viser_viz_spheres = (
            self.get_parameter("viser_visualize_robot_spheres")
            .get_parameter_value()
            .bool_value
        )
        viser_viz_meshes = (
            self.get_parameter("viser_visualize_collision_meshes")
            .get_parameter_value()
            .bool_value
        )
        viser_add_control_frames = (
            self.get_parameter("viser_add_control_frames")
            .get_parameter_value()
            .bool_value
        )

        self.__viz_cameras_enabled = (
            self.get_parameter("visualize_cameras").get_parameter_value().bool_value
        )
        camera_rgb_topics = (
            self.get_parameter("rgb_image_topics")
            .get_parameter_value()
            .string_array_value
        )
        camera_rgb_info_topics_str = (
            self.get_parameter("camera_rgb_info_topics").get_parameter_value().string_value
        )
        camera_rgb_info_topics = ast.literal_eval(camera_rgb_info_topics_str)

        if os.path.exists(self.__workspace_file_path):
            min_corner, max_corner = load_grid_corners_from_workspace_file(
                self.__workspace_file_path
            )
            self.__grid_size_m = get_grid_size(
                min_corner, max_corner, self.__voxel_size
            )
            self.__grid_center_m = get_grid_center(min_corner, self.__grid_size_m)

        if is_grid_valid(self.__grid_size_m, self.__voxel_size):
            raise SystemExit

        try:
            from curobo.viewer import ViserVisualizer
        except ImportError as e:
            raise SystemExit from e

        cp = None
        if viser_add_robot:
            if not viser_content_path or not viser_urdf_path:
                viser_add_robot = False
            else:
                cp = _build_robot_data_dict(
                    viser_content_path, viser_urdf_path, viser_asset_path
                )

        self.__locked_joint_names = (
            set(
                cp.get("robot_cfg", {})
                .get("kinematics", {})
                .get("lock_joints", {})
                .keys()
            )
            if cp
            else set()
        )
        self.__viz = ViserVisualizer(
            content_path=cp,
            connect_ip=viser_host,
            connect_port=viser_port,
            add_robot_to_scene=viser_add_robot,
            initialize_viser=viser_init,
            add_control_frames=viser_add_control_frames,
            visualize_robot_spheres=viser_viz_spheres,
            visualize_collision_meshes=viser_viz_meshes,
        )

        self.__tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=60.0))
        self.__tf_listener = TransformListener(self.__tf_buffer, self)

        self.__draw_grid_box()
        self.__draw_origin_gizmo()
        self._setup_esdf_slice()
        self.__batch_ik_client = self.create_client(ComputeIK, "/cumotion/compute_ik")
        self.__interactive_target_srv = self.create_service(
            GetInteractiveTarget, "/cumotion/get_interactive_target",
            self.__get_interactive_target_cb,
        )
        self._setup_reachability_slice()

        self.__esdf_service_name = esdf_service_name
        self.__esdf_client = None

        joint_states_topic = (
            self.get_parameter("joint_states_topic")
            .get_parameter_value()
            .string_value
        )
        self.create_subscription(
            JointState, joint_states_topic, self.__joint_state_cb, 10
        )

        planning_scene_topic = (
            self.get_parameter("planning_scene_topic")
            .get_parameter_value()
            .string_value
        )
        self.create_subscription(
            PlanningScene, planning_scene_topic, self.__planning_scene_cb, 10
        )

        self.create_subscription(
            PointCloud2, "/curobo_mapper/colored_surface",
            self.__colored_surface_cb, 10
        )

        num_cameras = min(len(camera_rgb_topics), len(camera_rgb_info_topics))
        if self.__viz_cameras_enabled and num_cameras > 0:
            self.__latest_camera_images = [None] * num_cameras
            self.__latest_camera_infos = [None] * num_cameras
            self.__camera_frame_handles = [None] * num_cameras
            self.__camera_image_handles = [None] * num_cameras
            for i in range(num_cameras):
                self.create_subscription(
                    Image, camera_rgb_topics[i],
                    lambda msg, idx=i: self.__rgb_cb(msg, idx), 10
                )
                self.create_subscription(
                    CameraInfo, camera_rgb_info_topics[i],
                    lambda msg, idx=i: self.__cam_info_cb(msg, idx), 10
                )
            self.__camera_viz_timer = self.create_timer(
                0.1, self.__update_camera_visualization
            )
            self.get_logger().info(
                f"Camera visualization enabled: {num_cameras} camera(s)"
            )

        # --- Text query GUI ---
        self.__text_query_pub = self.create_publisher(
            String, "/curobo_mapper/text_query", 1
        )
        vserver = self.__viz._server
        gui = vserver.gui
        self.__text_input = gui.add_text(
            "Semantic Search", initial_value="",
        )
        self.__text_top_k = gui.add_number(
            "top_k", min=1, max=2000, step=1, initial_value=500,
        )
        self.__text_min_score = gui.add_slider(
            "min_score", min=0.0, max=1.0, step=0.01, initial_value=0.05,
        )
        self.__text_search_btn = gui.add_button("Search")
        self.__text_clear_btn = gui.add_button("Clear Matches")

        @self.__text_search_btn.on_click
        def _(_):
            txt = self.__text_input.value.strip()
            if not txt:
                return
            q = String(data=txt)
            self.__text_query_pub.publish(q)
            self.get_logger().info(f"Text query sent: '{txt}'")

        @self.__text_clear_btn.on_click
        def _(_):
            self.__text_query_pub.publish(String(data=""))

        self.create_subscription(
            PointCloud2, "/curobo_mapper/matched_features",
            self.__matched_features_cb, 10
        )

        self.create_subscription(
            PointCloud2, "/curobo_mapper/features_pca",
            self.__features_pca_cb, 1
        )

        self.__feature_pca_image: Optional[np.ndarray] = None
        self.create_subscription(
            Image, "/curobo_mapper/feature_pca_image",
            self.__feature_pca_image_cb, 1
        )

        vserver = self.__viz._server
        with vserver.gui.add_folder("Live View"):
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            self.__current_rgb_image = vserver.gui.add_image(
                blank, label="Current RGB", format="jpeg", jpeg_quality=80
            )
            self.__current_feature_pca_image = vserver.gui.add_image(
                blank, label="Current Feature PCA", format="jpeg", jpeg_quality=80
            )

        timer_cb_group = MutuallyExclusiveCallbackGroup()
        self.timer = self.create_timer(
            period, self.timer_callback, callback_group=timer_cb_group
        )
        self.__image_timer = self.create_timer(
            0.2, self.__update_image_panels,
        )

    def __draw_origin_gizmo(self):
        """Add a transform gizmo at the grid origin for visualization."""
        # Calculate the grid origin (min corner)
        min_corner = get_grid_min_corner(self.__grid_center_m, self.__grid_size_m)

        # Add transform controls gizmo at the origin
        self.__origin_gizmo = self.__viz._server.scene.add_transform_controls(
            "/esdf_origin_gizmo",
            scale=0.3,
            position=tuple(min_corner),
        )

    def __draw_grid_box(self):
        min_c = get_grid_min_corner(self.__grid_center_m, self.__grid_size_m)
        max_c = min_c + np.array(self.__grid_size_m)
        corners = np.array(
            [
                [min_c[0], min_c[1], min_c[2]],
                [max_c[0], min_c[1], min_c[2]],
                [max_c[0], max_c[1], min_c[2]],
                [min_c[0], max_c[1], min_c[2]],
                [min_c[0], min_c[1], max_c[2]],
                [max_c[0], min_c[1], max_c[2]],
                [max_c[0], max_c[1], max_c[2]],
                [min_c[0], max_c[1], max_c[2]],
            ],
            dtype=np.float32,
        )
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        lines = np.array([[corners[i], corners[j]] for i, j in edges], dtype=np.float32)
        self.__viz._server.scene.add_line_segments(
            "/esdf_grid_box",
            points=lines,
            colors=np.array([255, 255, 0], dtype=np.uint8),
            line_width=2.0,
        )

    def _setup_esdf_slice(self):
        server = self.__viz._server
        self.__slice_show = server.gui.add_checkbox(
            "Show ESDF Slice", initial_value=False
        )

        self.__slice_slider = server.gui.add_slider(
            "Slice Height",
            min=float(self.__grid_center_m[2] - self.__grid_size_m[2] / 2.0),
            max=float(self.__grid_center_m[2] + self.__grid_size_m[2] / 2.0),
            step=float(self.__voxel_size),
            initial_value=float(self.__grid_center_m[2]),
        )

        self.__slice_gizmo = server.scene.add_transform_controls(
            "/esdf_slice_gizmo",
            scale=0.2,
            position=(
                float(self.__grid_center_m[0]),
                float(self.__grid_center_m[1]),
                float(self.__grid_center_m[2]),
            ),
            visible=False,
        )

        self.__slice_image = server.scene.add_image(
            "/esdf_slice_gizmo/slice_image",
            image=np.zeros((1, 1, 3), dtype=np.uint8),
            render_width=float(self.__grid_size_m[0]),
            render_height=float(self.__grid_size_m[1]),
            visible=False,
        )

        @self.__slice_gizmo.on_update
        def _on_slice_update(_):
            if self.__slice_show.value:
                self.__slice_slider.value = self.__slice_gizmo.position[2]
                self._update_esdf_slice()

        @self.__slice_show.on_update
        def _on_slice_toggle(_):
            self.__slice_image.visible = self.__slice_show.value
            self.__slice_gizmo.visible = self.__slice_show.value
            if self.__slice_show.value:
                self._update_esdf_slice()

        @self.__slice_slider.on_update
        def _on_slider(_):
            if self.__slice_show.value:
                pos = list(self.__slice_gizmo.position)
                pos[2] = self.__slice_slider.value
                self.__slice_gizmo.position = tuple(pos)
                self._update_esdf_slice()

    def _update_esdf_slice(self):
        if self.__esdf_grid_data is None or self.__esdf_grid_center is None:
            return
        import trimesh

        q = self.__slice_gizmo.wxyz
        slice_pose = trimesh.transformations.quaternion_matrix([q[0], q[1], q[2], q[3]])
        slice_pose[:3, 3] = self.__slice_gizmo.position

        grid_shape = self.__esdf_grid_data.shape
        actual_grid_size = np.array([
            grid_shape[0] * self.__esdf_voxel_size,
            grid_shape[1] * self.__esdf_voxel_size,
            grid_shape[2] * self.__esdf_voxel_size,
        ])
        slice_colors = _extract_esdf_slice(
            esdf_grid=self.__esdf_grid_data,
            origin=torch.as_tensor(self.__esdf_grid_center),
            voxel_size=self.__esdf_voxel_size,
            slice_pose=slice_pose,
            grid_size_m=actual_grid_size,
            slice_resolution=256,
        )
        self.__slice_image.image = slice_colors

    def _setup_reachability_slice(self):
        server = self.__viz._server
        self.__reachability_show = server.gui.add_checkbox(
            "Show Reachability", initial_value=False
        )

        max_extent = min(self.__grid_size_m[0], self.__grid_size_m[1]) / 2.0
        self.__reachability_extent_slider = server.gui.add_slider(
            "Reachability Extent",
            min=0.05,
            max=max_extent,
            step=0.01,
            initial_value=min(1.0, max_extent),
        )

        self.__reachability_gizmo = server.scene.add_transform_controls(
            "/reachability_gizmo",
            scale=0.2,
            position=(
                float(self.__grid_center_m[0]),
                float(self.__grid_center_m[1]),
                float(self.__grid_center_m[2]),
            ),
            visible=False,
        )

        self.__reachability_service = self.create_service(
            Trigger, "cumotion/refresh_reachability", self._reachability_service_cb
        )
        self.__reachability_poll_timer = self.create_timer(
            0.2, self._reachability_poll_cb
        )

        @self.__reachability_gizmo.on_update
        def _on_reach_gizmo_update(_):
            if self.__reachability_show.value:
                self._trigger_reachability_update()

        @self.__reachability_show.on_update
        def _on_reach_toggle(_):
            self.__reachability_gizmo.visible = self.__reachability_show.value
            if self.__reachability_show.value:
                self._trigger_reachability_update()
            else:
                if self.__reachability_image_handle is not None:
                    self.__reachability_image_handle.visible = False
                if self.__reachability_bounds_handle is not None:
                    self.__reachability_bounds_handle.visible = False

        @self.__reachability_extent_slider.on_update
        def _on_reach_slider(_):
            if self.__reachability_show.value:
                self._trigger_reachability_update()

    def _blacken_reachability_image(self):
        if self.__reachability_image_handle is not None:
            n = self._REACHABILITY_N
            black = np.zeros((n, n, 3), dtype=np.uint8)
            self.__reachability_image_handle.image = black

    def _trigger_reachability_update(self):
        if self.__reachability_updating:
            self.__reachability_pending = True
            return
        self.__reachability_pending = False
        self.__reachability_updating = True
        self._blacken_reachability_image()
        self._update_reachability_slice()

    def _reachability_service_cb(self, request, response):
        self.__reachability_pending = True
        self._blacken_reachability_image()
        if self.__reachability_show:
            self.__reachability_show.value = True
        if self.__reachability_gizmo:
            self.__reachability_gizmo.visible = True
        if not self.__reachability_updating:
            self._trigger_reachability_update()
        response.success = True
        response.message = "Reachability update triggered"
        return response

    def _reachability_poll_cb(self):
        if self.__reachability_future is None:
            return
        if not self.__reachability_future.done():
            return
        self._render_reachability(self.__reachability_future)
        self.__reachability_future = None
        self.__reachability_updating = False
        if self.__reachability_pending:
            self._trigger_reachability_update()

    def _update_reachability_slice(self):
        import trimesh

        if self.__latest_joint_state is None:
            self.get_logger().warn("No joint state available for reachability")
            self.__reachability_updating = False
            return

        q = self.__reachability_gizmo.wxyz
        gizmo_mat = trimesh.transformations.quaternion_matrix([q[0], q[1], q[2], q[3]])
        gizmo_mat[:3, 3] = self.__reachability_gizmo.position

        extent = self.__reachability_extent_slider.value
        n_per_axis = self._REACHABILITY_N
        hs = extent / 2.0

        ros_poses = []
        lin = np.linspace(-hs, hs, n_per_axis, dtype=np.float32)
        for i in range(n_per_axis):
            for j in range(n_per_axis):
                local_p = np.array([lin[j], lin[i], 0.0, 1.0])
                world_p = gizmo_mat @ local_p

                pose = RosPose()
                pose.position.x = float(world_p[0])
                pose.position.y = float(world_p[1])
                pose.position.z = float(world_p[2])
                pose.orientation.w = float(q[0])
                pose.orientation.x = float(q[1])
                pose.orientation.y = float(q[2])
                pose.orientation.z = float(q[3])
                ros_poses.append(pose)

        if not self.__batch_ik_client.service_is_ready():
            self.__reachability_updating = False
            return

        req = ComputeIK.Request()
        req.goal_poses = ros_poses
        req.seed_state = self.__latest_joint_state

        self.__reachability_future = self.__batch_ik_client.call_async(req)
        self.get_logger().info(
            f"Sent ComputeIK request for {len(ros_poses)} poses"
        )

    def _render_reachability(self, fut):
        import trimesh

        try:
            result = fut.result()
        except Exception as e:
            self.get_logger().error(f"ComputeIK call failed: {e}")
            return
        if result is None:
            self.get_logger().warn("ComputeIK returned None")
            return
        server = self.__viz._server
        n_per_axis = self._REACHABILITY_N
        success_arr = np.array(result.success, dtype=bool).reshape(n_per_axis, n_per_axis)

        n_ok = int(success_arr.sum())
        self.get_logger().info(
            f"Reachability: {n_ok}/{n_per_axis * n_per_axis} reachable"
        )

        img = np.zeros((n_per_axis, n_per_axis, 3), dtype=np.uint8)
        img[success_arr] = [0, 200, 0]
        img[~success_arr] = [200, 0, 0]

        extent = self.__reachability_extent_slider.value
        visible = self.__reachability_show.value
        if self.__reachability_image_handle is not None:
            self.__reachability_image_handle.remove()
        self.__reachability_image_handle = server.scene.add_image(
            "/reachability_gizmo/reachability_image",
            image=img,
            render_width=extent,
            render_height=extent,
            visible=visible,
        )

        half = extent / 2.0
        gizmo_pos = self.__reachability_gizmo.position
        rot = trimesh.transformations.quaternion_matrix(
            list(self.__reachability_gizmo.wxyz)
        )[:3, :3]
        corners_local = np.array(
            [[-half, -half, 0], [half, -half, 0],
             [half, half, 0], [-half, half, 0]],
            dtype=np.float32,
        )
        corners_world = (rot @ corners_local.T).T + gizmo_pos
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        lines = np.array(
            [[corners_world[i], corners_world[j]] for i, j in edges],
            dtype=np.float32,
        )
        yellow = np.array([255, 255, 0], dtype=np.uint8)
        if self.__reachability_bounds_handle is not None:
            self.__reachability_bounds_handle.remove()
        self.__reachability_bounds_handle = server.scene.add_line_segments(
            "/reachability_bounds",
            points=lines,
            colors=yellow,
            line_width=3.0,
        )
        self.__reachability_bounds_handle.visible = visible

    def __joint_state_cb(self, msg: JointState):
        self.__latest_joint_state = msg

        if not hasattr(self, "_ESDFViserNode__viz"):
            return
        names = [n for n in msg.name if n not in self.__locked_joint_names]
        positions = [
            msg.position[i]
            for i, n in enumerate(msg.name)
            if n not in self.__locked_joint_names
        ]
        self.__viz.set_joint_positions(
            torch.tensor(positions, dtype=torch.float32, device="cuda"), names
        )

    def __planning_scene_cb(self, msg: PlanningScene):
        if not hasattr(self, "_ESDFViserNode__viz"):
            return

        for h in self.__obstacle_handles.values():
            h.remove()
        self.__obstacle_handles.clear()
        for co in msg.world.collision_objects:
            for prim, pose_msg in zip(co.primitives, co.primitive_poses):
                name = "/obstacles/" + co.id + "/" + str(co.primitives.index(prim))
                self.__add_prim_mesh(name, prim, pose_msg)

    def __build_prim_mesh(self, prim):
        import trimesh

        if prim.type == SolidPrimitive.BOX:
            return trimesh.creation.box(extents=list(prim.dimensions))
        if prim.type == SolidPrimitive.SPHERE:
            return trimesh.creation.icosphere(radius=prim.dimensions[0])
        if prim.type == SolidPrimitive.CYLINDER:
            return trimesh.creation.cylinder(
                radius=prim.dimensions[1], height=prim.dimensions[0], sections=16
            )
        return None

    def __add_prim_mesh(self, name, prim, pose_msg):
        import trimesh

        m = self.__build_prim_mesh(prim)
        if m is None:
            return
        q = [
            pose_msg.orientation.w,
            pose_msg.orientation.x,
            pose_msg.orientation.y,
            pose_msg.orientation.z,
        ]
        mat = trimesh.transformations.quaternion_matrix(q)
        mat[:3, 3] = [pose_msg.position.x, pose_msg.position.y, pose_msg.position.z]
        m.apply_transform(mat)
        self.__obstacle_handles[name] = self.__viz._server.scene.add_mesh_trimesh(
            name=name, mesh=m
        )

    def __rgb_cb(self, msg: Image, idx: int):
        if msg.encoding == "rgb8":
            rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3
            )
        elif msg.encoding == "bgr8":
            rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3
            )[:, :, ::-1]
        elif msg.encoding == "rgba8":
            rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 4
            )[:, :, :3]
        elif msg.encoding == "bgra8":
            rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 4
            )[:, :, 2::-1]
        else:
            return
        self.__latest_camera_images[idx] = rgb

    def __cam_info_cb(self, msg: CameraInfo, idx: int):
        self.__latest_camera_infos[idx] = msg

    def _lookup_camera_pose(self, camera_frame: str):
        try:
            t = self.__tf_buffer.lookup_transform(
                self.__robot_base_frame,
                camera_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1),
            )
            return _CuPoseCompat.from_list([
                t.transform.translation.x,
                t.transform.translation.y,
                t.transform.translation.z,
                t.transform.rotation.w,
                t.transform.rotation.x,
                t.transform.rotation.y,
                t.transform.rotation.z,
            ])
        except TransformException:
            return None

    def __update_camera_visualization(self):
        for i in range(len(self.__latest_camera_infos)):
            info = self.__latest_camera_infos[i]
            if info is None:
                continue
            camera_frame = info.header.frame_id
            pose = self._lookup_camera_pose(camera_frame)
            if pose is None:
                continue
            if self.__camera_frame_handles[i] is not None:
                self.__camera_frame_handles[i].remove()
            self.__camera_frame_handles[i] = self.__viz.add_frame(
                f"/cameras/frame_{i}", pose, scale=0.12,
            )

    def timer_callback(self):
        if not self.__enable_mapper:
            return
        if self.__esdf_client is None:
            self.__esdf_client = self.create_client(
                GetEsdf, self.__esdf_service_name
            )
        if not self.__esdf_client.service_is_ready() or self.__esdf_future is not None:
            return

        req = GetEsdf.Request()
        req.visualize_esdf = True
        req.update_esdf = self.__update_esdf_on_request

        # --- THE FIX: Force nvblox to dynamically calculate bounds from active memory ---
        req.use_aabb = True
        req.frame_id = self.__robot_base_frame

        # Pass fallbacks for safety interfaces
        req.aabb_min_m = Point(
            x=0.0 - self.__grid_size_m[0] * 0.5,
            y=0.0 - self.__grid_size_m[1] * 0.5,
            z=0.0 - self.__grid_size_m[2] * 0.5,
        )
        req.aabb_size_m = Vector3(
            x=self.__grid_size_m[0],
            y=self.__grid_size_m[1],
            z=self.__grid_size_m[2],
        )

        self.__esdf_future = self.__esdf_client.call_async(req)
        self.__esdf_future.add_done_callback(self.__esdf_response_cb)

    def __esdf_response_cb(self, future):
        try:
            response = future.result()
            if response.success:
                self.__update_visualization(response)
            else:
                self.get_logger().warn("GetEsdf returned success=False")
        except Exception as e:
            self.get_logger().error(f"GetEsdf service call failed: {e}")
        finally:
            self.__esdf_future = None

    def __get_interactive_target_cb(self, request, response):
        try:
            all_poses = self.__viz.get_control_frame_pose()
        except Exception:
            response.success = False
            response.message = "No interactive control frames available"
            return response
        if request.frame_names:
            names = request.frame_names
        else:
            names = list(all_poses.keys())
        for name in names:
            if name not in all_poses:
                continue
            pose = all_poses[name]
            ros_pose = RosPose()
            ros_pose.position.x = float(pose.position[0].cpu()) if hasattr(pose.position, 'cpu') else float(pose.position[0])
            ros_pose.position.y = float(pose.position[1].cpu()) if hasattr(pose.position, 'cpu') else float(pose.position[1])
            ros_pose.position.z = float(pose.position[2].cpu()) if hasattr(pose.position, 'cpu') else float(pose.position[2])
            ros_pose.orientation.w = float(pose.quaternion[0].cpu()) if hasattr(pose.quaternion, 'cpu') else float(pose.quaternion[0])
            ros_pose.orientation.x = float(pose.quaternion[1].cpu()) if hasattr(pose.quaternion, 'cpu') else float(pose.quaternion[1])
            ros_pose.orientation.y = float(pose.quaternion[2].cpu()) if hasattr(pose.quaternion, 'cpu') else float(pose.quaternion[2])
            ros_pose.orientation.z = float(pose.quaternion[3].cpu()) if hasattr(pose.quaternion, 'cpu') else float(pose.quaternion[3])
            response.poses.append(ros_pose)
            response.frame_names.append(name)
        response.success = True
        response.message = f"Returned {len(response.poses)} frame pose(s)"
        return response

    def __update_visualization(self, esdf_data):
        esdf_array = esdf_data.esdf_and_gradients

        # 1. Dynamically read dimensions directly from the message layout
        #    nvblox stores dims in C-order (x-slowest, z-fastest):
        #    dim[0] = nx, dim[1] = ny, dim[2] = nz
        nx = esdf_array.layout.dim[0].size
        ny = esdf_array.layout.dim[1].size
        nz = esdf_array.layout.dim[2].size

        if nx == 0 or ny == 0 or nz == 0:
            self.get_logger().warn(
                "Received empty or uninitialized ESDF layout dimensions."
            )
            return

        # 2. Reshape to (nx, ny, nz) matching physical x-y-z ordering
        raw_data = np.array(esdf_array.data, dtype=np.float32).reshape(nx, ny, nz)

        # 3. No transpose needed — already in correct C-order
        data = np.transpose(raw_data, (0, 1, 2))

        voxel_size = esdf_data.voxel_size_m
        origin = np.array(
            [esdf_data.origin_m.x, esdf_data.origin_m.y, esdf_data.origin_m.z]
        )

        # 4. Store the raw ESDF grid so the slice always has the latest data,
        #    regardless of whether occupied voxels are found below.
        self.__esdf_grid_data = torch.as_tensor(data, dtype=torch.float32).contiguous()
        self.__esdf_origin = origin
        self.__esdf_grid_center = origin + (np.array([nx, ny, nz]) * voxel_size / 2.0)
        self.__esdf_voxel_size = voxel_size

        actual_grid_size = np.array([nx, ny, nz]) * voxel_size
        actual_center = origin + (actual_grid_size / 2.0)

        # Don't reset slice_gizmo.position — user's manual placement is preserved

        # Update origin gizmo position
        if self.__origin_gizmo:
            self.__origin_gizmo.position = tuple(origin)

        if self.__slice_show and self.__slice_show.value:
            self._update_esdf_slice()

        # 5. ESDF voxel point cloud is intentionally not published — the colored
        # surface from __colored_surface_cb already visualizes occupied regions.

    def __colored_surface_cb(self, msg: PointCloud2):
        n = msg.width
        if n == 0:
            return
        packed = np.frombuffer(msg.data, dtype=np.float32).reshape(n, 4)
        positions = packed[:, :3].copy()
        rgb_uint32 = packed[:, 3].view(np.uint32).copy()
        colors = np.zeros((n, 3), dtype=np.uint8)
        colors[:, 0] = (rgb_uint32 >> 16) & 0xFF
        colors[:, 1] = (rgb_uint32 >> 8) & 0xFF
        colors[:, 2] = rgb_uint32 & 0xFF
        if np.all(colors == 0):
            return
        self.__viz.add_point_cloud(
            pointcloud=positions,
            colors=colors,
            point_size=float(self.__voxel_size),
            name="/colored_surface",
        )

    def __features_pca_cb(self, msg: PointCloud2):
        n = msg.width
        if n == 0:
            return
        packed = np.frombuffer(msg.data, dtype=np.float32).reshape(n, 4)
        positions = packed[:, :3].copy()
        rgb_uint32 = packed[:, 3].view(np.uint32).copy()
        colors = np.zeros((n, 3), dtype=np.uint8)
        colors[:, 0] = (rgb_uint32 >> 16) & 0xFF
        colors[:, 1] = (rgb_uint32 >> 8) & 0xFF
        colors[:, 2] = rgb_uint32 & 0xFF
        self.__viz.add_point_cloud(
            pointcloud=positions,
            colors=colors,
            point_size=float(self.__voxel_size),
            name="/features_pca",
        )

    def __matched_features_cb(self, msg: PointCloud2):
        n = msg.width
        if n == 0:
            self.__viz.add_point_cloud(
                pointcloud=np.zeros((0, 3), dtype=np.float32),
                colors=np.zeros((0, 3), dtype=np.uint8),
                point_size=float(self.__voxel_size) * 1.5,
                name="/matched_features",
            )
            return
        packed = np.frombuffer(msg.data, dtype=np.float32).reshape(n, 4)
        positions = packed[:, :3].copy()
        rgb_uint32 = packed[:, 3].view(np.uint32).copy()
        colors = np.zeros((n, 3), dtype=np.uint8)
        colors[:, 0] = (rgb_uint32 >> 16) & 0xFF
        colors[:, 1] = (rgb_uint32 >> 8) & 0xFF
        colors[:, 2] = rgb_uint32 & 0xFF
        self.__viz.add_point_cloud(
            pointcloud=positions,
            colors=colors,
            point_size=float(self.__voxel_size) * 1.5,
            name="/matched_features",
        )

    def __feature_pca_image_cb(self, msg: Image):
        if msg.encoding == "rgb8":
            rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3
            )
        elif msg.encoding == "rgba8":
            rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 4
            )[:, :, :3]
        else:
            return
        self.__feature_pca_image = rgb

    def __update_image_panels(self):
        if self.__latest_camera_images[0] is not None:
            self.__current_rgb_image.image = self.__latest_camera_images[0]
        if self.__feature_pca_image is not None:
            self.__current_feature_pca_image.image = self.__feature_pca_image


def main(args=None):
    rclpy.init(args=args)
    node = ESDFViserNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()


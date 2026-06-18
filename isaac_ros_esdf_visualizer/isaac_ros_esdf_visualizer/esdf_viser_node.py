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
from geometry_msgs.msg import Point, Vector3
from moveit_msgs.msg import CollisionObject, PlanningScene
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from isaac_ros_cumotion_python_utils.utils import (
    get_grid_center,
    get_grid_min_corner,
    get_grid_size,
    is_grid_valid,
    load_grid_corners_from_workspace_file,
)
from isaac_ros_cumotion_interfaces.srv import GetEsdf
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node


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
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("planning_scene_topic", "/planning_scene")
        self.declare_parameter("esdf_service_call_period_secs", 1.0)
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
        viser_add_frames = (
            self.get_parameter("viser_add_control_frames")
            .get_parameter_value()
            .bool_value
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
            add_control_frames=viser_add_frames,
            visualize_robot_spheres=viser_viz_spheres,
            visualize_collision_meshes=viser_viz_meshes,
        )

        self.__draw_grid_box()
        self.__draw_origin_gizmo()
        self._setup_esdf_slice()

        self.__esdf_service_name = esdf_service_name
        self.__esdf_client = None

        if viser_add_robot:
            joint_state_topic = (
                self.get_parameter("joint_state_topic")
                .get_parameter_value()
                .string_value
            )
            self.create_subscription(
                JointState, joint_state_topic, self.__joint_state_cb, 10
            )

        planning_scene_topic = (
            self.get_parameter("planning_scene_topic")
            .get_parameter_value()
            .string_value
        )
        self.create_subscription(
            PlanningScene, planning_scene_topic, self.__planning_scene_cb, 10
        )

        timer_cb_group = MutuallyExclusiveCallbackGroup()
        self.timer = self.create_timer(
            period, self.timer_callback, callback_group=timer_cb_group
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

    def __joint_state_cb(self, msg: JointState):
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

    def timer_callback(self):
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
        finally:
            self.__esdf_future = None

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

        # 4. Filter out unobserved regions (nvblox uses -1000 for unobserved)
        # ESDF convention: positive = outside (free), negative = inside (occupied)
        # Negate so occupied voxels have positive values
        unobserved_mask = data < -999.0
        data_clean = data.copy()
        data_clean[unobserved_mask] = 1000.0
        data_occupied = -data_clean
        occupied_mask = data_occupied > 0.0
        if not np.any(occupied_mask):
            self.get_logger().warn("No occupied voxels found in ESDF grid.")
            return

        indices = np.argwhere(occupied_mask)
        positions = origin + (indices.astype(np.float64) + 0.5) * voxel_size
        values = data_occupied[occupied_mask]

        # 5. Downsample if needed
        if len(positions) > self.__max_publish_voxels:
            step = max(1, len(positions) // self.__max_publish_voxels)
            positions = positions[::step]
            values = values[::step]

        # 6. Generate vertex color maps
        max_val = max(np.max(values), 0.01)
        colors = np.zeros((len(positions), 3), dtype=np.uint8)
        colors[:, 0] = np.clip((values / max_val) * 255, 50, 255).astype(np.uint8)
        colors[:, 1] = np.clip((1 - values / max_val) * 80, 0, 80).astype(np.uint8)

        # 7. Store parameters
        self.__esdf_grid_data = torch.as_tensor(data, dtype=torch.float32).contiguous()
        self.__esdf_origin = origin
        self.__esdf_grid_center = origin + (np.array([nx, ny, nz]) * voxel_size / 2.0)
        self.__esdf_voxel_size = voxel_size

        actual_grid_size = np.array([nx, ny, nz]) * voxel_size
        actual_center = origin + (actual_grid_size / 2.0)

        if self.__slice_gizmo:
            self.__slice_gizmo.position = tuple(actual_center)

        # Update origin gizmo position
        if self.__origin_gizmo:
            self.__origin_gizmo.position = tuple(origin)

        self.__viz.add_point_cloud(
            pointcloud=positions.astype(np.float32),
            colors=colors,
            point_size=float(voxel_size),
            name="/esdf/voxels",
        )

        if self.__slice_show and self.__slice_show.value:
            self._update_esdf_slice()


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


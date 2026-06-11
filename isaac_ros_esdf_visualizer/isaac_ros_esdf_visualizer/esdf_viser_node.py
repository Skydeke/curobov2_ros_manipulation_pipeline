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

import numpy as np
import torch
import yaml
import yourdfpy
from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3
from moveit_msgs.msg import CollisionObject, PlanningScene
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from isaac_ros_cumotion_python_utils.utils import \
    get_grid_center, get_grid_min_corner, get_grid_size, is_grid_valid, \
    load_grid_corners_from_workspace_file
from nvblox_msgs.srv import EsdfAndGradients
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node


def _build_robot_data_dict(
    xrdf_path: str, urdf_path: str, asset_path: str
) -> dict:
    with open(xrdf_path) as f:
        xrdf = yaml.safe_load(f)

    urdf = yourdfpy.URDF.load(urdf_path, build_scene_graph=True)

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
    kin["urdf_path"] = urdf_path
    kin["asset_root_path"] = asset_path
    kin["mesh_link_names"] = mesh_link_names

    return {"robot_cfg": {"kinematics": kin}}


class ESDFViserNode(Node):

    def __init__(self):
        super().__init__('esdf_viser_node')
        self.__obstacle_handles = {}
        self.__attached_object_handles = {}
        self.__attached_object_meshes = []

        self.declare_parameter('workspace_file_path', '')
        self.declare_parameter('grid_center_m', [0.0, 0.0, 0.0])
        self.declare_parameter('grid_size_m', [2.0, 2.0, 2.0])
        self.declare_parameter('update_esdf_on_request', True)
        self.declare_parameter('use_aabb_on_request', True)
        self.declare_parameter('voxel_size', 0.01)
        self.declare_parameter('max_publish_voxels', 500000)
        self.declare_parameter('esdf_service_name', '/nvblox_node/get_esdf_and_gradient')
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('planning_scene_topic', '/planning_scene')
        self.declare_parameter('esdf_service_call_period_secs', 1.0)
        self.declare_parameter('viser_host', '0.0.0.0')
        self.declare_parameter('viser_port', 8080)
        self.declare_parameter('viser_content_path', '')
        self.declare_parameter('viser_urdf_path', '')
        self.declare_parameter('viser_asset_path', '')
        self.declare_parameter('viser_add_robot_to_scene', False)
        self.declare_parameter('viser_initialize', True)
        self.declare_parameter('viser_add_control_frames', False)
        self.declare_parameter('viser_visualize_robot_spheres', False)
        self.declare_parameter('viser_visualize_collision_meshes', False)
        self.__esdf_future = None

        self.__workspace_file_path = (
            self.get_parameter('workspace_file_path').get_parameter_value().string_value
        )
        self.__grid_size_m = (
            self.get_parameter('grid_size_m').get_parameter_value().double_array_value
        )
        self.__update_esdf_on_request = (
            self.get_parameter('update_esdf_on_request').get_parameter_value().bool_value
        )
        self.__use_aabb_on_request = (
            self.get_parameter('use_aabb_on_request').get_parameter_value().bool_value
        )
        self.__grid_center_m = (
            self.get_parameter('grid_center_m').get_parameter_value().double_array_value
        )
        self.__voxel_size = self.get_parameter('voxel_size').get_parameter_value().double_value
        self.__max_publish_voxels = (
            self.get_parameter('max_publish_voxels').get_parameter_value().integer_value
        )
        esdf_service_name = (
            self.get_parameter('esdf_service_name').get_parameter_value().string_value
        )
        self.__robot_base_frame = (
            self.get_parameter('robot_base_frame').get_parameter_value().string_value
        )
        period = (
            self.get_parameter('esdf_service_call_period_secs').get_parameter_value().double_value
        )
        viser_host = (
            self.get_parameter('viser_host').get_parameter_value().string_value
        )
        viser_port = (
            self.get_parameter('viser_port').get_parameter_value().integer_value
        )
        viser_content_path = (
            self.get_parameter('viser_content_path').get_parameter_value().string_value
        )
        viser_urdf_path = (
            self.get_parameter('viser_urdf_path').get_parameter_value().string_value
        )
        viser_asset_path = (
            self.get_parameter('viser_asset_path').get_parameter_value().string_value
        )
        viser_add_robot = (
            self.get_parameter('viser_add_robot_to_scene').get_parameter_value().bool_value
        )
        viser_init = (
            self.get_parameter('viser_initialize').get_parameter_value().bool_value
        )
        viser_add_frames = (
            self.get_parameter('viser_add_control_frames').get_parameter_value().bool_value
        )
        viser_viz_spheres = (
            self.get_parameter('viser_visualize_robot_spheres').get_parameter_value().bool_value
        )
        viser_viz_meshes = (
            self.get_parameter('viser_visualize_collision_meshes').get_parameter_value().bool_value
        )

        if os.path.exists(self.__workspace_file_path):
            self.get_logger().info(
                f'Loading grid center and dims from workspace file: {self.__workspace_file_path}.')
            min_corner, max_corner = load_grid_corners_from_workspace_file(
                self.__workspace_file_path)
            self.__grid_size_m = get_grid_size(min_corner, max_corner, self.__voxel_size)
            self.__grid_center_m = get_grid_center(min_corner, self.__grid_size_m)
        else:
            self.get_logger().info(
                'Loading grid position and dims from grid_center_m and grid_size_m parameters.')
            self.get_logger().info(
                f'Params: grid_center={self.__grid_center_m}, '
                f'grid_size={self.__grid_size_m}, '
                f'voxel_size={self.__voxel_size}, '
                f'max_voxels={self.__max_publish_voxels}, '
                f'service={esdf_service_name}')

        if is_grid_valid(self.__grid_size_m, self.__voxel_size):
            self.get_logger().fatal('Number of voxels should be at least 1 in every dimension.')
            raise SystemExit

        try:
            from curobo.viewer import ViserVisualizer
        except ImportError as e:
            self.get_logger().fatal(
                'curobo.viewer not available. Ensure curobo_core is installed.')
            raise SystemExit from e

        cp = None
        if viser_add_robot:
            if not viser_content_path or not viser_urdf_path:
                self.get_logger().warn(
                    'viser_add_robot_to_scene is true but viser_content_path or '
                    'viser_urdf_path is empty; disabling robot visualization')
                viser_add_robot = False
            else:
                cp = _build_robot_data_dict(
                    viser_content_path, viser_urdf_path, viser_asset_path
                )
        self.__locked_joint_names = set(
            cp.get("robot_cfg", {}).get("kinematics", {}).get("lock_joints", {}).keys()
        ) if cp else set()
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
        self.get_logger().info(f'Viser visualizer started at http://{viser_host}:{viser_port}')

        self.__draw_grid_box()

        self.__esdf_service_name = esdf_service_name
        self.__esdf_client = None

        if viser_add_robot:
            joint_state_topic = (
                self.get_parameter('joint_state_topic').get_parameter_value().string_value
            )
            self.create_subscription(JointState, joint_state_topic, self.__joint_state_cb, 10)
        planning_scene_topic = (
            self.get_parameter('planning_scene_topic').get_parameter_value().string_value
        )
        self.create_subscription(
            PlanningScene, planning_scene_topic, self.__planning_scene_cb, 10
        )

        timer_cb_group = MutuallyExclusiveCallbackGroup()
        self.timer = self.create_timer(
            period, self.timer_callback, callback_group=timer_cb_group
        )

    def __draw_grid_box(self):
        min_c = get_grid_min_corner(self.__grid_center_m, self.__grid_size_m)
        max_c = min_c + np.array(self.__grid_size_m)
        corners = np.array([
            [min_c[0], min_c[1], min_c[2]],
            [max_c[0], min_c[1], min_c[2]],
            [max_c[0], max_c[1], min_c[2]],
            [min_c[0], max_c[1], min_c[2]],
            [min_c[0], min_c[1], max_c[2]],
            [max_c[0], min_c[1], max_c[2]],
            [max_c[0], max_c[1], max_c[2]],
            [min_c[0], max_c[1], max_c[2]],
        ], dtype=np.float32)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        lines = np.array([[corners[i], corners[j]] for i, j in edges], dtype=np.float32)
        yellow = np.array([255, 255, 0], dtype=np.uint8)
        self.__viz._server.scene.add_line_segments(
            "/esdf_grid_box", points=lines, colors=yellow, line_width=2.0,
        )

    def __joint_state_cb(self, msg: JointState):
        if not hasattr(self, '_ESDFViserNode__viz'):
            return
        names = [n for n in msg.name if n not in self.__locked_joint_names]
        positions = [msg.position[i] for i, n in enumerate(msg.name)
                     if n not in self.__locked_joint_names]
        self.__viz.set_joint_positions(
            torch.tensor(positions, dtype=torch.float32, device='cuda'), names
        )

    def __planning_scene_cb(self, msg: PlanningScene):
        if not hasattr(self, '_ESDFViserNode__viz'):
            return

        import trimesh

        # Clear world obstacles
        for h in self.__obstacle_handles.values():
            h.remove()
        self.__obstacle_handles.clear()

        # Add world collision objects
        for co in msg.world.collision_objects:
            for prim, pose_msg in zip(co.primitives, co.primitive_poses):
                name = "/obstacles/" + co.id + "/" + str(co.primitives.index(prim))
                self.__add_prim_mesh(name, prim, pose_msg)

        # Handle attached objects — store mesh + link-relative pose for tracking
        for h in self.__attached_object_handles.values():
            h.remove()
        self.__attached_object_handles.clear()
        self.__attached_object_meshes.clear()

        for aco in msg.robot_state.attached_collision_objects:
            obj = aco.object
            for prim, pose_msg in zip(obj.primitives, obj.primitive_poses):
                name = "/attached/" + obj.id + "/" + str(obj.primitives.index(prim))
                m = self.__build_prim_mesh(prim)
                if m is None:
                    continue
                # pose is relative to aco.link_name
                self.__attached_object_meshes.append(
                    (m, name, aco.link_name, pose_msg)
                )

    def __build_prim_mesh(self, prim):
        import trimesh
        if prim.type == SolidPrimitive.BOX:
            return trimesh.creation.box(extents=list(prim.dimensions))
        if prim.type == SolidPrimitive.SPHERE:
            return trimesh.creation.icosphere(radius=prim.dimensions[0])
        if prim.type == SolidPrimitive.CYLINDER:
            return trimesh.creation.cylinder(
                radius=prim.dimensions[1], height=prim.dimensions[0], sections=16)
        return None

    def __add_prim_mesh(self, name, prim, pose_msg):
        import trimesh
        m = self.__build_prim_mesh(prim)
        if m is None:
            return
        q = [pose_msg.orientation.w, pose_msg.orientation.x,
             pose_msg.orientation.y, pose_msg.orientation.z]
        mat = trimesh.transformations.quaternion_matrix(q)
        mat[:3, 3] = [pose_msg.position.x, pose_msg.position.y, pose_msg.position.z]
        m.apply_transform(mat)
        h = self.__viz._server.scene.add_mesh_trimesh(name=name, mesh=m)
        self.__obstacle_handles[name] = h

    def timer_callback(self):
        if self.__esdf_client is None:
            self.__esdf_client = self.create_client(
                EsdfAndGradients, self.__esdf_service_name
            )

        if not self.__esdf_client.service_is_ready():
            return

        if self.__esdf_future is None:
            min_corner = get_grid_min_corner(self.__grid_center_m, self.__grid_size_m)
            aabb_min = Point()
            aabb_min.x = min_corner[0]
            aabb_min.y = min_corner[1]
            aabb_min.z = min_corner[2]
            aabb_size = Vector3()
            aabb_size.x = self.__grid_size_m[0]
            aabb_size.y = self.__grid_size_m[1]
            aabb_size.z = self.__grid_size_m[2]

            req = EsdfAndGradients.Request()
            req.visualize_esdf = True
            req.update_esdf = self.__update_esdf_on_request
            req.use_aabb = self.__use_aabb_on_request
            req.frame_id = self.__robot_base_frame
            req.aabb_min_m = aabb_min
            req.aabb_size_m = aabb_size
            req.aabbs_to_clear_min_m = []
            req.aabbs_to_clear_size_m = []
            req.spheres_to_clear_center_m = []
            req.spheres_to_clear_radius_m = []

            self.__esdf_future = self.__esdf_client.call_async(req)

        if self.__esdf_future.done():
            response = self.__esdf_future.result()
            if response.success:
                self.__update_visualization(response)
            else:
                self.get_logger().info('ESDF request failed. Not updating the grid.')
            self.__esdf_future = None

    def __update_visualization(self, esdf_data):
        esdf_array = esdf_data.esdf_and_gradients
        shape = [
            esdf_array.layout.dim[0].size,
            esdf_array.layout.dim[1].size,
            esdf_array.layout.dim[2].size,
        ]
        data = np.array(esdf_array.data, dtype=np.float32).reshape(shape)

        # Same sign convention as esdf_visualizer.py:
        # nvblox uses negative distance inside obstacles, flip for visualization.
        # nvblox assigns -1000.0 for unobserved voxels.
        data[data < -999.9] = 1000.0
        data = -data
        data += 0.5 * esdf_data.voxel_size_m

        occupied = data > 0
        indices = np.argwhere(occupied)

        if len(indices) == 0:
            self.get_logger().info('No occupied voxels found.')
            return

        voxel_size = esdf_data.voxel_size_m
        origin = np.array([
            esdf_data.origin_m.x,
            esdf_data.origin_m.y,
            esdf_data.origin_m.z,
        ])
        positions = origin + (indices.astype(np.float64) + 0.5) * voxel_size

        values = data[occupied]
        if len(positions) > self.__max_publish_voxels:
            step = max(1, len(positions) // self.__max_publish_voxels)
            positions = positions[::step]
            values = values[::step]

        # Color: red channel varies with distance (deeper inside = brighter red)
        max_val = max(np.max(values), 0.01)
        colors = np.zeros((len(positions), 3), dtype=np.uint8)
        colors[:, 0] = np.clip((values / max_val) * 255, 50, 255).astype(np.uint8)
        colors[:, 1] = np.clip((1 - values / max_val) * 80, 0, 80).astype(np.uint8)

        self.__viz.add_point_cloud(
            pointcloud=positions.astype(np.float32),
            colors=colors,
            point_size=float(voxel_size),
            name="/esdf/voxels",
        )
        self.get_logger().info(
            f'Published {len(positions)} voxels to Viser')


def main(args=None):
    rclpy.init(args=args)
    node = ESDFViserNode()
    try:
        node.get_logger().info('Starting ESDFViserNode')
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Destroying ESDFViserNode')
    except Exception as e:
        node.get_logger().info(f'Shutting down due to exception of type {type(e)}: {e}')
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()

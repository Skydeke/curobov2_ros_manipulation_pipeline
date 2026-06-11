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

from isaac_ros_cumotion_python_utils.utils import \
    get_grid_center, get_grid_min_corner, get_grid_size, is_grid_valid, \
    load_grid_corners_from_workspace_file
import numpy as np
from nvblox_msgs.srv import EsdfAndGradients
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
import torch
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Vector3


class ESDFVisualizer(Node):

    def __init__(self):
        super().__init__('esdf_visualizer')

        self.declare_parameter('workspace_file_path', '')
        self.declare_parameter('grid_center_m', [0.0, 0.0, 0.0])
        self.declare_parameter('grid_size_m', [2.0, 2.0, 2.0])
        self.declare_parameter('update_esdf_on_request', True)
        self.declare_parameter('use_aabb_on_request', True)

        self.declare_parameter('clear_shapes_on_request', False)
        self.declare_parameter('clear_shapes_subsampling_factor', 2)
        self.declare_parameter('aabbs_to_clear_min_m', [0.7, 0.6, -0.1])
        self.declare_parameter('aabbs_to_clear_size_m', [0.2, 0.2, 0.9])
        self.declare_parameter('spheres_to_clear_center_m', [1.5, 0.5, 0.0])
        self.declare_parameter('spheres_to_clear_radius_m', 0.2)

        self.declare_parameter('voxel_size', 0.01)
        self.declare_parameter('publish_voxel_size', 0.01)
        self.declare_parameter('max_publish_voxels', 500000)

        self.declare_parameter('esdf_service_name', '/nvblox_node/get_esdf_and_gradient')
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('esdf_service_call_period_secs', 0.01)
        self.__esdf_future = None
        self.__voxel_pub = self.create_publisher(Marker, '/curobo/voxels', 10)

        esdf_service_name = (
            self.get_parameter('esdf_service_name').get_parameter_value().string_value
        )
        esdf_service_cb_group = MutuallyExclusiveCallbackGroup()
        self.__esdf_client = self.create_client(
            EsdfAndGradients, esdf_service_name, callback_group=esdf_service_cb_group
        )
        while not self.__esdf_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'Service({esdf_service_name}) not available, waiting again...')
        self.__esdf_req = EsdfAndGradients.Request()

        timer_cb_group = MutuallyExclusiveCallbackGroup()
        self.timer = self.create_timer(
            self.get_parameter('esdf_service_call_period_secs').get_parameter_value().double_value,
            self.timer_callback, callback_group=timer_cb_group
        )

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
        self.__publish_voxel_size = (
            self.get_parameter('publish_voxel_size').get_parameter_value().double_value
        )
        self.__max_publish_voxels = (
            self.get_parameter('max_publish_voxels').get_parameter_value().integer_value
        )
        self.__clear_shapes_on_request = (
            self.get_parameter('clear_shapes_on_request').get_parameter_value().bool_value)
        self.__clear_shapes_subsampling_factor = (
            self.get_parameter(
                'clear_shapes_subsampling_factor').get_parameter_value().integer_value)
        self.__clear_shapes_counter = 0
        self.__aabbs_to_clear_min_m = (
            self.get_parameter('aabbs_to_clear_min_m').get_parameter_value().double_array_value)
        self.__aabbs_to_clear_size_m = (
            self.get_parameter('aabbs_to_clear_size_m').get_parameter_value().double_array_value)
        self.__spheres_to_clear_center_m = (
            self.get_parameter(
                'spheres_to_clear_center_m').get_parameter_value().double_array_value)
        self.__spheres_to_clear_radius_m = (
            self.get_parameter('spheres_to_clear_radius_m').get_parameter_value().double_value)

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

        if is_grid_valid(self.__grid_size_m, self.__voxel_size):
            self.get_logger().fatal('Number of voxels should be at least 1 in every dimension.')
            raise SystemExit

        self.__robot_base_frame = (
            self.get_parameter('robot_base_frame').get_parameter_value().string_value
        )
        self.__tensor_args = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        self.__device = torch.device(self.__tensor_args['device'])

        self.__grid_shape = (
            int(self.__grid_size_m[0] / self.__voxel_size),
            int(self.__grid_size_m[1] / self.__voxel_size),
            int(self.__grid_size_m[2] / self.__voxel_size),
        )

    def timer_callback(self):
        if self.__esdf_future is None:
            self.get_logger().debug('Calling ESDF service')

            min_corner = get_grid_min_corner(self.__grid_center_m, self.__grid_size_m)
            aabb_min = Point()
            aabb_min.x = min_corner[0]
            aabb_min.y = min_corner[1]
            aabb_min.z = min_corner[2]
            aabb_size = Vector3()
            aabb_size.x = self.__grid_size_m[0]
            aabb_size.y = self.__grid_size_m[1]
            aabb_size.z = self.__grid_size_m[2]

            self.__esdf_future = self.send_request(aabb_min, aabb_size)
        if self.__esdf_future is not None and self.__esdf_future.done():
            response = self.__esdf_future.result()
            if response.success:
                self.publish_voxels_from_esdf(response)
            else:
                self.get_logger().info('ESDF request failed. Not updating the grid.')
            self.__esdf_future = None

    def send_request(self, aabb_min_m, aabb_size_m):
        self.__esdf_req.visualize_esdf = True
        self.__esdf_req.update_esdf = self.__update_esdf_on_request
        self.__esdf_req.use_aabb = self.__use_aabb_on_request
        self.__esdf_req.frame_id = self.__robot_base_frame
        self.__esdf_req.aabb_min_m = aabb_min_m
        self.__esdf_req.aabb_size_m = aabb_size_m
        if self.__clear_shapes_on_request and \
                self.__clear_shapes_counter % self.__clear_shapes_subsampling_factor == 0:
            aabbs_to_clear_min_m = Point(
                x=self.__aabbs_to_clear_min_m[0],
                y=self.__aabbs_to_clear_min_m[1],
                z=self.__aabbs_to_clear_min_m[2])
            aabbs_to_clear_size_m = Point(
                x=self.__aabbs_to_clear_size_m[0],
                y=self.__aabbs_to_clear_size_m[1],
                z=self.__aabbs_to_clear_size_m[2])
            spheres_to_clear_center_m = Point(
                x=self.__spheres_to_clear_center_m[0],
                y=self.__spheres_to_clear_center_m[1],
                z=self.__spheres_to_clear_center_m[2])
            self.__esdf_req.aabbs_to_clear_min_m = [aabbs_to_clear_min_m]
            self.__esdf_req.aabbs_to_clear_size_m = [aabbs_to_clear_size_m]
            self.__esdf_req.spheres_to_clear_center_m = [spheres_to_clear_center_m]
            self.__esdf_req.spheres_to_clear_radius_m = [self.__spheres_to_clear_radius_m]
        else:
            self.__esdf_req.aabbs_to_clear_min_m = []
            self.__esdf_req.aabbs_to_clear_size_m = []
            self.__esdf_req.spheres_to_clear_center_m = []
            self.__esdf_req.spheres_to_clear_radius_m = []
        self.__clear_shapes_counter += 1

        self.get_logger().debug(
            f'ESDF  req = {self.__esdf_req.aabb_min_m}, {self.__esdf_req.aabb_size_m}'
        )
        esdf_future = self.__esdf_client.call_async(self.__esdf_req)
        return esdf_future

    def publish_voxels_from_esdf(self, esdf_data):
        esdf_array = esdf_data.esdf_and_gradients
        array_shape = [
            esdf_array.layout.dim[0].size,
            esdf_array.layout.dim[1].size,
            esdf_array.layout.dim[2].size,
        ]
        array_data = np.array(esdf_array.data, dtype=np.float32)
        esdf_grid = torch.as_tensor(array_data, device=self.__device)
        esdf_grid = esdf_grid.view(array_shape[0], array_shape[1], array_shape[2])

        esdf_grid[esdf_grid < -999.9] = 1000.0
        esdf_grid = -1.0 * esdf_grid
        esdf_grid += 0.5 * self.__voxel_size

        occupied_mask = esdf_grid > 0.0
        occupied_indices = torch.nonzero(occupied_mask)
        if occupied_indices.size(0) == 0:
            return

        if occupied_indices.size(0) > self.__max_publish_voxels:
            indices = torch.randperm(occupied_indices.size(0), device=self.__device)[:self.__max_publish_voxels]
            occupied_indices = occupied_indices[indices]

        origin = esdf_data.origin_m
        xs = (occupied_indices[:, 0].float() + 0.5) * esdf_data.voxel_size_m + origin.x
        ys = (occupied_indices[:, 1].float() + 0.5) * esdf_data.voxel_size_m + origin.y
        zs = (occupied_indices[:, 2].float() + 0.5) * esdf_data.voxel_size_m + origin.z
        voxel_positions = torch.stack([xs, ys, zs], dim=-1)
        voxel_positions_np = voxel_positions.cpu().numpy()
        voxel_size = float(esdf_data.voxel_size_m)

        self._publish_marker(voxel_positions_np, voxel_size)

    def _publish_marker(self, voxel_positions, voxel_size):
        if len(voxel_positions) == 0:
            return

        marker = Marker()
        marker.header.frame_id = self.__robot_base_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        marker.type = 6
        marker.ns = 'curobo_world'
        marker.action = 0
        marker.pose.orientation.w = 1.0
        marker.lifetime = rclpy.duration.Duration(seconds=0.0).to_msg()
        marker.frame_locked = False
        marker.scale.x = voxel_size
        marker.scale.y = voxel_size
        marker.scale.z = voxel_size
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        for pos in voxel_positions:
            pt = Point()
            pt.x = float(pos[0])
            pt.y = float(pos[1])
            pt.z = float(pos[2])
            marker.points.append(pt)

        self.__voxel_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    esdf_client = ESDFVisualizer()
    try:
        esdf_client.get_logger().info('Starting ESDFVisualizer node')
        rclpy.spin(esdf_client)
    except KeyboardInterrupt:
        esdf_client.get_logger().info('Destroying ESDFVisualizer node')
    except Exception as e:
        esdf_client.get_logger().info(f'Shutting down due to exception of type {type(e)}: {e}')
    esdf_client.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()

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

"""Getting-started feature-mapping example: continuously fetches the voxel grid
from the get_voxel_grid service on curobo_server and renders coloured voxels in
a viser viewer. Colour is a height-based placeholder, ready to be replaced by
per-voxel semantic labels in a real feature-mapping pipeline."""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.task import Future
from sensor_msgs.msg import JointState

from isaac_ros_cumotion_interfaces.srv import GetVoxelGrid

from ._viser_helpers import (
    _viz_set_positions,
    active_joint_names_from_content,
    start_service_poll,
    viser_serve_forever,
)


class FeatureViserNode(Node):
    """Repolls the voxel grid from curobo_server and displays coloured voxels
    in viser. Colour is assigned based on voxel height (placeholder for
    per-voxel semantic features)."""

    def __init__(self):
        super().__init__('feature_viser_node')

        # Parameters
        self.declare_parameter('robot_config', 'franka.yml')
        self.declare_parameter('content_path', '')
        self.declare_parameter('urdf_path', '')
        self.declare_parameter('asset_path', '')
        self.declare_parameter('server_node', 'curobo_server')
        self.declare_parameter('viser_host', '0.0.0.0')
        self.declare_parameter('viser_port', 8080)
        self.declare_parameter('add_robot_to_scene', True)
        self.declare_parameter('add_control_frames', True)

        robot_config = self.get_parameter('robot_config').value
        content_path = self.get_parameter('content_path').value or robot_config
        urdf_path = self.get_parameter('urdf_path').value
        asset_path = self.get_parameter('asset_path').value
        server_node = self.get_parameter('server_node').value
        viser_host = self.get_parameter('viser_host').value
        viser_port = int(self.get_parameter('viser_port').value)
        add_robot = bool(self.get_parameter('add_robot_to_scene').value)
        add_frames = bool(self.get_parameter('add_control_frames').value)

        cp = None
        if add_robot and content_path and urdf_path:
            from ._viser_helpers import _build_robot_data_dict
            cp = _build_robot_data_dict(content_path, urdf_path, asset_path)

        from curobo.viewer import ViserVisualizer
        self._viz = ViserVisualizer(
            content_path=cp,
            connect_ip=viser_host,
            connect_port=viser_port,
            add_robot_to_scene=add_robot and cp is not None,
            add_control_frames=add_frames,
        )

        prefix = f'/{server_node}'
        self._voxel_client = self.create_client(
            GetVoxelGrid, f'{prefix}/get_voxel_grid'
        )

        self._js_sub = self.create_subscription(
            JointState, '/joint_states', self._on_js, 10
        )
        self._js = None

        self._point_cloud_handle = None
        self._ready = False
        self._busy = False

        self.create_timer(0.05, self._update_viser)
        self._fetch_timer = self.create_timer(1.0, self._fetch_tick)

        self.get_logger().info(
            'FeatureViserNode ready - waiting for voxel grid service'
        )
        start_service_poll(self, self._voxel_client, self._on_service_ready)

    def _on_service_ready(self, client):
        self._ready = True
        self.get_logger().info('GetVoxelGrid service is up')

    def _on_js(self, msg):
        self._js = msg

    def _update_viser(self):
        if self._js is not None:
            _viz_set_positions(self._viz, list(self._js.position), list(self._js.name))

    def _fetch_tick(self):
        if not self._ready or self._busy:
            return
        req = GetVoxelGrid.Request()
        req.bbox_min_x = -1.0
        req.bbox_min_y = -1.0
        req.bbox_min_z = -1.0
        req.bbox_max_x = 1.0
        req.bbox_max_y = 1.0
        req.bbox_max_z = 1.0

        self._busy = True
        fut = self._voxel_client.call_async(req)
        fut.add_done_callback(self._on_voxel_done)

    def _on_voxel_done(self, future: Future):
        self._busy = False
        resp = future.result()
        positions = self._parse_positions(resp.voxel_grid)
        self.get_logger().debug(
            f'Voxel grid refreshed - {len(positions)} occupied voxels'
        )
        colours = self._assign_feature_colours(positions)
        self._display_coloured_voxels(positions, colours)

    @staticmethod
    def _parse_positions(voxel_grid):
        nx = int(voxel_grid.size_x)
        ny = int(voxel_grid.size_y)
        nz = int(voxel_grid.size_z)

        data = np.asarray(voxel_grid.data, dtype=np.float64)
        occupied = np.flatnonzero(data)
        if occupied.size == 0:
            return np.empty((0, 3))

        rx = voxel_grid.resolutions.x
        ry = voxel_grid.resolutions.y
        rz = voxel_grid.resolutions.z
        ox = voxel_grid.origin.x
        oy = voxel_grid.origin.y
        oz = voxel_grid.origin.z

        idx = occupied
        z = idx % nz
        rem = idx // nz
        y = rem % ny
        x = rem // ny

        return np.stack(
            [ox + x * rx, oy + y * ry, oz + z * rz], axis=1
        ).astype(np.float32)

    @staticmethod
    def _assign_feature_colours(positions):
        """Height-based placeholder colour ramp: blue -> green -> red."""
        if len(positions) == 0:
            return np.empty((0, 3))

        zs = positions[:, 2]
        z_min, z_max = zs.min(), zs.max()
        span = z_max - z_min if z_max > z_min else 1.0
        t = (zs - z_min) / span

        colours = np.zeros((len(t), 3), dtype=np.uint8)
        colours[:, 0] = (t * 255).astype(np.uint8)
        colours[:, 1] = ((1.0 - np.abs(2.0 * t - 1.0)) * 255).astype(np.uint8)
        colours[:, 2] = ((1.0 - t) * 255).astype(np.uint8)
        return colours

    def _display_coloured_voxels(self, positions, colours):
        if self._point_cloud_handle is not None:
            self._point_cloud_handle.remove()
            self._point_cloud_handle = None

        if len(positions) == 0:
            return

        self._point_cloud_handle = self._viz._server.scene.add_point_cloud(
            name='feature_cloud',
            points=positions,
            colors=colours,
            point_size=0.012,
        )


def main(args=None):
    rclpy.init(args=args)
    node = FeatureViserNode()
    try:
        viser_serve_forever(node, node._viz, node.get_logger())
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

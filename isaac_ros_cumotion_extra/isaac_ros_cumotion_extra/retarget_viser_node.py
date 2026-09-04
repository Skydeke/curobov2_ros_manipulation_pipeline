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

"""Getting-started humanoid retargeting example: loads the robot skeleton and
plays back a synthetic sinusoidal joint trajectory through a viser viewer to
demonstrate the rendering pipeline for humanoid retargeting workflows
(placeholder - no actual retargeting solver is invoked)."""

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from ._viser_helpers import (
    _viz_set_positions,
    active_joint_names_from_content,
    viser_serve_forever,
)


class RetargetViserNode(Node):
    """Displays the robot skeleton and replays a synthetic joint-state
    trajectory through the viser viewer.

    This is a placeholder node for humanoid retargeting workflows. In a
    production pipeline the joint states would come from a retargeting
    solver; here we generate a sinusoidal motion on each joint for
    demonstration purposes."""

    def __init__(self):
        super().__init__('retarget_viser_node')

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

        # Active robot joints (kortex: joint_1..joint_7).
        self._joint_names = active_joint_names_from_content(cp) or [
            f'joint_{i + 1}' for i in range(7)
        ]
        self._num_joints = len(self._joint_names)

        self._js_sub = self.create_subscription(
            JointState, '/joint_states', self._on_js, 10
        )
        self._js = None

        # Playback timer (synthetic retarget trajectory at 10 Hz) and viser
        # update timer to keep the published robot pose in sync.
        self._playback_timer = self.create_timer(0.1, self._playback_tick)
        self._tick = 0
        self._t = 0.0

        self.create_timer(0.05, self._update_viser)

        self.get_logger().info(
            'RetargetViserNode ready - playing synthetic trajectory'
        )

    def _on_js(self, msg):
        self._js = msg

    def _update_viser(self):
        if self._js is not None:
            _viz_set_positions(self._viz, list(self._js.position), list(self._js.name))

    def _playback_tick(self):
        t = self._t
        self._t += 0.1
        self._tick += 1

        positions = []
        for i in range(self._num_joints):
            freq = 0.3 + 0.1 * i
            positions.append(0.2 * math.sin(freq * t + 0.4 * i))

        _viz_set_positions(self._viz, positions, self._joint_names)


def main(args=None):
    rclpy.init(args=args)
    node = RetargetViserNode()
    try:
        viser_serve_forever(node, node._viz, node.get_logger())
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

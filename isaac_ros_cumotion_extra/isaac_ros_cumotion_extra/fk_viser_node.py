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

"""Getting-started FK example with an interactive viser GUI.

Keeps the ROS2 service architecture (FK is computed on ``curobo_server`` via the
``Fk`` service) while adding per-joint GUI sliders so the user can manually
drive each joint angle. The resulting tool-frame poses are rendered as frames
in the viser viewer. A "Sweep" toggle re-enables the automatic demo sweep.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.task import Future
from sensor_msgs.msg import JointState

from isaac_ros_cumotion_interfaces.srv import Fk, WarmupFK

from ._viser_helpers import (
    _CuPoseCompat,
    _viz_set_positions,
    active_joint_names_from_content,
    start_service_poll,
    viser_serve_forever,
)


class FkViserNode(Node):
    """ROS2 FK with per-joint sliders in a viser viewer."""

    def __init__(self):
        super().__init__('fk_viser_node')

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
        self._warmup_fk_client = self.create_client(WarmupFK, f'{prefix}/warmup_fk')
        self._fk_client = self.create_client(Fk, f'{prefix}/fk')

        self._joint_names = active_joint_names_from_content(cp) or [
            f'joint_{i + 1}' for i in range(7)
        ]
        self._num_joints = len(self._joint_names)

        self._tool_frame_handles = []
        self._busy = False
        self._warmup_issued = False
        self._fk_ready = False

        self._sliders = {}
        self._joint_values = [0.0] * self._num_joints
        self._sweep_enabled = True
        self._sweep_t = 0.0
        self._updating_sliders = False
        self._setup_viser_ui()

        self._js = None
        self._js_sub = self.create_subscription(
            JointState, '/joint_states', self._on_js, 10
        )
        self._js_synced = False

        self._sweep_timer = self.create_timer(0.1, self._sweep_tick)

        self.get_logger().info('FkViserNode ready - waiting for FK service')
        start_service_poll(self, self._warmup_fk_client, self._on_service_ready)
        start_service_poll(self, self._fk_client, self._on_service_ready)

    def _setup_viser_ui(self):
        """Create one GUI slider per joint plus a Sweep toggle."""
        try:
            server = getattr(self._viz, '_server', None)
            if server is None:
                return
            for i, name in enumerate(self._joint_names):
                slider = server.gui.add_slider(
                    f'Joint {name}',
                    min=-3.14,
                    max=3.14,
                    step=0.01,
                    initial_value=0.0,
                )
                slider.on_update(lambda _e, idx=i: self._on_slider(idx))
                self._sliders[i] = slider
            self._sweep_btn = server.gui.add_button(
                'Sweep: ON', color='green'
            )
            self._sweep_btn.on_click(lambda _: self._toggle_sweep())
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'Could not set up viser GUI controls: {exc}')

    def _on_js(self, msg):
        self._js = msg
        if not self._js_synced:
            self._sync_from_joint_state()

    def _sync_from_joint_state(self):
        """Initialize the sliders from the live ``/joint_states``.

        The sliders start at 0.0 by default; pull the real joint values so the
        GUI reflects the actual robot pose on startup. This runs once (guarded
        by ``_js_synced``); afterwards the automatic sweep (via
        ``_update_sliders``) or manual slider drags own the values.
        """
        js = self._js
        if js is None or not js.name or len(js.position) != len(js.name):
            return
        js_map = dict(zip(js.name, js.position))
        if not any(n in js_map for n in self._joint_names):
            return
        values = [float(js_map.get(n, self._joint_values[i]))
                  for i, n in enumerate(self._joint_names)]
        self._joint_values = values
        self._js_synced = True
        self._update_sliders()
        _viz_set_positions(self._viz, values, self._joint_names)

    def _update_sliders(self):
        """Push the current joint values into the GUI sliders.

        Setting a slider's value programmatically re-triggers its
        ``on_update``; the ``_updating_sliders`` guard makes ``_on_slider``
        ignore those synthetic events so they never disable the sweep.
        """
        self._updating_sliders = True
        try:
            for i in range(self._num_joints):
                try:
                    self._sliders[i].value = self._joint_values[i]
                except Exception:  # noqa: BLE001
                    pass
        finally:
            self._updating_sliders = False

    def _on_slider(self, idx):
        """Read the slider value and apply it to the robot, disabling the
        automatic sweep so manual control isn't fought."""
        if self._updating_sliders:
            return
        try:
            val = self._sliders[idx].value
        except Exception:  # noqa: BLE001
            return
        if self._sweep_enabled:
            self._sweep_enabled = False
            self._set_sweep_button(False)
        self._joint_values[idx] = float(val)
        _viz_set_positions(self._viz, self._joint_values, self._joint_names)
        self._send_fk_request(self._joint_values)

    def _toggle_sweep(self):
        self._sweep_enabled = not self._sweep_enabled
        self._set_sweep_button(self._sweep_enabled)

    def _set_sweep_button(self, enabled):
        try:
            btn = getattr(self, '_sweep_btn', None)
            if btn is not None:
                btn.text = 'Sweep: ON' if enabled else 'Sweep: OFF'
                btn.color = 'green' if enabled else 'gray'
        except Exception:  # noqa: BLE001
            pass

    def _on_service_ready(self, client):
        if self._warmup_issued:
            return
        if not self._warmup_fk_client.service_is_ready():
            return
        self._warmup_issued = True

        req = WarmupFK.Request()
        req.batch_size = 1
        fut = self._warmup_fk_client.call_async(req)
        fut.add_done_callback(self._on_warmup_fk_done)

    def _on_warmup_fk_done(self, future: Future):
        resp = future.result()
        self.get_logger().info(
            f'Warmup FK: success={resp.success}, msg={resp.message}'
        )
        self._fk_ready = True

    def _sweep_tick(self):
        if not self._fk_ready or self._busy:
            return
        if not self._sweep_enabled:
            # Manual mode: don't override the slider-driven configuration.
            return
        t = self._sweep_t
        self._sweep_t += 0.1

        for i in range(self._num_joints):
            freq = 0.4 + 0.15 * i
            self._joint_values[i] = 0.6 * math.sin(freq * t + 0.4 * i)

        self._update_sliders()
        _viz_set_positions(self._viz, self._joint_values, self._joint_names)
        self._send_fk_request(self._joint_values)

    def _send_fk_request(self, positions):
        self._busy = True
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = list(self._joint_names)
        joint_state.position = list(positions)

        req = Fk.Request()
        req.joint_states = [joint_state]

        fut = self._fk_client.call_async(req)
        fut.add_done_callback(self._on_fk_done)

    def _on_fk_done(self, future: Future):
        self._busy = False
        resp = future.result()
        self._display_tool_frames(resp.poses)

    def _display_tool_frames(self, poses):
        for h in self._tool_frame_handles:
            h.remove()
        self._tool_frame_handles.clear()

        for i, pose in enumerate(poses):
            comp = _CuPoseCompat.from_list([
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.w,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
            ])
            handle = self._viz.add_frame(
                f'fk_frame_{i}', comp, scale=0.12
            )
            self._tool_frame_handles.append(handle)

        self.get_logger().info(f'Displayed {len(poses)} tool frames in viser')


def main(args=None):
    rclpy.init(args=args)
    node = FkViserNode()
    try:
        viser_serve_forever(node, node._viz, node.get_logger())
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()


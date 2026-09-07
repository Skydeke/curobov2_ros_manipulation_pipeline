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

"""Getting-started reactive / MPC control example.

Mirrors cuRobo's ``reactive_control.py`` example: the end-effector gizmo is
dragged and the MPC controller tracks it continuously. Every solver call runs
on ``curobo_server`` through ROS services only (never an in-node MotionPlanner):

- ``SetPlanner`` selects the MPC (reactive) controller,
- ``SendTrajectory`` action starts the continuous servo loop (goal = gizmo),
- a ``Pose`` topic supplies live goal updates as the gizmo is dragged,
- the server's joint commands stream back through the action feedback and are
  rendered in viser.
"""

import math
import numpy as np

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.task import Future
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

from isaac_ros_cumotion_interfaces.action import SendTrajectory
from isaac_ros_cumotion_interfaces.srv import SetPlanner
from isaac_ros_cumotion_interfaces.msg import Goalset

from ._viser_helpers import (
    _start_positions,
    _viz_set_positions,
    active_joint_names_from_content,
    start_service_poll,
    viser_serve_forever,
)


class MpcViserNode(Node):
    """Reactive-MPC viser node with a live-tracking goal gizmo.

    The target is set by dragging the control frame. A ``SetPlanner`` call
    selects the MPC (reactive) controller, a ``SendTrajectory`` action starts
    the continuous servo loop, and ``Pose`` messages on the mpc_goal topic
    retarget it live. The server's joint commands (action feedback) are rendered.
    """

    def __init__(self):
        super().__init__('mpc_viser_node')

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
        self._content = cp

        from curobo.viewer import ViserVisualizer
        self._viz = ViserVisualizer(
            content_path=cp,
            connect_ip=viser_host,
            connect_port=viser_port,
            add_robot_to_scene=add_robot and cp is not None,
            add_control_frames=add_frames,
        )

        prefix = f'/{server_node}'
        self._set_planner_client = self.create_client(
            SetPlanner, f'{prefix}/set_planner'
        )
        self._action_client = ActionClient(self, SendTrajectory, f'{prefix}/execute_trajectory')

        # Live MPC goal: publishing a Pose retargets the active reactive loop.
        from geometry_msgs.msg import Pose as PoseMsg
        self._goal_pub = self.create_publisher(PoseMsg, f'{prefix}/mpc_goal', 10)

        self._js_sub = self.create_subscription(
            JointState, '/joint_states', self._on_js, 10
        )
        self._js = None

        self._joint_names = active_joint_names_from_content(cp) or [
            f'joint_{i + 1}' for i in range(7)
        ]

        self._goal_center = np.array([0.4, 0.0, 0.3])
        self._goal_radius = 0.08
        self._last_goal_key = None
        self._servo_active = False
        self._ready = False
        self._t = 0.0

        self._setup_viser_ui()

        self.create_timer(0.05, self._update_viser)
        self.create_timer(0.1, self._watch_goal)

        self.get_logger().info(
            'MpcViserNode ready - MPC closed loop (waiting for action server)'
        )
        start_service_poll(self, self._set_planner_client, self._on_service_ready)

    def _setup_viser_ui(self):
        """Create the Start/Stop buttons and a status text.

        The draggable goal gizmo is created automatically by the visualizer for
        each tool frame (``add_control_frames=True``).
        """
        try:
            server = getattr(self._viz, '_server', None)
            if server is not None:
                self._gui_start_btn = server.gui.add_button(
                    'Start', color='green'
                )
                self._gui_start_btn.on_click(lambda _: self._start_servo())
                self._gui_stop_btn = server.gui.add_button(
                    'Stop', color='red'
                )
                self._gui_stop_btn.on_click(lambda _: self._stop_servo())
                self._gui_status = server.gui.add_text('Idle')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'Could not set up viser GUI controls: {exc}')

    def _set_status(self, text, ok=None):
        try:
            if getattr(self, '_gui_status', None) is None:
                return
            self._gui_status.text = text
            if ok is not None:
                self._gui_status.color = 'green' if ok else 'red'
        except Exception:  # noqa: BLE001
            pass

    def _on_service_ready(self, client):
        self._ready = True
        self.get_logger().info('SetPlanner service is up, selecting MPC planner')

        req = SetPlanner.Request()
        req.planner_type = SetPlanner.Request.MPC
        fut = self._set_planner_client.call_async(req)
        fut.add_done_callback(self._on_planner_switched)

    def _on_planner_switched(self, future: Future):
        try:
            resp = future.result()
            if not resp.success:
                self._set_status('MPC switch failed', ok=False)
                self.get_logger().error(f'MPC planner switch failed: {resp.message}')
                return
            self.get_logger().info(
                f'MPC planner active (was {resp.previous_planner})'
            )
            self._set_status('MPC ready - press Start', ok=True)
        except Exception as e:  # noqa: BLE001
            self._set_status('MPC switch error', ok=False)
            self.get_logger().error(f'MPC planner switch error: {e}')

    def _on_js(self, msg):
        self._js = msg

    def _update_viser(self):
        if self._servo_active:
            # During closed-loop execution the MPC feedback (joint_command)
            # owns the visualization; don't fight it with the measured state.
            return
        if self._js is not None and self._js.name and len(self._js.position) == len(self._js.name):
            js_map = dict(zip(self._js.name, self._js.position))
            if all(n in js_map for n in self._joint_names):
                _viz_set_positions(self._viz, [js_map[n] for n in self._joint_names], list(self._joint_names))

    def _current_goal(self):
        """Return (pos[3], quat[4]) of the gizmo, else a slow moving circle.

        The draggable control frame is created automatically by the visualizer
        for each tool frame; reuse the first one.
        """
        try:
            poses = self._viz.get_control_frame_pose()
        except Exception:  # noqa: BLE001
            poses = None
        if poses:
            frame = next(iter(poses.values()))
            pos = frame.position.cpu().squeeze().numpy()
            quat = frame.quaternion.cpu().squeeze().numpy()
            return pos[:3], quat[:4]

        t = self._t
        pos = np.array([
            self._goal_center[0] + self._goal_radius * math.sin(t),
            self._goal_center[1] + self._goal_radius * math.cos(t),
            self._goal_center[2],
        ])
        return pos, np.array([1.0, 0.0, 0.0, 0.0])

    def _watch_goal(self):
        """Republish the gizmo pose to the MPC live-goal topic as it moves."""
        self._t += 0.1
        if not self._servo_active:
            return
        pos, quat = self._current_goal()
        key = tuple(round(float(v), 4) for v in pos) + tuple(round(float(v), 4) for v in quat)
        if self._last_goal_key == key:
            return
        self._last_goal_key = key
        msg = Pose()
        msg.position.x = float(pos[0])
        msg.position.y = float(pos[1])
        msg.position.z = float(pos[2])
        msg.orientation.w = float(quat[0])
        msg.orientation.x = float(quat[1])
        msg.orientation.y = float(quat[2])
        msg.orientation.z = float(quat[3])
        self._goal_pub.publish(msg)

    def _start_servo(self):
        """Start the reactive (MPC) servo loop on the server via the action."""
        if self._servo_active or not self._ready:
            return
        if not self._action_client.wait_for_server(timeout_sec=2.0):
            self._set_status('Action server not available', ok=False)
            self.get_logger().error('execute_trajectory action server not available')
            return
        pos, quat = self._current_goal()

        goal = SendTrajectory.Goal()
        start_js = JointState()
        start_js.header.stamp = self.get_clock().now().to_msg()
        start_js.name = list(self._joint_names)
        start_js.position = _start_positions(self._content, self._joint_names, self._js)
        goal.start_pose = start_js
        p = Pose()
        p.position.x = float(pos[0])
        p.position.y = float(pos[1])
        p.position.z = float(pos[2])
        p.orientation.w = float(quat[0])
        p.orientation.x = float(quat[1])
        p.orientation.y = float(quat[2])
        p.orientation.z = float(quat[3])
        goal.goalsets = [Goalset(poses=[p])]
        goal.allow_cached = False

        self._set_status('Starting MPC...')
        self._servo_active = True
        self._last_goal_key = None
        send_goal_future = self._action_client.send_goal_async(
            goal, feedback_callback=self._on_feedback
        )
        send_goal_future.add_done_callback(self._on_goal_accepted)

    def _on_goal_accepted(self, future: Future):
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self._servo_active = False
            self._set_status('MPC goal rejected', ok=False)
            self.get_logger().error('MPC execution goal was rejected')
            return
        self.get_logger().info('MPC execution goal accepted')

    def _stop_servo(self):
        if not self._servo_active:
            return
        self._servo_active = False
        if self._action_client.server_is_available():
            self.get_logger().info('Cancelling MPC execution goal')
        self._set_status('Stopped')

    def _on_feedback(self, feedback_msg):
        fb = feedback_msg.feedback
        if not self._servo_active:
            return
        state = getattr(fb, 'state', '')
        on_target = bool(getattr(fb, 'on_target', False))
        pos_err = getattr(fb, 'position_error', 0.0)
        self._set_status(
            f'{state}  |  err {pos_err:.3f} m' + ('  (on target)' if on_target else ''),
            ok=on_target,
        )
        cmd = getattr(fb, 'joint_command', None)
        if cmd is not None and cmd.name and len(cmd.position) == len(cmd.name):
            pos_map = dict(zip(cmd.name, cmd.position))
            if all(n in pos_map for n in self._joint_names):
                _viz_set_positions(self._viz, [pos_map[n] for n in self._joint_names], list(self._joint_names))


def main(args=None):
    rclpy.init(args=args)
    node = MpcViserNode()
    try:
        viser_serve_forever(node, node._viz, node.get_logger())
    finally:
        node._stop_servo()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

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

"""Interactive motion-planning viser node (mirrors cuRobo ``motion_planning.py``).

The GUI matches the upstream example -- draggable goal gizmo plus two buttons
-- and every plan is computed on ``curobo_server`` via ROS services only (never
an in-node MotionPlanner):

- "Move" (green)  -> Classic single-pose plan toward the goal gizmo,
- "Grasp" (blue)  -> Classic approach->grasp->lift, three single-pose plans
  chained and animated in sequence toward the gizmo.

The returned trajectory is animated in the viewer; the joint-trajectory plot
image in the GUI panel is NOT computed locally — the node enables the server's
``publish_plan_debug_image`` param and displays the latched ``/<node>/motion_plan_debug``
image the server publishes per plan.
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.task import Future
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image as ImageMsg
from sensor_msgs.msg import JointState

from isaac_ros_cumotion_interfaces.srv import SetPlanner, TrajectoryGeneration
from isaac_ros_cumotion_interfaces.msg import Goalset

from ._viser_helpers import (
    _start_positions,
    _viz_set_positions,
    active_joint_names_from_content,
    start_service_poll,
    viser_serve_forever,
)


class MpViserNode(Node):
    """ROS2 motion planning with an interactive viser GUI.

    The user drags the goal control frame in viser and presses "Plan"; the goal
    is sent to ``curobo_server`` via ``TrajectoryGeneration`` and the resulting
    path is animated in the viewer and plotted in the GUI panel.
    """

    def __init__(self):
        super().__init__('mp_viser_node')

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
        self._traj_client = self.create_client(
            TrajectoryGeneration, f'{prefix}/generate_trajectory'
        )
        self._set_planner_client = self.create_client(
            SetPlanner, f'{prefix}/set_planner'
        )
        self._set_params_client = self.create_client(
            SetParameters, f'{prefix}/set_parameters'
        )

        # The server publishes the plan plot on /<node>/motion_plan_debug with a
        # transient_local (latched) depth-1 QoS, so a matching latched subscription
        # keeps the GUI panel in sync without consuming extra bandwidth.
        latched = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._plot_image_sub = self.create_subscription(
            ImageMsg, f'{prefix}/motion_plan_debug', self._on_plan_image, latched
        )

        self._js_sub = self.create_subscription(
            JointState, '/joint_states', self._on_js, 10
        )
        self._js = None

        self._joint_names = active_joint_names_from_content(cp) or [
            f'joint_{i + 1}' for i in range(7)
        ]

        self._waypoint_handles = []
        self._busy = False
        self._ready = False

        # Interactive goal control frame + GUI (mirrors cuRobo motion_planning.py)
        self._traj_plot_handle = None
        self._default_goal = np.array([0.4, 0.2, 0.3])
        self._mode = 'Move'
        self._grasp_queue = []
        self._setup_viser_ui()

        self.create_timer(0.05, self._update_viser)

        self.get_logger().info('MpViserNode ready - waiting for trajectory service')
        start_service_poll(self, self._traj_client, self._on_service_ready)
        start_service_poll(self, self._set_params_client, self._on_params_ready)

    # ------------------------------------------------------------------
    # Viser UI
    # ------------------------------------------------------------------

    def _setup_viser_ui(self):
        """Create the draggable goal gizmo, Move/Grasp buttons, and the
        trajectory plot image panel.

        Mirrors the upstream cuRobo ``motion_planning.py`` GUI exactly, but all
        planning goes through ROS services on ``curobo_server`` (never an in-node
        MotionPlanner): "Move" = Classic single-pose plan, "Grasp" =
        approach->grasp->lift as three chained Classic single-pose plans.
        """
        try:
            server = getattr(self._viz, '_server', None)
            if server is not None:
                self._gui_move_btn = server.gui.add_button(
                    'Move', color='green'
                )
                self._gui_move_btn.on_click(lambda _: self._on_move())
                self._gui_grasp_btn = server.gui.add_button(
                    'Grasp', color='blue'
                )
                self._gui_grasp_btn.on_click(lambda _: self._on_grasp())
                self._gui_status = server.gui.add_text('Idle')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'Could not set up viser GUI controls: {exc}')

    # ------------------------------------------------------------------
    # Move / Grasp (upstream UI semantics, via ROS services only)
    # ------------------------------------------------------------------

    def _read_goal_pose(self):
        """Read the current draggable goal gizmo pose (position + wxyz quat).

        The draggable control frames are created automatically by the
        visualizer for each tool frame (``add_control_frames=True``); reuse the
        first one rather than adding a custom frame that
        ``get_control_frame_pose()`` would never return.
        """
        try:
            poses = self._viz.get_control_frame_pose()
        except Exception:
            poses = None
        if poses:
            frame = next(iter(poses.values()))
            pos = frame.position.cpu().squeeze().numpy()
            quat = frame.quaternion.cpu().squeeze().numpy()
            return pos[:3], quat[:4]
        return self._default_goal, np.array([1.0, 0.0, 0.0, 0.0])

    def _on_move(self):
        """'Move' -> Classic single-pose plan toward the goal gizmo."""
        if not self._ready or self._busy:
            return
        pos, quat = self._read_goal_pose()
        self._mode = 'Move'
        self._switch_and_plan(
            SetPlanner.Request.CLASSIC,
            self._build_move_request(pos, quat),
        )

    def _on_grasp(self):
        """'Grasp' -> Classic sequential approach->grasp->lift toward the gizmo.

        Each waypoint is a separate single-goalset Classic plan; the three are
        queued and sent one after another (plan + animate each in turn), which
        replaces the removed MultiPoint planner with identical waypoint
        semantics.
        """
        if not self._ready or self._busy:
            return
        pos, quat = self._read_goal_pose()
        self._mode = 'Grasp'
        self._grasp_queue = list(self._build_grasp_requests(pos, quat))
        if not self._grasp_queue:
            return
        self._switch_and_plan(
            SetPlanner.Request.CLASSIC,
            self._grasp_queue.pop(0),
        )

    @staticmethod
    def _quat_z_axis(quat):
        """Return the tool-frame Z axis (right-handed) from a wxyz quaternion."""
        w, x, y, z = (float(v) for v in quat)
        return np.array([
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ])

    def _build_pose(self, pos, quat):
        msg = Pose()
        msg.position.x = float(pos[0])
        msg.position.y = float(pos[1])
        msg.position.z = float(pos[2])
        msg.orientation.w = float(quat[0])
        msg.orientation.x = float(quat[1])
        msg.orientation.y = float(quat[2])
        msg.orientation.z = float(quat[3])
        return msg

    def _build_start(self):
        start_js = JointState()
        start_js.header.stamp = self.get_clock().now().to_msg()
        start_js.name = list(self._joint_names)
        start_js.position = _start_positions(self._content, self._joint_names, self._js)
        return start_js

    def _build_move_request(self, pos, quat):
        def build():
            req = TrajectoryGeneration.Request()
            req.start_pose = self._build_start()
            req.goalsets = [Goalset(poses=[self._build_pose(pos, quat)])]
            return req
        return build

    def _build_grasp_requests(self, pos, quat, approach_offset=0.1, lift_offset=0.1):
        """Generate the approach->grasp->lift waypoint request list.

        Mirrors upstream ``plan_grasp`` (approach offset back along tool -Z,
        grasp at the goal, lift along tool +Z): each waypoint becomes one
        single-goalset Classic request, planned and animated in sequence.
        """
        z = self._quat_z_axis(quat)
        approach = np.asarray(pos) - approach_offset * z
        lift = np.asarray(pos) + lift_offset * z

        def make(target):
            def build():
                req = TrajectoryGeneration.Request()
                req.start_pose = self._build_start()
                req.goalsets = [Goalset(poses=[self._build_pose(target, quat)])]
                return req
            return build

        return [make(approach), make(pos), make(lift)]

    def _switch_and_plan(self, planner_type, build_req):
        """Switch the server planner, then send the plan once the switch lands."""
        if self._busy:
            return
        self._busy = True
        self._pending_build = build_req
        self._set_status('Switching planner...')
        req = SetPlanner.Request()
        req.planner_type = planner_type
        fut = self._set_planner_client.call_async(req)
        fut.add_done_callback(self._on_planner_switched)

    # ------------------------------------------------------------------
    # ROS plumbing
    # ------------------------------------------------------------------

    def _on_service_ready(self, client):
        self._ready = True
        self.get_logger().info('TrajectoryGeneration service is up')

    def _on_params_ready(self, client):
        """Enable the server's ``publish_plan_debug_image`` param so it publishes
        the per-plan joint-trajectory plot on ``/<node>/motion_plan_debug`` for
        the GUI panel (no local re-compute of the plot)."""
        req = SetParameters.Request()
        param = Parameter()
        param.name = 'publish_plan_debug_image'
        param.value = ParameterValue(
            type=ParameterType.PARAMETER_BOOL, bool_value=True
        )
        req.parameters = [param]
        fut = client.call_async(req)
        fut.add_done_callback(self._on_params_set)

    def _on_params_set(self, future: Future):
        try:
            resp = future.result()
            ok = bool(resp.results and resp.results[0].successful)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Could not enable plan debug image: {exc}')
            return
        if ok:
            self.get_logger().info('publish_plan_debug_image = true on server')
        else:
            self.get_logger().warn('Server rejected publish_plan_debug_image')

    def _on_planner_switched(self, future: Future):
        build = getattr(self, '_pending_build', None)
        self._pending_build = None
        try:
            resp = future.result()
            if not resp.success:
                self._busy = False
                self._set_status('Planner switch failed')
                self.get_logger().error(f'Planner switch failed: {resp.message}')
                return
            self.get_logger().info(
                f'Planner active: {resp.current_planner} (was {resp.previous_planner})'
            )
        except Exception as e:  # noqa: BLE001
            self._busy = False
            self._set_status('Planner switch error')
            self.get_logger().error(f'Planner switch error: {e}')
            return

        if build is None:
            self._busy = False
            return

        req = build()
        self._set_status('Planning...')
        fut = self._traj_client.call_async(req)
        fut.add_done_callback(self._on_trajectory_done)

    def _set_status(self, text):
        try:
            if getattr(self, '_gui_status', None) is not None:
                self._gui_status.text = text
        except Exception:  # noqa: BLE001
            pass

    def _on_js(self, msg):
        self._js = msg

    def _update_viser(self):
        if self._js is not None:
            _viz_set_positions(self._viz, list(self._js.position), list(self._js.name))

    # ------------------------------------------------------------------
    # Trajectory rendering
    # ------------------------------------------------------------------

    def _on_trajectory_done(self, future: Future):
        resp = future.result()
        if not resp.success:
            self._busy = False
            self._grasp_queue = []
            self._set_status('Plan failed')
            return
        trajectory = resp.trajectory
        self.get_logger().debug(f'{self._mode} plan - {len(trajectory)} waypoints, dt={resp.dt}')
        self._set_status(f'{self._mode} OK - {len(trajectory)} waypoints')
        self._animate_trajectory(trajectory)
        if self._mode == 'Grasp' and self._grasp_queue:
            # Chain the next waypoint of the approach->grasp->lift sequence.
            next_build = self._grasp_queue.pop(0)
            self._busy = True
            self._set_status('Planning...')
            req = next_build()
            fut = self._traj_client.call_async(req)
            fut.add_done_callback(self._on_trajectory_done)
            return
        self._busy = False

    def _animate_trajectory(self, waypoints):
        """Move the viser robot through the returned joint waypoints."""
        import time

        for h in self._waypoint_handles:
            h.remove()
        self._waypoint_handles.clear()

        n = len(waypoints)
        for i in range(0, n, max(1, n // 40)):
            js = waypoints[i]
            names = list(self._joint_names)
            pos = list(js.position)[: len(names)]
            _viz_set_positions(self._viz, pos, names)
            time.sleep(0.01)
        if n:
            js = waypoints[-1]
            _viz_set_positions(
                self._viz, list(js.position)[: len(self._joint_names)], self._joint_names
            )

    def _on_plan_image(self, msg: ImageMsg):
        """Display the server-published plan plot in the GUI panel.

        The server renders the joint-trajectory plot from the actual cuRobo
        trajectory (``/<node>/motion_plan_debug``, latched rgb8); this node
        only decodes and shows it -- no local re-compute.
        """
        try:
            if msg.encoding not in ('rgb8', 'bgr8'):
                self.get_logger().debug(
                    f'Unsupported plan-image encoding: {msg.encoding}'
                )
                return
            if not msg.height or not msg.width or not msg.data:
                return
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, int(msg.step // msg.width)
            )
            if img.shape[2] < 3:
                return
            img = img[:, :, :3]
            if msg.encoding == 'bgr8':
                img = img[:, :, ::-1].copy()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'Could not decode plan image: {exc}')
            return
        self._render_plan_image(img)

    def _render_plan_image(self, img):
        """Put the decoded plan plot image into the viser GUI panel."""
        server = getattr(self._viz, '_server', None)
        if server is None:
            return
        try:
            if self._traj_plot_handle is not None:
                self._traj_plot_handle.remove()  # type: ignore[attr-defined]
                self._traj_plot_handle = None
            self._traj_plot_handle = server.gui.add_image(
                img, label='Joint trajectory', format='png'
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'Could not render plan image: {exc}')


def main(args=None):
    rclpy.init(args=args)
    node = MpViserNode()
    try:
        viser_serve_forever(node, node._viz, node.get_logger())
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

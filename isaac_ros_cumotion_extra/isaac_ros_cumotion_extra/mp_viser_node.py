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
- "Grasp" (blue)  -> MultiPoint approach->grasp->lift plan toward the gizmo.

The returned trajectory is animated in the viewer and rendered as a
joint-trajectory plot in the GUI panel.
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.task import Future
from geometry_msgs.msg import Pose
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
        self._setup_viser_ui()

        self.create_timer(0.05, self._update_viser)

        self.get_logger().info('MpViserNode ready - waiting for trajectory service')
        start_service_poll(self, self._traj_client, self._on_service_ready)

    # ------------------------------------------------------------------
    # Viser UI
    # ------------------------------------------------------------------

    def _setup_viser_ui(self):
        """Create the draggable goal gizmo, Move/Grasp buttons, and the
        trajectory plot image panel.

        Mirrors the upstream cuRobo ``motion_planning.py`` GUI exactly, but all
        planning goes through ROS services on ``curobo_server`` (never an in-node
        MotionPlanner): "Move" = Classic single-pose plan, "Grasp" = MultiPoint
        approach->grasp->lift plan.
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
        """'Grasp' -> MultiPoint approach->grasp->lift plan toward the gizmo."""
        if not self._ready or self._busy:
            return
        pos, quat = self._read_goal_pose()
        self._mode = 'Grasp'
        self._switch_and_plan(
            SetPlanner.Request.MULTIPOINT,
            self._build_grasp_request(pos, quat),
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

    def _build_grasp_request(self, pos, quat, approach_offset=0.1, lift_offset=0.1):
        """Synthesize an approach->grasp->lift MultiPoint request from the gizmo.

        Mirrors upstream ``plan_grasp`` (approach offset back along tool -Z,
        grasp at the goal, lift along tool +Z) as three sequential waypoints
        consumed by the server's MultiPointPlanner.
        """
        z = self._quat_z_axis(quat)
        approach = np.asarray(pos) - approach_offset * z
        lift = np.asarray(pos) + lift_offset * z

        def build():
            req = TrajectoryGeneration.Request()
            req.start_pose = self._build_start()
            req.goalsets = [
                Goalset(poses=[self._build_pose(approach, quat)]),
                Goalset(poses=[self._build_pose(pos, quat)]),
                Goalset(poses=[self._build_pose(lift, quat)]),
            ]
            return req
        return build

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
        self._busy = False
        resp = future.result()
        if not resp.success:
            self._set_status('Plan failed')
            return
        trajectory = resp.trajectory
        self.get_logger().debug(f'{self._mode} plan - {len(trajectory)} waypoints, dt={resp.dt}')
        self._set_status(f'{self._mode} OK - {len(trajectory)} waypoints')
        self._animate_trajectory(trajectory)
        title = (f'{self._mode} Plan  |  {len(trajectory) * resp.dt:.2f}s   '
                 f'({len(trajectory)} waypoints)')
        self._plot_trajectory(trajectory, resp.dt, title)

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

    def _plot_trajectory(self, waypoints, dt, title=''):
        """Render a joint trajectory plot as a PNG image in the viser GUI.

        Mirrors cuRobo's ``motion_planning.py`` plot: stacked Position /
        Velocity / Accel / Jerk panels using the real joint names, shown in a
        GUI image panel.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import io
            from PIL import Image
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'Could not import plotting: {exc}')
            return

        n = len(waypoints)
        if not n:
            return
        dof = len(self._joint_names)
        t = np.arange(n) * dt

        pos = np.zeros((n, dof))
        vel = np.zeros((n, dof))
        acc = np.zeros((n, dof))
        jrk = np.zeros((n, dof))
        for i, js in enumerate(waypoints):
            p = np.asarray(js.position)
            v = np.asarray(js.velocity) if len(js.velocity) else np.zeros(len(p))
            pos[i, : min(dof, len(p))] = p[:dof]
            vel[i, : min(dof, len(v))] = v[:dof]

        d = max(dt, 1e-6)
        if not np.any(vel):
            vel = np.diff(pos, axis=0, prepend=pos[:1]) / d
        acc = np.diff(vel, axis=0, prepend=vel[:1]) / d
        jrk = np.diff(acc, axis=0, prepend=acc[:1]) / d

        names = self._joint_names
        plot_data = [
            (pos, 'Position (rad)'),
            (vel, 'Velocity (rad/s)'),
            (acc, 'Accel (rad/s^2)'),
            (jrk, 'Jerk (rad/s^3)'),
        ]

        n_plots = len(plot_data)
        fig, axes = plt.subplots(n_plots, 1, figsize=(6, 2 * n_plots), dpi=100, sharex=True)
        if n_plots == 1:
            axes = [axes]

        for ax, (data, ylabel) in zip(axes, plot_data):
            for j in range(dof):
                label = names[j] if j < len(names) else f'J{j}'
                if len(label) > 10:
                    label = label[:8] + '..'
                ax.plot(t, data[:, j], linewidth=1.2, label=label)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

        axes[0].legend(loc='upper right', fontsize=7, ncol=2)
        axes[-1].set_xlabel('Time (s)', fontsize=9)
        if title:
            fig.suptitle(title, fontsize=11, fontweight='bold')
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img = np.array(Image.open(buf).convert('RGB'))

        server = getattr(self._viz, '_server', None)
        if server is None:
            return
        if getattr(self, '_traj_plot_handle', None) is not None:
            try:
                self._traj_plot_handle.remove()
            except Exception:  # noqa: BLE001
                pass
        self._traj_plot_handle = server.gui.add_image(
            img, label='Joint trajectory', format='png'
        )


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

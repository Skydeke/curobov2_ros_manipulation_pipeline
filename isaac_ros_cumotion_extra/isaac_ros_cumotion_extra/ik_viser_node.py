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

"""Interactive IK viser node.

Keeps the ROS2 service architecture (IK is computed on ``curobo_server`` via the
``Ik`` service) while adding an interactive viser GUI on top:

- a draggable 6-DOF control frame on the tool link sets the IK target,
- dragging it re-solves IK continuously (mirroring curobo_core's interactive
  examples), and the solved configuration is rendered on the robot,
- optional reachability mode: a draggable slice gizmo with a green/red heatmap
  showing which workspace positions are IK-solvable.
"""

import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.task import Future
from geometry_msgs.msg import Pose

from isaac_ros_cumotion_interfaces.srv import Ik, IkBatch, WarmupIK

from ._viser_helpers import (
    _viz_set_positions,
    active_joint_names_from_content,
    start_service_poll,
    viser_serve_forever,
)

_GRID_BATCH = 500  # ~22x22 grid


class IkViserNode(Node):
    """Interactive IK with a draggable goal gizmo.

    Scenes are solved on ``curobo_server``; the node drives the gizmo and maps
    the solved joint configurations onto the viser robot.
    """

    def __init__(self):
        super().__init__('ik_viser_node')

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
        self._warmup_ik_client = self.create_client(WarmupIK, f'{prefix}/warmup_ik')
        self._ik_client = self.create_client(Ik, f'{prefix}/ik')
        self._ik_batch_client = self.create_client(IkBatch, f'{prefix}/ik_batch')

        self._joint_names = active_joint_names_from_content(cp) or [
            f'joint_{i + 1}' for i in range(7)
        ]

        self._busy = False
        self._warmup_issued = False
        self._ik_ready = False

        # Default target in front of the arm, tool pitched forward to clear the
        # wrist self-collision.
        self._default_goal = [0.4, 0.0, 0.4]
        self._tool_pitch = -55.0
        self._last_solve_pose = None

        # Reachability mode state
        self._reachability_enabled = False
        self._reachability_busy = False
        self._reachability_needs_update = True
        self._reachability_pending = False
        self._reachability_gen = 0
        self._reachability_settle_tol = 2e-3
        self._reachability_settle_time = 0.45
        self._reachability_last_moved = time.monotonic()
        self._reachability_prev_pos = None
        self._reachability_prev_wxyz = None
        self._reachability_prev_extent = None
        self._prev_gizmo_pos = None
        self._prev_gizmo_wxyz = None
        self._prev_grid_extent = None
        self._gui_reachability_cb = None
        self._gui_grid_extent_slider = None
        self._gui_reachability_slice = None
        self._gui_reachability_bounds = None
        self._n_per_axis = int(_GRID_BATCH ** 0.5)

        self._setup_viser_ui()

        # Re-solve whenever the gizmo moves.
        self._solve_timer = self.create_timer(0.1, self._solve_tick)
        # Reachability heatmap timer.
        self._reachability_timer = self.create_timer(0.2, self._reachability_tick)

        self.get_logger().info('IkViserNode ready - waiting for IK service')
        start_service_poll(self, self._warmup_ik_client, self._on_service_ready)
        start_service_poll(self, self._ik_client, self._on_service_ready)
        start_service_poll(self, self._ik_batch_client, self._on_service_ready)

    def _setup_viser_ui(self):
        """Create the interactive IK gizmo, status text, and reachability controls."""
        try:
            server = getattr(self._viz, '_server', None)
            if server is not None:
                self._gui_status = server.gui.add_text('status', 'Idle')
                self._gui_reachability_cb = server.gui.add_checkbox(
                    'reachability', initial_value=False,
                )
                self._gui_reachability_cb.on_update(self._on_reachability_toggle)
                self._gui_grid_extent_slider = server.gui.add_slider(
                    'grid_extent_m',
                    min=0.1, max=2.0, step=0.05, initial_value=1.0,
                )
                self._gui_grid_extent_slider.on_update(
                    lambda _: setattr(self, '_reachability_needs_update', True),
                )
                self._gui_grid_extent_slider.visible = False
                self.get_logger().info('Viser GUI controls created successfully')
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.get_logger().error(f'Could not set up viser GUI controls: {exc}\n{traceback.format_exc()}')

    def _on_reachability_toggle(self, event):
        value = bool(event.target.value)
        self.get_logger().info(f'Reachability toggle: {value}')
        self._reachability_enabled = value
        self._reachability_needs_update = True
        # Clear previous solve/settle state so the first solve runs promptly.
        self._prev_gizmo_pos = None
        self._prev_gizmo_wxyz = None
        self._prev_grid_extent = None
        self._reachability_prev_pos = None
        self._reachability_prev_wxyz = None
        self._reachability_prev_extent = None
        self._reachability_last_moved = time.monotonic()
        if self._gui_grid_extent_slider is not None:
            self._gui_grid_extent_slider.visible = value
        if not self._reachability_enabled:
            self._clear_reachability_viz()

    def _clear_reachability_viz(self):
        server = getattr(self._viz, '_server', None)
        if server is None:
            return
        if self._gui_reachability_slice is not None:
            try:
                server.scene.remove_node(self._gui_reachability_slice)
            except Exception:
                pass
            self._gui_reachability_slice = None
        if self._gui_reachability_bounds is not None:
            try:
                server.scene.remove_node(self._gui_reachability_bounds)
            except Exception:
                pass
            self._gui_reachability_bounds = None

    @staticmethod
    def _tilt_quat(pitch_deg: float) -> list:
        """Quaternion (w, x, y, z) for a rotation about Y by ``pitch_deg``."""
        half = math.radians(pitch_deg) / 2.0
        return [math.cos(half), 0.0, math.sin(half), 0.0]

    def _blacken_reachability_image(self, pos, wxyz, extent):
        """Show a black slice plane while a reachability solve is in progress."""
        server = getattr(self._viz, '_server', None)
        if server is None:
            return
        n = self._n_per_axis
        black = np.zeros((n, n, 3), dtype=np.uint8)
        if self._gui_reachability_slice is not None:
            self._gui_reachability_slice.position = tuple(pos)
            self._gui_reachability_slice.wxyz = tuple(wxyz)
            self._gui_reachability_slice.image = black
            return
        self._gui_reachability_slice = server.scene.add_image(
            name='/reachability_bounds/slice_image',
            image=black,
            render_width=extent,
            render_height=extent,
            wxyz=tuple(wxyz),
            position=tuple(pos),
        )

    def _set_status(self, text, ok=None):
        try:
            if getattr(self, '_gui_status', None) is None:
                return
            if hasattr(self._gui_status, 'value'):
                self._gui_status.value = text
            else:
                self._gui_status.text = text
            if ok is not None:
                if hasattr(self._gui_status, 'color'):
                    self._gui_status.color = 'green' if ok else 'red'
        except Exception:  # noqa: BLE001
            pass

    def _on_service_ready(self, client):
        if self._warmup_issued:
            return
        if not self._warmup_ik_client.service_is_ready():
            return
        self._warmup_issued = True

        req = WarmupIK.Request()
        req.batch_size = 1
        fut = self._warmup_ik_client.call_async(req)
        fut.add_done_callback(self._on_warmup_ik_done)

    def _on_warmup_ik_done(self, future: Future):
        resp = future.result()
        self.get_logger().info(
            f'Warmup IK: success={resp.success}, msg={resp.message}'
        )
        self._ik_ready = True

    # ------------------------------------------------------------------
    # Gizmo helpers
    # ------------------------------------------------------------------

    def _goal_from_gizmo(self):
        """Read the current gizmo pose, or fall back to defaults.

        The draggable control frame is created automatically by the visualizer
        for the tool frame; reuse the first one (``get_control_frame_pose``
        never returns custom frames added with ``add_control_frame``).
        """
        try:
            poses = self._viz.get_control_frame_pose()
        except Exception:  # noqa: BLE001
            poses = None
        if poses:
            frame = next(iter(poses.values()))
            pos = frame.position.cpu().squeeze().numpy()
            quat = frame.quaternion.cpu().squeeze().numpy()
            return list(pos), list(quat)
        return list(self._default_goal), self._tilt_quat(self._tool_pitch)

    def _gizmo_pose(self):
        """Return (pos[3], quat[4]) as numpy arrays, or None."""
        try:
            poses = self._viz.get_control_frame_pose()
        except Exception:  # noqa: BLE001
            return None
        if not poses:
            return None
        frame = next(iter(poses.values()))
        return (
            frame.position.cpu().squeeze().numpy().astype(np.float32),
            frame.quaternion.cpu().squeeze().numpy().astype(np.float32),
        )

    # ------------------------------------------------------------------
    # Single-pose IK (original mode)
    # ------------------------------------------------------------------

    def _solve_tick(self):
        if not self._ik_ready or self._busy or self._reachability_enabled:
            return
        pos, quat = self._goal_from_gizmo()
        # Skip replanning if the gizmo hasn't moved since the last solve.
        key = tuple(round(float(v), 4) for v in pos) + tuple(round(float(v), 4) for v in quat)
        if self._last_solve_pose == key:
            return
        self._last_solve_pose = key

        goal_pose = Pose()
        goal_pose.position.x = float(pos[0])
        goal_pose.position.y = float(pos[1])
        goal_pose.position.z = float(pos[2])
        goal_pose.orientation.w = float(quat[0])
        goal_pose.orientation.x = float(quat[1])
        goal_pose.orientation.y = float(quat[2])
        goal_pose.orientation.z = float(quat[3])

        self._busy = True
        self._set_status('Solving...')
        req = Ik.Request()
        req.pose = goal_pose
        fut = self._ik_client.call_async(req)
        fut.add_done_callback(self._on_ik_done)

    def _on_ik_done(self, future: Future):
        self._busy = False
        resp = future.result()
        if not resp.success:
            self._set_status('IK failed', ok=False)
            return

        valid = resp.joint_states_valid.data
        positions = list(resp.joint_states.position)
        self.get_logger().info(
            f'IK solved - valid={valid}, {len(positions)} joints'
        )
        self._set_status(f'IK ok - valid={valid}', ok=bool(valid))
        _viz_set_positions(self._viz, positions, self._joint_names)

    # ------------------------------------------------------------------
    # Reachability mode
    # ------------------------------------------------------------------

    def _reachability_tick(self):
        if not self._reachability_enabled or not self._ik_ready:
            return

        if not self._ik_batch_client.service_is_ready():
            self.get_logger().warn('IkBatch service not ready')
            return

        gp = self._gizmo_pose()
        if gp is None:
            return
        cur_pos, cur_wxyz = gp
        cur_extent = self._gui_grid_extent_slider.value

        moved_since_last_tick = (
            self._reachability_prev_pos is None
            or not np.allclose(cur_pos, self._reachability_prev_pos, atol=self._reachability_settle_tol)
            or not np.allclose(cur_wxyz, self._reachability_prev_wxyz, atol=self._reachability_settle_tol)
            or cur_extent != self._reachability_prev_extent
        )
        self._reachability_prev_pos = cur_pos.copy()
        self._reachability_prev_wxyz = cur_wxyz.copy()
        self._reachability_prev_extent = cur_extent

        changed_since_solve = not (
            self._prev_gizmo_pos is not None
            and self._prev_gizmo_wxyz is not None
            and self._prev_grid_extent is not None
            and np.allclose(cur_pos, self._prev_gizmo_pos, atol=self._reachability_settle_tol)
            and np.allclose(cur_wxyz, self._prev_gizmo_wxyz, atol=self._reachability_settle_tol)
            and cur_extent == self._prev_grid_extent
            and not self._reachability_needs_update
        )

        now = time.monotonic()
        if moved_since_last_tick:
            # Anchor the settle timer on the most recent gizmo movement.
            self._reachability_last_moved = now
            if self._reachability_busy:
                # Invalidate the in-flight solve so its result is discarded,
                # keep the map black, and queue a re-run after the user lets go.
                self._reachability_gen += 1
                self._reachability_pending = True
                self._blacken_reachability_image(cur_pos, cur_wxyz, cur_extent)
            return

        # Gizmo is not moving right now.
        if changed_since_solve and now - self._reachability_last_moved < self._reachability_settle_time:
            # Not yet settled after the last change: keep the map black.
            self._blacken_reachability_image(cur_pos, cur_wxyz, cur_extent)
            return

        if self._reachability_busy:
            # Single solve in flight: wait for it to finish.
            if self._reachability_pending:
                self._blacken_reachability_image(cur_pos, cur_wxyz, cur_extent)
            return

        if not changed_since_solve and not self._reachability_pending:
            # Nothing new to solve.
            return

        # A change settled long enough (user let go), or a pending re-run after a
        # busy solve finished. Snapshot and solve.
        self._prev_gizmo_pos = cur_pos.copy()
        self._prev_gizmo_wxyz = cur_wxyz.copy()
        self._prev_grid_extent = cur_extent
        self._reachability_needs_update = False
        self._reachability_pending = False

        self._reachability_gen += 1
        gen = self._reachability_gen

        self._blacken_reachability_image(cur_pos, cur_wxyz, cur_extent)
        self._reachability_busy = True
        self._set_status('Reachability: solving...')

        import viser.transforms as vtf

        n = self._n_per_axis
        extent = cur_extent
        half = extent / 2.0

        rot = vtf.SO3(cur_wxyz).as_matrix().astype(np.float32)
        pose_matrix = np.eye(4, dtype=np.float32)
        pose_matrix[:3, :3] = rot
        pose_matrix[:3, 3] = cur_pos

        lin = np.linspace(-half, half, n, dtype=np.float32)
        uu, vv = np.meshgrid(lin, lin, indexing='xy')
        local_pts = np.stack([uu.ravel(), vv.ravel(),
                              np.zeros(n * n, dtype=np.float32),
                              np.ones(n * n, dtype=np.float32)], axis=-1)
        grid_world = (pose_matrix @ local_pts.T).T[:, :3]

        # Every grid cell uses the gizmo's exact orientation (position varies,
        # orientation = gizmo's), matching the esdf_viser reachability mode.
        orientations = np.tile(cur_wxyz, (n * n, 1))

        req = IkBatch.Request()
        for i in range(n * n):
            p = Pose()
            p.position.x = float(grid_world[i, 0])
            p.position.y = float(grid_world[i, 1])
            p.position.z = float(grid_world[i, 2])
            p.orientation.w = float(orientations[i, 0])
            p.orientation.x = float(orientations[i, 1])
            p.orientation.y = float(orientations[i, 2])
            p.orientation.z = float(orientations[i, 3])
            req.poses.append(p)

        fut = self._ik_batch_client.call_async(req)
        fut.add_done_callback(lambda f, g=gen: self._on_reachability_done(f, g))

    def _on_reachability_done(self, future: Future, gen: int):
        self._reachability_busy = False
        if not self._reachability_enabled:
            return
        if gen != self._reachability_gen:
            self.get_logger().info(
                f'Reachability: discarding stale solve (gen {gen} != {self._reachability_gen})'
            )
            return
        resp = future.result()
        if not resp.success:
            self._set_status('Reachability: batch IK failed', ok=False)
            return

        n = self._n_per_axis
        actual_batch = n * n

        success = np.array(
            [v.data for v in resp.joint_states_valid[:actual_batch]],
            dtype=bool,
        ).reshape(n, n)

        img = np.zeros((n, n, 3), dtype=np.uint8)
        img[success] = [0, 200, 0]
        img[~success] = [200, 0, 0]

        server = getattr(self._viz, '_server', None)
        if server is None:
            return

        cur_pos = self._prev_gizmo_pos
        cur_wxyz = self._prev_gizmo_wxyz
        cur_extent = self._prev_grid_extent
        if cur_pos is None or cur_wxyz is None or cur_extent is None:
            return
        half = cur_extent / 2.0

        import viser.transforms as vtf
        rot = vtf.SO3(cur_wxyz).as_matrix().astype(np.float32)

        corners_local = np.array(
            [[-half, -half, 0], [half, -half, 0],
             [half, half, 0], [-half, half, 0]],
            dtype=np.float32,
        )
        corners_world = (rot @ corners_local.T).T + cur_pos
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        lines = np.array(
            [[corners_world[i], corners_world[j]] for i, j in edges],
            dtype=np.float32,
        )

        if self._gui_reachability_bounds is not None:
            try:
                server.scene.remove_node(self._gui_reachability_bounds)
            except Exception:
                pass
        self._gui_reachability_bounds = server.scene.add_line_segments(
            '/reachability_bounds',
            points=lines,
            colors=np.array([255, 255, 0], dtype=np.uint8),
            line_width=3.0,
        )

        if self._gui_reachability_slice is not None:
            try:
                server.scene.remove_node(self._gui_reachability_slice)
            except Exception:
                pass
        self._gui_reachability_slice = server.scene.add_image(
            name='/reachability_bounds/slice_image',
            image=img,
            render_width=cur_extent,
            render_height=cur_extent,
            wxyz=tuple(cur_wxyz),
            position=tuple(cur_pos),
        )

        n_success = int(success.sum())
        self._set_status(
            f'Reachability: {n_success}/{actual_batch} '
            f'({100 * n_success / actual_batch:.0f}%)',
            ok=True,
        )

        last = resp.joint_states_valid[actual_batch].data if len(resp.joint_states_valid) > actual_batch else False
        if last and len(resp.joint_states) > actual_batch:
            _viz_set_positions(
                self._viz,
                list(resp.joint_states[actual_batch].position),
                self._joint_names,
            )


def main(args=None):
    rclpy.init(args=args)
    node = IkViserNode()
    try:
        viser_serve_forever(node, node._viz, node.get_logger())
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

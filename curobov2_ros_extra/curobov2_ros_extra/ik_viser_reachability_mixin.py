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
- dragging it re-solves IK continuously (mirroring curobo's interactive
  examples), and the solved configuration is rendered on the robot,
- optional reachability mode: a draggable slice gizmo with a green/red heatmap
  showing which workspace positions are IK-solvable. The whole grid is solved
  on the server in one ``/<node>/generate_rm`` (GenerateRM) call; this node
  only sends the plane pose/extent/grid and renders the returned per-cell
  joint configurations.
"""

import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.task import Future
from geometry_msgs.msg import Pose

from curobov2_ros_interfaces.srv import GenerateRM, Ik, WarmupIK

from ._viser_helpers import (
    _viz_set_positions,
    active_joint_names_from_content,
    start_service_poll,
    viser_serve_forever,
)

_GRID_CELLS = 100  # 10x10 grid, one GenerateRM solve on the server



class _IkReachabilityMixin:
    def _reachability_tick(self):
        if not self._reachability_enabled or not self._ik_ready:
            return

        if not self._rm_client.service_is_ready():
            self.get_logger().warn('GenerateRM service not ready')
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

        # Delegate the whole grid solve to the server: /<server_node>/generate_rm
        # solves one IK per grid cell on a plane through the gizmo pose (the
        # server composes the tool-frame orientation internally, so the client
        # only sends the plane pose + extents + grid count).
        req = GenerateRM.Request()
        req.plane_position_x = float(cur_pos[0])
        req.plane_position_y = float(cur_pos[1])
        req.plane_position_z = float(cur_pos[2])
        req.plane_orientation_w = float(cur_wxyz[0])
        req.plane_orientation_x = float(cur_wxyz[1])
        req.plane_orientation_y = float(cur_wxyz[2])
        req.plane_orientation_z = float(cur_wxyz[3])
        req.plane_size_x = float(cur_extent)
        req.plane_size_y = float(cur_extent)
        req.grid_size_x = self._n_per_axis
        req.grid_size_y = self._n_per_axis

        fut = self._rm_client.call_async(req)
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
        metrics = resp.metrics
        if metrics is None or metrics.n_total == 0:
            self._set_status('Reachability: map solve failed', ok=False)
            return
        n_total = int(metrics.n_total)
        if not resp.success:
            self.get_logger().warn(
                f'Reachability: solve reported failure ({resp.message}) - '
                'showing returned map (all-red if nothing converged)'
            )

        gx = int(metrics.grid_size_x) or self._n_per_axis
        gy = int(metrics.grid_size_y) or self._n_per_axis
        n_solved = int(metrics.n_solved)

        valid = np.array(
            [v.data for v in metrics.joint_states_valid[:n_total]],
            dtype=bool,
        )
        success = valid.reshape(gy, gx)

        img = np.zeros((gy, gx, 3), dtype=np.uint8)
        img[success] = [0, 200, 0]
        img[~success] = [200, 0, 0]

        server = getattr(self._viz, '_server', None)
        if server is None:
            return

        # The metrics plane pose is authoritative (the server normalizes the
        # input quaternion); use it so the rendered slice always matches the
        # cells that were actually solved.
        pp = metrics.plane_pose.position
        quat = metrics.plane_pose.orientation
        cur_wxyz = np.array(
            [quat.w, quat.x, quat.y, quat.z], dtype=np.float32
        )
        cur_pos = np.array(
            [pp.x, pp.y, pp.z], dtype=np.float32
        )
        extent_x = float(metrics.plane_size_x) or 1.0
        extent_y = float(metrics.plane_size_y) or 1.0
        hx, hy = extent_x / 2.0, extent_y / 2.0

        import viser.transforms as vtf
        rot = vtf.SO3(cur_wxyz).as_matrix().astype(np.float32)

        corners_local = np.array(
            [[-hx, -hy, 0], [hx, -hy, 0],
             [hx, hy, 0], [-hx, hy, 0]],
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
            render_width=extent_x,
            render_height=extent_y,
            wxyz=tuple(cur_wxyz),
            position=tuple(cur_pos),
        )

        self._set_status(
            f'Reachability: {n_solved}/{n_total} '
            f'({100 * n_solved / n_total:.0f}%)',
            ok=True,
        )

        # Render the robot at the first solved cell so the user sees a reachable
        # pose without guessing.
        first = int(np.argmax(valid)) if np.any(valid) else -1
        if (
            first >= 0
            and first < len(metrics.joint_states)
            and len(metrics.joint_states[first].position) > 0
        ):
            _viz_set_positions(
                self._viz,
                list(metrics.joint_states[first].position),
                self._joint_names,
            )

__all__ = [
    '_GRID_CELLS',
    '_IkReachabilityMixin',
]

#!/usr/bin/env python3
"""
Reachability-map service for the unified planner node (v2).

Solves a uniform grid of IK goals on a user-facing plane and returns per-cell
outcomes so an RViz display can render the map. The plane is defined by a pose
(position + orientation quaternion) and extents; ``grid_size_x * grid_size_y``
cells centred on the pose all share the plane's orientation (only the position
varies). The solve reuses the shared IK solver from ``IKServices``.

Per-cell results reuse the same message types as the Ik / IkBatch services
(sensor_msgs/JointState[] + std_msgs/Bool[]) inside ReachabilityMetrics.

Service exposed (prefixed with the node name):
  /<node>/generate_rm   (GenerateRM) - plane -> ReachabilityMetrics
"""

import math
import time
import traceback

import numpy as np
import std_msgs.msg
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

from isaac_ros_cumotion_interfaces.srv import GenerateRM
from isaac_ros_cumotion_interfaces.msg import ReachabilityMetrics

# Cap on the total number of grid cells solved per request. The IK solver is
# created with max_batch_size == grid size and allocates per-batch buffers, so
# an absurd grid (e.g. 200x200) would blow up GPU memory. Legacy viser
# reachability used ~500 (22x22); 1200 (~35x34) is a generous ceiling for the
# reduced seed count reachability uses.
_MAX_GRID_CELLS = 1200


def _normalize_quat(q):
    """Normalise a (w, x, y, z) quaternion; identity for an all-zero one."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return 1.0, 0.0, 0.0, 0.0
    return w / n, x / n, y / n, z / n


def _quat_to_matrix(quat) -> np.ndarray:
    """Rotation matrix for the (w, x, y, z) quaternion ``quat``."""
    w, x, y, z = _normalize_quat(quat)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def _quat_multiply(a, b):
    """Hamilton product of two (w, x, y, z) quaternions (``a`` o ``b``)."""
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    return (
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
        a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
        a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
        a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0)


class ReachabilityServices:
    """Grid-IK reachability map service on an existing node.

    Depends on config_wrapper (for the frame id / robot config) and an
    IKServices instance (for the shared IK solver). Worlds are kept up to date
    by the node's existing update_all_solvers_world(), so the solve accounts
    for the current obstacles.
    """

    def __init__(self, node, config_wrapper, ik_services):
        self._node = node
        self._config = config_wrapper
        self._ik_services = ik_services
        self._map_id = 0

        # The map solves the grid in ONE cuRobo batch with the SAME num_seeds
        # as the regular /ik services (default 20), for uniform solve quality.
        # VRAM grows with grid cells x seeds; keep the grid modest (see
        # _MAX_GRID_CELLS) rather than lowering the seed count.

        name = node.get_name()
        node.create_service(
            GenerateRM, f"{name}/generate_rm", self._generate_rm_callback)

        node.get_logger().info(
            f"Reachability services registered (/<node>/generate_rm, "
            f"grid cap {_MAX_GRID_CELLS} cells)")

    # ------------------------------------------------------------------
    # Service
    # ------------------------------------------------------------------

    def _generate_rm_callback(self, request: GenerateRM.Request,
                              response: GenerateRM.Response):
        t0 = time.perf_counter()
        try:
            gx, gy = self._grid_size(request)
            goals = self._build_goals(request, gx, gy)
            # Guard the shared IK solver's rebuild/solve with the planner's
            # gpu_lock so it never races an open-loop plan's CUDA graph
            # capture (capture is process-global; cf. unified_planner_node).
            # solve_poses() uses the same default seed count as /ik & /ik_batch.
            gpu_lock = getattr(self._node, "gpu_lock", None)
            if gpu_lock is not None:
                with gpu_lock:
                    ok, positions, flags, joint_names = (
                        self._ik_services.solve_poses(goals))
            else:
                ok, positions, flags, joint_names = (
                    self._ik_services.solve_poses(goals))

            metrics = ReachabilityMetrics()
            metrics.header.stamp = self._node.get_clock().now().to_msg()
            metrics.header.frame_id = self._config.base_link
            metrics.map_id = self._map_id
            self._map_id += 1

            metrics.plane_size_x = request.plane_size_x
            metrics.plane_size_y = request.plane_size_y
            metrics.grid_size_x = gx
            metrics.grid_size_y = gy
            metrics.plane_pose.position.x = request.plane_position_x
            metrics.plane_pose.position.y = request.plane_position_y
            metrics.plane_pose.position.z = request.plane_position_z
            quat = _normalize_quat((
                request.plane_orientation_w,
                request.plane_orientation_x,
                request.plane_orientation_y,
                request.plane_orientation_z))
            metrics.plane_pose.orientation.w, \
                metrics.plane_pose.orientation.x, \
                metrics.plane_pose.orientation.y, \
                metrics.plane_pose.orientation.z = quat

            metrics.goals = goals
            for i in range(len(goals)):
                js = JointState()
                valid = std_msgs.msg.Bool()
                if ok and positions is not None and flags[i]:
                    js.name = list(joint_names)
                    js.position = list(positions[i])
                    valid.data = True
                metrics.joint_states.append(js)
                metrics.joint_states_valid.append(valid)

            metrics.n_total = len(goals)
            metrics.n_solved = sum(
                1 for v in metrics.joint_states_valid if v.data)
            metrics.solve_time_ms = (time.perf_counter() - t0) * 1e3

            response.metrics = metrics
            response.success = ok
            response.message = (
                f"solved {metrics.n_solved}/{metrics.n_total}"
                if ok else "IK solve failed; map shown all-red")
            return response
        except Exception as e:
            self._node.get_logger().error(
                f"Reachability generate_rm failed: {e}\n{traceback.format_exc()}")
            response.success = False
            response.message = str(e)
            return response

    # ------------------------------------------------------------------
    # Grid construction
    # ------------------------------------------------------------------

    @staticmethod
    def _grid_size(request):
        """Clamp an absurd grid down to ``_MAX_GRID_CELLS`` (aspect preserved)."""
        gx = max(1, int(request.grid_size_x))
        gy = max(1, int(request.grid_size_y))
        if gx * gy > _MAX_GRID_CELLS:
            scale = math.sqrt(_MAX_GRID_CELLS / float(gx * gy))
            gx = max(1, int(gx * scale))
            gy = max(1, int(gy * scale))
            while gx * gy > _MAX_GRID_CELLS:
                if gx > gy:
                    gx -= 1
                else:
                    gy -= 1
        return gx, gy

    @staticmethod
    def _build_goals(request, gx: int, gy: int):
        """World-frame tool poses for every grid cell, row-major (x fastest).

        The plane's local XY spans [-size/2, size/2]; the local Z axis is the
        plane's normal. Every cell's tool is the plane frame composed with
        +90deg about its local Y axis: the tool frame[0] (grasping_frame) is a
        90-deg-offset frame on the flange whose +X is the physical axis the
        end effector visibly extends along, so tool +X points INTO the plane
        (along -normal). Mimicking an IK-goal call in the tool frame, this is
        the "inverse" tool-frame goal: solve_pose() then lands the solved
        robot's end effector pointing at the plane, matching the arrows.
        """
        quat = _normalize_quat((
            request.plane_orientation_w,
            request.plane_orientation_x,
            request.plane_orientation_y,
            request.plane_orientation_z))
        rot = _quat_to_matrix(quat)
        # +90deg about the plane's local Y: rotates the plane frame so tool
        # +X (the grasping_frame's physical pointing axis) maps to the plane's
        # -Z / inward normal. Ry(90): (x,y,z) -> (-z,y,x).
        tool_quat = _quat_multiply(
            quat, (math.cos(math.pi / 4.0), 0.0, math.sin(math.pi / 4.0), 0.0))
        origin = np.array([request.plane_position_x,
                           request.plane_position_y,
                           request.plane_position_z], dtype=np.float64)

        hx = request.plane_size_x / 2.0
        hy = request.plane_size_y / 2.0
        us = np.linspace(-hx, hx, gx, dtype=np.float64) \
            if gx > 1 else np.zeros(1, dtype=np.float64)
        vs = np.linspace(-hy, hy, gy, dtype=np.float64) \
            if gy > 1 else np.zeros(1, dtype=np.float64)

        goals = []
        for iy in range(gy):
            for ix in range(gx):
                local = np.array([us[ix], vs[iy], 0.0], dtype=np.float64)
                world = origin + rot @ local
                p = Pose()
                p.position.x = float(world[0])
                p.position.y = float(world[1])
                p.position.z = float(world[2])
                p.orientation.w = float(tool_quat[0])
                p.orientation.x = float(tool_quat[1])
                p.orientation.y = float(tool_quat[2])
                p.orientation.z = float(tool_quat[3])
                goals.append(p)
        return goals
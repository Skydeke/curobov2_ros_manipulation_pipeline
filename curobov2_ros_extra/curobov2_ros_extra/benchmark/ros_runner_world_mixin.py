# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ROS leg: replay the benchmark problems through the running planner server. Talks to the ``unified_planner`` node started by ``gen_traj.launch.py``:

- world for each problem is pushed via ``/unified_planner/add_object``
  (obstacles converted from the same cuRobo obstacle dicts the native leg's
  ``SceneCfg.create`` consumes) after ``/unified_planner/remove_all_objects``
- each problem is planned through
  ``/unified_planner/generate_trajectory`` (classic single-goal path)
- success is read from the response; path length / motion time / waypoint
  count are computed client-side from the returned interpolated trajectory
  with the same formulas as the native leg (``compare.trajectory_metrics``)
- position error (mm) is the winner's solver-reported ``position_error``,
  read from its ``PlanningStats.considered`` row (same convergence metric the
  native leg records as ``result.position_error * 1000``; the row's
  ``max_waypoint_error`` carries the solver's per-seed residual in meters,
  ×1000 here). No row → the field is left ``None`` (row omitted): the
  interpolated trajectory's final waypoint is pinned to the goal joint state
  by implicit-goal interpolation, so FK'ing it would report ~0 mm by
  construction rather than the solver's honest residual."""

# Standard Library
import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Sequence

import rclpy
from geometry_msgs.msg import Point as RosPoint
from geometry_msgs.msg import Pose as RosPose
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.node import Node
from sensor_msgs.msg import JointState as RosJointState

from curobov2_ros_interfaces.msg import Goalset, SceneObject, TrajectoryGoal
from curobov2_ros_interfaces.srv import (
    AddObject,
    SetCollisionCache,
    SetJointLocks,
    SetObjects,
    TrajectoryGeneration,
)
from std_srvs.srv import Trigger

from .compare import (
    trajectory_jerk,
    trajectory_metrics,
)
from .obstacle_convert import obstacles_dict_to_add_requests
from .problems import (
    collision_cache_sizes,
    filter_scenes,
    load_problems,
    mpinets_lock_for_scene,
    mpinets_scene_keys,
)

# Franka active (cspace) joints, in cuRobo order — matches the robot YAML's
# cspace.joint_names (fingers are locked by the robot config, not sent).
FRANKA_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]

SERVER_NODE = "unified_planner"

GENERATE_TRAJECTORY_SRV = f"{SERVER_NODE}/generate_trajectory"
ADD_OBJECT_SRV = f"{SERVER_NODE}/add_object"
REMOVE_ALL_OBJECTS_SRV = f"{SERVER_NODE}/remove_all_objects"
SET_COLLISION_CACHE_SRV = f"{SERVER_NODE}/set_collision_cache"
# Runtime torque-mode switch plumbing (both stock ROS2 services — no new
# wire schema): the server's dynamics mode is baked into its robot_cfg at
# solver build time, so flipping it = set_parameters + update_motion_gen_config.
GET_PARAMETERS_SRV = f"{SERVER_NODE}/get_parameters"
SET_PARAMETERS_SRV = f"{SERVER_NODE}/set_parameters"
UPDATE_MOTION_GEN_CONFIG_SRV = f"{SERVER_NODE}/update_motion_gen_config"
# Runtime joint-lock switch: the gripper spread is part of the robot model's
# collision geometry (curobo bakes lock_joints into the parsed kinematics
# config), so a change re-parses the robot YAML and rebuilds the model and
# every solver. Per-joint: we name the joints to re-pin and leave the rest of
# the model's locks alone.
SET_JOINT_LOCKS_SRV = f"{SERVER_NODE}/set_joint_locks"
# Bulk scene replacement: clear + N adds, ONE solver-world refresh (see
# set_objects / SetObjects.srv). Optional: servers predating the service
# only offer clear/add_object, which the runner falls back to.
SET_OBJECTS_SRV = f"{SERVER_NODE}/set_objects"


def obstacle_hash(obstacles: Dict[str, Any]) -> str:
    """Stable hash of a problem's obstacle dict (scene-change detection)."""
    canonical = json.dumps(obstacles, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()



class _RosRunnerWorldMixin:
    def set_world_bulk(
        self, client, obstacles: Dict[str, Any], timeout: float = 120.0
    ) -> bool:
        """Replace the scene via one ``set_objects`` call.

        Returns False (never raises) on any failure — transport error,
        timeout or server rejection — so the caller can fall back to the
        clear + per-object loop, which surfaces the real error.
        """
        try:
            request = SetObjects.Request()
            request.clear_first = True
            for payload in obstacles_dict_to_add_requests(obstacles):
                obj = SceneObject()
                obj.type = int(payload["type"])
                obj.name = payload["name"]
                px, py, pz, qw, qx, qy, qz = payload["pose"]
                obj.pose.position.x = px
                obj.pose.position.y = py
                obj.pose.position.z = pz
                obj.pose.orientation.w = qw
                obj.pose.orientation.x = qx
                obj.pose.orientation.y = qy
                obj.pose.orientation.z = qz
                dx, dy, dz = payload["dims"]
                obj.dimensions.x = dx
                obj.dimensions.y = dy
                obj.dimensions.z = dz
                cr, cg, cb, ca = payload["color"]
                obj.color.r = cr
                obj.color.g = cg
                obj.color.b = cb
                obj.color.a = ca
                obj.mesh_file_path = payload.get("mesh_file_path") or ""
                for v in payload.get("vertices", []):
                    obj.vertices.append(RosPoint(x=v[0], y=v[1], z=v[2]))
                obj.triangles.extend(payload.get("triangles", []))
                request.objects.append(obj)
            response = self._call(client, request, timeout)
            if response is None or not bool(response.success):
                self.get_logger().warn(
                    "set_objects failed "
                    f"({getattr(response, 'message', 'no response')}) — "
                    "falling back to clear + per-object adds")
                return False
            return True
        except Exception as exc:  # noqa: BLE001 - fallback covers it
            self.get_logger().warn(f"set_objects failed ({exc}) — falling back")
            return False

    def add_world(self, obstacles: Dict[str, Any], timeout: float = 60.0) -> None:
        for payload in obstacles_dict_to_add_requests(obstacles):
            request = AddObject.Request()
            request.type = int(payload["type"])
            request.name = payload["name"]
            px, py, pz, qw, qx, qy, qz = payload["pose"]
            request.pose.position.x = px
            request.pose.position.y = py
            request.pose.position.z = pz
            request.pose.orientation.w = qw
            request.pose.orientation.x = qx
            request.pose.orientation.y = qy
            request.pose.orientation.z = qz
            dx, dy, dz = payload["dims"]
            request.dimensions.x = dx
            request.dimensions.y = dy
            request.dimensions.z = dz
            cr, cg, cb, ca = payload["color"]
            request.color.r = cr
            request.color.g = cg
            request.color.b = cb
            request.color.a = ca
            if payload.get("mesh_file_path"):
                request.mesh_file_path = payload["mesh_file_path"]
            for v in payload.get("vertices", []):
                request.vertices.append(RosPoint(x=v[0], y=v[1], z=v[2]))
            request.triangles.extend(payload.get("triangles", []))

            response = self._call(self._add_client, request, timeout)
            if not response.success:
                raise RuntimeError(
                    f"add_object({request.name}) failed: "
                    f"{getattr(response, 'message', '')}"
                )

    # ------------------------------------------------------------------
    # planning
    # ------------------------------------------------------------------

    def _build_goal(self, problem: Dict[str, Any]) -> TrajectoryGoal:
        start = RosJointState(
            name=FRANKA_JOINT_NAMES,
            position=[float(v) for v in problem["start"]],
        )
        gp = problem["goal_pose"]
        pose = RosPose()
        pose.position.x = float(gp["position_xyz"][0])
        pose.position.y = float(gp["position_xyz"][1])
        pose.position.z = float(gp["position_xyz"][2])
        pose.orientation.w = float(gp["quaternion_wxyz"][0])
        pose.orientation.x = float(gp["quaternion_wxyz"][1])
        pose.orientation.y = float(gp["quaternion_wxyz"][2])
        pose.orientation.z = float(gp["quaternion_wxyz"][3])
        goalset = Goalset(poses=[pose])
        return TrajectoryGoal(
            start_pose=start,
            goalsets=[goalset],
        )

    @staticmethod
    def _client_energy_torque(
        waypoints,
        waypoint_velocities,
        dt,
        robot_model_data,
    ):
        """Reconstruct Energy (J) / max Torque (N·m) client-side.

        Mirrors the upstream ``compute_trajectory_energy`` arithmetic on the
        returned interpolated trajectory: ``torque[t] = pin.rnea(model, data,
        q[t], qd[t], qdd[t])``, ``energy = sum(|torque * qd|) * dt``,
        ``max_torque = max(|torque|)``. The wire carries no acceleration, so
        ``qdd`` is finite-differenced from the returned per-waypoint velocity
        (or from ``q`` when velocity is absent) — the accepted approximation
        (exact for the native leg's ``js_solution``).
        """
        import numpy as np
        import pinocchio as pin

        model, data, _torque_limits = robot_model_data
        q = np.asarray(waypoints, dtype=float)
        if q.ndim > 2:
            q = q.reshape(-1, q.shape[-1])
        num_dof = model.nq
        if waypoint_velocities is not None:
            qd = np.asarray(waypoint_velocities, dtype=float)
            if qd.ndim > 2:
                qd = qd.reshape(q.shape[0], -1)
        else:
            qd = np.gradient(q, dt, axis=0)
        qdd = np.gradient(qd, dt, axis=0)
        horizon = q.shape[0]
        torques = np.zeros((horizon, num_dof))
        for t in range(horizon):
            torques[t, :] = pin.rnea(
                model, data, q[t, :num_dof], qd[t, :num_dof], qdd[t, :num_dof]
            )[:num_dof]
        power = torques[:, :num_dof] * qd[:, :num_dof]
        energy = float(np.sum(np.abs(power)) * dt)
        max_torque = float(np.max(np.abs(torques[:, :num_dof])))
        return energy, max_torque

    def plan_one(
        self,
        problem: Dict[str, Any],
        problem_name: str,
        scene_key: str,
        timeout: float = 120.0,
        robot_model_data=None,
    ) -> Dict[str, Any]:
        goal = self._build_goal(problem)
        request = TrajectoryGeneration.Request()
        request.request = goal

        t_start = time.perf_counter()
        response = self._call(self._traj_client, request, timeout)
        wall_time = time.perf_counter() - t_start

        result = response.response
        entry: Dict[str, Any] = {
            "problem_name": problem_name,
            "scene_key": scene_key,
            "capability": "planning",
            "success": bool(result.success),
            "time_s": wall_time,
            "n_waypoints": 0,
            "path_length": None,
            "motion_time_s": None,
            "solve_time_s": None,
            "jerk": None,
            "position_error_mm": None,
            "energy_j": None,
            "torque_nm": None,
        }

        if not result.success:
            entry["message"] = result.message
            return entry

        waypoints = [list(w.position) for w in result.trajectory]
        n, path_length, _ = trajectory_metrics(waypoints, float(result.dt))
        entry["n_waypoints"] = n
        entry["path_length"] = path_length
        # Server-reported metrics, same definitions as the native leg. No
        # fallbacks: servers without these fields are too old for this
        # runner (DDS already requires matched interfaces anyway).
        entry["motion_time_s"] = self._result_motion_time(result)
        entry["solve_time_s"] = self._result_solver_time(result)
        entry["jerk"] = trajectory_jerk(waypoints, float(result.dt))
        entry["position_error_mm"] = self._result_position_error(result)
        # Retain the winning endpoint for client-side FK verification
        # (position + orientation vs the goal pose). Server joint order
        # matches the requesting robot config, so callers FK with the
        # kinematics built from that same config.
        entry["final_joints"] = list(waypoints[-1]) if waypoints else None
        if robot_model_data is not None:
            try:
                velocities = None
                first_wp = result.trajectory[0] if result.trajectory else None
                if first_wp is not None and list(
                    getattr(first_wp, "velocity", []) or []
                ):
                    velocities = [list(w.velocity) for w in result.trajectory]
                energy_j, torque_nm = self._client_energy_torque(
                    waypoints,
                    velocities,
                    float(result.dt),
                    robot_model_data,
                )
                entry["energy_j"] = energy_j
                entry["torque_nm"] = torque_nm
            except Exception as exc:  # noqa: BLE001 - degrade, not fail
                self.get_logger().warn(f"Failed to reconstruct energy/torque: {exc}")
        return entry

    @staticmethod
    def _finite_or_none(value) -> Optional[float]:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        return v if v == v and v not in (float("inf"), float("-inf")) else None

    @staticmethod
    def _result_motion_time(result) -> Optional[float]:
        """Server-reported motion time (PlanningStats.motion_time_s).

        Same definition as the native leg (boundary knots excluded). None
        when unavailable.
        """
        stats = getattr(result, "stats", None)
        return _RosRunnerWorldMixin._finite_or_none(
            getattr(stats, "motion_time_s", None))

    @staticmethod
    def _result_solver_time(result) -> Optional[float]:
        """Top-level solver time (PlanningStats.solver_time_s).

        Same metric as the native leg (cuRobo result.solve_time, seconds),
        filled server-side for every solve. None when unavailable.
        """
        stats = getattr(result, "stats", None)
        return _RosRunnerWorldMixin._finite_or_none(
            getattr(stats, "solver_time_s", None))

    @staticmethod
    def _result_position_error(result) -> Optional[float]:
        """Top-level winner position error (PlanningStats.position_error_mm).

        Same metric and computation as the native leg, done server-side.
        None when unavailable.
        """
        stats = getattr(result, "stats", None)
        return _RosRunnerWorldMixin._finite_or_none(
            getattr(stats, "position_error_mm", None))

__all__ = [
    'FRANKA_JOINT_NAMES',
    'SERVER_NODE',
    'GENERATE_TRAJECTORY_SRV',
    'ADD_OBJECT_SRV',
    'REMOVE_ALL_OBJECTS_SRV',
    'SET_COLLISION_CACHE_SRV',
    'GET_PARAMETERS_SRV',
    'SET_PARAMETERS_SRV',
    'UPDATE_MOTION_GEN_CONFIG_SRV',
    'SET_JOINT_LOCKS_SRV',
    'SET_OBJECTS_SRV',
    'obstacle_hash',
    '_RosRunnerWorldMixin',
]

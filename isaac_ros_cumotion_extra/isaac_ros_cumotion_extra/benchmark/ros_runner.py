# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ROS leg: replay the benchmark problems through the running planner server.

Talks to the ``unified_planner`` node started by ``gen_traj.launch.py``:

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
  construction rather than the solver's honest residual.
- wall-clock timing is measured client-side around the service call;
  solver-reported solve time is read from the winner
  ``PlanningStats.considered`` row (the request sets
  ``log_considered_trajectories`` — a reporting-only flag, accepted on the
  classic planner — so the server fills the gated detail block)
"""

# Standard Library
import time
from typing import Any, Dict, List, Optional

import rclpy
from geometry_msgs.msg import Point as RosPoint
from geometry_msgs.msg import Pose as RosPose
from rclpy.node import Node
from sensor_msgs.msg import JointState as RosJointState

from isaac_ros_cumotion_interfaces.msg import Goalset, PlanningOptions, TrajectoryGoal
from isaac_ros_cumotion_interfaces.srv import (
    AddObject,
    SetCollisionCache,
    TrajectoryGeneration,
)
from std_srvs.srv import Trigger

from .compare import (
    trajectory_jerk,
    trajectory_metrics,
    winner_position_error_mm,
    winner_solve_time,
)
from .obstacle_convert import obstacles_dict_to_add_requests
from .problems import collision_cache_sizes, filter_scenes, load_problems

# Franka active (cspace) joints, in cuRobo order — matches the robot YAML's
# cspace.joint_names (fingers are locked by the robot config, not sent).
FRANKA_JOINT_NAMES = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7",
]

SERVER_NODE = "unified_planner"

GENERATE_TRAJECTORY_SRV = f"{SERVER_NODE}/generate_trajectory"
ADD_OBJECT_SRV = f"{SERVER_NODE}/add_object"
REMOVE_ALL_OBJECTS_SRV = f"{SERVER_NODE}/remove_all_objects"
SET_COLLISION_CACHE_SRV = f"{SERVER_NODE}/set_collision_cache"


class RosBenchmarkRunner(Node):

    def __init__(self, service_timeout: float = 30.0):
        super().__init__("curobo_benchmark_ros_runner")
        self._traj_client = self.create_client(
            TrajectoryGeneration, GENERATE_TRAJECTORY_SRV
        )
        self._add_client = self.create_client(AddObject, ADD_OBJECT_SRV)
        self._clear_client = self.create_client(Trigger, REMOVE_ALL_OBJECTS_SRV)
        self._cache_client = self.create_client(
            SetCollisionCache, SET_COLLISION_CACHE_SRV
        )

        for label, client in (
            ("generate_trajectory", self._traj_client),
            ("add_object", self._add_client),
            ("remove_all_objects", self._clear_client),
            ("set_collision_cache", self._cache_client),
        ):
            if not client.wait_for_service(timeout_sec=service_timeout):
                self.get_logger().error(f"{label} service not available")
                raise RuntimeError(f"{label} service not available")
            self.get_logger().info(f"Connected to {client.srv_name}")

    # ------------------------------------------------------------------
    # service plumbing
    # ------------------------------------------------------------------

    def _call(self, client, request, timeout: float):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if future.result() is None:
            raise RuntimeError(
                f"{client.srv_name}: service call failed (timeout={timeout}s)"
            )
        return future.result()

    # ------------------------------------------------------------------
    # collision cache
    # ------------------------------------------------------------------

    def size_collision_cache(self, problems, timeout: float = 120.0) -> None:
        """Size the server's collision cache to the dataset (native parity).

        curobo's Warp collision kernels launch one thread per (robot sphere,
        padded obstacle slot) per obstacle *type*, so an over-padded cache —
        the server's former deployment default
        ``{cuboid: 100, mesh: 100, voxel: ...}`` (the launch defaults are now
        32/4 via ``collision_cache_cuboid`` / ``collision_cache_mesh``) —
        makes every solver iteration run a far larger kernel grid than the
        native leg's ``{obb: n_cubes}`` cache — the residual ~7x
        single-attempt solve gap after the
        ``obstacle_collision_mode:=cuboid`` fix. Sizing the cache to
        the dataset's actual per-type counts (the same computation as the
        native leg's ``check_problems``) and disabling the empty no-camera
        voxel layer makes both legs collide through native-equivalent kernel
        grids.

        One synchronous solver rebuild (~25-37 s, inside the service call) is
        paid here, once, before the warmup probe — after it the sized kernels
        are live for the whole timed run. ``--no-size-cache`` on the CLI skips
        this to reproduce the padded (slow) behaviour.
        """
        cuboid_slots, mesh_slots = collision_cache_sizes(problems)
        request = SetCollisionCache.Request()
        request.obb = cuboid_slots
        request.mesh = mesh_slots
        request.blox = 0  # no cameras in the benchmark: disable the voxel layer
        self.get_logger().info(
            f"Sizing solver collision cache to dataset needs "
            f"(obb={request.obb}, mesh={request.mesh}, blox={request.blox}) — "
            f"one synchronous solver rebuild, then native-sized kernel grids"
        )
        response = self._call(self._cache_client, request, timeout)
        if not response.success:
            raise RuntimeError(
                f"set_collision_cache failed: {getattr(response, 'message', '')}"
            )
        self.get_logger().info(
            f"Collision cache set: obb={response.obb_cache}, "
            f"mesh={response.mesh_cache}, blox={response.blox_cache}"
        )

    # ------------------------------------------------------------------
    # world management
    # ------------------------------------------------------------------

    def clear_world(self, timeout: float = 30.0) -> None:
        response = self._call(self._clear_client, Trigger.Request(), timeout)
        if not response.success:
            raise RuntimeError(
                f"remove_all_objects failed: {getattr(response, 'message', '')}"
            )

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
        # log_considered_trajectories is a *reporting-only* flag (no search
        # effect; accepted on the classic planner): the server fills
        # stats.considered with one row per seed, so the runner can read the
        # winner's solver-reported solve time and position error
        # (see _winner_solve_time / _winner_position_error_mm).
        return TrajectoryGoal(
            start_pose=start,
            goalsets=[goalset],
            options=PlanningOptions(log_considered_trajectories=True),
        )

    def plan_one(
        self, problem: Dict[str, Any], problem_name: str, scene_key: str,
        timeout: float = 120.0,
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
        }

        if not result.success:
            entry["message"] = result.message
            return entry

        waypoints = [list(w.position) for w in result.trajectory]
        n, path_length, motion_time = trajectory_metrics(waypoints, float(result.dt))
        entry["n_waypoints"] = n
        entry["path_length"] = path_length
        entry["motion_time_s"] = motion_time
        entry["solve_time_s"] = self._winner_solve_time(result)
        entry["jerk"] = trajectory_jerk(waypoints, float(result.dt))
        entry["position_error_mm"] = self._winner_position_error_mm(result)
        return entry

    @staticmethod
    def _winner_position_error_mm(result) -> Optional[float]:
        """Solver-reported position error (mm) of the winning candidate.

        The request sets ``log_considered_trajectories`` (a reporting-only
        flag), so the server fills ``stats.considered`` with one row per seed;
        the winning row's ``max_waypoint_error`` carries the solver's own
        per-seed ``position_error`` (m, the same convergence metric the native
        leg records as ``result.position_error``), ×1000 here. When the rows
        are absent (or the winner can't be attributed) the field is ``None``
        and the row is omitted — deliberately no client-side FK fallback: the
        returned interpolated trajectory's final waypoint is pinned to the
        goal joint state by implicit-goal interpolation, so FK'ing it would
        report ~0 mm by construction rather than the solver's honest residual.
        """
        stats = getattr(result, "stats", None)
        rows = list(getattr(stats, "considered", None) or [])
        return winner_position_error_mm(
            rows,
            getattr(result, "selected_goal_index", None),
            getattr(result, "selected_seed_index", None),
        )

    @staticmethod
    def _winner_solve_time(result) -> Optional[float]:
        """Solver-reported solve time from the winner's considered row.

        The request sets ``log_considered_trajectories`` (a reporting-only
        flag), so the server fills ``stats.considered`` with one row per seed;
        the winning row's ``solve_time`` is the solver's own time — the same
        ``result.solve_time`` the native leg records, not client-side wall
        time.
        """
        stats = getattr(result, "stats", None)
        rows = list(getattr(stats, "considered", None) or [])
        return winner_solve_time(
            rows,
            getattr(result, "selected_goal_index", None),
            getattr(result, "selected_seed_index", None),
        )


def run_ros(
    dataset: str = "demo",
    scene: Optional[str] = None,
    service_timeout: float = 30.0,
    call_timeout: float = 120.0,
    warmup_probe: bool = True,
    size_collision_cache: bool = True,
) -> List[Dict[str, Any]]:
    """Run the ROS-wrapped leg over a robometrics dataset.

    ``scene`` restricts the run to one scene key within the dataset
    (``None`` = all scenes). One warmup probe (first valid problem, result
    discarded) is sent before the timed run so CUDA-graph warmup / first-solve
    JIT on the server does not skew (or stall) the measurements.

    ``size_collision_cache`` (default True) first resizes the server's
    collision cache to the dataset's actual per-type obstacle counts and
    disables the (empty, no-camera) voxel layer — the native leg runs with a
    ``{obb: n_cubes}`` cache, and the server's oversized deployment default
    would otherwise inflate every solver iteration's Warp collision kernel
    grid ~7x (see ``size_collision_cache`` and the README). Pass False to
    reproduce the padded behaviour for A/B diagnosis.
    """
    rclpy.init()
    node: Optional[RosBenchmarkRunner] = None
    try:
        node = RosBenchmarkRunner(service_timeout=service_timeout)
        problems = filter_scenes(load_problems(dataset), scene)
        if not problems:
            raise ValueError("No problems to run after scene filtering.")
        node.get_logger().info(
            f"ROS leg dataset={dataset} scenes: {', '.join(sorted(problems))}"
        )

        if size_collision_cache:
            node.size_collision_cache(problems, timeout=call_timeout)
        else:
            node.get_logger().warn(
                "size_collision_cache=False: leaving the server's default "
                "collision cache in place (padded kernel grids — diagnostic)"
            )

        ready_problems = [
            (scene_key, i, p)
            for scene_key, scene_problems in problems.items()
            for i, p in enumerate(scene_problems, start=1)
            if p.get("collision_buffer_ik", 0.0) >= 0.0
        ]
        if not ready_problems:
            return []

        if warmup_probe:
            scene_key, i, first_problem = ready_problems[0]
            node.get_logger().info(
                f"Warmup probe: {scene_key}_{i} (world clear + add + plan)"
            )
            node.clear_world(timeout=call_timeout)
            node.add_world(first_problem["obstacles"], timeout=call_timeout)
            try:
                node.plan_one(first_problem, f"{scene_key}_{i}", scene_key, timeout=call_timeout)
            except RuntimeError as exc:
                node.get_logger().warn(f"Warmup probe failed (continuing): {exc}")

        results: List[Dict[str, Any]] = []
        for scene_key, i, problem in ready_problems:
            problem_name = f"{scene_key}_{i}"
            node.get_logger().info(f"Solving {problem_name} ...")
            node.clear_world(timeout=call_timeout)
            node.add_world(problem["obstacles"], timeout=call_timeout)
            results.append(
                node.plan_one(problem, problem_name, scene_key, timeout=call_timeout)
            )

        node.get_logger().info(
            f"ROS leg done: {len(results)} problems, "
            f"{sum(1 for r in results if r['success'])} succeeded"
        )
        return results
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
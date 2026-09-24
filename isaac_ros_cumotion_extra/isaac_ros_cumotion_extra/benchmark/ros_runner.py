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

Energy (J) / Torque (N·m) follow the accepted "server plans; client
computes" split: the server runs torque-limited planning from its loaded
robot config (no response schema change — ``use_dynamics`` only labels the
run / client-side reconstruction), and this client reconstructs the two
columns from the returned trajectory with the upstream Pinocchio helper's
arithmetic. The server's torque mode is a build-time property of its robot
cfg, so ``run_ros(..., server_dynamics=…)`` switches the running server
between the reference page's two tables at runtime through the standard
``set_parameters`` service + the ``update_motion_gen_config`` trigger (see
``RosBenchmarkRunner.set_server_torque_mode``) — one server life services
BOTH motion ROS rows, nothing is ever skipped. Because the wire carries
positions + velocity + dt but *no acceleration*, ``qdd`` is
finite-differenced from the returned velocity: the numbers are an
approximation of the native leg's (exact ``js_solution``) values — see the
README "timing attribution" section for the accepted divergence.
"""

# Standard Library
import time
from typing import Any, Dict, List, Optional

import rclpy
from geometry_msgs.msg import Point as RosPoint
from geometry_msgs.msg import Pose as RosPose
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
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
# Runtime torque-mode switch plumbing (both stock ROS2 services — no new
# wire schema): the server's dynamics mode is baked into its robot_cfg at
# solver build time, so flipping it = set_parameters + update_motion_gen_config.
GET_PARAMETERS_SRV = f"{SERVER_NODE}/get_parameters"
SET_PARAMETERS_SRV = f"{SERVER_NODE}/set_parameters"
UPDATE_MOTION_GEN_CONFIG_SRV = f"{SERVER_NODE}/update_motion_gen_config"


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
        self._get_params_client = self.create_client(
            GetParameters, GET_PARAMETERS_SRV
        )
        self._params_client = self.create_client(
            SetParameters, SET_PARAMETERS_SRV
        )
        self._rebuild_client = self.create_client(
            Trigger, UPDATE_MOTION_GEN_CONFIG_SRV
        )

        for label, client in (
            ("generate_trajectory", self._traj_client),
            ("add_object", self._add_client),
            ("remove_all_objects", self._clear_client),
            ("set_collision_cache", self._cache_client),
            ("get_parameters", self._get_params_client),
            ("set_parameters", self._params_client),
            ("update_motion_gen_config", self._rebuild_client),
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

    def set_server_torque_mode(
        self,
        load_dynamics: bool,
        payload_mass: float,
        rebuild_timeout: float = 300.0,
    ) -> None:
        """Switch the running server's torque-limited planning mode at runtime.

        The server's dynamics mode is baked into its ``robot_cfg`` at solver
        build time, so this sets the ``load_dynamics`` / ``robot_payload_mass``
        node parameters through the stock ``set_parameters`` service and then
        triggers ``update_motion_gen_config`` (``std_srvs/Trigger`` — the same
        build-time-parameter path as ``max_goalset`` / ``num_trajopt_seeds``,
        see docs/concepts/parameters.md) to rebuild the MotionPlanner from
        them. No new wire schema: both services are standard ROS2 interfaces.

        Idempotent: the server's current parameter values are queried first,
        and the expensive rebuild is only triggered when the mode actually
        changes — the without-torque leg against an already-plain server costs
        nothing, and the with-torque leg pays one rebuild (~20-60 s:
        allocation + CUDA-graph re-capture; ``rebuild_timeout`` defaults
        high). This is what lets a single server life fill BOTH motion ROS
        rows of the reference page reproduction.
        """
        # Query the server's current values; skip the expensive rebuild when
        # they already match this leg's requested mode.
        get_req = GetParameters.Request()
        get_req.names = ["load_dynamics", "robot_payload_mass"]
        current = self._call(
            self._get_params_client, get_req, timeout=rebuild_timeout
        )
        if len(current.values) == 2:
            dyn = current.values[0].bool_value
            mass = current.values[1].double_value
            if bool(dyn) == bool(load_dynamics) and abs(
                mass - float(payload_mass)
            ) < 1e-9:
                self.get_logger().info(
                    f"Server already in the requested torque mode "
                    f"(load_dynamics={str(load_dynamics).lower()}, "
                    f"robot_payload_mass={payload_mass}) — no rebuild needed"
                )
                return

        self.get_logger().info(
            f"Switching server torque mode: load_dynamics="
            f"{str(load_dynamics).lower()}, robot_payload_mass={payload_mass} "
            f"— triggering update_motion_gen_config rebuild (~20-60 s)"
        )
        set_req = SetParameters.Request()
        set_req.parameters = [
            Parameter(
                name="load_dynamics",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_BOOL,
                    bool_value=bool(load_dynamics),
                ),
            ),
            Parameter(
                name="robot_payload_mass",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_DOUBLE,
                    double_value=float(payload_mass),
                ),
            ),
        ]
        set_resp = self._call(
            self._params_client, set_req, timeout=rebuild_timeout
        )
        failed = [res.reason for res in set_resp.results if not res.successful]
        if failed:
            raise RuntimeError(
                f"set_parameters failed switching torque mode: {failed}"
            )
        self._call(
            self._rebuild_client, Trigger.Request(), timeout=rebuild_timeout
        )
        self.get_logger().info("update_motion_gen_config rebuild complete")

    def set_server_max_attempts(
        self, attempts: int, timeout: float = 30.0
    ) -> None:
        """Pin the server's plan_pose retry budget to match this run's.

        ``max_attempts`` is a **plan-time** parameter: the node reads it fresh
        per request (``_get_planner_config`` -> ``plan_pose``), so a plain
        ``set_parameters`` takes effect immediately — no solver rebuild, unlike
        the torque mode. The runner pins it to the run's own ``--max-attempts``
        so native and ROS legs always share one retry envelope: the whole
        benchmark defaults to the page's 100-attempt budget (its 99.73 %
        success). Idempotent — the current value is queried first and
        nothing is sent when the server already matches.
        """
        get_req = GetParameters.Request()
        get_req.names = ["max_attempts"]
        current = self._call(
            self._get_params_client, get_req, timeout=timeout
        )
        if len(current.values) == 1 and (
            current.values[0].integer_value == int(attempts)
        ):
            self.get_logger().info(
                f"Server max_attempts already {attempts} — no change needed"
            )
            return

        self.get_logger().info(
            f"Pinning server max_attempts to {attempts} "
            f"(plan-time parameter — no rebuild needed)"
        )
        set_req = SetParameters.Request()
        set_req.parameters = [
            Parameter(
                name="max_attempts",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_INTEGER,
                    integer_value=int(attempts),
                ),
            ),
        ]
        set_resp = self._call(
            self._params_client, set_req, timeout=timeout
        )
        failed = [res.reason for res in set_resp.results if not res.successful]
        if failed:
            raise RuntimeError(
                f"set_parameters failed pinning max_attempts: {failed}"
            )

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

    @staticmethod
    def _client_energy_torque(
        waypoints, waypoint_velocities, dt, robot_model_data,
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
        self, problem: Dict[str, Any], problem_name: str, scene_key: str,
        timeout: float = 120.0, robot_model_data=None,
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
        n, path_length, motion_time = trajectory_metrics(waypoints, float(result.dt))
        entry["n_waypoints"] = n
        entry["path_length"] = path_length
        entry["motion_time_s"] = motion_time
        entry["solve_time_s"] = self._winner_solve_time(result)
        entry["jerk"] = trajectory_jerk(waypoints, float(result.dt))
        entry["position_error_mm"] = self._winner_position_error_mm(result)
        if robot_model_data is not None:
            try:
                velocities = None
                first_wp = result.trajectory[0] if result.trajectory else None
                if first_wp is not None and list(
                    getattr(first_wp, "velocity", []) or []
                ):
                    velocities = [list(w.velocity) for w in result.trajectory]
                energy_j, torque_nm = self._client_energy_torque(
                    waypoints, velocities, float(result.dt), robot_model_data,
                )
                entry["energy_j"] = energy_j
                entry["torque_nm"] = torque_nm
            except Exception as exc:  # noqa: BLE001 - degrade, not fail
                self.get_logger().warn(
                    f"Failed to reconstruct energy/torque: {exc}"
                )
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
    use_dynamics: bool = False,
    mass: float = 3.0,
    server_dynamics: Optional[bool] = None,
    server_payload_mass: float = 0.0,
    server_max_attempts: Optional[int] = None,
    verbose: bool = True,
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

    ``use_dynamics`` / ``mass`` label this run as the torque-limited variant at
    the given payload mass (the page's "with torque limits" table; the server
    runs torque-limited planning from its own loaded robot config — nothing
    changes on the wire). Energy (J) / Torque (N·m) are reconstructed
    client-side from every successful returned trajectory at ``mass`` via the
    upstream Pinocchio model (see ``_client_energy_torque``); when the model
    can't be loaded (e.g. pinocchio absent) the columns stay ``None`` and the
    compare rows are omitted.

    ``server_dynamics`` / ``server_payload_mass`` (default ``None`` / ``0.0``)
    drive the runtime torque-mode switch: when ``server_dynamics`` is given,
    the leg FIRST ensures the running server's solver is in that mode —
    plain, or torque-limited at ``server_payload_mass`` — via
    ``RosBenchmarkRunner.set_server_torque_mode`` (stock ``set_parameters`` +
    ``update_motion_gen_config`` trigger, no relaunch). The webpage
    reproduction passes ``False/0.0`` for the without-torque ROS leg and
    ``True/mass`` for the with-torque leg, so one server life fills both rows
    and nothing is skipped.

    ``server_max_attempts`` (default ``None`` = leave the server's param
    alone) pins the server's plan-time ``max_attempts`` to this value via
    ``set_server_max_attempts`` (one ``set_parameters``, no rebuild) so the
    ROS leg retries exactly like its native counterpart. Every benchmark
    subcommand passes its own ``--max-attempts`` — default 100, the page's
    budget, for the whole benchmark.
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

        if server_dynamics is not None:
            # Ensure the server's solver is in the mode this leg needs (see
            # set_server_torque_mode): the without-torque leg asks for plain,
            # the with-torque leg asks for torque limits + the payload mass.
            # Idempotent — no rebuild when the server already matches.
            node.set_server_torque_mode(
                bool(server_dynamics), float(server_payload_mass or 0.0)
            )

        if server_max_attempts is not None:
            # Pin the server's plan_pose retry budget to this run's (see
            # set_server_max_attempts): every subcommand defaults both legs to
            # the page's 100-attempt budget. Plan-time parameter — one
            # set_parameters, no rebuild, idempotent.
            node.set_server_max_attempts(int(server_max_attempts))

        robot_model_data = _load_client_dynamics_model(node, mass)

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
                node.plan_one(
                    first_problem, f"{scene_key}_{i}", scene_key,
                    timeout=call_timeout, robot_model_data=robot_model_data,
                )
            except RuntimeError as exc:
                node.get_logger().warn(f"Warmup probe failed (continuing): {exc}")

        results: List[Dict[str, Any]] = []
        for scene_key, i, problem in ready_problems:
            problem_name = f"{scene_key}_{i}"
            if verbose:
                node.get_logger().info(f"Solving {problem_name} ...")
            node.clear_world(timeout=call_timeout)
            node.add_world(problem["obstacles"], timeout=call_timeout)
            results.append(
                node.plan_one(
                    problem, problem_name, scene_key,
                    timeout=call_timeout, robot_model_data=robot_model_data,
                )
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


def _load_client_dynamics_model(node, mass: float):
    """Load the upstream Pinocchio dynamics model for the client-side
    Energy/Torque reconstruction (``load_robot_model_for_dynamics`` with the
    run's ``mass``). Returns ``None`` — columns left ``None`` — when the
    model can't be loaded (pinocchio/curobo not available, upstream helper
    absent, or load failure), so a hiccup degrades to no rows instead of
    failing the run.
    """
    try:
        from .core_runner import _reference_benchmark_module

        reference = _reference_benchmark_module()
        loader = getattr(reference, "load_robot_model_for_dynamics", None)
        if loader is None:
            node.get_logger().warn(
                "load_robot_model_for_dynamics not found in the reference "
                "benchmark — Energy/Torque rows omitted"
            )
            return None
        model_data = loader(robot_name="franka", attached_object_mass=mass)
        node.get_logger().info(
            f"Client dynamics model loaded (franka, attached "
            f"object mass={mass} kg) — Energy/Torque will be reconstructed "
            f"from returned trajectories"
        )
        return model_data
    except Exception as exc:  # noqa: BLE001 - degrade, not fail
        node.get_logger().warn(
            f"Dynamics model unavailable ({exc}) — Energy/Torque rows omitted"
        )
        return None
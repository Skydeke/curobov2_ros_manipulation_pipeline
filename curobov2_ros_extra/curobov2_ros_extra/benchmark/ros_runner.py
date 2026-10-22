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



from curobov2_ros_extra.benchmark.ros_runner_server_mixin import _RosRunnerServerMixin
from curobov2_ros_extra.benchmark.ros_runner_world_mixin import _RosRunnerWorldMixin

class RosBenchmarkRunner(_RosRunnerServerMixin, _RosRunnerWorldMixin, Node):

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
        self._get_params_client = self.create_client(GetParameters, GET_PARAMETERS_SRV)
        self._params_client = self.create_client(SetParameters, SET_PARAMETERS_SRV)
        self._rebuild_client = self.create_client(Trigger, UPDATE_MOTION_GEN_CONFIG_SRV)
        self._lock_client = self.create_client(SetJointLocks, SET_JOINT_LOCKS_SRV)

        for label, client in (
            ("generate_trajectory", self._traj_client),
            ("add_object", self._add_client),
            ("remove_all_objects", self._clear_client),
            ("set_collision_cache", self._cache_client),
            ("get_parameters", self._get_params_client),
            ("set_parameters", self._params_client),
            ("update_motion_gen_config", self._rebuild_client),
            ("set_joint_locks", self._lock_client),
        ):
            if not client.wait_for_service(timeout_sec=service_timeout):
                self.get_logger().error(f"{label} service not available")
                raise RuntimeError(f"{label} service not available")
            self.get_logger().info(f"Connected to {client.srv_name}")

    # ------------------------------------------------------------------
    # service plumbing
    # ------------------------------------------------------------------

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
    server_reset_seed: Optional[bool] = None,
    apply_mpinets_locks: bool = True,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Run the ROS-wrapped leg over a robometrics dataset. ``scene`` restricts the run to one scene key within the dataset
    (``None`` = all scenes). One warmup probe (first valid problem, result
    discarded) is sent before the timed run so CUDA-graph warmup / first-solve
    JIT on the server does not skew (or stall) the measurements."""
    rclpy.init()
    node: Optional[RosBenchmarkRunner] = None
    switched_locks = False
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

        if server_reset_seed is not None:
            # Same idea for the RNG rewind: the server only rewinds before a
            # plan when its reset_seed_per_plan parameter says so, and the
            # compose launch turns it on. Pin it from the run's own seed
            # diagnostic so this leg matches the native one whatever the server
            # was launched with (plan-time — one set_parameters, no rebuild).
            node.set_server_reset_seed(bool(server_reset_seed))

        robot_model_data = _load_client_dynamics_model(node, mass)

        lock_joint_names = None
        if apply_mpinets_locks:
            lock_joint_names = node.server_lock_joint_names()
        else:
            node.get_logger().info(
                "apply_mpinets_locks=False: leaving the server's gripper locks "
                "at the robot config value for every scene (diagnostic)"
            )
        mpinets_scenes = mpinets_scene_keys() if dataset == "full" else None

        # Upstream builds one planner per dataset file, and the only robot-config
        # difference is the gripper lock. ready_problems is scene-major, so this
        # fires once per dataset block, and the server no-ops the switch when the
        # value already matches.
        active_lock = None

        def apply_lock(scene_key: str) -> None:
            nonlocal active_lock, switched_locks
            if lock_joint_names is None:
                return
            lock = mpinets_lock_for_scene(scene_key, mpinets_scenes, dataset)
            if lock != active_lock:
                node.set_server_lock_joints(
                    [lock] * len(lock_joint_names), joint_names=lock_joint_names
                )
                active_lock = lock
                switched_locks = True

        # Per-scene cache sizes (native parity, see above). Fires on scene
        # changes with new counts only; returns True when it rebuilt.
        active_cache: Optional[tuple] = None

        def apply_cache(scene_key: str) -> bool:
            nonlocal active_cache
            if not size_collision_cache:
                return False
            need = collision_cache_sizes({scene_key: problems[scene_key]})
            if need == active_cache:
                return False
            node.size_collision_cache({scene_key: problems[scene_key]},
                                      timeout=call_timeout)
            active_cache = need
            return True

        # Cache sizing happens per SCENE inside the loop (apply_cache): upstream
        # builds one planner per dataset file with check_problems' exact OBB
        # count, and a persistent server matches those kernel grids only by
        # rebuilding for each scene's exact counts. Consecutive scenes with
        # identical counts skip the rebuild.
        if not size_collision_cache:
            node.get_logger().warn(
                "size_collision_cache=False: leaving the server's default "
                "collision cache in place (padded kernel grids — diagnostic)"
            )

        # PORT of upstream motion_plan_benchmark: the per-problem world the
        # NATIVE leg builds is SceneCfg.create(obstacles).get_obb_world(). Convert
        # the problems' obstacles ONCE here so the ROS leg collides against the
        # same world - and so the collision cache is sized for what is actually
        # pushed (all cuboids) rather than for the raw primitives/meshes.
        from curobov2_ros_extra.benchmark.obstacle_convert import (
            native_obb_obstacles,
        )

        for _scene_problems in problems.values():
            for _p in _scene_problems:
                if _p.get("obstacles"):
                    _p["obstacles"] = native_obb_obstacles(_p["obstacles"])
        node.get_logger().info(
            "world: converted problem obstacles with the native leg's "
            "get_obb_world() so both legs collide against identical geometry")

        ready_problems = [
            (scene_key, i, p)
            for scene_key, scene_problems in problems.items()
            for i, p in enumerate(scene_problems, start=1)
            if p.get("collision_buffer_ik", 0.0) >= 0.0
        ]
        if not ready_problems:
            return []

        current_hash: Optional[str] = None
        world_sets = 0
        bulk_client = node.bulk_client_or_none(service_timeout)
        if bulk_client is not None:
            node.get_logger().info(
                "Using set_objects bulk scene API "
                "(one solver refresh per scene)")

        def ensure_world(obstacles: Dict[str, Any]) -> None:
            """Push the world only when it differs from the registered scene.

            Prefers one ``set_objects`` call (single solver-world refresh);
            falls back to the clear + per-object loop on servers without the
            service.
            """
            nonlocal current_hash, world_sets
            digest = obstacle_hash(obstacles)
            if digest == current_hash:
                return
            if bulk_client is not None and node.set_world_bulk(
                    bulk_client, obstacles, timeout=call_timeout):
                current_hash = digest
                world_sets += 1
                return
            node.clear_world(timeout=call_timeout)
            node.add_world(obstacles, timeout=call_timeout)
            current_hash = digest
            world_sets += 1

        def warmup_probe_for(scene_key: str, i: int, first_problem: Dict[str, Any]) -> None:
            """One discarded plan so CUDA-graph warmup / first-solve JIT never
            lands in a timed problem (native warms 3 solves per scene)."""
            node.get_logger().info(
                f"Warmup probe: {scene_key}_{i} (world clear + add + plan)"
            )
            ensure_world(first_problem["obstacles"])
            try:
                node.plan_one(
                    first_problem,
                    f"{scene_key}_{i}",
                    scene_key,
                    timeout=call_timeout,
                    robot_model_data=robot_model_data,
                )
            except RuntimeError as exc:
                node.get_logger().warn(f"Warmup probe failed (continuing): {exc}")

        if warmup_probe:
            scene_key, i, first_problem = ready_problems[0]
            # Switch to the first scene's lock BEFORE warming up, so the probe
            # exercises the geometry the run actually solves with (the rebuild
            # it triggers is the same one the loop would have paid anyway).
            # Size the cache for this scene first for the same reason.
            apply_lock(scene_key)
            apply_cache(scene_key)
            warmup_probe_for(scene_key, i, first_problem)

        results: List[Dict[str, Any]] = []
        _split_acc = {"world": 0.0, "plan": 0.0, "n": 0}
        for scene_key, i, problem in ready_problems:
            problem_name = f"{scene_key}_{i}"
            if verbose:
                node.get_logger().info(f"Solving {problem_name} ...")
            apply_lock(scene_key)
            if warmup_probe and apply_cache(scene_key):
                # Fresh solver/graphs for this scene's exact cache: warm them
                # with a discarded probe so no timed problem pays first-solve
                # JIT (native warms per scene the same way).
                warmup_probe_for(scene_key, i, problem)
            _t0 = time.perf_counter()
            ensure_world(problem["obstacles"])
            _t1 = time.perf_counter()
            results.append(
                node.plan_one(
                    problem,
                    problem_name,
                    scene_key,
                    timeout=call_timeout,
                    robot_model_data=robot_model_data,
                )
            )
            _t2 = time.perf_counter()
            # Per-problem runner-side split (world-set vs plan-call round
            # trips, measured client-side). Logged as a running average so a
            # systematic gap between server-measured spans shows up here.
            _split_acc["world"] += _t1 - _t0
            _split_acc["plan"] += _t2 - _t1
            _split_acc["n"] += 1
            if verbose and _split_acc["n"] % 25 == 0:
                _n = _split_acc["n"]
                node.get_logger().info(
                    f"runner-split avg over {_n}: "
                    f"world-set {_split_acc['world'] / _n * 1e3:.0f} ms, "
                    f"plan-call {_split_acc['plan'] / _n * 1e3:.0f} ms"
                )

        node.get_logger().info(
            f"ROS leg done: {len(results)} problems, "
            f"{sum(1 for r in results if r['success'])} succeeded "
            f"({world_sets} world sets)"
        )
        return results
    finally:
        if node is not None and switched_locks:
            # Leave the server on the robot config's own lock values, the state
            # it was launched in: a later leg on the same server (the webpage's
            # IK / kinematics suites, say) must not inherit this run's
            # dataset-specific gripper. No-ops when already restored.
            try:
                node.restore_server_lock_joints()
            except Exception as exc:  # noqa: BLE001 - never mask the run's result
                node.get_logger().warn(
                    f"Could not restore the server's joint locks: {exc}"
                )
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

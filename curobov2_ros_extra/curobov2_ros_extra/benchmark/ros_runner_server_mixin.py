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



class _RosRunnerServerMixin:
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
        """Switch the running server's torque-limited planning mode at runtime. The server's dynamics mode is baked into its ``robot_cfg`` at solver
        build time, so this sets the ``load_dynamics`` / ``robot_payload_mass``
        node parameters through the stock ``set_parameters`` service and then
        triggers ``update_motion_gen_config`` (``std_srvs/Trigger`` — the same
        build-time-parameter path as ``max_goalset`` / ``num_trajopt_seeds``,
        the node's ROS parameters) to rebuild the MotionPlanner from
        them. No new wire schema: both services are standard ROS2 interfaces."""
        # Query the server's current values; skip the expensive rebuild when
        # they already match this leg's requested mode.
        get_req = GetParameters.Request()
        get_req.names = ["load_dynamics", "robot_payload_mass"]
        current = self._call(self._get_params_client, get_req, timeout=rebuild_timeout)
        if len(current.values) == 2:
            dyn = current.values[0].bool_value
            mass = current.values[1].double_value
            if (
                bool(dyn) == bool(load_dynamics)
                and abs(mass - float(payload_mass)) < 1e-9
            ):
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
        set_resp = self._call(self._params_client, set_req, timeout=rebuild_timeout)
        failed = [res.reason for res in set_resp.results if not res.successful]
        if failed:
            raise RuntimeError(f"set_parameters failed switching torque mode: {failed}")
        self._call(self._rebuild_client, Trigger.Request(), timeout=rebuild_timeout)
        self.get_logger().info("update_motion_gen_config rebuild complete")

    def server_lock_joint_names(self, rebuild_timeout: float = 300.0) -> List[str]:
        """Return the joints the server's robot config locks at launch.

        Asks the server to drop any override and reload the robot YAML's own
        lock values, which is the server's launch state, so this costs nothing
        beyond one round trip — and it means the caller learns the lock-joint
        names from the server instead of hardcoding the Franka's two fingers.
        Captured once per run; ``set_server_lock_joints`` then names those
        joints with one value each.
        """
        names = self.restore_server_lock_joints(rebuild_timeout=rebuild_timeout)
        if not names:
            raise RuntimeError(
                "server's robot config declares no kinematics.lock_joints — "
                "nothing to switch"
            )
        return names

    def restore_server_lock_joints(self, rebuild_timeout: float = 300.0) -> List[str]:
        """Drop the server's lock override, reloading the robot YAML's values. The inverse of :meth:`set_server_lock_joints`, and the way a run hands
        the server back in its launch state. Idempotent server-side."""
        request = SetJointLocks.Request()
        request.restore = True
        response = self._call(self._lock_client, request, timeout=rebuild_timeout)
        if not response.success:
            raise RuntimeError(f"set_joint_locks failed: {response.message}")
        self.get_logger().info(
            f"set_joint_locks (restore): {dict(zip(response.joint_names, response.positions))}"
            f" — {response.message}"
        )
        return list(response.joint_names)

    def set_server_lock_joints(
        self,
        positions: Sequence[float],
        joint_names: Sequence[str],
        rebuild_timeout: float = 300.0,
    ) -> None:
        """Set the running server's gripper joint locks at runtime. Upstream's reference benchmark builds a SEPARATE planner per dataset
        file, and the only difference in the robot config is the gripper lock:
        ``load_curobo`` pins ``panda_finger_joint1/2`` at **0.025 for the
        mpinets file** and leaves the bundled ``franka.yml`` value (**0.04**)
        in place for motion_benchmaker. That value is collision geometry, not
        a tuning knob — it is how wide the gripper's collision spheres are, so
        at 0.04 the goal configuration of a tight mpinets reach can itself sit
        in collision and the plan fails where the native leg succeeds."""
        request = SetJointLocks.Request()
        request.joint_names = [str(name) for name in joint_names]
        request.positions = [float(v) for v in positions]
        self.get_logger().info(
            f"Setting server joint locks {dict(zip(request.joint_names, request.positions))} — "
            "rebuilds the robot model and solvers only if they changed"
        )
        response = self._call(self._lock_client, request, timeout=rebuild_timeout)
        if not response.success:
            raise RuntimeError(f"set_joint_locks failed: {response.message}")
        self.get_logger().info(
            f"set_joint_locks: {dict(zip(response.joint_names, response.positions))}"
            f" — {response.message}"
        )

    def set_server_max_attempts(self, attempts: int, timeout: float = 30.0) -> None:
        """Pin the server's plan_pose retry budget to match this run's. ``max_attempts`` is a **plan-time** parameter: the node reads it fresh
        per request (``_get_planner_config`` -> ``plan_pose``), so a plain
        ``set_parameters`` takes effect immediately — no solver rebuild, unlike
        the torque mode. The runner pins it to the run's own ``--max-attempts``
        so native and ROS legs always share one retry envelope: the whole
        benchmark defaults to the page's 100-attempt budget (its 99.73 %
        success)."""
        get_req = GetParameters.Request()
        get_req.names = ["max_attempts"]
        current = self._call(self._get_params_client, get_req, timeout=timeout)
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
        set_resp = self._call(self._params_client, set_req, timeout=timeout)
        failed = [res.reason for res in set_resp.results if not res.successful]
        if failed:
            raise RuntimeError(f"set_parameters failed pinning max_attempts: {failed}")

    def set_server_reset_seed(self, enabled: bool, timeout: float = 30.0) -> None:
        """Pin the server's per-plan RNG rewind to this run's. ``reset_seed_per_plan`` is a **plan-time** parameter: the node reads it
        fresh per request (``_get_planner_config`` -> ``SinglePlanner.plan``),
        so a plain ``set_parameters`` takes effect immediately and nothing is
        re-parsed or rebuilt. The runner pins it from the same ``--no-reset-seed``
        / ``--unseeded`` diagnostic that drives the native leg, so the two legs
        rewind (or both drift) and the seed-rewind A/B is a real two-leg
        comparison instead of a one-sided one."""
        get_req = GetParameters.Request()
        get_req.names = ["reset_seed_per_plan"]
        current = self._call(self._get_params_client, get_req, timeout=timeout)
        if len(current.values) == 1 and bool(current.values[0].bool_value) == bool(
            enabled
        ):
            self.get_logger().info(
                f"Server reset_seed_per_plan already {bool(enabled)} — no "
                "change needed"
            )
            return

        self.get_logger().info(
            f"Pinning server reset_seed_per_plan to {bool(enabled)} "
            "(plan-time parameter — no rebuild needed)"
        )
        set_req = SetParameters.Request()
        set_req.parameters = [
            Parameter(
                name="reset_seed_per_plan",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_BOOL,
                    bool_value=bool(enabled),
                ),
            ),
        ]
        set_resp = self._call(self._params_client, set_req, timeout=timeout)
        failed = [res.reason for res in set_resp.results if not res.successful]
        if failed:
            raise RuntimeError(
                f"set_parameters failed pinning reset_seed_per_plan: {failed}"
            )
        self.get_logger().info(
            f"Server reset_seed_per_plan = {bool(enabled)} (the server's "
            "launch default, set here so both legs rewind alike)"
        )

    # ------------------------------------------------------------------
    # collision cache
    # ------------------------------------------------------------------

    def size_collision_cache(self, problems, timeout: float = 120.0) -> None:
        """Size the server's collision cache to the dataset (native parity). curobo's Warp collision kernels launch one thread per (robot sphere,
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
        grids."""
        # The server refuses a shrink below the loaded scene (it would
        # leave the solvers blind), so the world goes first: empty fits
        # every size, and the scene is pushed back right after the rebuild.
        self.clear_world(timeout=timeout)
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

    def bulk_client_or_none(self, service_timeout: float = 30.0):
        """A ``set_objects`` client, or None when the server lacks the service.

        Older servers only offer clear/add_object, which callers fall back
        to — hence optional here instead of a hard constructor requirement.
        """
        client = self.create_client(SetObjects, SET_OBJECTS_SRV)
        if client.wait_for_service(timeout_sec=min(float(service_timeout), 5.0)):
            return client
        self.destroy_client(client)
        return None

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
    '_RosRunnerServerMixin',
]

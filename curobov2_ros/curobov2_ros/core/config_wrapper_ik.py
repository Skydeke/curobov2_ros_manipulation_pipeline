# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config wrapper for the cuRobo v2 standalone IK solver (`InverseKinematics`).

Same pattern as ``core/config_wrapper_motion.py`` (ConfigWrapperMotion): this
wrapper owns the solver's build-time configuration and lifecycle — seed
budget, CUDA-graph mode, tolerances — resolved from ROS params, plus the
``update_ik_config`` rebuild service. ``core/ik_services.py`` (IKServices) is
the thin ROS service layer over it (``/ik``, ``/ik_batch``, ``/warmup_ik``),
just as the planner classes plan over the MotionPlanner the motion wrapper
builds.

Shared context: the wrapper adopts the motion wrapper's managers (robot
model, obstacles, scene, collision cache, perception) by reference instead of
rebuilding them, so there is exactly one live robot/scene context per node —
the same reason the reactive (MPC) solver builds from the shared
ConfigWrapperMotion context rather than owning one. It must therefore be
constructed with the node's already-built ``ConfigWrapperMotion``, never
standalone: there is deliberately no second ``ConfigManager`` /
``RobotModelManager`` / ``ObstacleManager`` here (a second set would re-parse
the URDF, re-allocate the voxel grid, and double-register ROS services).

v2 notes:
- IKSolver / IKSolverConfig → InverseKinematics / InverseKinematicsCfg.
- CollisionCheckerType is gone; collision is part of InverseKinematicsCfg.create.
- load_from_robot_config(...) replaced by Cfg.create(robot=<shared cfg>, ...).
"""

from contextlib import nullcontext
from functools import partial

import torch
from std_srvs.srv import Trigger

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.scene import Scene
from curobo.types import JointState as CuRoboJS

from curobov2_ros_interfaces.srv import GetCollisionDistance

from .config_wrapper import ConfigWrapper, resolve_use_cuda_graph
from .collision_distance import _compute_sphere_distance


class ConfigWrapperIK(ConfigWrapper):
    """Standalone IK solver config wrapper (v2 `InverseKinematics`)."""

    def __init__(self, node, config_motion: ConfigWrapper) -> None:
        # Adopt the shared context — see the module docstring. Every property
        # inherited from ConfigWrapper (scene, collision_cache, kin_model,
        # _device, _ops_dtype, ...) reads through these references, so both
        # wrappers always see the same robot and world.
        self.node = node
        self.robot = config_motion.robot
        self.config_manager = config_motion.config_manager
        self.robot_model_manager = config_motion.robot_model_manager
        self.obstacle_manager = config_motion.obstacle_manager
        self.camera_system_manager = config_motion.camera_system_manager
        self.laser_system_manager = config_motion.laser_system_manager
        # Shared, not rebuilt: the service manager was constructed against the
        # motion wrapper (world-update callbacks, robot strategies). A second
        # one would double-register services.
        self.ros_service_manager = config_motion.ros_service_manager

        # v2 IK tunables, mirroring ConfigWrapperMotion's motion tunables.
        self.num_ik_seeds = self._resolve_num_ik_seeds(node)
        # ROS param 'use_cuda_graph' (default True), overridable via the
        # CUROBO_USE_CUDA_GRAPH env var — the same flag the MotionPlanner
        # follows, so one param governs every solver's graph capture.
        self.use_cuda_graph = resolve_use_cuda_graph(node)
        self.position_tolerance = 0.005
        self.orientation_tolerance = 0.05
        # Collision checking in the standalone solver (self + scene), read
        # from the node's ``ik_collision_check`` param (default True). False
        # = pure-kinematics IK matching the official MoveIt methodology
        # (sceneless plugin call): solver built with self_collision_check=False
        # and no scene/cache, world pushes become no-ops for it. Baked in at
        # build time like num_ik_seeds (update_ik_config rebuild).
        self.collision_check = self._resolve_collision_check(node)
        self.use_lm_seed = self._resolve_use_lm_seed(node)

        # Live solver state. The solver is created only on build() (startup
        # warmup or on-demand /warmup_ik); batch 0 means "not yet warmed up".
        self.ik_solver: InverseKinematics | None = None
        self.ik_batch_size: int = 0
        self.ik_num_seeds: int = 0
        self.ik_use_cuda_graph: bool | None = None
        self.ik_collision_check: bool | None = None
        self.ik_use_lm_seed: bool | None = None

        self.ik_config_srv = node.create_service(
            Trigger,
            node.get_name() + "/update_ik_config",
            partial(self.set_ik_config, node),
        )

    def _resolve_num_ik_seeds(self, node) -> int:
        """Per-pose IK seed budget, read from the node's ``num_ik_seeds``.

        Shared with the planner's internal IK (``ConfigWrapperMotion``): the
        standalone ``/ik`` / ``/ik_batch`` services resolve a goal pose with
        the SAME seed count the MotionPlanner's IK uses for pose/goalset
        goals, so both IK paths behave identically. ``num_seeds`` / VRAM scale
        as batch x seeds. Default 32 (cuRobo's default at
        ``max_batch_size == 1``). Baked into solver buffers / the CUDA graph
        at build time, so a runtime change takes effect via the
        ``update_ik_config`` rebuild.
        """
        if node.has_parameter("num_ik_seeds"):
            return max(
                1,
                int(
                    node.get_parameter("num_ik_seeds")
                    .get_parameter_value()
                    .integer_value
                ),
            )
        return 32

    def _resolve_collision_check(self, node) -> bool:
        """Whether the standalone IK solver checks collisions.

        Read from the node's ``ik_collision_check`` param (default True).
        False gives a pure-kinematics solve (official MoveIt methodology):
        no self-collision, no scene, no collision cache.
        """
        if node.has_parameter("ik_collision_check"):
            return bool(
                node.get_parameter("ik_collision_check")
                .get_parameter_value()
                .bool_value
            )
        return True

    def _resolve_use_lm_seed(self, node) -> bool:
        """Whether the standalone IK solver runs LM seed generation.

        Read from the node's ``ik_use_lm_seed`` param (default True).
        """
        if node.has_parameter("ik_use_lm_seed"):
            return bool(
                node.get_parameter("ik_use_lm_seed")
                .get_parameter_value()
                .bool_value
            )
        return True

    def build(self, batch_size: int, num_seeds: int | None = None) -> None:
        """Create (or recreate) the IK solver for the given batch size/seeds."""
        if num_seeds is None:
            num_seeds = self._resolve_num_ik_seeds(self.node)
        # Build against an EMPTY scene, never the live one: after a cache
        # shrink the live world may not fit the new cache, and constructing
        # against it raises "Cannot load N cuboids" (observed live, killing a
        # benchmark rebuild). Empty always fits; the real scene is pushed
        # right below.
        scene = Scene()

        self.node.get_logger().info(
            f"Initializing IK solver (batch_size={batch_size}, num_seeds={num_seeds})..."
        )

        # Single shared curobo kinematic from RobotModelManager (one URDF
        # parse for the whole node) — do not re-build from the YAML path.
        # num_seeds runs that many parallel optimisation trajectories per
        # pose in the batch, so GPU memory grows with batch x seeds.
        # Collision mode resolves live (like num_seeds): a runtime param flip
        # takes effect on the next build/rebuild without relaunch.
        collision_check = self._resolve_collision_check(self.node)
        use_lm_seed = self._resolve_use_lm_seed(self.node)
        use_cuda_graph = resolve_use_cuda_graph(self.node)
        cfg = InverseKinematicsCfg.create(
            robot=self.robot_model_manager.robot_cfg,
            scene_model=scene if collision_check else None,
            num_seeds=num_seeds,
            position_tolerance=self.position_tolerance,
            orientation_tolerance=self.orientation_tolerance,
            self_collision_check=collision_check,
            collision_cache=self.collision_cache if collision_check else None,
            use_cuda_graph=use_cuda_graph,
            max_batch_size=max(1, batch_size),
        )
        # use_lm_seed is a dataclass default (True), not a create() kwarg.
        cfg.use_lm_seed = use_lm_seed
        self.node.get_logger().info(
            "IK solver envelope: "
            f"use_cuda_graph={use_cuda_graph}, "
            f"num_ik_seeds={num_seeds}, "
            f"collision_check={collision_check}, "
            f"use_lm_seed={use_lm_seed}, "
            f"batch_size={max(1, batch_size)}"
        )
        self.ik_solver = InverseKinematics(cfg)
        self.ik_batch_size = batch_size
        self.ik_num_seeds = num_seeds
        self.ik_use_cuda_graph = use_cuda_graph
        self.ik_collision_check = collision_check
        self.ik_use_lm_seed = use_lm_seed
        # Keep the resolved tunables in sync with what was actually built, so
        # the next set_ik_config can tell a param change from a no-op.
        self.num_ik_seeds = num_seeds
        self.use_cuda_graph = use_cuda_graph
        self.collision_check = collision_check
        self.use_lm_seed = use_lm_seed

        # Warmup: solve a batch of random configs to prime CUDA kernels.
        q_sample = self.ik_solver.sample_configs(batch_size)
        js = CuRoboJS.from_position(
            q_sample, joint_names=self.ik_solver.kinematics.joint_names
        )
        kin_state = self.ik_solver.compute_kinematics(js)
        goal = kin_state.tool_poses.as_goal()
        self.ik_solver.solve_pose(goal_tool_poses=goal)
        # CPU/GPU ordering bridge — off by default (torch_sync param), see
        # node.torch_sync_enabled().
        sync = getattr(self.node, "torch_sync_enabled", None)
        if callable(sync) and sync():
            torch.cuda.synchronize()

        # Push the real scene after the empty build — collision mode only.
        # Best-effort: after a cache shrink the previous world may not fit
        # the new cache — warn loudly and continue empty; the next world
        # event (or an explicit world set) pushes a fitting world.
        # Same-cache rebuilds (batch-size changes, startup) always fit, so
        # those paths are unchanged.
        if collision_check:
            try:
                self.update_world(
                    self.obstacle_manager.primitives_only_scene())
            except Exception as e:
                self.node.get_logger().warn(
                    f"IK solver rebuilt empty: current world does not fit the new "
                    f"cache ({e}); push a fitting world before planning")

        self.node.get_logger().info("IK solver ready")

    def set_ik_config(self, node, _, response):
        """Build (or rebuild) the IK solver from the current ROS params.

        Called on demand via the `update_ik_config` service — the IK analogue
        of the motion wrapper's `update_motion_gen_config`: after
        `ros2 param set ... num_ik_seeds` (or `use_cuda_graph`,
        `ik_collision_check`, `ik_use_lm_seed`), triggering this service
        rebuilds the solver from the new values without a relaunch.
        """
        # (Re)captures CUDA graphs, so the whole body must hold gpu_lock —
        # the same invariant as every other graph-capturing path in the node.
        # Standalone nodes without the lock fall back to a no-op context.
        gpu_lock = getattr(node, "gpu_lock", None)
        lock_ctx = gpu_lock if gpu_lock is not None else nullcontext()
        with lock_ctx:
            self.build(max(1, self.ik_batch_size), num_seeds=None)

        if response is not None:
            response.success = True
            response.message = "IK solver config set"
        return response

    def needs_rebuild(self, batch_size: int, num_seeds: int) -> bool:
        """Whether the live solver no longer matches the requested config.

        Compares batch size, seed count, the currently resolved CUDA-graph
        mode (a `ros2 param set ... use_cuda_graph` flip), and the collision
        mode (an `ik_collision_check` flip) against the live solver, so a
        solve can rebuild lazily instead of serving a stale configuration.
        """
        return (
            batch_size != self.ik_batch_size
            or num_seeds != self.ik_num_seeds
            or resolve_use_cuda_graph(self.node) != self.ik_use_cuda_graph
            or self._resolve_collision_check(self.node) != self.ik_collision_check
            or self._resolve_use_lm_seed(self.node) != self.ik_use_lm_seed
        )

    def invalidate(self) -> None:
        """Drop the live solver after an unrecoverable build/solve failure."""
        self.ik_solver = None
        self.ik_batch_size = 0
        self.ik_num_seeds = 0
        self.ik_use_cuda_graph = None
        self.ik_collision_check = None
        self.ik_use_lm_seed = None

    def rebuild(self) -> None:
        """Recreate the IK solver after a collision-cache change.

        The collision cache is fixed at solver creation, so a cache change
        requires a rebuild (reusing the last batch size; seeds re-resolve from
        the current ``num_ik_seeds`` param). No-op if the solver was never
        initialized.
        """
        if self.ik_solver is None:
            return
        self.build(max(1, self.ik_batch_size), num_seeds=None)
        self.node.get_logger().info(
            "ConfigWrapperIK: solver rebuilt after cache change")

    def update_world(self, scene=None) -> None:
        """Propagate obstacle changes to the IK solver. No-op if not initialized.

        ``scene`` is the node's already-resolved solver scene from
        ``update_all_solvers_world`` (normalized to cuboid/mesh/voxel, and
        possibly degraded to primitives-only). It must be passed, because that
        method is where the two ESDF-withholding decisions live —

          * ``push_esdf_to_solvers:=false`` (diagnostic: solvers see analytic
            primitives only), and
          * ``collision_cache['voxel'] is None`` (``SetCollisionCache blox=0``,
            where a voxel-carrying scene would raise "Voxel cache not
            initialized" inside the solver's update_world).

        Deriving the scene from the obstacle manager here instead would ignore
        both decisions: /ik would keep seeing camera obstacles in diagnostic
        mode, and would raise on every world update under blox=0. The solver is
        built WITH ``collision_cache`` (see ``build``), so unlike
        ``FKServices`` it can accept the voxel layer.

        No-op when the solver runs collision-free (``ik_collision_check``):
        motion-leg world pushes must not touch a kinematics-only solver.
        """
        if self.ik_solver is None:
            return
        if self.ik_collision_check is False:
            return
        if scene is None:
            # Normalize primitives to solver-supported collision types
            # (sphere/cylinder/capsule -> mesh), or they silently don't collide.
            scene = self.obstacle_manager.collision_world_scene()
        self.ik_solver.update_world(scene)

    def update_world_config(self, node) -> None:
        """Push the current Scene into the IK solver (standalone fallback).

        Used when no node-level orchestration exists (see
        ``ConfigWrapper._register_world_observers``): the unified planner
        routes through ``update_all_solvers_world`` instead, which passes its
        already-resolved scene to ``update_world`` above.
        """
        self.update_world(self.obstacle_manager.collision_world_scene())

    def callback_get_collision_distance(
        self, node, request: GetCollisionDistance, response
    ):
        """Fill a GetCollisionDistance response (shared harness, same as motion)."""
        return _compute_sphere_distance(self, node, response)

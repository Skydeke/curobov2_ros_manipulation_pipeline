"""
Config wrapper for the cuRobo v2 motion planner (`MotionPlanner`).

IK is not wrapped here — it lives in ``core/ik_services.py``, built from the
same shared context. MPC is built by ``MPCController`` from this wrapper.

v2 notes:
- MotionGen/MpcSolver/IKSolver → MotionPlanner / ModelPredictiveControl / InverseKinematics.
- No more `MotionGenConfig` + `MotionGenPlanConfig` split — solver params live
  on `*Cfg.create(...)`, and per-call params are keyword args of `plan_pose`.
- Collision is integrated into `*Cfg.create()` via `scene_model`, `collision_cache`
  and `self_collision_check`. `CollisionCheckerType` no longer exists.
- `CollisionQueryBuffer` is gone; collision distance goes through the scene
  model directly.
"""

from contextlib import nullcontext
from functools import partial
import os
import rclpy

from std_srvs.srv import Trigger
from isaac_ros_cumotion_interfaces.srv import GetCollisionDistance

# v2 runtime flags — must be set before any cuRobo objects are instantiated.
# cuda_graph_reset lets solvers rebuild captured graphs when buffer shapes
# change between plan calls (e.g. different interpolated trajectory lengths).
# Without this, a second plan with a different horizon raises
# "CUDA graph reset is not available." Requires CUDA 12.0+.
# cuda_streams=False routes every cost/eval kernel onto the current (capture)
# stream instead of per-cost workspace streams. On solver rebuild the planner
# re-captures CUDA graphs; the workspace streams' record_event / wait_stream
# and the cuda_core_backend launches that touch a capturing stream are illegal
# and invalidate the capture (CUDA_ERROR_STREAM_CAPTURE_INVALIDATED), crashing
# the node whenever the collision cache is rebuilt. Keeping cuda_streams=True
# makes that rebuild crash nondeterministically.
# Note: curobo.runtime re-exports (and shadows) curobo._src.runtime values at
# import time — torch_util.is_cuda_graph_reset_available() reads from
# curobo.runtime and cuda_stream_util.cuda_stream_context() reads
# curobo.runtime.cuda_streams, so we must flip the flags on the public module too.
import curobo._src.runtime as _curobo_runtime
_curobo_runtime.cuda_graph_reset = True
import curobo.runtime as _curobo_runtime_public
_curobo_runtime_public.cuda_graph_reset = True

# Diagnostic only: CUROBO_DEBUG_CUDA_GRAPH=1 turns on PyTorch CUDA graph debug
# mode (torch.cuda.CUDAGraph.enable_debug_mode) so a capture-illegal operation
# is reported at the moment it happens — naming the exact culprit — instead of
# surfacing a stale CUDA_ERROR_STREAM_CAPTURE_INVALIDATED from a later launch.
# No effect when unset/0.
if os.environ.get("CUROBO_DEBUG_CUDA_GRAPH", "").strip().lower() not in (
    "",
    "0",
    "false",
    "no",
    "off",
):
    _curobo_runtime.debug_cuda_graphs = True
    _curobo_runtime_public.debug_cuda_graphs = True

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo._src.util.config_io import join_path, resolve_config
from curobo.content import get_task_configs_path

from .config_wrapper import ConfigWrapper, resolve_use_cuda_graph
from .collision_distance import _compute_sphere_distance, _query_sphere_collision


class ConfigWrapperMotion(ConfigWrapper):
    """Motion planner config wrapper (v2 `MotionPlanner`)."""

    def __init__(self, node, robot):
        super().__init__(node, robot)

        # v2 trajopt / IK / batch tunables
        self.num_ik_seeds = self._resolve_num_ik_seeds(node)
        self.num_trajopt_seeds = self._resolve_num_trajopt_seeds(node)
        # ROS param 'use_cuda_graph' (default True), overridable via the
        # CUROBO_USE_CUDA_GRAPH env var. Disabling avoids the MPC->Classic
        # captured-graph replay segfault at the cost of per-plan latency.
        self.use_cuda_graph = resolve_use_cuda_graph(node)
        # Solver recipe mirrors curobo core's benchmark: particle + LBFGS for
        # both IK and trajopt (applied unconditionally in set_motion_gen_config
        # below) — the same optimizer set the native benchmark leg runs, by
        # default in every startup.
        self.self_collision_check = True
        self.position_tolerance = 0.005
        self.orientation_tolerance = 0.05
        self.max_batch_size = self._resolve_max_batch_size(node)
        self.multi_env = False
        self.max_goalset = self._resolve_max_goalset(node)

        self.motion_gen_srv = node.create_service(
            Trigger,
            node.get_name() + "/update_motion_gen_config",
            partial(self.set_motion_gen_config, node),
        )

        self.init_services(node)

    def _resolve_max_goalset(self, node) -> int:
        """Per-segment candidate-set cap, read from the node's ROS param.

        Default 16 (the core hard-errors when a goalset exceeds
        ``config.max_goalset``, and the solver buffer is sized from it at build
        time). Specified via launch parameters and re-read on every
        ``update_motion_gen_config`` so a runtime change takes effect on
        rebuild.
        """
        if node.has_parameter("max_goalset"):
            return int(
                node.get_parameter("max_goalset").get_parameter_value().integer_value
            )
        return 16

    def _resolve_num_trajopt_seeds(self, node) -> int:
        """Trajopt candidate trajectories per problem, read from the node param.

        Each seed is a full trajectory-optimization solve, so per-plan latency
        scales ~linearly with this count (12 seeds ≈ 12 solves per plan; 1 seed
        is the fast lane for single deterministic segments). Default 12 (the
        pre-batch, full-ranking behavior). Baked into solver buffers / the CUDA
        graph at build time, so a runtime change takes effect only via the
        ``update_motion_gen_config`` rebuild (same caveat as max_goalset).
        """
        if node.has_parameter("num_trajopt_seeds"):
            return max(
                1,
                int(
                    node.get_parameter("num_trajopt_seeds")
                    .get_parameter_value()
                    .integer_value
                ),
            )
        return 12

    def _resolve_num_ik_seeds(self, node) -> int:
        """Per-pose IK seeds, read from the node param.

        Used for pose/goalset planning: each candidate goal pose is resolved
        into joint configurations with this many parallel IK seeds, and the
        standalone ``/ik`` / ``/ik_batch`` services use the SAME count (see
        ``IKServices._resolve_num_seeds``) so both IK paths behave identically.
        Default 32 (cuRobo's default at ``max_batch_size == 1``). Baked into
        solver buffers / the CUDA graph at build time, so a runtime change
        takes effect only via the ``update_motion_gen_config`` rebuild (same
        caveat as num_trajopt_seeds).
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

    _GRAPH_PLANNER_YAML = "graph_planner/exact_graph_planner.yml"

    def _resolve_graph_planner_config(self, node):
        """Graph-planner config for the build, with ROS-param search-budget overrides.

        The v2 PRM graph planner has no per-plan "seed count": its search
        effort is (nodes sampled per iteration) x (path-finding iterations),
        capped by ``max_nodes``. Those three knobs are exposed as node params
        (``graph_new_nodes_per_iteration``, ``graph_max_path_finding_iterations``,
        ``graph_max_nodes``) and merged over the shipped
        ``exact_graph_planner.yml`` defaults. When none are set this returns
        the default YAML path — identical behavior to passing nothing.
        """
        overrides = {}
        if node.has_parameter("graph_new_nodes_per_iteration"):
            overrides["new_nodes_per_iteration"] = max(
                1,
                int(
                    node.get_parameter("graph_new_nodes_per_iteration")
                    .get_parameter_value()
                    .integer_value
                ),
            )
        if node.has_parameter("graph_max_path_finding_iterations"):
            overrides["max_path_finding_iterations"] = max(
                1,
                int(
                    node.get_parameter("graph_max_path_finding_iterations")
                    .get_parameter_value()
                    .integer_value
                ),
            )
        if node.has_parameter("graph_max_nodes"):
            overrides["max_nodes"] = max(
                1,
                int(
                    node.get_parameter("graph_max_nodes")
                    .get_parameter_value()
                    .integer_value
                ),
            )
        if not overrides:
            return self._GRAPH_PLANNER_YAML
        base = resolve_config(
            join_path(get_task_configs_path(), self._GRAPH_PLANNER_YAML)
        )
        base["graph_planner"].update(overrides)
        return base

    def _resolve_max_batch_size(self, node) -> int:
        """Batched-solve capacity (problems stacked into ONE solver call).

        The trajectory_generation_batch surface (Alternatives/plan_batch fan-out)
        plans N problems in a single plan_cspace solve, with the problems on the
        batch dimension of the solver. The solver buffers are sized from this at
        build time (like max_goalset), so it must cover the largest fan-out the
        deployment actually sends — the node's launch parameter sets it, default 1
        (no batch, matches the pre-batch behavior exactly).
        """
        if node.has_parameter("max_batch_size"):
            return max(
                1,
                int(
                    node.get_parameter("max_batch_size")
                    .get_parameter_value()
                    .integer_value
                ),
            )
        return 1

    def set_motion_gen_config(self, node, _, response):
        """
        Build (or rebuild) the MotionPlanner and warmup.

        Called at init and on demand via the `update_motion_gen_config` service.
        """
        # Rebuilding + warmup re-captures CUDA graphs, so the whole body must
        # hold gpu_lock: a concurrent viz timer (collision spheres / sparse voxel
        # grid) doing host->device copies or a depth-camera integrate while a
        # stream is capturing invalidates the capture
        # (cudaErrorStreamCaptureUnsupported then Invalidated) and crashes the
        # node — the same invariant every other graph-capturing path in the node
        # follows. RLock: reentrant when this is called from
        # rebuild_solvers_for_cache_change (which already holds it). Standalone
        # nodes without the lock fall back to a no-op context.
        gpu_lock = getattr(node, "gpu_lock", None)
        lock_ctx = gpu_lock if gpu_lock is not None else nullcontext()
        with lock_ctx:
            # No perception voxel layer at construction — collision_cache allocates
            # the voxel storage and update_world fills it by copy. Passing the live
            # layer aliases the solver's buffer onto our ESDF tensor, which the first
            # update_world then clears to "solid". See primitives_only_scene().
            scene = self.obstacle_manager.primitives_only_scene()
            collision_activation_distance = (
                node.get_parameter("collision_activation_distance")
                .get_parameter_value()
                .double_value
            )

            cfg_kwargs: dict = {
                "robot": self.robot_model_manager.robot_cfg,
                "scene_model": scene,
                "num_ik_seeds": self._resolve_num_ik_seeds(node),
                "num_trajopt_seeds": self.num_trajopt_seeds,
                "graph_planner_config": self._resolve_graph_planner_config(node),
                "position_tolerance": self.position_tolerance,
                "orientation_tolerance": self.orientation_tolerance,
                "use_cuda_graph": self.use_cuda_graph,
                "self_collision_check": self.self_collision_check,
                "collision_cache": self.collision_cache,
                "optimizer_collision_activation_distance": collision_activation_distance,
                "max_batch_size": self._resolve_max_batch_size(node),
                "multi_env": self.multi_env,
                "max_goalset": self._resolve_max_goalset(node),
            }
            # Same solver recipe as curobo core's benchmark
            # (motion_plan_benchmark.py load_curobo): particle + LBFGS for both
            # IK and trajopt. Unconditional — this is the default in every
            # startup, no flag or env var.
            cfg_kwargs.update(
                {
                    "ik_optimizer_configs": [
                        "ik/particle_ik.yml",
                        "ik/lbfgs_ik.yml",
                    ],
                    "ik_transition_model": "ik/transition_ik.yml",
                    "metrics_rollout": "metrics_base.yml",
                    "trajopt_optimizer_configs": [
                        "trajopt/particle_trajopt.yml",
                        "trajopt/lbfgs_bspline_trajopt.yml",
                    ],
                    "trajopt_transition_model": "trajopt/transition_bspline_trajopt.yml",
                    "store_debug": False,
                }
            )
            node.get_logger().info(
                "MotionPlanner solver recipe: particle+LBFGS ik/trajopt"
            )

            node.get_logger().info(
                "MotionPlanner solver envelope: "
                f"use_cuda_graph={cfg_kwargs['use_cuda_graph']}, "
                f"num_ik_seeds={cfg_kwargs['num_ik_seeds']}, "
                f"num_trajopt_seeds={cfg_kwargs['num_trajopt_seeds']}, "
                f"collision_activation_distance={collision_activation_distance}"
            )

            cfg = MotionPlannerCfg.create(**cfg_kwargs)

            node.motion_planner = MotionPlanner(cfg)
            # Legacy alias — some downstream code still references `node.motion_gen`.
            node.motion_gen = node.motion_planner

            # Output sampling step of the interpolated plan. It's a trajopt config
            # field (not a MotionPlannerCfg.create arg), so set it post-build, before
            # warmup so the interpolation buffer picks it up. Guarded: the standalone
            # node doesn't declare this param.
            if node.has_parameter("interpolation_dt"):
                interp_dt = (
                    node.get_parameter("interpolation_dt")
                    .get_parameter_value()
                    .double_value
                )
                try:
                    node.motion_planner.trajopt_solver.config.interpolation_dt = interp_dt
                    node.get_logger().info(f"interpolation_dt set to {interp_dt}s")
                except AttributeError:
                    node.get_logger().warn(
                        "Could not set interpolation_dt on trajopt_solver"
                    )

            node.get_logger().info("warming up..")

            self.node_is_available = False
            node.set_parameters(
                [
                    rclpy.parameter.Parameter(
                        "node_is_available", rclpy.Parameter.Type.BOOL, False
                    )
                ]
            )

            try:
                node.motion_planner.warmup()
            except Exception:
                node.motion_planner = None
                node.motion_gen = None
                raise

            node.set_parameters(
                [
                    rclpy.parameter.Parameter(
                        "node_is_available", rclpy.Parameter.Type.BOOL, True
                    )
                ]
            )
            self.node_is_available = True

            node.get_logger().info("Motion planner configured")

        if response is not None:
            response.success = True
            response.message = "Motion planner config set"
        return response

    def update_world_config(self, node):
        """Push the current Scene into all active solvers."""
        # Sphere/cylinder/capsule obstacles must be converted to a solver-
        # supported collision type (cuboid/mesh) or they are silently dropped
        # from collision checking. obstacle_collision_mode picks the conversion:
        # 'cuboid' (fast OBB approximation, default) or 'mesh' (exact trimesh,
        # legacy). See collision_world_scene().
        scene = self.obstacle_manager.collision_world_scene()
        if getattr(node, "motion_planner", None) is not None:
            node.motion_planner.update_world(scene)
        if getattr(node, "mpc", None) is not None:
            # MPCSolver has no top-level update_world(); go through its checker.
            node.mpc.scene_collision_checker.load_collision_model(scene)

        # The pushes re-added every obstacle with enable=1 (load_from_scene_cfg
        # clears + re-adds); re-assert any attached-obstacle disable.
        self.obstacle_manager.reapply_attached_disables(node)

        self.node.get_logger().info(
            f"Updated world: {len(scene.cuboid)} cuboids, {len(scene.mesh)} meshes"
        )

    def callback_get_collision_distance(
        self, node, request: GetCollisionDistance, response
    ):
        return _compute_sphere_distance(self, node, response)

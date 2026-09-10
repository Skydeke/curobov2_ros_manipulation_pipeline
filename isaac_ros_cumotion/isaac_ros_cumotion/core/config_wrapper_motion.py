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

from functools import partial
import rclpy

from std_srvs.srv import Trigger
from isaac_ros_cumotion_interfaces.srv import GetCollisionDistance

# v2 runtime flags — must be set before any cuRobo objects are instantiated.
# cuda_graph_reset lets solvers rebuild captured graphs when buffer shapes
# change between plan calls (e.g. different interpolated trajectory lengths).
# Without this, a second plan with a different horizon raises
# "CUDA graph reset is not available." Requires CUDA 12.0+.
# Note: curobo.runtime re-exports (and shadows) curobo._src.runtime values at
# import time — torch_util.is_cuda_graph_reset_available() reads from
# curobo.runtime, so we must flip the flag on the public module too.
import curobo._src.runtime as _curobo_runtime
_curobo_runtime.cuda_graph_reset = True
import curobo.runtime as _curobo_runtime_public
_curobo_runtime_public.cuda_graph_reset = True

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

from .config_wrapper import ConfigWrapper, resolve_use_cuda_graph
from .collision_distance import _compute_sphere_distance, _query_sphere_collision


class ConfigWrapperMotion(ConfigWrapper):
    """Motion planner config wrapper (v2 `MotionPlanner`)."""

    def __init__(self, node, robot):
        super().__init__(node, robot)

        # v2 trajopt / IK / batch tunables
        self.num_ik_seeds = 32
        self.num_trajopt_seeds = 12
        # ROS param 'use_cuda_graph' (default True), overridable via the
        # CUROBO_USE_CUDA_GRAPH env var. Disabling avoids the MPC->Classic
        # captured-graph replay segfault at the cost of per-plan latency.
        self.use_cuda_graph = resolve_use_cuda_graph(node)
        self.self_collision_check = True
        self.position_tolerance = 0.005
        self.orientation_tolerance = 0.05
        self.max_batch_size = 1
        self.multi_env = False
        self.max_goalset = self._resolve_max_goalset(node)

        self.motion_gen_srv = node.create_service(
            Trigger,
            node.get_name() + '/update_motion_gen_config',
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
        if node.has_parameter('max_goalset'):
            return int(node.get_parameter('max_goalset').get_parameter_value().integer_value)
        return 16

    def set_motion_gen_config(self, node, _, response):
        """
        Build (or rebuild) the MotionPlanner and warmup.

        Called at init and on demand via the `update_motion_gen_config` service.
        """
        # No perception voxel layer at construction — collision_cache allocates
        # the voxel storage and update_world fills it by copy. Passing the live
        # layer aliases the solver's buffer onto our ESDF tensor, which the first
        # update_world then clears to "solid". See primitives_only_scene().
        scene = self.obstacle_manager.primitives_only_scene()
        collision_activation_distance = node.get_parameter(
            'collision_activation_distance'
        ).get_parameter_value().double_value

        cfg = MotionPlannerCfg.create(
            robot=self.robot_model_manager.robot_cfg,
            scene_model=scene,
            num_ik_seeds=self.num_ik_seeds,
            num_trajopt_seeds=self.num_trajopt_seeds,
            position_tolerance=self.position_tolerance,
            orientation_tolerance=self.orientation_tolerance,
            use_cuda_graph=self.use_cuda_graph,
            self_collision_check=self.self_collision_check,
            collision_cache=self.collision_cache,
            optimizer_collision_activation_distance=collision_activation_distance,
            max_batch_size=self.max_batch_size,
            multi_env=self.multi_env,
            max_goalset=self._resolve_max_goalset(node),
        )

        node.motion_planner = MotionPlanner(cfg)
        # Legacy alias — some downstream code still references `node.motion_gen`.
        node.motion_gen = node.motion_planner

        # Output sampling step of the interpolated plan. It's a trajopt config
        # field (not a MotionPlannerCfg.create arg), so set it post-build, before
        # warmup so the interpolation buffer picks it up. Guarded: the standalone
        # node doesn't declare this param.
        if node.has_parameter('interpolation_dt'):
            interp_dt = node.get_parameter('interpolation_dt').get_parameter_value().double_value
            try:
                node.motion_planner.trajopt_solver.config.interpolation_dt = interp_dt
                node.get_logger().info(f"interpolation_dt set to {interp_dt}s")
            except AttributeError:
                node.get_logger().warn("Could not set interpolation_dt on trajopt_solver")

        node.get_logger().info("warming up..")

        self.node_is_available = False
        node.set_parameters([
            rclpy.parameter.Parameter('node_is_available', rclpy.Parameter.Type.BOOL, False)
        ])

        try:
            node.motion_planner.warmup()
        except Exception:
            node.motion_planner = None
            node.motion_gen = None
            raise

        node.set_parameters([
            rclpy.parameter.Parameter('node_is_available', rclpy.Parameter.Type.BOOL, True)
        ])
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
        # from collision checking. See collision_world_scene().
        scene = self.obstacle_manager.collision_world_scene()
        if getattr(node, 'motion_planner', None) is not None:
            node.motion_planner.update_world(scene)
        if getattr(node, 'mpc', None) is not None:
            # MPCSolver has no top-level update_world(); go through its checker.
            node.mpc.scene_collision_checker.load_collision_model(scene)

        # The pushes re-added every obstacle with enable=1 (load_from_scene_cfg
        # clears + re-adds); re-assert any attached-obstacle disable.
        self.obstacle_manager.reapply_attached_disables(node)

        self.node.get_logger().info(
            f"Updated world: {len(scene.cuboid)} cuboids, {len(scene.mesh)} meshes"
        )

    def callback_get_collision_distance(self, node, request: GetCollisionDistance, response):
        return _compute_sphere_distance(self, node, response)

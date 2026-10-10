#!/usr/bin/env python3
"""
Unified trajectory planner node (v2) using Strategy Pattern.

Supports multiple planning strategies (Classic, MPC, Multi-Point, Joint-Space)
and allows dynamic switching between them.

v2 notes:
- MotionGen → MotionPlanner (wired via ConfigWrapperMotion as `node.motion_planner`).
- MpcSolver → ModelPredictiveControl (built by MPCController from the shared
  context, published as `node.mpc`).
- TensorDeviceType → DeviceCfg. We read device/dtype from the wrapper.
- WorldConfig → Scene (obstacle_manager.get_scene()), the single source of
  truth, propagated to solvers via update_world(scene).
- Perception: camera data → Mapper (ObstacleManager) → ESDF VoxelGrid in the
  Scene, refreshed on-demand before each plan via refresh_perception_world().
- ground plane lives on the Scene via ObstacleManager, not `world_cfg.add_obstacle`.
"""

import contextlib
import logging
import os
import sys
import threading
import time
import traceback

# PyTorch allocator hint recommended by the OOM errors themselves: reduces
# VRAM fragmentation on the 7.65 GiB cards this stack runs on. Must be set
# before the first CUDA allocation, so before rclpy/torch/curobo import here.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import rclpy
import torch
from rclpy.action import ActionServer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from sensor_msgs.msg import JointState as JointStateMsg
from std_srvs.srv import Trigger
from curobov2_ros_interfaces.srv import (
    TrajectoryGeneration,
    TrajectoryGenerationBatch,
    SetPlanner,
    GetPlanners,
)
from curobov2_ros_interfaces.action import SendTrajectory
from curobov2_ros_interfaces.msg import (
    WorldCollisionContact,
    SelfCollisionContact,
    JointLimitViolation,
    StateCollisions,
    PlanningStats,
    ConsideredTrajectory,
    TrajectoryGoal,
    TrajectoryResult,
)

from curobo.types import DeviceCfg, JointState
from curobo.logging import setup_logger as setup_curobo_logger

from curobov2_ros.robot.robot_context import RobotContext
from curobov2_ros.core.config_wrapper_motion import ConfigWrapperMotion
from curobov2_ros.core.joint_order import (
    resolve_joint_targets,
    resolve_start_pose,
)
from curobov2_ros.core.collision_distance import (
    _attributed_collisions,
    _attributed_joint_limit_violations,
    _attributed_self_collisions,
)
from curobov2_ros.core.attachment_services import AttachmentServices
from curobov2_ros.core.ik_services import IKServices
from curobov2_ros.core.fk_services import FKServices
from curobov2_ros.core.reachability_services import ReachabilityServices
from curobov2_ros.core.robot_segmentation import RobotSegmentation
from curobov2_ros.core.gpu_lanes import GpuScheduler, PRIO_REACTIVE, PRIO_PLAN, PRIO_KIN, PRIO_PERCEPTION, PRIO_BULK
from curobov2_ros.planners import (
    PlannerFactory,
    PlannerManager,
    ReactiveController,
    SinglePlanner,
)

# cuRobo logs through Python's `logging` module ('curobo' logger) which the
# node configures at startup; by default its messages are plain. Color them
# with the same level palette rclpy uses so `[WARNING] [curobo] ...` lines
# look like the colored ROS logs next to them. Both streams go to stderr, so
# any pipeline that keeps ROS colors will keep curobo colors too.
_CUROBO_LOG_FORMAT = "[%(levelname)s] [%(name)s] %(message)s"
_CUROBO_COLOR_RESET = "\033[0m"
_CUROBO_LEVEL_COLORS = {
    logging.DEBUG: "\033[96m",
    logging.INFO: "\033[32m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[1;31m",
}


def _curobo_colors_enabled() -> bool:
    """Whether ANSI colors should be emitted for curobo log lines.

    Mirrors rcutils' decision (RCUTILS_COLORIZED_OUTPUT env var, then the
    stderr tty check) so curobo lines follow exactly the same colorization
    rules as the adjacent ROS logs. ``NO_COLOR`` disables them.
    """
    if os.environ.get("NO_COLOR"):
        return False
    colorized = os.environ.get("RCUTILS_COLORIZED_OUTPUT", "").strip().lower()
    if colorized:
        return colorized not in ("0", "false", "no", "off")
    return sys.stderr.isatty()


class _CuroboColorFormatter(logging.Formatter):
    """Level-colored formatter for the 'curobo' logger.

    Colors the WHOLE line with the severity color, matching rclpy's colored
    output (rclpy wraps the entire formatted log message, not just the
    ``[INFO]``/``[WARN]`` token).
    """

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if _curobo_colors_enabled():
            color = _CUROBO_LEVEL_COLORS.get(record.levelno)
            if color:
                return f"{color}{msg}{_CUROBO_COLOR_RESET}"
        return msg


def install_colored_curobo_logger(
    level: int = logging.WARNING, logger_name: str = "curobo"
) -> None:
    """Route the ``logger_name`` logger through a single colored stderr handler.

    Replaces any pre-existing handlers on the logger and stops propagation to
    the root logger so each curobo line is emitted exactly once.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.handlers[:] = []
    handler = logging.StreamHandler()
    handler.setFormatter(_CuroboColorFormatter(_CUROBO_LOG_FORMAT))
    logger.addHandler(handler)
    logger.propagate = False


class _ThrottleMeshCacheReuseWarnings(logging.Filter):
    """Rate-limit curobo's benign "Mesh already in cache" WARNING.

    curobo keeps ONE mesh store per process (a global cache). Every time an
    obstacle with a mesh prim is registered again — e.g. the planner's own
    scene prims on startup, or a ``remove_all_objects`` + re-``add_object``
    cycle from a client (the parity benchmark does this per problem) — the
    cache hits and curobo logs the same informational WARNING each time
    ("Mesh already in cache, reusing existing instance: <name>"). The message
    is expected behavior, not a problem; throttle it so each DISTINCT message
    (i.e. each mesh name) escapes at most once per ``interval`` seconds while
    the churn continues. Repeats of the same line are the spam and get slowed;
    a line that differs even slightly (a new mesh name) still logs
    immediately. ``interval <= 0`` restores full silence.

    A single filter instance is shared by every record, so the throttle state
    is guarded by a lock (the node runs a multi-threaded executor). State is
    message-keyed and pruned once it outgrows a small bound, so memory stays
    flat across a session.
    """

    _MESH_CACHE_REUSE = "Mesh already in cache"

    def __init__(self, interval: float = 5.0):
        super().__init__()
        self.interval = float(interval)
        # message text -> last emission time (monotonic); one window per
        # distinct message, so a slightly different line is never throttled
        # by an earlier one's window.
        self._last_emitted = {}
        self._lock = threading.Lock()

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if self._MESH_CACHE_REUSE not in message:
            return True
        if self.interval <= 0:
            return False
        with self._lock:
            now = time.monotonic()
            last = self._last_emitted.get(message)
            if last is None or now - last >= self.interval:
                self._last_emitted[message] = now
                if len(self._last_emitted) > 64:
                    # drop keys that have been quiet for 2+ windows; actively
                    # churning keys (last < 2*interval ago) are kept.
                    quiet_before = now - 2 * self.interval
                    self._last_emitted = {
                        k: v for k, v in self._last_emitted.items() if v >= quiet_before
                    }
                return True
            return False


def throttle_curobo_mesh_cache_warnings(
    logger_name: str = "curobo", interval: float = 5.0
) -> None:
    """Install :class:`_ThrottleMeshCacheReuseWarnings` on the ``curobo`` logger."""
    logging.getLogger(logger_name).addFilter(
        _ThrottleMeshCacheReuseWarnings(interval=interval)
    )


class UnifiedPlannerNode(Node):
    """Unified trajectory planning node with dynamic strategy switching (v2)."""

    def __init__(self):
        super().__init__("unified_planner")

        # v2 device config — kept as `tensor_args` for backward-compat with
        # code paths that read `.device` / `.dtype` from it.
        self.tensor_args = DeviceCfg(device="cuda", dtype=torch.float32)

        # Serializes curobo CUDA-graph *capture* against concurrent GPU work on
        # other executor threads — notably the depth-camera callback's
        # mapper.integrate() and the viz timer's _spheres_in_collision .cpu().
        # A GPU op launched while a stream is capturing invalidates the capture
        # (cudaErrorStreamCaptureUnsupported / cudaErrorStreamCaptureInvalidated).
        # Created before cameras/solvers so it always exists when callbacks
        # fire. RLock: the holder thread may re-enter; a different thread
        # (depth callback) fails the non-blocking acquire and simply skips its
        # frame.  Open-loop plans (use_cuda_graph=True) hold gpu_lock for the
        # entire plan() call — cuRobo can re-capture GraphExecutors mid-plan
        # (reset_shape from _prepare_goal_buffer), so the old single-shot
        # pending-flag guard was insufficient (exit 1, 2026-09-11).
        self.declare_parameter("exit_on_gpu_fault", True)
        self.declare_parameter("gpu_exclusive_plans", True)   # S4/S7: True = old safe behaviour
        self.gpu = GpuScheduler(on_poisoned=self._on_gpu_poisoned)
        # Legacy adapter: `with node.gpu_lock:` == barrier.exclusive(); acquire(blocking=False) works too.
        self.gpu_lock = self.gpu.legacy_lock()
        self.lane_reactive = self.gpu.lane("reactive")
        self.lane_plan = self.gpu.lane("plan")
        self.lane_kin = self.gpu.lane("kin")
        self.lane_perception = self.gpu.lane("perception")
        self.lane_viz = self.gpu.lane("viz")

        # Callback groups for the MultiThreadedExecutor (num_threads=8): without
        # these every subscription/timer lands in the default MutuallyExclusive
        # group, so the executor serializes all of them and the camera callbacks
        # block the viz publishers (and vice versa) — broadcast starvation
        # symptoms. Explicit groups let the GPU-heavy work run in parallel:
        #  - perception: camera subscriptions (depth integrate + segmentation) —
        #    the sustained GPU-bound path, so it never shares a thread slot with
        #    viz.
        #  - viz: the GPU-backed viz timers (collision spheres, sparse voxel
        #    grid, workspace marker); CPU-light and rate-limited.
        #  - markers: the CPU-only scene-obstacle marker timer so it always
        #    ticks even while perception/viz are busy.
        # Created before cameras/solvers so they always exist when callbacks
        # fire. cf. manager-architecture.md (callback groups / gpu_lock).
        self._perception_callback_group = MutuallyExclusiveCallbackGroup()
        self._viz_callback_group = MutuallyExclusiveCallbackGroup()
        self._marker_callback_group = MutuallyExclusiveCallbackGroup()

        # Serializes goal admission: only one execute_trajectory goal may be
        # active at a time (open-loop or reactive). Guards against two
        # concurrent servo/execute loops driving the robot at once. Set in
        # goal_callback (accept-time, closing the race with execute_callback)
        # and cleared in execute_callback's finally. Also consulted by
        # set_planner_callback to refuse switching planners mid-goal.
        self._goal_lock = threading.Lock()
        self._goal_active = False

        # Output sampling step (s) of the interpolated trajectory (trajopt) —
        # curobo_ros is the authority on this value (see resolve_interpolation_dt);
        # it is what every JointCommandStrategy stamps into time_from_start.
        # MUST be declared before RobotContext is constructed below: RobotContext
        # reads it (via resolve_interpolation_dt) to build its strategies.
        self.declare_parameter("interpolation_dt", 0.025)

        # cuRobo-internal log level. Mirrors the legacy curobo_server toggle:
        # ``True`` raises cuRobo's logger to INFO, ``False`` keeps it at WARNING.
        # Only affects messages routed through ``curobo``'s own logging, not ROS
        # node logs.
        self.declare_parameter("enable_curobo_debug_mode", False)
        curobo_debug = self.get_parameter("enable_curobo_debug_mode").value
        setup_curobo_logger("info" if curobo_debug else "warning", "curobo")
        curobo_level = logging.INFO if curobo_debug else logging.WARNING
        # Route curobo's logger through a colored formatter so its lines match
        # the level colors of the adjacent ROS logs (same stderr stream).
        install_colored_curobo_logger(curobo_level)
        if hasattr(logging, "lastResort") and logging.lastResort is not None:
            logging.lastResort.setLevel(curobo_level)
        # Throttle curobo's benign "Mesh already in cache" WARNING (repeated
        # obstacle registration hits its global mesh cache; see the filter):
        # at most one line per interval seconds while the churn continues;
        # 0/negative restores full silence. Read once at startup (installed
        # before the solver build, so the first registrations count).
        self.declare_parameter("curobo_mesh_cache_log_interval", 5.0)
        throttle_curobo_mesh_cache_warnings(
            interval=self.get_parameter("curobo_mesh_cache_log_interval").value
        )

        self.robot_context = RobotContext(self)

        self.declare_parameter("planner_type", "classic")
        # Planning retries per request (MotionPlanner.plan_pose). Plumbed
        # through the launch arg and read by every SinglePlanner subclass at
        # plan time (classic / joint-space / multi-waypoint). Default 1 = the
        # single-attempt behavior.
        self.declare_parameter("max_attempts", 1)
        # Rewind the solver RNG before every plan (MotionPlanner.reset_seed),
        # reproducing the reference benchmark's per-problem seed reset so each
        # request sees the same ik/trajopt candidate stream. Read per request
        # (like max_attempts), so a `ros2 param set` takes effect immediately.
        # OFF by default: a live server should draw fresh seeds per request
        # rather than replay the same first candidates forever — only the
        # parity benchmark needs the rewind, and it launches with
        # reset_seed_per_plan:=true.
        self.declare_parameter("reset_seed_per_plan", False)
        # Torque-limited planning (the reference benchmarks page's "with
        # torque limits" table): `load_dynamics` builds the robot's pinocchio
        # inverse-dynamics model (RobotModelManager must see it at startup —
        # the robot_cfg is built once), and `robot_payload_mass` patches the
        # attached_object payload inertia onto the MotionPlanner like the
        # reference script's update_links_inertial({"attached_object":
        # {"mass": args.mass}}). Payload only applies with load_dynamics (a
        # mass patch without inverse dynamics is undefined in curobo).
        self.declare_parameter("load_dynamics", False)
        self.declare_parameter("robot_payload_mass", 0.0)
        # torch.cuda.synchronize() bridges the executor's Python threads to the
        # GPU but blocks the calling thread every frame/kernel — off by default
        # so the depth callback and viz timers keep running while the GPU works
        # asynchronously (cf. the 1-core executor behaviour that starved the
        # collision-spheres publisher). Enable only when you need deterministic
        # GPU/CPU ordering (e.g. debugging a race).
        self.declare_parameter("torch_sync", False)
        # Trajectory RE-TIMING: scales the stamped time of every sent trajectory
        # (see JointCommandStrategy._dilated_dt). 1.0 = nominal interpolation_dt
        # pacing; <1.0 slows the motion down, >1.0 speeds it up (cuRobo
        # convention). The same value still gates how often execute() re-reads
        # progression.
        self.declare_parameter("time_dilation_factor", 1.0)
        self.declare_parameter("voxel_size", 0.05)
        # Publish rate (Hz) of the sparse voxel grid topic. <= 0 disables it.
        self.declare_parameter("sparse_voxel_publish_rate", 7.0)
        # Publish robot collision spheres + scene obstacles as RViz markers by
        # default. Disable at launch if a live CUDA-graph capture is wanted with
        # no competing visualization traffic.
        self.declare_parameter("publish_collision_spheres", True)
        # Publish the motion-plan debug image (joint-trajectory pos/vel/acc/jerk
        # plot as an RGB Image on /<node>/motion_plan_debug). Independent of
        # enable_curobo_debug_mode; off by default. See
        # SinglePlanner._publish_plan_image().
        self.declare_parameter("publish_plan_debug_image", False)
        # Publish a wireframe box of the Mapper's TSDF/ESDF workspace extent
        # (mapper_extent_xyz centered at mapper_grid_center) on
        # /<node>/mapper_workspace for RViz. Disable at launch if the marker
        # traffic is unwanted (e.g. during CUDA-graph profiling).
        self.declare_parameter("publish_workspace_visualisation", True)
        # NOTE: the TSDF decay knob is `decay_half_life_s` (seconds), declared by
        # ObstacleManager._load_perception_params. The raw per-integrate
        # `decay_factor` is no longer exposed: it is derived from the half-life
        # and the cameras' combined frame rate.

        self.declare_parameter("collision_activation_distance", 0.025)
        self.declare_parameter("convergence_threshold", 0.01)
        self.declare_parameter("max_mpc_iterations", 1000)
        # Capture/replay CUDA graphs in the solvers (faster, but a captured
        # MotionGen graph can be invalidated by intervening MPC activity — see
        # the MPC->Classic re-warmup in _setup_planner). Override at runtime with
        # the CUROBO_USE_CUDA_GRAPH env var (0 disables) for A/B testing.
        self.declare_parameter("use_cuda_graph", True)
        # Per-segment candidate-set cap for the open-loop planners (a goalset is
        # "N acceptable poses, planner picks the best"). The cap sizes solver
        # buffers at build time and is baked into the CUDA graph, so changing
        # it at runtime requires a rebuild via update_motion_gen_config (same
        # caveat as max_batch_size). Read by ConfigWrapperMotion.
        self.declare_parameter("max_goalset", 16)
        # Batched-solve capacity: problems stacked into ONE plan_cspace call on
        # the trajectory_generation_batch surface (batch dim = problems × seed
        # dim = candidate trajectories). Cap sizes solver buffers at build time
        # and is baked into the CUDA graph (same caveat as max_goalset). Read by
        # ConfigWrapperMotion; the launch sets it to the task's max fan-out.
        self.declare_parameter("max_batch_size", 1)
        # Trajopt candidate trajectories per problem (the seed axis: each
        # problem is solved from this many warmstarts and the best kept). Each
        # seed is a FULL trajectory-optimization solve, so per-plan latency
        # scales ~linearly with it — 1 seed is the fast lane (single
        # deterministic segments need no ranking). Sized into solver buffers /
        # baked into the CUDA graph at build time, so a runtime change needs
        # the update_motion_gen_config rebuild (same caveat as max_goalset).
        # Read by ConfigWrapperMotion; the launch sets the deployment default.
        self.declare_parameter("num_trajopt_seeds", 12)
        # Per-pose IK seeds: planner-internal IK for pose/goalset goals AND the
        # standalone /ik, /ik_batch services read the SAME param — one seed
        # budget for both IK paths (previously hardcoded 32/20 split). Baked
        # into solver buffers / the CUDA graph at build time (same caveat as
        # num_trajopt_seeds). Read by ConfigWrapperMotion and IKServices.
        self.declare_parameter("num_ik_seeds", 32)
        # --- default kinematics warmup (IK solver + FK model) ---
        # IKServices/FKServices register their services in their constructors and
        # build nothing, so /ik, /ik_batch, /fk and /fk_batch all answer
        # "not initialized" (and return an empty success=false) until somebody
        # calls /warmup_ik or /warmup_fk. That pushed the warm-up obligation
        # onto every client, and a client that forgot simply got silently
        # useless kinematics — the task constructor's cartesian_path stage was
        # the one that tripped over it, failing every straight-line segment with
        # "fk failed: ''" while the trajectory stages planned fine.
        #
        # Both default ON so a plain `ros2 launch` gives a server whose
        # kinematics services answer. IK is the expensive build; set
        # `warmup_ik:=false` to skip it and keep the on-demand path (a client
        # that needs IK must then call /warmup_ik itself, and a client that
        # needs a batch size other than warmup_ik_batch_size must still call
        # /warmup_ik — the later call re-initializes the solver).
        self.declare_parameter("warmup_ik", True)
        self.declare_parameter("warmup_ik_batch_size", 1)
        # FK is cheap (one Kinematics + a warmup batch of random configs).
        self.declare_parameter("warmup_fk", True)
        # Batch size for the FK warmup batch. Only compiles the kernels for that
        # batch shape — _compute_poses accepts any batch at call time.
        self.declare_parameter("warmup_fk_batch_size", 1)
        # PRM graph-planner search budget for the first plan attempt
        # (enable_graph_attempt): nodes sampled per iteration x path-finding
        # iterations, capped by max_nodes. There is NO per-plan "seed count" in
        # the v2 graph planner — these are the effort knobs, merged over
        # graph_planner/exact_graph_planner.yml at build time (same caveat as
        # num_trajopt_seeds). Read by ConfigWrapperMotion.
        self.declare_parameter("graph_new_nodes_per_iteration", 20)
        self.declare_parameter("graph_max_path_finding_iterations", 10)
        self.declare_parameter("graph_max_nodes", 20000)
        # Diagnostic toggle (see update_all_solvers_world): set false to withhold
        # the perception ESDF from the solvers. Leave true for normal operation —
        # false disables camera-based collision avoidance.
        self.declare_parameter("push_esdf_to_solvers", True)
        # Fold the depth-map robot segmentation INTO this server node (rather
        # than a standalone node): when true, each camera whose `camera_purpose`
        # is 'all' or 'segmentation' has its depth stream masked against the
        # robot's own collision spheres before the mapper integrates it, so the
        # arm / mount never become ESDF voxels. On by default — "use cameras
        # for everything" unless a camera is purpose-limited. The masking
        # component subscribes to each segmented camera's raw depth (the shared
        # camera_* params) and republishes the masked streams, which the mapper
        # camera strategies are derived to consume. Camera identity comes from
        # the shared camera_* config, not duplicated params.
        self.declare_parameter("enable_robot_segmentation", True)
        # Segmenter tuning (declared here — not just inside the component — so
        # launch-file overrides are applied at node construction like every
        # other param). Min distance (m) to a robot collision sphere for a
        # depth point to be kept; inflation (m) of the mask shapes. The masked
        # OUTPUT topic is derived per camera from that camera's raw `camera_topic`
        # (leaf segment replaced by `masked_depth`), so there is no segmenter
        # output-topic parameter.
        self.declare_parameter("robot_segmentation_distance_threshold", 0.05)
        self.declare_parameter("robot_segmentation_mask_margin", 0.0)
        # Reactive (MPC) solver build params — read by MPCController.build_solver().
        self.declare_parameter("mpc_step_dt", 0.03)
        self.declare_parameter("mpc_horizon_steps", 30)
        # LBFGS iterations per optimize call. cuRobo's defaults (200 warm-start,
        # 300 cold-start) are tuned for the offline getting-started demo, not
        # real-time control — the old per-tick full-horizon re-optimization
        # (optimize_action_sequence, pre-2026-09) took ~1-1.3s per call on this
        # hardware (vs. optimization_dt=0.03s), so the arm went long stretches
        # uncorrected then jumped, causing overshoot/oscillation. The native
        # optimize_next_action loop (plan-ahead, pop one command per solve)
        # keeps the same iteration counts for the warm-start re-solves that
        # refill its command buffer.
        # NOTE: cuRobo's LBFGS requires num_iters to be a MULTIPLE of its inner
        # loop size (25) — e.g. 25/50/75/100 are valid, 10 raises ValueError.
        # Isolated test (franka.yml, no real hardware): 25/100 -> ~18ms/call
        # (vs ~74ms/call at 200/300) and converges FASTER in wall-clock terms
        # despite fewer iterations per call (more, cheaper corrections beat
        # fewer, expensive ones for a receding-horizon controller).
        # 5/10 are the MPPI values; the 25/100 in the note above are the L-BFGS
        # ones, kept here because they pair with mpc_solver_type below.
        self.declare_parameter("mpc_warm_start_iters", 5)
        self.declare_parameter("mpc_cold_start_iters", 10)
        # MPC solver selection. The default is 'mppi_acceleration' (MPPI in
        # ACCELERATION space), the recipe validated on the real M1013 -- it holds
        # in the postures where L-BFGS + B-spline stalls.
        # The two iteration counts above are tuned FOR THIS DEFAULT. Switching to
        # 'lbfgs_bspline' means also passing mpc_warm_start_iters:=25
        # mpc_cold_start_iters:=100 -- cuRobo's L-BFGS requires multiples of 25
        # (see the note above), and 5/10 would leave it untuned.
        self.declare_parameter("mpc_solver_type", "mppi_acceleration")
        self.declare_parameter("mpc_mppi_num_particles", 400)
        # Deprecated (pre-2026-09 velocity feedback cap, _v_bc): the native
        # optimize_next_action loop warm-starts internally, so this is declared
        # only so existing configs that still set it keep loading.
        self.declare_parameter("mpc_vel_feedback_alpha", 1.0)
        # Fixed-interval command pacing (seconds). 0.0 = off (re-solve/re-send as
        # fast as the solve allows, ~70ms — replaces the previous window before the
        # bridge finishes it). >0 = hold each command window for this long before
        # re-solving/re-sending, and read the real robot state only AFTER it has
        # executed (fresh + velocity-consistent). Set to the window duration
        # (interpolation_steps*2 * mpc_step_dt = 8*0.03 = 0.24) to fully execute
        # each window. cf. debug 2026-07-16.
        self.declare_parameter("mpc_command_interval", 0.24)
        # Reactive (Retarget/teleop) build params — read by RetargetController.
        self.declare_parameter("retarget_position_weight", 1.0)
        self.declare_parameter("retarget_orientation_weight", 1.0)
        self.declare_parameter("retarget_use_mpc", False)
        # Lifetime (s) of a pre-planned (preview) trajectory cached by
        # generate_trajectory and reused by the execute action.
        self.declare_parameter("trajectory_cache_ttl", 30.0)
        # Max open-loop trajectories kept in the reuse cache. Multi-entry so
        # EVERY segment of a chain is replayable at execute time (see
        # _store_pending_plan); this cap bounds the GPU-side trajectory tensors
        # a long multi-task run accumulates between TTL prunes.
        self.declare_parameter("trajectory_cache_size", 64)
        # How far the arm may have strayed from the state a request says it
        # starts in, for a cache MISS in the execute path to be trusted. See
        # _check_execution_start_drift. 0.01 rad matches the MoveIt pipeline's
        # `allowed_start_tolerance` (iki_kortex_moveit_config/config/
        # moveit_controllers.yaml): the curobo chain must not accept a start
        # state that MoveIt would have refused to plan from - one number for
        # the wall, whichever planner is behind it. It absorbs ordinary
        # encoder/servo noise and stays far under the fraction of a joint's
        # range that a wrong IK branch moves.
        self.declare_parameter("execution_start_tolerance", 0.01)
        # Drift below this (but above 0) is reported and allowed: ordinary
        # servo tracking error, where re-solving from the arm's ACTUAL pose is
        # exactly the right thing to do. Sits an order of magnitude below the
        # refuse tier, as before.
        self.declare_parameter("execution_start_warn_drift", 0.005)

        # Single shared context for every solver (robot + obstacles + scene +
        # collision cache). The MPC solver is built lazily from this same context.
        self.config_wrapper_motion = ConfigWrapperMotion(self, self.robot_context)

        # Adopt the canonical joint order/DOF into the RobotContext descriptor now
        # that the kinematics exist (RobotContext is built before the kin model).
        self.robot_context.bind_kinematics(self.config_wrapper_motion.kin_model)

        # Depth-map robot segmentation, folded into this node (not a standalone
        # node). Gated by `enable_robot_segmentation`; shares this node's
        # RobotContext + Kinematics so it can never disagree with the planner's
        # own joint state / collision spheres, and the SHARED perception camera
        # configs (raw topic, camera-info topic, camera frame) so it can
        # never disagree with the mapper about which depth stream is which. One
        # RobotSegmentationCameraStrategy is created per segmented camera
        # (purpose 'all' or 'segmentation' in `camera_purpose`) and registered
        # through CameraContext just like the mapper's camera strategies; each
        # publishes on a topic derived from that camera's own raw topic
        # (e.g. /kortex_vision/depth/masked_depth), which is what the mapper's
        # strategy for that camera subscribes to. The mask services are
        # registered through the RosServiceManager (shared across all streams).
        self.robot_segmentation = None
        if self.get_parameter("enable_robot_segmentation").value:
            camera_cfgs = (
                self.config_wrapper_motion.camera_system_manager.camera_cfgs
            ) or []
            # An empty-topic camera slot (the [''] default when no camera is
            # configured) is INACTIVE: it must not reach RobotSegmentation,
            # which would subscribe to its (empty) camera-info topic and crash.
            seg_cfgs = [c for c in camera_cfgs if c.for_segmentation and c.depth_topic]
            if not seg_cfgs:
                self.get_logger().warn(
                    "enable_robot_segmentation requires at least one camera "
                    "with a depth topic and a camera_purpose of 'all' or "
                    "'segmentation': set 'camera_topic' (and camera_info_topic) "
                    "at launch. Segmentation disabled."
                )
            else:
                self.robot_segmentation = RobotSegmentation(
                    self,
                    self.robot_context,
                    self.config_wrapper_motion.kin_model,
                    camera_cfgs=seg_cfgs,
                    base_frame=self.config_wrapper_motion.base_link,
                    ops_dtype=self.tensor_args.dtype,
                    device=self.tensor_args.device,
                )
                self.config_wrapper_motion.ros_service_manager.register_robot_segmentation(
                    self.robot_segmentation
                )

        # Shared Scene for all planners — references ObstacleManager's Scene.
        # All planners see the same obstacles after update_world(scene).
        self.shared_scene = self.config_wrapper_motion.obstacle_manager.get_scene()

        # Solvers (created on demand).
        self.motion_planner = None  # v2 alias
        self.motion_gen = None  # legacy alias, kept for older code paths
        self.mpc = None  # reactive: ModelPredictiveControl
        self.retargeter = None  # reactive: MotionRetargeter (teleop)

        # Which solver currently owns the single live CUDA graph. Two captured
        # graphs cannot safely coexist in one CUDA context: the inactive one's
        # baked device addresses get invalidated by the active one's allocator
        # activity, so replaying it segfaults in cuGraphLaunch. We enforce
        # "exactly one live graph" by releasing every other solver's graph
        # whenever the owner changes (see _ensure_exclusive_graph). Keys:
        # 'motion' for all open-loop planners (they share one MotionPlanner),
        # 'reactive:<name>' for each reactive controller. None = no owner yet.
        self._active_graph_owner = None

        # True whenever the next call into a reactive solver may CAPTURE a CUDA
        # graph (fresh solver, or right after a release) rather than just replay
        # one. Capture is process-global — no other thread may issue ANY CUDA op
        # while it's in progress — so ReactiveController._step_guard uses
        # take_graph_capture_pending() to decide whether its next step() needs
        # gpu_lock. Open-loop plans (generate_trajectory / execute) now ALWAYS
        # hold gpu_lock when use_cuda_graph is on, because cuRobo can re-capture
        # graphs mid-plan via reset_shape() (see _plan_lock docstring,
        # 2026-09-11). cf. debug 2026-07-28.
        self._graph_capture_pending = True
        self._graph_capture_pending_lock = threading.Lock()

        # Attach/detach a scene obstacle to the arm's attached_object link
        # (standalone feature — pre-positioned objects, simulation, tests).
        # Registers its own attach_object/detach_object services.
        self.attachment_services = AttachmentServices(self, self.config_wrapper_motion)

        # Cached open-loop plans from generate_trajectory (preview) calls,
        # reused by the execute action when a target matches. MULTI-entry now:
        # one entry per planned segment (keyed by target signature, holding
        # that segment's trajectory), because a chain executes segment-by-
        # segment and a single slot could only ever serve the LAST segment
        # planned - the rest re-solved at execute time, and a re-solve of a
        # multi-solution pose goal can pick the OTHER IK branch (the exact
        # divergence the chain-continuity check exists to catch). See
        # _store_pending_plan / _pending_plan_matches / _prune_plan_cache.
        self._pending_plan = []
        self._last_pushed_epoch = -1

        # Shared IK — same Scene as MotionPlanner.
        self.ik_services = IKServices(self, self.config_wrapper_motion)

        # Reachability map (uniform grid IK on a plane) — reuses the IK solver.
        self.reachability_services = ReachabilityServices(
            self, self.config_wrapper_motion, self.ik_services
        )

        # FK — needs the robot YAML (geometry) and the shared Scene (batch
        # collision validation for FkBatch).
        self.fk_services = FKServices(self, self.config_wrapper_motion)

        self.planner_manager = PlannerManager(self, self.config_wrapper_motion)

        initial_planner = (
            self.get_parameter("planner_type").get_parameter_value().string_value
        )
        self._warmup_initial_planner(initial_planner)
        # The planner has its own internal IK, but /ik, /ik_batch, /fk and
        # /fk_batch allocate their own models and are dead until warmed. Warm
        # them by default so a client never has to know to ask.
        self._warmup_kinematics_services()
        self.planner_manager.set_current_planner(initial_planner)
        # Startup warms exactly one solver, so it is already the sole graph
        # owner — record it so the first plan reuses that warmup graph instead
        # of needlessly releasing and re-capturing it.
        self._active_graph_owner = self._graph_owner_key(
            self.planner_manager.get_current_planner()
        )

        # Pre-warm the voxel-grid primitive rasterization path while the executor
        # is not yet spinning (no service can be served until __init__ returns),
        # so the first get_voxel_grid call never pays kernel compilation.
        self.config_wrapper_motion.obstacle_manager.prewarm_voxel_rasterization(self)

        self.generate_trajectory_srv = self.create_service(
            TrajectoryGeneration,
            f"{self.get_name()}/generate_trajectory",
            self.generate_trajectory_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.trajectory_generation_batch_srv = self.create_service(
            TrajectoryGenerationBatch,
            f"{self.get_name()}/trajectory_generation_batch",
            self.trajectory_generation_batch_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.set_planner_srv = self.create_service(
            SetPlanner,
            f"{self.get_name()}/set_planner",
            self.set_planner_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.get_planners_srv = self.create_service(
            GetPlanners,
            f"{self.get_name()}/get_planners",
            self.get_planners_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.clear_trajectory_srv = self.create_service(
            Trigger,
            f"{self.get_name()}/clear_trajectory",
            self.clear_trajectory_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.clear_voxel_map_srv = self.create_service(
            Trigger,
            f"{self.get_name()}/clear_voxel_map",
            self.clear_voxel_map_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Reentrant group so cancel_callback can run while a long-running
        # reactive execute_callback is still servoing (MultiThreadedExecutor).
        self._action_server = ActionServer(
            self,
            SendTrajectory,
            f"{self.get_name()}/execute_trajectory",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )

        from geometry_msgs.msg import Pose as PoseMsg

        self.mpc_goal_sub = self.create_subscription(
            PoseMsg,
            f"{self.get_name()}/mpc_goal",
            self.mpc_goal_callback,
            10,
        )

        self.get_logger().info(
            "Unified planner ready with initial planner: "
            f"{self.planner_manager.get_current_planner().get_planner_name()}"
        )

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def _warmup_kinematics_services(self):
        """Build the standalone IK solver and FK model during startup.

        Both are lazy by design (IKServices/FKServices only allocate on an
        explicit warm-up call) so a node that never uses those services does not
        pay for them. The default flips that: the planner itself has its own IK,
        but /ik, /ik_batch, /fk and /fk_batch are separately allocated, and any
        client that reaches for them without warming first gets a silent failure
        rather than an error. Warming here makes the services usable from the
        moment the node is up, and leaves the /warmup_* services for the one
        thing they are still needed for: changing the batch size later.

        Each build is independent and failure-isolated: if the IK solver cannot
        be built the node still comes up (and still logs that /ik is
        unavailable), because a failed kinematics warm-up must not take the
        planner down with it.
        """
        if self.get_parameter("warmup_ik").get_parameter_value().bool_value:
            batch = max(
                1,
                int(
                    self.get_parameter("warmup_ik_batch_size")
                    .get_parameter_value()
                    .integer_value
                ),
            )
            self.get_logger().info(
                f"Warming up standalone IK solver (batch_size={batch})..."
            )
            try:
                self.ik_services.warmup(batch)
                self.get_logger().info("  -> IK solver ready")
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(
                    f"Startup IK warmup failed: {e}. /ik and /ik_batch will "
                    "report 'not initialized' until /warmup_ik succeeds."
                )
        else:
            self.get_logger().info(
                "Skipping IK warmup (warmup_ik:=false); call "
                f"{self.get_name()}/warmup_ik before /ik"
            )

        if self.get_parameter("warmup_fk").get_parameter_value().bool_value:
            batch = max(
                1,
                int(
                    self.get_parameter("warmup_fk_batch_size")
                    .get_parameter_value()
                    .integer_value
                ),
            )
            self.get_logger().info(f"Warming up FK model (batch_size={batch})...")
            try:
                self.fk_services.warmup(batch)
                self.get_logger().info("  -> FK model ready")
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(
                    f"Startup FK warmup failed: {e}. /fk and /fk_batch will "
                    "report 'not initialized' until /warmup_fk succeeds."
                )
        else:
            self.get_logger().info(
                "Skipping FK warmup (warmup_fk:=false); call "
                f"{self.get_name()}/warmup_fk before /fk"
            )

    def _warmup_initial_planner(self, planner_type: str):
        self.get_logger().info(f"Warming up {planner_type} planner...")

        # Startup captures CUDA graphs (seed-IK graph in _warmup_classic; the
        # camera workers are already live and would invalidate an unguarded
        # capture). Hold the barrier exclusive across the warmup; in-flight
        # lane jobs drain in milliseconds and nothing else contends yet.
        with self.gpu_lock:
            self._warmup_initial_planner_locked(planner_type)

        self.get_logger().info(f"{planner_type} planner ready")

    def _warmup_initial_planner_locked(self, planner_type: str):
        """Warmup dispatch. Runs with the GPU barrier held (see above)."""

        if planner_type in ("mpc", "model_predictive_control"):
            self._warmup_mpc()
        elif planner_type in ("retarget", "motion_retargeting", "teleop"):
            self._warmup_reactive("retarget")
        elif planner_type in ("classic", "joint_space", "motion_gen"):
            self._warmup_classic()
        else:
            self._warmup_classic()
            self.get_logger().warn(
                f"Planner '{planner_type}' not fully wired, falling back to classic warmup"
            )

    def _warmup_classic(self):
        """Warm up MotionPlanner for Classic/JointSpace planners."""
        if self.motion_planner is None:
            self.get_logger().info("  -> Initializing MotionPlanner...")
            self.config_wrapper_motion.set_motion_gen_config(self, None, None)
            # The wrapper sets both self.motion_planner and self.motion_gen.
            SinglePlanner.set_motion_planner(self.motion_planner)
            self.get_logger().info(
                "  -> MotionPlanner ready and shared with SinglePlanner"
            )
        else:
            self.get_logger().info("  -> MotionPlanner already initialized (cache)")

    def _ensure_ground_plane(self):
        """No-op: the ground is the `floor` cuboid of the loaded world file.

        Both shipped worlds put it at z=-0.8 (config/floor_world.yml here,
        leeloo_world.yaml in the leeloo deployment).

        This method used to add a `ground` cuboid at z=-0.1 at runtime. That
        extra cuboid landed AFTER the voxelization SceneCollision had been
        pre-allocated (sized on the world file's three cuboids), so it overflowed
        that cache on every perception refresh -> CPU fallback (~10s) that froze
        the MPC loop. Removed: the robot base sits ~80cm above the floor, which
        is therefore collision ground enough.

        Kept as a no-op rather than deleted because the reactive warmup path
        still calls it (see _warmup_reactive).
        """
        return

    def _warmup_reactive(self, key: str):
        """Build a reactive controller's solver on demand from the shared context.

        Each reactive controller's build_solver() publishes its solver where the
        node expects it (self.mpc / self.retargeter).
        """
        self._ensure_ground_plane()
        self.planner_manager.get_planner(key).ensure_solver()
        self.get_logger().info(f"  -> Reactive '{key}' solver ready")

    def _warmup_mpc(self):
        """Warm up the v2 ModelPredictiveControl solver on demand."""
        if self.mpc is not None:
            self.get_logger().info("  -> MPC solver already initialized (cache)")
            return
        self.get_logger().info("  -> Initializing MPC solver...")
        self._warmup_reactive("mpc")

    def update_all_solvers_world(self, scene=None):
        """Propagate scene updates to all initialized solvers.

        The pushes upload the Scene to every solver's CUDA collision model —
        a CUDA op, so it must be serialised against CUDA graph captures
        (gpu_lock docstring). Holds gpu_lock (blocking, reentrant RLock): most
        callers already run under it (refresh_perception_world, clear_voxel,
        attachment), but add_object/remove_object reach us through the world-
        changed observer WITHOUT a lock, and those run on a service thread that
        can overlap a capture. RLock reentrancy makes the nested acquires free.
        """
        with self.gpu_lock:
            obstacle_manager = self.config_wrapper_motion.obstacle_manager
            # Normalize whatever scene we're handed to the solver-supported types
            # (sphere/cylinder/capsule -> mesh), or they're silently dropped from
            # collision. See obstacle_manager.collision_world_scene().
            scene = self._solver_bound_scene(scene, obstacle_manager)

            # DIAGNOSTIC (default off => normal behaviour). Withholds the perception
            # ESDF voxel layer from the solvers so only analytic primitives remain,
            # to test whether the voxel layer is what pegs con_scene_collision at its
            # sentinel value. SAFETY: with this enabled the solvers do NOT see
            # camera-observed obstacles, so run it only with a clear workspace.
            if (
                not self.get_parameter("push_esdf_to_solvers")
                .get_parameter_value()
                .bool_value
            ):
                scene = obstacle_manager.primitives_only_scene()
                self.get_logger().warn(
                    "push_esdf_to_solvers=false: solvers see analytic primitives ONLY "
                    "(no camera obstacles) - diagnostic mode",
                    throttle_duration_sec=5.0,
                )
            elif obstacle_manager.collision_cache["voxel"] is None:
                # No voxel cache pre-allocated in the solvers (SetCollisionCache
                # blox=0) — a scene carrying an ESDF layer would raise "Voxel cache
                # not initialized" inside update_world. Degrade gracefully: no
                # camera-based collision avoidance instead of a hard planning failure.
                scene = obstacle_manager.primitives_only_scene()
                self.get_logger().warn(
                    "Voxel collision cache disabled (SetCollisionCache blox=0): "
                    "solvers see analytic primitives ONLY, no camera obstacles",
                    throttle_duration_sec=5.0,
                )

            _t_push0 = time.monotonic()
            if self.motion_planner is not None:
                self.motion_planner.update_world(scene)
            _t_push_motion = time.monotonic()

            # Reactive controllers each own their collision model; delegate to the
            # controller's update_world() override (no node dependency on internals).
            if self.mpc is not None:
                self.planner_manager.get_planner("mpc").update_world(scene)
            _t_push_mpc = time.monotonic()
            if self.retargeter is not None:
                self.planner_manager.get_planner("retarget").update_world(scene)
            _t_push_retarget = time.monotonic()

            # Both services take the scene we just resolved — they must not
            # re-derive it from the obstacle manager, or the two degradation
            # paths above silently do not apply to them. fk_services ignores
            # the voxel layer by construction (its checker has no voxel cache,
            # see FKServices.update_world).
            self.ik_services.update_world(scene)
            _t_push_ik = time.monotonic()
            self.fk_services.update_world()
            _t_push_fk = time.monotonic()

            # Every update_world above clears + re-adds all obstacles with enable=1,
            # which wipes the flag cuRobo set for an attached obstacle — re-assert
            # the disabled set now or the static copy collides with the attached
            # spheres again.
            obstacle_manager.reapply_attached_disables(self)
            _t0 = getattr(obstacle_manager, "_world_set_t0", None)
            _handler_ms = (
                f", handler_total {(time.monotonic() - _t0) * 1e3:.1f} ms"
                if _t0 is not None else "")
            obstacle_manager._world_set_t0 = None
            self.get_logger().info(
                f"world-push: motion {(_t_push_motion - _t_push0) * 1e3:.1f} ms, "
                f"mpc {(_t_push_mpc - _t_push_motion) * 1e3:.1f} ms, "
                f"retarget {(_t_push_retarget - _t_push_mpc) * 1e3:.1f} ms, "
                f"ik {(_t_push_ik - _t_push_retarget) * 1e3:.1f} ms, "
                f"fk {(_t_push_fk - _t_push_ik) * 1e3:.1f} ms, "
                f"total {(_t_push_fk - _t_push0) * 1e3:.1f} ms{_handler_ms}"
            )

    def _solver_bound_scene(self, scene, obstacle_manager):
        """Resolve/normalize a scene before it is pushed to the solvers.

        Sphere/cylinder/capsule obstacles are converted to a solver-supported
        collision type (cuboid/mesh); without this they render in RViz but never
        collide. See obstacle_manager.collision_world_scene()."""
        if scene is None:
            scene = self.shared_scene
        # Round-trip through the collision-supported scene so non-cuboid
        # primitives become meshes.
        return obstacle_manager.collision_world_scene_from(scene)

    def refresh_perception_world(self):
        epoch = getattr(self, "perception_epoch", 0)
        if epoch == self._last_pushed_epoch:
            return                                  # nothing integrated since the last push
        obs = self.config_wrapper_motion.obstacle_manager
        with self.gpu_lock:
            if obs.refresh_esdf():
                self.update_all_solvers_world(obs.get_scene())
                self._last_pushed_epoch = epoch

    def rebuild_solvers_for_cache_change(self):
        """Rebuild all active solvers after a collision-cache change.

        The collision cache is allocated at solver creation, so a change
        requires recreating the solvers (a world update is not sufficient).
        Registered as ObstacleManager's cache-change observer.
        """
        self._rebuild_all_solvers("collision cache changed")

    def _rebuild_all_solvers(self, reason: str, robot_model_changed: bool = False):
        """Rebuild every solver built from the current robot cfg / world.

        Shared by the two build-time switches a plain world update cannot
        absorb: a collision-cache change (the cache is allocated at solver
        creation) and a joint-lock change (which re-parses the robot YAML
        first — see ``RobotModelManager.set_lock_joints``).

        ``robot_model_changed`` says the robot model itself was re-parsed, not
        just the world. Only then is the FK model stale — it holds its own GPU
        ``Kinematics`` built from the robot cfg at init (see ``FKServices``),
        whereas a cache change leaves the robot cfg untouched.

        Held under gpu_lock: this (re)captures CUDA graphs, same invariant as
        every other graph-capturing path in this node (see gpu_lock's other
        acquisitions) — without it, a concurrent camera integrate()/perception
        refresh could invalidate the graph being captured here
        (cudaErrorStreamCaptureInvalidated).
        """
        self.get_logger().info(f"{reason} - rebuilding solvers (blocking, ~20s)...")

        with self.gpu_lock:
            # Motion planner (present after the initial warmup).
            if self.motion_planner is not None:
                self.config_wrapper_motion.set_motion_gen_config(self, None, None)
                SinglePlanner.set_motion_planner(self.motion_planner)

            # IK (only if it was initialized). IKServices reads the canonical
            # (motion) cache directly, so no sync is needed.
            self.ik_services.rebuild()

            # FK keeps its own GPU model, so it only needs the rebuild when the
            # robot model behind it was re-parsed.
            if robot_model_changed:
                fk_services = getattr(self, "fk_services", None)
                if fk_services is not None:
                    fk_services.rebuild()

            # Reactive controllers (only if initialized). Built from the SAME
            # shared cache, so just rebuild their solvers — no manual cache copy.
            if self.mpc is not None:
                self.planner_manager.get_planner("mpc").rebuild_solver()
            if self.retargeter is not None:
                self.planner_manager.get_planner("retarget").rebuild_solver()

            # Rebuilt solvers hold no graph yet — their first step() will
            # capture, so mark it pending (see take_graph_capture_pending).
            self._set_graph_capture_pending()

        self.get_logger().info(f"Solvers rebuilt after {reason}")

    # ------------------------------------------------------------------
    # Plan / execute callbacks
    # ------------------------------------------------------------------

    def generate_trajectory_callback(self, request: TrajectoryGeneration, response):
        """Single-request trajectory generation (preview workflow).

        DRY schema: the core request rides in ``request.request``
        (TrajectoryGoal) and the core result in ``response.response``
        (TrajectoryResult); the srv-level ``start/end_state_collisions``
        siblings mirror the embedded ones.
        """
        try:
            planner = self.planner_manager.get_current_planner()
            if planner is None:
                response.response = TrajectoryResult()
                response.response.success = False
                response.response.message = "No planner selected"
                response.response.stats = self._empty_stats()
                response.start_state_collisions = StateCollisions()
                response.end_state_collisions = StateCollisions()
                return response

            result = self._plan_trajectory_goal(planner, request.request)
            response.response = result
            # srv-level StateCollisions siblings mirror the embedded ones.
            response.start_state_collisions = result.start_state_collisions
            response.end_state_collisions = result.end_state_collisions

            # Open-loop caching now lives inside _plan_trajectory_goal (it is
            # shared by the batch surfaces too), so every planned segment lands
            # in the multi-entry cache. Reactive controllers have no
            # trajectory to cache, and a reactive solve invalidates any cached
            # open-loop plan.
            if not planner.is_open_loop():
                self._pending_plan = []
            return response
        except Exception as e:
            self.get_logger().error(f"Trajectory generation error: {e}")
            self.get_logger().error(traceback.format_exc())
            response.response = TrajectoryResult()
            response.response.success = False
            response.response.message = f"Error: {e}"
            response.response.stats = self._empty_stats()
            response.start_state_collisions = StateCollisions()
            response.end_state_collisions = StateCollisions()
            return response

    def _plan_trajectory_goal(self, planner, goal):
        """Serialise this plan on the 'plan' lane (sole owner of the MotionPlanner + its graphs)."""
        return self.lane_plan.call(self._plan_trajectory_goal_impl, planner, goal, prio=PRIO_PLAN)

    def _plan_trajectory_goal_impl(self, planner, goal: TrajectoryGoal) -> TrajectoryResult:
        """Plan one TrajectoryGoal and produce its TrajectoryResult.

        Shared by the generate_trajectory srv and the trajectory_generation_batch
        srv (one call per problem) so every surface serializes results
        identically: stats + winner/per-waypoint insight are ALWAYS populated,
        trajectory/dt on success (empty otherwise), StateCollisions diagnostics
        on failure.
        """
        result_msg = TrajectoryResult()
        _t_total = time.monotonic()
        try:
            ok, reason = self._check_goal_request(planner, goal)
            if not ok:
                result_msg.success = False
                result_msg.message = reason
                result_msg.trajectory = []
                result_msg.dt = 0.0
                result_msg.stats = self._build_planning_stats(goal, {}, problems=1)
                return result_msg

            _, start_state = self._resolve_start_state(goal)
            config = self._get_planner_config(planner)
            self._setup_planner(planner)
            _t_setup = time.monotonic()

            # Refresh the perception-based collision world before planning so
            # the plan accounts for the latest camera data.
            self.refresh_perception_world()
            _t_world = time.monotonic()

            self.get_logger().info(f"Planning with {planner.get_planner_name()}")
            # _plan_lock() holds gpu_lock for the entire plan when CUDA graphs
            # are enabled: cuRobo's optimizer can re-capture graphs mid-plan
            # (reset_shape from _prepare_goal_buffer), which races with the viz
            # timer if no lock is held (see _plan_lock docstring, 2026-09-11).
            with self._plan_lock():
                result = planner.plan(start_state, goal, config, self.robot_context)
            _t_plan = time.monotonic()
            self.get_logger().info(
                f"[plan-perf] {planner.get_planner_name()}: "
                f"node_setup(start-state+config+planner-setup+lock) {(_t_setup - _t_total) * 1e3:.1f} ms, "
                f"world_refresh(ESDF+world-push) {(_t_world - _t_setup) * 1e3:.1f} ms, "
                f"planner_plan(=curobo_solve+finalize) {(_t_plan - _t_world) * 1e3:.1f} ms, "
                f"end_to_end {(_t_plan - _t_total) * 1e3:.1f} ms"
            )

            self._fill_result_insight(result_msg, planner, goal, result, start_state)
            # Every successful open-loop plan lands in the multi-entry cache
            # here (single srv AND the batch surfaces, which reuse this path):
            # a chain executing later replays each planned segment instead of
            # re-solving it. See _store_pending_plan.
            if result_msg.success and planner.is_open_loop():
                self._store_pending_plan(start_state, goal)
            return result_msg

        except Exception as e:
            self.get_logger().error(f"Trajectory generation error: {e}")
            self.get_logger().error(traceback.format_exc())
            result_msg.success = False
            result_msg.message = f"Error: {e}"
            result_msg.trajectory = []
            result_msg.dt = 0.0
            result_msg.start_state_collisions = StateCollisions()
            result_msg.end_state_collisions = StateCollisions()
            result_msg.stats = self._build_planning_stats(goal, {}, problems=1)
            return result_msg

    def _plan_trajectory_goal_batch(self, planner, goals):
        """Serialise this batch plan on the 'plan' lane (sole owner of the MotionPlanner + its graphs)."""
        return self.lane_plan.call(self._plan_trajectory_goal_batch_impl, planner, goals, prio=PRIO_PLAN)

    def _plan_trajectory_goal_batch_impl(self, planner, goals) -> list:
        """Plan an array of goals in ONE planner.plan_batch call.

        Mirrors _plan_trajectory_goal's ordering (validate → resolve starts →
        config → setup planner → refresh world → plan under _plan_lock) but
        runs the planner-independent steps ONCE for the whole batch: one
        perception-world refresh, one lock acquisition, one solver call (the
        planner stacks the problems into the solver's batch dimension, so N
        problems share one CUDA-graph capture instead of N).

        Falls back to one sequential _plan_trajectory_goal per goal when any
        request fails validation or the batched solve raises — the per-problem
        response contract (success OR failure for every request) always holds.

        Returns:
            One TrajectoryResult per goal, in request order, serialized through
            the same _fill_result_insight as the single-goal service.
        """
        if any(not self._check_goal_request(planner, g)[0] for g in goals):
            return [self._plan_trajectory_goal(planner, g) for g in goals]
        try:
            start_states = [self._resolve_start_state(g)[1] for g in goals]
            config = self._get_planner_config(planner)
            self._setup_planner(planner)

            # ONE perception refresh for the whole batch (the per-goal path
            # re-runs it per problem). The staleness window is unchanged: no
            # new depth frames integrate mid-batch under _plan_lock.
            self.refresh_perception_world()

            self.get_logger().info(
                f"Batched planning: {len(goals)} problem(s) with "
                f"{planner.get_planner_name()}"
            )
            # See _plan_lock: holds gpu_lock for the WHOLE batched solve, since
            # cuRobo may re-capture CUDA graphs mid-plan (even more likely when
            # one call covers N problems).
            with self._plan_lock():
                planned = planner.plan_batch(
                    start_states, goals, config, self.robot_context
                )
            if len(planned) != len(goals):
                raise RuntimeError(
                    f"plan_batch returned {len(planned)} results for "
                    f"{len(goals)} problems"
                )
            response_msgs = []
            for g, r, s in zip(goals, planned, start_states):
                result_msg = self._fill_result_insight(
                    TrajectoryResult(), planner, g, r, s
                )
                # Each batched problem keeps its OWN trajectory: the
                # planner's planned_trajectory slot ends up holding only the
                # last problem's after plan_batch, so store per-result.
                if result_msg.success and planner.is_open_loop():
                    self._store_pending_plan(s, g, traj=getattr(r, "trajectory", None))
                response_msgs.append(result_msg)
            return response_msgs
        except Exception as e:
            self.get_logger().error(
                f"Batch planning error (falling back to "
                f"sequential per-goal planning): {e}"
            )
            self.get_logger().error(traceback.format_exc())
            return [self._plan_trajectory_goal(planner, g) for g in goals]

    def trajectory_generation_batch_callback(self, request, response):
        """Batch trajectory generation: plan every problem, rollup total_stats.

        Problems are solved in ONE curobo call when the active planner exposes
        ``plan_batch`` (batch dimension = problems, seed dimension = candidate
        trajectories — N problems cost one GPU solve and one world refresh);
        otherwise the callback falls back to one sequential plan per problem.
        Either way it always produces one ``responses[i]`` (TrajectoryResult)
        per problem — success or failure — and a cross-problem ``total_stats``
        rollup (always populated once dispatched, even when every problem
        fails).
        """
        try:
            planner = self.planner_manager.get_current_planner()
            if planner is None:
                response.success = False
                response.error_msg = "No planner selected"
                response.responses = []
                response.total_stats = self._empty_stats()
                return response

            goals = list(getattr(request, "requests", None) or [])
            if not goals:
                response.responses = []
                response.success = True
                response.error_msg = ""
                response.total_stats = self._empty_stats()
                return response
            if getattr(planner, "plan_batch", None) is not None:
                response.responses = self._plan_trajectory_goal_batch(planner, goals)
            else:
                response.responses = [
                    self._plan_trajectory_goal(planner, g) for g in goals
                ]
            response.success = all(r.success for r in response.responses)
            response.error_msg = ""
            response.total_stats = self._rollup_stats(response.responses)
            self.get_logger().info(
                f"Batch trajectory generation: {len(response.responses)} "
                f"problem(s), success={response.success} "
                f"(total_stats: problems={response.total_stats.problems}, "
                f"candidates generated={response.total_stats.candidates_generated}, "
                f"solved={response.total_stats.candidates_solved}, "
                f"pruned={response.total_stats.candidates_pruned}, "
                f"waypoints={response.total_stats.waypoints_planned})"
            )
            return response
        except Exception as e:
            self.get_logger().error(f"Batch trajectory generation error: {e}")
            self.get_logger().error(traceback.format_exc())
            response.success = False
            response.error_msg = f"Error: {e}"
            response.responses = []
            response.total_stats = self._empty_stats()
            return response

    # ------------------------------------------------------------------
    # Result → TrajectoryResult / PlanningStats serialization (DRY schema)
    # ------------------------------------------------------------------

    def _fill_result_insight(
        self,
        result_msg: TrajectoryResult,
        planner,
        goal,
        result,
        start_state,
        *,
        fill_arrays: bool = True,
    ) -> TrajectoryResult:
        """Populate a TrajectoryResult from a fresh PlannerResult.

        Writes success/message, the winner goal+seed / per-waypoint status
        arrays, always-populated stats (with gated considered rows) and the
        StateCollisions diagnostics (cleared on success, populated on failure).
        ``fill_arrays`` toggles the trajectory/dt serialization. The execute
        action keeps it ON for successful re-solves: the executor's
        chain-continuity check compares the plan-time endpoints against the
        DRIVEN trajectory, and a result without waypoints is skipped as
        "nothing to compare", silently driving a diverged chain on. It stays
        OFF on the execution-failure path (already returning False).
        """
        meta = result.metadata or {}
        # Solver-reported solve time for PlanningStats.solver_time_s — the
        # same number the native leg records as "Solve Time"
        # (result.solve_time, seconds; NOT total_time, which is what the
        # planner-phases line reports as solver_total for the wrapper split).
        # result.solve_time is not always a plain float (solver internals may
        # leave a tensor), so try scalar, single-element tensor, then the
        # winner considered row — NaN only when nothing usable exists.
        solver_time_s = self._solver_time_s(result, meta)
        self._fill_insight_fields(
            result_msg,
            goal,
            meta,
            planner,
            success=result.success,
            message=result.message,
            start_state=start_state,
            solver_time_s=solver_time_s,
        )

        if not result.success:
            self.get_logger().error(
                f"Planning failed: {result.message}\n"
                + self._format_collision_feedback(
                    result_msg.start_state_collisions, result_msg.end_state_collisions
                )
            )
            return result_msg

        if fill_arrays and result.trajectory is not None:
            waypoints, dt, n = self._result_trajectory(planner, result.trajectory)
            result_msg.trajectory = waypoints
            result_msg.dt = dt
            result_msg.start_state_collisions = StateCollisions()
            result_msg.end_state_collisions = StateCollisions()
            self.get_logger().info(
                f"Planning succeeded: {result.message} "
                f"(trajectory: {n} waypoints, dt: {dt}s)"
            )
        else:
            self.get_logger().info(f"Planning succeeded: {result.message}")
        return result_msg

    def _fill_insight_fields(
        self,
        result_msg: TrajectoryResult,
        goal,
        meta: dict,
        planner,
        *,
        success: bool,
        message: str,
        start_state=None,
        solver_time_s: float = float("nan"),
    ) -> TrajectoryResult:
        """Write winner arrays + stats + collision diagnostics from ``meta``.

        Shared by the fresh-plan and cached-preview-execute paths: every field
        (except trajectory/dt, filled by the caller) is written from a metadata
        block whose keys match the planner layer's ``_result_metadata``.
        """
        result_msg.success = bool(success)
        result_msg.message = message

        sel = meta.get("selected_goal_index")
        result_msg.selected_goal_index = (
            [int(x) for x in sel] if sel is not None else []
        )
        seeds = meta.get("selected_seed_index")
        result_msg.selected_seed_index = (
            [int(x) for x in seeds] if seeds is not None else []
        )
        status = meta.get("waypoint_status")
        result_msg.waypoint_status = (
            [int(x) for x in status] if status is not None else []
        )

        result_msg.stats = self._build_planning_stats(goal, meta, problems=1,
                                                      solver_time_s=solver_time_s)

        if not success:
            goal_joints = self._goalset_joint_target(goal)
            self._fill_collision_diagnostics(
                result_msg,
                planner=planner,
                start_joints=self._diagnostic_joints(start_state),
                goal_joints=list(goal_joints) if goal_joints else None,
            )
        return result_msg

    def _result_trajectory(self, planner, traj):
        """Flip a successful plan's trajectory into ([JointState], dt, n).

        Shared by every surface so the trajectory serialization (interpolated
        [B, T, D] → one JointState per waypoint, dt from the planner's trajopt
        config) stays identical across the single srv, batch srv, and action.
        Returns ``([], 0.0, 0)`` when ``traj`` carries no position data.
        """
        if traj is None or getattr(traj, "position", None) is None:
            return [], 0.0, 0
        dt = self.get_parameter("interpolation_dt").get_parameter_value().double_value
        mp = getattr(planner, "motion_planner", None)
        trajopt = getattr(mp, "trajopt_solver", None) if mp is not None else None
        trajopt_cfg = getattr(trajopt, "config", None)
        dt_val = getattr(trajopt_cfg, "interpolation_dt", None)
        if dt_val is not None:
            dt = float(dt_val)

        pos_tensor = traj.position
        vel_tensor = traj.velocity
        while pos_tensor.ndim > 2:
            pos_tensor = pos_tensor[0]
            if vel_tensor is not None:
                vel_tensor = vel_tensor[0]

        n_waypoints = pos_tensor.shape[0]
        pos_list = pos_tensor.detach().cpu().tolist()
        vel_list = (
            vel_tensor.detach().cpu().tolist() if vel_tensor is not None else None
        )
        trajectory_msgs = []
        for i in range(n_waypoints):
            waypoint = JointStateMsg()
            if hasattr(traj, "joint_names") and traj.joint_names is not None:
                waypoint.name = list(traj.joint_names)
            waypoint.position = pos_list[i]
            if vel_list is not None:
                waypoint.velocity = vel_list[i]
            trajectory_msgs.append(waypoint)
        return trajectory_msgs, dt, n_waypoints

    def _fill_reused_trajectory(
        self, result_msg: TrajectoryResult, planner, traj=None
    ) -> None:
        """Fill trajectory/dt on a cache-hit execute from the replayed plan.

        With the multi-entry cache a hit can replay a MID-chain segment,
        whose plan is no longer the planner's own slot (``planned_trajectory``
        only ever holds the last segment planned), so the replaying entry's
        trajectory is passed in; it falls back to the planner slot for the
        single-segment preview workflow. That trajectory IS what gets driven,
        and it has to reach the caller's TrajectoryResult for the same reason
        the re-solve path fills arrays: the executor's chain-continuity check
        compares the plan-time endpoints against the DRIVEN trajectory, and a
        result without waypoints is skipped as "nothing to compare" - which
        would let a bad reuse drive a chain silently.
        """
        if traj is None:
            traj = getattr(planner, "planned_trajectory", None)
        if traj is None or getattr(traj, "position", None) is None:
            return
        waypoints, dt, n = self._result_trajectory(planner, traj)
        if n:
            result_msg.trajectory = waypoints
            result_msg.dt = dt

    def _build_planning_stats(
        self, goal, meta: dict, problems: int = 1,
        solver_time_s: float = float("nan"),
    ) -> PlanningStats:
        """PlanningStats for one problem — ALWAYS populated.

        ``waypoints_planned`` = the number of goal segments solved (one per
        ``goalsets`` entry); the candidate accounting and considered rows ride
        the planner metadata when reported (considered rows gated on the
        request's ``log_considered_trajectories``). ``solver_time_s`` is
        cuRobo's self-reported solve time (result.total_time, seconds) — the
        same metric the native leg records; NaN when no solve ran.
        """
        stats = PlanningStats()
        stats.problems = int(problems)
        stats.waypoints_planned = len(getattr(goal, "goalsets", None) or [])
        stats.candidates_generated = int(meta.get("candidates_generated", 0))
        stats.candidates_solved = int(meta.get("candidates_solved", 0))
        stats.candidates_pruned = int(meta.get("candidates_pruned", 0))
        try:
            stats.solver_time_s = float(solver_time_s)
        except Exception:
            stats.solver_time_s = float("nan")
        stats.position_error_mm = self._winner_position_error_mm(meta)
        try:
            _mt = float(meta.get("motion_time_s", float("nan")))
            stats.motion_time_s = _mt if _mt == _mt else float("nan")
        except Exception:
            stats.motion_time_s = float("nan")
        if self._log_considered_requested(goal) and meta.get("considered"):
            for row in meta["considered"]:
                stats.considered.append(self._to_considered_trajectory(row))
        return stats

    @staticmethod
    def _solver_time_s(result, meta: dict) -> float:
        """Solver-reported solve time (s) for PlanningStats, or NaN.

        Prefers ``result.solve_time`` as a plain scalar (what the native leg
        records); accepts single-element tensors via ``.item()``; falls back
        to the winner considered row's ``solve_time`` (same rows/indices as
        :meth:`_winner_position_error_mm`). NaN when nothing usable exists.
        """
        try:
            return float(getattr(result, "solve_time", None))
        except Exception:
            pass
        try:
            import torch
            _v = getattr(result, "solve_time", None)
            if isinstance(_v, torch.Tensor) and _v.numel() == 1:
                return float(_v.detach().flatten()[0].item())
        except Exception:
            pass
        try:
            rows = [r for r in (meta.get("considered") or []) if r is not None]
            goal_index = [int(v) for v in (meta.get("selected_goal_index") or [])]
            seed_index = [int(v) for v in (meta.get("selected_seed_index") or [])]
            if rows and goal_index and len(seed_index) == len(goal_index):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    if (row.get("problem", 0) != 0
                            or row.get("segment", 0) >= len(goal_index)):
                        continue
                    if (int(row.get("goalset_candidate", 0)) == goal_index[row.get("segment", 0)]
                            and int(row.get("seed", 0)) == seed_index[row.get("segment", 0)]):
                        return float(row.get("solve_time", float("nan")))
        except Exception:
            pass
        return float("nan")

    @staticmethod
    def _winner_position_error_mm(meta: dict) -> float:
        """Winner candidate's solver-reported position error (mm), or NaN.

        Mirrors the benchmark's ``winner_position_error_mm`` (compare.py) on
        the planner's considered rows: match problem 0, the selected
        goalset-candidate/seed per segment, read ``max_waypoint_error`` (m)
        × 1000. Same metric, same computation as the native leg's
        ``result.position_error * 1000`` — but server-side, so it needs no
        request flag and is always present.
        """
        try:
            rows = [r for r in (meta.get("considered") or []) if r is not None]
            goal_index = [int(v) for v in (meta.get("selected_goal_index") or [])]
            seed_index = [int(v) for v in (meta.get("selected_seed_index") or [])]
            if not rows or not goal_index or len(seed_index) != len(goal_index):
                return float("nan")
            for row in rows:
                gv = row.get("goalset_candidate", 0) if isinstance(row, dict) else 0
                sv = row.get("seed", 0) if isinstance(row, dict) else 0
                seg = row.get("segment", 0) if isinstance(row, dict) else 0
                pv = row.get("problem", 0) if isinstance(row, dict) else 0
                if pv != 0 or seg >= len(goal_index):
                    continue
                if int(gv) == goal_index[seg] and int(sv) == seed_index[seg]:
                    return float(row.get("max_waypoint_error", float("nan"))) * 1000.0
        except Exception:
            pass
        return float("nan")

    @staticmethod
    def _rollup_stats(responses) -> PlanningStats:
        """Cross-problem PlanningStats rollup for the batch response.

        Sums the scalar counts across problems and concatenates the considered
        rows, preserving each row's per-problem ``problem`` attribution.
        """
        total = PlanningStats()
        total.problems = len(responses)
        total.solver_time_s = float("nan")  # times don't sum; per-problem rows carry them
        total.position_error_mm = float("nan")  # likewise per-problem
        total.candidates_generated = sum(
            int(r.stats.candidates_generated) for r in responses
        )
        total.candidates_solved = sum(int(r.stats.candidates_solved) for r in responses)
        total.candidates_pruned = sum(int(r.stats.candidates_pruned) for r in responses)
        total.waypoints_planned = sum(int(r.stats.waypoints_planned) for r in responses)
        for r in responses:
            total.considered.extend(r.stats.considered)
        return total

    @staticmethod
    def _empty_stats() -> PlanningStats:
        """Truly-empty PlanningStats (no problems were dispatched)."""
        stats = PlanningStats()
        stats.solver_time_s = float("nan")
        stats.position_error_mm = float("nan")
        return stats

    @staticmethod
    def _log_considered_requested(goal) -> bool:
        """Whether the request asked for the considered-trajectory rows."""
        opts = getattr(goal, "options", None)
        if opts is None:
            return False
        try:
            return bool(getattr(opts, "log_considered_trajectories", False))
        except Exception:
            return False

    def _to_considered_trajectory(self, row) -> ConsideredTrajectory:
        """Map one planner metadata 'considered' row onto ConsideredTrajectory.

        Defensive: every field is read via ``.get()`` so a row that lacks a
        whole-task-only measurement (``waypoint_cost`` / ``path_length`` /
        ``clearance`` — the per-segment machinery provides none) serializes as
        0 instead of raising.
        """
        if isinstance(row, ConsideredTrajectory):
            return row
        c = ConsideredTrajectory()
        # Message index fields are unsigned: saturate any negative index
        # (e.g. -1 from a failed/absent solve) to 0 so a failure response
        # still serializes instead of raising OverflowError.
        c.problem = max(0, int(row.get("problem", 0)))
        c.segment = max(0, int(row.get("segment", 0)))
        c.goalset_candidate = max(0, int(row.get("goalset_candidate", 0)))
        c.seed = max(0, int(row.get("seed", 0)))
        c.success = bool(row.get("success", False))
        c.cost = float(row.get("cost", 0.0))
        c.waypoint_cost = float(row.get("waypoint_cost", 0.0))
        c.path_length = float(row.get("path_length", 0.0))
        c.clearance = float(row.get("clearance", 0.0))
        c.max_waypoint_error = float(row.get("max_waypoint_error", 0.0))
        c.solve_time = float(row.get("solve_time", 0.0))
        return c

    @staticmethod
    def _result_meta_from_planner(planner) -> dict:
        """Planner-layer metadata block from the planner's CURRENT per-plan attrs.

        Used when the execute action reuses a cached preview plan: the
        planner's selected-index / status / tally attrs still hold the values
        the preview plan() call wrote (the signature matched, so they describe
        exactly the plan being executed).
        """
        meta = {}
        for key, attr in (
            ("selected_goal_index", "_selected_goal_indexes"),
            ("selected_seed_index", "_selected_seed_index"),
            ("waypoint_status", "_waypoint_status"),
        ):
            val = getattr(planner, attr, None)
            if val is not None:
                meta[key] = list(val)
        tally = getattr(planner, "_candidate_tally", None)
        if tally:
            meta["candidates_generated"] = int(tally.get("generated", 0))
            meta["candidates_solved"] = int(tally.get("solved", 0))
            meta["candidates_pruned"] = int(tally.get("pruned", 0))
        considered = getattr(planner, "_considered_rows", None)
        if considered:
            meta["considered"] = list(considered)
        return meta

    @staticmethod
    def _goalset_joint_target(request):
        """Last non-empty per-segment joint target (the plan's end state), if any.

        Joint-space segments carry their target in ``Goalset.target_joint_positions``
        (the srv/action no longer expose a top-level joint array). The last such
        segment is the plan's end configuration.
        """
        joints = None
        for g in getattr(request, "goalsets", None) or []:
            gt = getattr(g, "target_joint_positions", None)
            if gt:
                joints = list(gt)
        return joints

    @staticmethod
    def _diagnostic_joints(state):
        """Joint positions (list) from a JointState's first batch row, or None.

        Best-effort: any malformed state yields None, and the caller falls back
        to the robot's live pose for the "start state" clause.
        """
        try:
            if state is None:
                return None
            pos = state.position
            if pos is None:
                return None
            if pos.ndim > 1:
                pos = pos[0]
            return pos.detach().cpu().tolist()
        except Exception:
            return None

    def _build_collision_diagnostics(
        self, planner=None, start_joints=None, goal_joints=None
    ):
        """Build structured collision diagnostics for a srv Response or action Result.

        Returns ``(start_state_collisions, end_state_collisions)`` — two
        StateCollisions messages (each wrapping world/self/cspace contact
        arrays), labelled for the plan's start and goal (end) configurations.
        Best-effort: returns two empty StateCollisions on any failure.
        """
        try:
            wrapper = self.config_wrapper_motion
            kin = None
            attach_svc = getattr(self, "attachment_services", None)
            if attach_svc is not None:
                try:
                    kin = attach_svc.kinematics()
                except Exception:
                    kin = None

            def _eval_state(label, at):
                sc = StateCollisions()
                if at is None:
                    return sc
                # World collisions
                reports = _attributed_collisions(
                    wrapper, self, kin=kin, solver=planner, at_joints=at
                )
                if reports:
                    sc.world_contacts = [
                        WorldCollisionContact(
                            state_label=label,
                            link=r["link"],
                            obstacle=(
                                (
                                    ", ".join(r["obstacles"])
                                    if r["obstacles"]
                                    else r["clobber_note"]
                                )
                                or "unknown-obstacle"
                            ),
                            depth=round(r["depth"], 6),
                        )
                        for r in sorted(reports, key=lambda r: -r["depth"])
                    ]

                # Self-collisions
                selfs = _attributed_self_collisions(
                    wrapper, self, kin=kin, at_joints=at
                )
                if selfs:
                    sc.self_contacts = [
                        SelfCollisionContact(
                            state_label=label,
                            link_a=r["link"],
                            link_b=r["link_b"],
                            depth=round(r["depth"], 6),
                        )
                        for r in sorted(selfs, key=lambda r: -r["depth"])
                    ]

                # Joint-limit violations
                limits = _attributed_joint_limit_violations(
                    wrapper, self, kin=kin, at_joints=at
                )
                if limits:
                    sc.cspace_violations = [
                        JointLimitViolation(
                            state_label=label,
                            joint=r["joint"],
                            value=round(r["value"], 6),
                            lower=round(r["lower"], 6),
                            upper=round(r["upper"], 6),
                        )
                        for r in limits
                    ]
                return sc

            start_sc = _eval_state("start", start_joints)
            end_sc = (
                _eval_state("goal", list(goal_joints))
                if goal_joints
                else StateCollisions()
            )
            return start_sc, end_sc

        except Exception as e:
            self.get_logger().warn(
                f"collision diagnostics unavailable: {e}\n{traceback.format_exc()}",
                throttle_duration_sec=5.0,
            )
            return StateCollisions(), StateCollisions()

    def _fill_collision_diagnostics(
        self, response, planner=None, start_joints=None, goal_joints=None
    ):
        """Populate start/end StateCollisions on a Response or Result."""
        start_sc, end_sc = self._build_collision_diagnostics(
            planner=planner, start_joints=start_joints, goal_joints=goal_joints
        )
        response.start_state_collisions = start_sc
        response.end_state_collisions = end_sc

    def _format_collision_feedback(self, start_sc, end_sc):
        """Render start/goal StateCollisions messages as a readable multi-line
        block for node-side failure logs (mirrors the grasp orchestrator's
        formatter so both sides print the same layout)."""
        return "\n".join(
            [
                "collision feedback:",
                self._format_state_collisions("start", start_sc),
                self._format_state_collisions("goal", end_sc),
            ]
        )

    @staticmethod
    def _format_state_collisions(label, sc):
        """Multi-line summary of a StateCollisions message, one row per
        contact, aligned columns, deepest contact first (joint-limit rows
        sort last)."""
        if sc is None:
            return f"  {label}: not evaluated"
        rows = []
        for c in sc.world_contacts:
            rows.append(
                (
                    c.depth,
                    c.link,
                    "->",
                    c.obstacle,
                    f"{c.depth * 1000:.1f} mm past surface",
                )
            )
        for c in sc.self_contacts:
            rows.append(
                (
                    c.depth,
                    c.link_a,
                    "<->",
                    c.link_b,
                    f"{c.depth * 1000:.1f} mm penetration",
                )
            )
        for c in sc.cspace_violations:
            rows.append(
                (
                    float("-inf"),
                    c.joint,
                    "limit",
                    f"outside [{c.lower:.3f}, {c.upper:.3f}] (value {c.value:.3f})",
                    "",
                )
            )
        if not rows:
            return f"  {label}: clear (no world/self contacts, within joint limits)"
        rows.sort(key=lambda r: r[0], reverse=True)
        width = max(len(r[1]) for r in rows)
        body = "\n".join(
            f"    {a:<{width}}  {op}  {b}  {suffix}".rstrip()
            for _, a, op, b, suffix in rows
        )
        noun = "contact" if len(rows) == 1 else "contacts"
        return f"  {label}: {len(rows)} {noun}\n{body}"

    def execute_callback(self, goal_handle):
        """Unified 'generate + execute' action for every controller.

        Open-loop: reuse a matching cached (preview) trajectory or plan one, then
        execute it to completion (terminates on its own).
        Reactive: set the goal then servo continuously; terminates only on cancel
        or error, signalling `on_target` through the action feedback.

        DRY schema: ``goal_handle.request`` is SendTrajectory.Goal wrapping the
        core TrajectoryGoal (``.goal``); the core result rides in
        ``result_msg.result`` (TrajectoryResult).
        """
        result_msg = SendTrajectory.Result()
        try:
            planner = self.planner_manager.get_current_planner()
            if planner is None:
                result_msg.result = TrajectoryResult()
                result_msg.result.success = False
                result_msg.result.message = "No planner selected"
                result_msg.result.stats = self._empty_stats()
                goal_handle.abort()
                return result_msg

            goal = goal_handle.request.goal
            return self._execute_goal(goal_handle, goal, planner, result_msg)
        finally:
            with self._goal_lock:
                self._goal_active = False

    def _refresh_and_plan(self, planner, start_state, goal, config):
        """Runs ON a lane thread."""
        self.refresh_perception_world()
        with self._plan_lock():
            return planner.plan(start_state, goal, config, self.robot_context)

    def _refresh_and_plan_reactive(self, planner, start_state, goal, config):
        """Runs ON the reactive lane thread."""
        self.refresh_perception_world()
        return planner.plan(start_state, goal, config, self.robot_context)

    def _execute_goal(self, goal_handle, goal, planner, result_msg):
        _t_total = time.monotonic()
        try:
            ok, reason = self._check_goal_request(planner, goal)
            if not ok:
                self.get_logger().error(f"Invalid execution goal: {reason}")
                result_msg.result = TrajectoryResult()
                result_msg.result.success = False
                result_msg.result.message = f"Invalid goal: {reason}"
                result_msg.result.stats = self._build_planning_stats(
                    goal, {}, problems=1
                )
                goal_handle.abort()
                return result_msg

            _, start_state = self._resolve_start_state(goal)
            config = self._get_planner_config(planner)
            self._setup_planner(planner)
            _t_setup = time.monotonic()

            result = None
            if planner.is_open_loop():
                reuse = bool(
                    getattr(goal_handle.request, "allow_cached", True)
                ) and self.lane_plan.call(
                    self._pending_plan_matches, start_state, goal, prio=PRIO_PLAN
                )
                if reuse:
                    _t_reuse = time.monotonic()
                    self.get_logger().info(
                        f"[plan-perf] Reusing cached (pre-planned) trajectory "
                        f"(setup {(_t_reuse - _t_total) * 1e3:.1f} ms TOTAL)"
                    )
                else:
                    # The plan that was displayed and validated is about to be
                    # discarded and re-solved from scratch. The re-solve is
                    # anchored to the request's start_pose, which was baked in
                    # when the chain was PLANNED - so if the arm is not there
                    # now, the trajectory produced is a different one, not a
                    # re-run of the one that was checked. That is the
                    # execution/display mismatch, and it is silent: the goal
                    # succeeds, the controller reports success, and only the
                    # joint values disagree. Refuse rather than drive it.
                    if bool(getattr(goal_handle.request, "force_cached", False)):
                        # force_cached: the caller demands the validated plan
                        # and forbids a fresh solve here. With the multi-entry
                        # cache every chain segment has a matching entry, so a
                        # miss means the conditions genuinely changed (cache
                        # cleared, TTL expired, world mutated) - driving a
                        # fresh solve is exactly the IK-branch jump this
                        # pipeline exists to avoid. Fail loudly instead.
                        msg = (
                            "force_cached: no matching cached trajectory "
                            "for this target; refusing to re-solve"
                        )
                        self.get_logger().error(f"{planner.get_planner_name()}: {msg}")
                        result_msg.result = TrajectoryResult()
                        result_msg.result.success = False
                        result_msg.result.message = msg
                        goal_handle.abort()
                        return result_msg
                    drift_ok, drift_msg = self._check_execution_start_drift(goal)
                    if not drift_ok:
                        self.get_logger().error(drift_msg)
                        result_msg.result = TrajectoryResult()
                        result_msg.result.success = False
                        result_msg.result.message = drift_msg
                        goal_handle.abort()
                        return result_msg
                    result = self.lane_plan.call(
                        self._refresh_and_plan, planner, start_state, goal, config, prio=PRIO_PLAN)
                    _t_world = _t_setup
                    _t_plan = time.monotonic()
                    self.get_logger().info(
                        f"[plan-perf] {planner.get_planner_name()} (execute path): "
                        f"node_setup(start-state+config+planner-setup+lock) {(_t_setup - _t_total) * 1e3:.1f} ms, "
                        f"world_refresh(ESDF+world-push) {(_t_world - _t_setup) * 1e3:.1f} ms, "
                        f"planner_plan(=curobo_solve+finalize) {(_t_plan - _t_world) * 1e3:.1f} ms, "
                        f"end_to_end {(_t_plan - _t_total) * 1e3:.1f} ms"
                    )
                    if not result.success:
                        self.lane_plan.call(
                            self._fill_result_insight,
                            result_msg.result,
                            planner,
                            goal,
                            result,
                            start_state,
                            fill_arrays=False,
                            prio=PRIO_PLAN,
                        )
                        self.get_logger().error(
                            f"Planning failed in execute path: {result.message}\n"
                            + self._format_collision_feedback(
                                result_msg.result.start_state_collisions,
                                result_msg.result.end_state_collisions,
                            )
                        )
                        result_msg.result.success = False
                        result_msg.result.message = f"Planning failed: {result.message}"
                        goal_handle.abort()
                        return result_msg
                # A hit CONSUMED its entry; a miss re-solved this target, which
                # supersedes its plan-time entry. Either way the arm is no
                # longer where that plan's start_pose says — keep every
                # OTHER segment's entry for the rest of the chain.
                self.lane_plan.call(
                    self._drop_pending_plan, start_state, goal, entry=reuse, prio=PRIO_PLAN
                )
            else:
                # Reactive: (re)set the goal on the solver before servoing.
                result = self.lane_reactive.call(self._refresh_and_plan_reactive, planner, start_state, goal, config, prio=PRIO_REACTIVE)
                _t_world = _t_setup
                self.get_logger().info(f"Planning with {planner.get_planner_name()}")
                _t_plan = time.monotonic()
                self.get_logger().info(
                    f"[plan-perf] {planner.get_planner_name()} (execute path): "
                    f"node_setup(start-state+config+planner-setup+lock) {(_t_setup - _t_total) * 1e3:.1f} ms, "
                    f"world_refresh(ESDF+world-push) {(_t_world - _t_setup) * 1e3:.1f} ms, "
                    f"planner_plan(=curobo_solve+finalize) {(_t_plan - _t_world) * 1e3:.1f} ms, "
                    f"end_to_end {(_t_plan - _t_total) * 1e3:.1f} ms"
                )
                if not result.success:
                    self.lane_reactive.call(
                        self._fill_result_insight,
                        result_msg.result,
                        planner,
                        goal,
                        result,
                        start_state,
                        fill_arrays=False,
                        prio=PRIO_REACTIVE,
                    )
                    self.get_logger().error(
                        f"Planning failed in execute path: {result.message}\n"
                        + self._format_collision_feedback(
                            result_msg.result.start_state_collisions,
                            result_msg.result.end_state_collisions,
                        )
                    )
                    result_msg.result.success = False
                    result_msg.result.message = f"Planning failed: {result.message}"
                    goal_handle.abort()
                    return result_msg

            if result is not None:
                # Winner-per-waypoint insight + stats for the fresh plan, plus
                # the trajectory that is about to be driven. This must not be
                # fill_arrays=False: the re-solve can move the endpoints from
                # what the plan-time solve produced, and the executor's
                # chain-continuity check needs those DRIVEN endpoints. A result
                # with no waypoints reads as "nothing to compare" there, and
                # the chain silently drives on into segments anchored to the
                # old end state - which is the bug this trajectory reporting
                # exists to stop.
                self.lane_plan.call(
                    self._fill_result_insight,
                    result_msg.result,
                    planner,
                    goal,
                    result,
                    start_state,
                    fill_arrays=True,
                    prio=PRIO_PLAN,
                )
            else:
                # Reused a matching cached preview: report the planner's own
                # (preview-plan) insight fields and the replayed trajectory.
                # Same reasoning: a cache hit replays exactly the cached plan,
                # so the executor must be able to verify where it is driving,
                # or an off-spec reuse is indistinguishable from a clean one.
                meta = self._result_meta_from_planner(planner)
                self._fill_insight_fields(
                    result_msg.result,
                    goal,
                    meta,
                    planner,
                    success=True,
                    message="Execution completed",
                    start_state=start_state,
                )
                self.lane_plan.call(
                    self._fill_reused_trajectory,
                    result_msg.result, planner, reuse["trajectory"], prio=PRIO_PLAN,
                )

            self.get_logger().info(f"Executing with {planner.get_planner_name()}")
            if result is not None:
                # Freshly solved (re-solve or reactive): plan() already staged
                # it into robot_context's command buffer.
                success = planner.execute(self.robot_context, goal_handle)
            else:
                # Replayed cache entry: the planner's slot and the command
                # buffer only hold the LAST segment planned, which for a
                # mid-chain hit is NOT this segment — stage it then stream.
                success = planner.replay(
                    reuse["trajectory"], self.robot_context, goal_handle
                )

            # Cancel takes precedence over the planner's return value.
            if goal_handle.is_cancel_requested:
                result_msg.result.success = False
                result_msg.result.message = "Execution canceled"
                goal_handle.canceled()
                return result_msg

            result_msg.result.success = bool(success)
            result_msg.result.message = (
                "Execution completed" if success else "Execution failed"
            )
            if success:
                goal_handle.succeed()
            else:
                goal_handle.abort()
            return result_msg

        except Exception as e:
            self.get_logger().error(f"Execution error: {e}")
            self.get_logger().error(traceback.format_exc())
            result_msg.result = TrajectoryResult()
            result_msg.result.success = False
            result_msg.result.message = f"Error: {e}"
            result_msg.result.stats = self._build_planning_stats(goal, {}, problems=1)
            if goal_handle.is_active:
                goal_handle.abort()
            return result_msg

    def set_planner_callback(
        self, request: SetPlanner.Request, response: SetPlanner.Response
    ):
        try:
            previous = self.planner_manager.get_current_planner()
            previous_name = previous.get_planner_name() if previous else "None"

            with self._goal_lock:
                if self._goal_active:
                    response.success = False
                    response.message = (
                        "Cannot switch planner while an execution goal is active "
                        "— cancel it first."
                    )
                    response.previous_planner = previous_name
                    response.current_planner = previous_name
                    self.get_logger().error(response.message)
                    return response

            key, error = PlannerFactory.switch_planner(
                request.planner_type, self.planner_manager
            )
            if error:
                response.success = False
                response.message = error
                response.previous_planner = previous_name
                response.current_planner = previous_name
                self.get_logger().error(error)
                return response

            planner = self.planner_manager.get_current_planner()
            self._setup_planner(planner)

            response.success = True
            response.message = f"Successfully switched to {planner.get_planner_name()}"
            response.previous_planner = previous_name
            response.current_planner = planner.get_planner_name()
            self.get_logger().info(
                f"Planner switch: {previous_name} -> {planner.get_planner_name()}"
            )

        except Exception as e:
            response.success = False
            response.message = f"Failed to switch planner: {e}"
            response.previous_planner = (
                previous_name if "previous_name" in locals() else "Unknown"
            )
            response.current_planner = response.previous_planner
            self.get_logger().error(response.message)
            self.get_logger().error(traceback.format_exc())

        return response

    def mpc_goal_callback(self, msg):
        """Receive live reactive-goal updates from a topic; stored as raw data.

        This is the ROS mapping of cuRobo's continuous `update_goal_tool_poses`:
        the active reactive controller retargets on its next loop iteration.
        """
        planner = self.planner_manager.get_current_planner()
        if isinstance(planner, ReactiveController):
            planner.set_live_goal(
                [
                    msg.position.x,
                    msg.position.y,
                    msg.position.z,
                    msg.orientation.w,
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                ]
            )
            self.get_logger().debug(
                f"Reactive goal updated from topic: "
                f"[{msg.position.x:.3f}, {msg.position.y:.3f}, {msg.position.z:.3f}]"
            )
        else:
            name = planner.get_planner_name() if planner is not None else "none"
            self.get_logger().warn(
                f"Received reactive goal but current planner is {name} - goal ignored",
                throttle_duration_sec=5.0,
            )

    def get_planners_callback(
        self, request: GetPlanners.Request, response: GetPlanners.Response
    ):
        current_type = self.planner_manager.get_current_planner_type()
        catalog = PlannerFactory.get_catalog()

        response.planner_names = [name for _, _, name in catalog]
        response.planner_ids = [int(eid) for _, eid, _ in catalog]

        response.current_planner_name = "Unknown"
        response.current_planner_id = 255
        for key, eid, name in catalog:
            if key == current_type:
                response.current_planner_name = name
                response.current_planner_id = int(eid)
                break

        response.success = True
        self.get_logger().info(
            f"GetPlanners: {len(catalog)} planners, current={response.current_planner_name}"
        )
        return response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_goal_request(self, planner, request):
        """Validate a goal request against the active planner's rules.

        Enforced here (shared by the generate_trajectory srv and the execute
        action) so invalid requests never reach MotionGen:

        - All planners read `goalsets` (one Goalset per segment). The joint-space
          planner's target lives in `Goalset.target_joint_positions`; the
          Cartesian planners read `Goalset.poses`.
        - Every `goalsets[i]` must hold >= 1 pose OR a non-empty
          `target_joint_positions` (a zero-pose, zero-joint set would break
          winner/segment index alignment).
        - Reactive controllers (MPC / retarget) track a single pose: exactly one
          set of one pose.
        - Open-loop planners cap per-set candidates at `max_goalset`: the goalset
          buffer is sized at solver build, and larger sets 100% abort in cuRobo.

        Returns (ok, reason).
        """
        planner_name = planner.get_planner_name()
        goalsets = list(getattr(request, "goalsets", None) or [])
        sizes = [len(getattr(g, "poses", [])) for g in goalsets]
        joint_sizes = [
            len(getattr(g, "target_joint_positions", []) or []) for g in goalsets
        ]

        if not goalsets:
            return False, (
                f"{planner_name}: request must provide `goalsets` "
                f"(one Goalset entry per segment)"
            )

        if any(n == 0 and joint_sizes[i] == 0 for i, n in enumerate(sizes)):
            return False, (
                f"{planner_name}: every goalset entry must hold >= 1 pose or a "
                f"non-empty target_joint_positions (got pose sizes {sizes}, "
                f"joint sizes {joint_sizes})"
            )

        if not planner.is_open_loop():
            if len(goalsets) != 1 or sizes != [1]:
                return False, (
                    f"{planner_name}: requires exactly one goalset entry with a "
                    f"single pose, got {len(goalsets)} set(s) of sizes {sizes}"
                )
            return True, None

        max_goalset = (
            self.get_parameter("max_goalset").get_parameter_value().integer_value
        )
        if any(n > max_goalset for n in sizes):
            return False, (
                f"{planner_name}: goalset of {max(sizes)} poses exceeds "
                f"max_goalset={max_goalset} (per-segment cap; the goalset buffer "
                f"is sized at solver build)"
            )
        return True, None

    def _setup_planner(self, planner):
        # Held under gpu_lock, same invariant as every other graph-capturing path
        # here (refresh_perception_world, rebuild_solvers_for_cache_change, both
        # planner.plan() call sites). This one was MISSING it, and that is not
        # theoretical: switching MPC -> Classic mid-session with the cameras live
        # crashed the node (2026-08-08). _warmup_classic() captures the seed-IK
        # CUDA graph, and torch.cuda.graph() defaults to capture_error_mode
        # "global" -- while a capture is live, ANY CUDA op from ANY thread of the
        # process fails with cudaErrorStreamCaptureUnsupported. The depth callback
        # honours its half of the contract (non-blocking acquire then skip the
        # frame, camera_depth_map_strategy.py:157), but a lock nobody holds is
        # always free, so it ran mapper.integrate() straight into the capture.
        # The capture aborted while leaving the stream in
        # cudaStreamCaptureStatusActive, which poisoned the CUDA context for the
        # whole process: 106 consecutive CUDA failures, no recovery short of a
        # node restart.
        #
        # gpu_lock is an RLock and is the OUTERMOST lock in the documented order
        # (robot_context.py:17-37), so acquiring it here is deadlock-free even
        # when a caller already holds it. Cost: the depth callback drops its
        # frames during the ~6 s warmup -- exactly the designed behaviour.
        with self.gpu_lock:
            # Enforce a single live CUDA graph before this planner runs: if it
            # isn't the current graph owner, release every other solver's captured
            # graph so this one re-captures (on its next call) as the sole live
            # graph. Cheap no-op when the owner is unchanged (repeated plans reuse
            # the graph).
            self._ensure_exclusive_graph(planner)

            # Open-loop planners share one MotionPlanner instance (class-level).
            if isinstance(planner, SinglePlanner):
                if self.motion_planner is None:
                    self.get_logger().info("On-demand warmup: open-loop planner")
                    self._warmup_classic()
                planner.set_motion_gen(self.motion_planner)

            # Reactive controllers build their own solver from the shared context.
            # ensure_solver() is idempotent (no-op once built), so just make sure a
            # ground plane exists and let the active controller build/reuse its solver.
            elif isinstance(planner, ReactiveController):
                if planner.solver is None:
                    self.get_logger().info("On-demand warmup: reactive controller")
                    self._ensure_ground_plane()
                planner.ensure_solver()

    # ------------------------------------------------------------------
    # CUDA graph exclusivity (only the active solver holds a live graph)
    # ------------------------------------------------------------------

    def _graph_owner_key(self, planner):
        """Stable key identifying which captured graph a planner would use.

        All open-loop planners share one MotionPlanner (one graph) -> 'motion'.
        Each reactive controller owns its own solver/graph -> 'reactive:<name>'.
        """
        if isinstance(planner, SinglePlanner):
            return "motion"
        if isinstance(planner, ReactiveController):
            return f"reactive:{planner.get_planner_name()}"
        return None

    def _ensure_exclusive_graph(self, planner):
        """Guarantee the given planner is the sole owner of a live CUDA graph."""
        self._ensure_exclusive_graph_key(self._graph_owner_key(planner))

    def _ensure_exclusive_graph_key(self, owner_key):
        """Make owner_key the sole live-graph owner, releasing all others.

        No-op if graphs are disabled or the owner is unchanged. Otherwise release
        every captured graph (the incoming solver re-captures cleanly, alone, on
        its next call) and record the new owner. See _active_graph_owner.
        """
        if not self.config_wrapper_motion.use_cuda_graph:
            return
        if owner_key is None or owner_key == self._active_graph_owner:
            return
        self.get_logger().info(
            f"CUDA graph owner {self._active_graph_owner} -> {owner_key}: releasing "
            f"other captured graphs so only the active solver holds one"
        )
        self._release_all_solver_cuda_graphs()
        self._active_graph_owner = owner_key

    def _release_all_solver_cuda_graphs(self):
        """Free every solver's captured CUDA graph (frees the graph + its pool).

        reset_cuda_graph() -> GraphExecutor.reset() calls CUDAGraph.reset(), which
        releases the graph's private memory pool; the solver itself and its shared
        collision world are untouched (it re-captures on next use). Held under the
        GPU lock so it can't overlap a concurrent capture / depth integrate.
        """
        with self.gpu_lock:
            if self.motion_planner is not None:
                for name in ("trajopt_solver", "ik_solver"):
                    self._safe_reset_graph(getattr(self.motion_planner, name, None))
            self._safe_reset_graph(self.mpc)
            self._safe_reset_graph(self.retargeter)
            # Every solver now holds no graph — its next call captures.
            self._set_graph_capture_pending()

    def _set_graph_capture_pending(self):
        """Mark that the next reactive solver call may capture a CUDA graph."""
        with self._graph_capture_pending_lock:
            self._graph_capture_pending = True

    def take_graph_capture_pending(self) -> bool:
        """Atomically take-and-clear: True if the next reactive step may capture.

        Called by ReactiveController._step_guard to decide whether its next
        step() needs gpu_lock — capture is process-global (see
        _graph_capture_pending's docstring), replay is not. Take-and-clear
        mirrors _take_live_goal(): exactly one caller sees True per release.

        Open-loop plans no longer use this — they always hold gpu_lock when
        use_cuda_graph is on (see _plan_lock docstring, 2026-09-11).
        """
        with self._graph_capture_pending_lock:
            pending = self._graph_capture_pending
            self._graph_capture_pending = False
            return pending

    def _plan_lock(self):
        """Exclusive barrier around a plan only when asked (default) and graphs are on.

        Plans already run inside a lane job (shared). With gpu_exclusive_plans=False a plan
        only goes exclusive in the places that really capture (see S7).
        """
        if self.config_wrapper_motion.use_cuda_graph and \
                self.get_parameter("gpu_exclusive_plans").get_parameter_value().bool_value:
            return self.gpu.exclusive()
        return contextlib.nullcontext()

    def torch_sync_enabled(self) -> bool:
        """``torch_sync`` param: sync the CPU to the GPU after the guarded calls.

        Off by default — a per-frame torch.cuda.synchronize() in the depth
        callback blocks the executor thread on the GPU every frame and starves
        the viz timers (see the 'torch_sync' parameter declaration). Set True
        only when deterministic GPU/CPU ordering is needed (race debugging).
        """
        return bool(self.get_parameter("torch_sync").get_parameter_value().bool_value)

    def _on_gpu_poisoned(self, lane, exc):
        """A lane hit a fatal CUDA error: the context is unusable. Fail loudly, then exit."""
        try:
            self.get_logger().fatal(f"GPU lane '{lane.name}' poisoned: {exc!r}")
            exit_flag = self.get_parameter("exit_on_gpu_fault").get_parameter_value().bool_value
        except Exception:
            exit_flag = True
        if exit_flag:
            threading.Timer(0.5, lambda: os._exit(70)).start()

    def _safe_reset_graph(self, solver):
        """Release a solver's captured CUDA graph(s); never raise into callers.

        Uses reset_cuda_graph(), NOT destroy(): SolverCore.destroy() shares its
        body with reset_cuda_graph() (resets optimizer/metrics_rollout/additional
        rollouts) but drops both of reset_cuda_graph()'s guards (use_cuda_graph,
        _task_initialized). Calling it unconditionally on metrics_rollout was
        tried and found to orphan its constraint tensors -- con_scene_collision
        froze at a huge constant instead of tracking live geometry (see debug
        2026-07-25) -- while cost_tool_pose_pos kept updating from the same
        get_current_metrics() call, proving the rollout's buffers were left in a
        stale/half-reset state rather than actually corrupted geometry.

        reset_cuda_graph() alone still has the original gap: IKSolver.reset_cuda_graph()
        never reaches the nested SeedIKSolver's private Levenberg-Marquardt
        GraphExecutor. That gap is what caused the original cuGraphLaunch segfault
        on Classic after MPC. So instead of destroy(), reach seed_ik_solver
        explicitly and narrowly: SeedIKSolver.destroy() only resets its own two
        GraphExecutors and touches no rollout/collision state, so it's safe to
        call unconditionally. Also recurse into a nested ik_solver (MpcSolver owns
        one for goal-state IK; MpcSolver.reset_cuda_graph() only forwards to
        self.core and never releases it, so its seed-IK graphs were never being
        freed at all).
        """
        if solver is None:
            return
        try:
            if hasattr(solver, "reset_cuda_graph"):
                solver.reset_cuda_graph()
            seed_ik_solver = getattr(solver, "seed_ik_solver", None)
            if seed_ik_solver is not None:
                seed_ik_solver.destroy()
            nested_ik_solver = getattr(solver, "ik_solver", None)
            if nested_ik_solver is not None and nested_ik_solver is not solver:
                self._safe_reset_graph(nested_ik_solver)
        except Exception as e:
            self.get_logger().warn(
                f"CUDA graph release failed on {type(solver).__name__}: {e}"
            )

    def _get_planner_config(self, planner) -> dict:
        if isinstance(planner, SinglePlanner):
            # plan_pose only honors max_attempts in v2 (timeout / time_dilation
            # are not solver args anymore — speed lives in the robot YAML
            # cspace). The launch forwards `max_attempts` ("Planning retries per
            # request"); the node declares it (default 1, so existing launches
            # are unchanged) and we read it here so requesting N attempts
            # actually reaches plan_pose.
            return {
                "max_attempts": self.get_parameter("max_attempts")
                .get_parameter_value()
                .integer_value,
                "reset_seed_per_plan": self.get_parameter("reset_seed_per_plan")
                .get_parameter_value()
                .bool_value,
            }
        if isinstance(planner, ReactiveController):
            return {
                "convergence_threshold": self.get_parameter(
                    "convergence_threshold"
                ).value,
                "max_iterations": self.get_parameter("max_mpc_iterations").value,
            }
        return {}

    # ------------------------------------------------------------------
    # Start state + pending-plan (preview cache) helpers
    # ------------------------------------------------------------------

    def _resolve_start_state(self, src):
        """Return (start_joint_list, start_state) from a request/goal start_pose.

        Falls back to the robot's current joint pose when start_pose is empty.
        Shared by the generate_trajectory service and the execute action.
        """
        start_pose = getattr(src, "start_pose", None)
        current = list(self.robot_context.get_joint_pose())
        # The start pose may now name its joints (sensor_msgs/JointState.name).
        # Named poses are re-ordered into the model's active cspace order HERE,
        # so a client needs no out-of-band knowledge of that order. A nameless
        # pose keeps its historical meaning: already in active-DOF order.
        model_names = self.robot_context.description.joint_names
        start_joint_pose = resolve_start_pose(start_pose, model_names, current)
            # A short start pose (e.g. the MoveIt arm group sends only its 7
            # DOF) pads the trailing DOF (wrist-mounted gripper) from the
        # A short start pose (e.g. the MoveIt arm group sends only its 7
        # DOF) pads the trailing DOF (wrist-mounted gripper) from the
        # robot's current pose so the planner always gets a full-DOF
        # start state. Padding happens after the re-order, so the padding
        # lands on the DOF the caller did not name.
        if len(start_joint_pose) < len(current):
            padded = len(current) - len(start_joint_pose)
            start_joint_pose = start_joint_pose + current[len(start_joint_pose) :]
            self.get_logger().info(
                f"Using start position from request (padded {padded} DOF from "
                f"current): {[f'{x:.3f}' for x in start_joint_pose]}"
            )
        else:
            self.get_logger().info(
                f"Using start position from request: "
                f"{[f'{x:.3f}' for x in start_joint_pose]}"
            )
        start_state = JointState.from_position(
            torch.tensor(
                [start_joint_pose],
                dtype=self.tensor_args.dtype,
                device=self.tensor_args.device,
            )
        )
        return start_joint_pose, start_state

    def _check_execution_start_drift(self, goal):
        """Is the arm still where this goal says it starts?

        Only meaningful on the re-solve path. On a cache hit the signature
        already pins the start state, so a hit means the arm has not moved and
        there is nothing to compare. On a miss the request's ``start_pose`` is
        the plan-time pose, and ``_resolve_start_state`` prefers it over the
        arm's real one ("Using start position from request") - so a stale pose
        silently anchors the new solve to a configuration the arm is not in.
        cuRobo then finds a different IK branch, and the trajectory that gets
        driven is not the trajectory that was planned and displayed.

        Two outcomes, because the two situations mean opposite things:

          * small drift - ordinary servo tracking error. The arm really is
            slightly off, re-solving from where it actually is is the correct
            response, and the new plan IS the right plan. Warn, proceed.
          * large drift - the arm is somewhere the chain never put it (an
            earlier segment failed, was skipped, or the arm was moved
            externally). Re-solving produces a trajectory unrelated to the
            validated one. Refuse, and leave the arm where it is so the caller
            can recover deliberately rather than inherit a mystery motion.

        Returns ``(ok, message)``; ``message`` is the log/error text.
        """
        start_pose = getattr(goal, "start_pose", None)
        if start_pose is None or len(start_pose.position) == 0:
            # No start_pose in the goal: the planner is already anchored to the
            # arm's live pose, so there is nothing stale to catch.
            return True, ""
        try:
            requested = [float(x) for x in start_pose.position]
            current = [float(x) for x in self.robot_context.get_joint_pose()]
        except (TypeError, RuntimeError, ValueError) as exc:
            # A tensor that will not convert is not evidence of drift. Do not
            # turn a type problem into a refused motion.
            self.get_logger().warn(
                f"Could not compare requested start pose against the arm's "
                f"actual pose ({exc}); trusting the request."
            )
            return True, ""

        # A short start pose (e.g. a 7-DOF arm group) is padded from the live
        # pose in _resolve_start_state, so pad the same way before comparing.
        n = min(len(requested), len(current))
        if n == 0:
            return True, ""
        requested, current = requested[:n], current[:n]
        deltas = [abs(a - b) for a, b in zip(requested, current)]
        worst = max(range(n), key=deltas.__getitem__)
        drift = deltas[worst]
        fmt = lambda v: [f"{x:.3f}" for x in v]  # noqa: E731
        where = (
            f"joint {worst}: requested {requested[worst]:.3f} rad, "
            f"arm is at {current[worst]:.3f} rad ({drift:.3f} rad off)"
        )

        warn = (
            self.get_parameter("execution_start_warn_drift")
            .get_parameter_value()
            .double_value
        )
        tol = (
            self.get_parameter("execution_start_tolerance")
            .get_parameter_value()
            .double_value
        )
        if drift > tol:
            return False, (
                f"Refusing to re-solve for execution: the arm is {drift:.3f} rad "
                f"from the start state this goal was planned from, over the "
                f"{tol:.3f} rad execution_start_tolerance. The cached plan "
                f"missed (scene or cache state changed) and re-solving from the "
                f"planned start pose would drive a trajectory that is not the "
                f"one that was planned. Requested {fmt(requested)}, actual "
                f"{fmt(current)}; worst {where}."
            )
        if drift > warn:
            self.get_logger().warn(
                f"Execute re-solve with {drift:.3f} rad of start-pose drift "
                f"(within tolerance {tol:.3f}, warn above {warn:.3f}). "
                f"Re-solving from the arm's actual pose; worst {where}."
            )
        return True, ""

    def _prune_plan_cache(self):
        """Drop expired entries and bound the pool's size (oldest first).

        Called on store and on lookup, so an expired entry is never handed
        out and a long multi-task run cannot accumulate entries between TTL
        prunes. Bounded by ``trajectory_cache_size`` (default 64), far more
        than any single chain's segment count.
        """
        ttl = (
            self.get_parameter("trajectory_cache_ttl")
            .get_parameter_value()
            .double_value
        )
        cap = (
            self.get_parameter("trajectory_cache_size")
            .get_parameter_value()
            .integer_value
        )
        now = time.monotonic()
        alive = [e for e in self._pending_plan if (now - e["stamp"]) <= ttl]
        if len(alive) > cap:
            alive = alive[-cap:]
        self._pending_plan = alive

    def _store_pending_plan(self, start_state, req, traj=None):
        """Cache a just-planned open-loop trajectory for execute-time reuse.

        Multi-entry, keyed by target signature, so EVERY segment of a chain
        is replayable. The single-slot predecessor only ever knew the LAST
        segment planned, so a mid-chain execute always missed and re-solved —
        and a re-solve of a multi-solution pose goal is free to pick the
        OTHER IK branch, which is exactly the divergence the chain-continuity
        check exists to catch. Each entry keeps the trajectory itself: the
        planner's ``planned_trajectory`` slot is overwritten by the next
        segment's solve before that segment executes.
        """
        if traj is None:
            planner = self.planner_manager.get_current_planner()
            traj = getattr(planner, "planned_trajectory", None)
        self._prune_plan_cache()
        self._pending_plan.append(
            {
                "planner": self.planner_manager.get_current_planner_type(),
                "signature": self._target_signature(start_state, req),
                "stamp": time.monotonic(),
                "trajectory": traj,
            }
        )

    def _pending_plan_matches(self, start_state, req):
        """The fresh cached plan entry matching this target, or None.

        Returns the whole entry (with its stored trajectory), not a bool:
        the execute action replays THAT trajectory. The signature includes
        the start state, so a match also proves the arm has not drifted from
        the plan.
        """
        self._prune_plan_cache()
        sig = self._target_signature(start_state, req)
        planner_key = self.planner_manager.get_current_planner_type()
        for entry in self._pending_plan:
            if entry["planner"] != planner_key:
                continue
            if self._signatures_match(entry["signature"], sig):
                return entry
        return None

    def _drop_pending_plan(self, start_state, req, entry=None):
        """Drop plan-cache entries that execution consumed or superseded.

        A replay CONSUMES its entry (identity drop) so a later identical
        request cannot replay a trajectory the arm has already driven. A
        re-solve supersedes the plan-time entry for the same target
        (signature drop): the arm is now on the re-solved path, and replaying
        the old plan anchored to a stale ``start_pose`` would strand the
        chain somewhere the plan never validated. Either way only THIS
        segment's entries go; the rest of the chain's stay replayable.
        """
        if entry:
            self._pending_plan = [e for e in self._pending_plan if e is not entry]
            return
        sig = self._target_signature(start_state, req)
        planner_key = self.planner_manager.get_current_planner_type()
        self._pending_plan = [
            e
            for e in self._pending_plan
            if e["planner"] != planner_key
            or not self._signatures_match(e["signature"], sig)
        ]

    @staticmethod
    def _pose_tuple(p):
        return (
            p.position.x,
            p.position.y,
            p.position.z,
            p.orientation.w,
            p.orientation.x,
            p.orientation.y,
            p.orientation.z,
        )

    def _target_signature(self, start_state, req) -> dict:
        start = [float(x) for x in start_state.position[0].cpu().tolist()]
        goalsets = [
            [self._pose_tuple(p) for p in g.poses]
            for g in (getattr(req, "goalsets", None) or [])
        ]
        # Joint targets are per-segment (Goalset.target_joint_positions); each
        # entry stays aligned with its goalset ([] for Cartesian segments).
        # The NAMES are part of the key: a name-keyed goal and a nameless one
        # that happen to hold the same numbers must not share a cache entry,
        # because the node resolves them differently.
        joints = [
            [float(x) for x in (getattr(g, "target_joint_positions", None) or [])]
            for g in (getattr(req, "goalsets", None) or [])
        ]
        joint_names = [
            [str(n) for n in (getattr(g, "target_joint_names", None) or [])]
            for g in (getattr(req, "goalsets", None) or [])
        ]
        # Planning options shape the plan (waypoint_tolerance gates waypoint
        # status, log_considered_trajectories gates the considered rows), so a
        # cache reuse must not cross an options boundary.
        opts = getattr(req, "options", None)
        return {
            "start": start,
            "goalsets": goalsets,
            "target_joints": joints,
            "target_joint_names": joint_names,
            "options": (
                (
                    float(getattr(opts, "waypoint_tolerance", 0.0)),
                    bool(getattr(opts, "log_considered_trajectories", False)),
                )
                if opts is not None
                else (0.0, False)
            ),
        }

    @staticmethod
    def _poses_match(a, b, pos_tol, ori_tol) -> bool:
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        if any(abs(a[i] - b[i]) > pos_tol for i in range(3)):
            return False
        dot = sum(a[3 + i] * b[3 + i] for i in range(4))
        return (1.0 - abs(dot)) < ori_tol

    def _signatures_match(
        self, a, b, pos_tol=1e-3, ori_tol=1e-2, joint_tol=1e-3
    ) -> bool:
        # Options must be identical (exact tuple equality); a signature stored
        # before options were part of the signature has no key -> None != a
        # tuple, so that cache never replays across the migration boundary.
        if a.get("options") != b.get("options"):
            return False
        if len(a["start"]) != len(b["start"]) or any(
            abs(x - y) > joint_tol for x, y in zip(a["start"], b["start"])
        ):
            return False
        aj, bj = a["target_joints"], b["target_joints"]
        if len(aj) != len(bj):
            return False
        for aj_seg, bj_seg in zip(aj, bj):
            if len(aj_seg) != len(bj_seg):
                return False
            if any(abs(x - y) > joint_tol for x, y in zip(aj_seg, bj_seg)):
                return False
        ag, bg = a["goalsets"], b["goalsets"]
        if len(ag) != len(bg):
            return False
        for gset_a, gset_b in zip(ag, bg):
            if len(gset_a) != len(gset_b):
                return False
            if any(
                not self._poses_match(x, y, pos_tol, ori_tol)
                for x, y in zip(gset_a, gset_b)
            ):
                return False
        return True

    def clear_trajectory_callback(self, request, response):
        """Discard every cached (preview) trajectory on user request."""
        had = bool(self._pending_plan)
        self._pending_plan = []
        response.success = True
        response.message = (
            "Cached trajectory cleared" if had else "No cached trajectory"
        )
        self.get_logger().info(response.message)
        return response

    def clear_voxel_map_callback(self, request, response):
        """Clear ONLY the dynamic (depth-derived) voxel channel.

        Analytic collision objects (boxes, spheres, capsules, cylinders, meshes
        added via ``add_object``) are stored outside the TSDF in the Scene's
        cuboid/mesh buffers, so they always stay. The mapper's dynamic blocks
        are cleared in place, the perception ESDF is recomputed, and the world
        is pushed to all solvers — all under ``gpu_lock`` — so the next plan
        and every servoing step already sees the cleared map.
        """
        obs = self.config_wrapper_motion.obstacle_manager
        with self.gpu_lock:
            try:
                n_blocks = obs.clear_dynamic_voxels(self)
            except Exception as e:
                response.success = False
                response.message = f"clear_voxel_map failed: {e}"
                self.get_logger().error(response.message)
                return response
            refreshed = obs.refresh_esdf()
            if refreshed:
                self.update_all_solvers_world(obs.get_scene())
        response.success = True
        response.message = f"Cleared dynamic voxel channel ({n_blocks} block(s))" + (
            "" if refreshed else "; WARNING: perception ESDF not refreshed"
        )
        self.get_logger().info(response.message)
        return response

    def goal_callback(self, goal):
        """Admit at most one execute_trajectory goal at a time.

        Two concurrent execute() loops (open-loop or reactive) would both
        stream commands to the robot. The flag is set here (accept-time) so
        there is no race with execute_callback starting; it is cleared in
        execute_callback's finally, however the goal ends (succeed / abort /
        cancel / exception).
        """
        with self._goal_lock:
            if self._goal_active:
                self.get_logger().warn(
                    "Rejecting execution goal: another goal is already active "
                    "- cancel it first."
                )
                return rclpy.action.GoalResponse.REJECT
            self._goal_active = True
        self.get_logger().info("Received execution goal")
        return rclpy.action.GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.robot_context.stop_robot()
        planner = self.planner_manager.get_current_planner()
        if hasattr(planner, "cancel"):
            planner.cancel()
        self.get_logger().info("Goal cancelled")
        return rclpy.action.CancelResponse.ACCEPT


def main(args=None):
    # Dump a Python traceback on a fatal signal (SIGSEGV/SIGABRT). A GPU illegal
    # access surfaces as a native segfault (process exit -11) with no Python
    # error otherwise — faulthandler shows which CuRobo call was executing.
    # Redundant with PYTHONFAULTHANDLER=1 set in the launch file; harmless if run
    # standalone without that env.
    import faulthandler

    faulthandler.enable()

    rclpy.init(args=args)
    node = UnifiedPlannerNode()

    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    try:
        node.get_logger().info("Unified planner running, shut down with CTRL-C")
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down.\n")

    if getattr(node, "robot_segmentation", None) is not None:
        node.robot_segmentation.destroy()
    # Stop the depth integration workers BEFORE tearing down the node so the
    # daemon threads don't fall out mid-integrate (camera_depth_map_strategy.py).
    cam_mgr = getattr(
        getattr(node, "config_wrapper_motion", None), "camera_system_manager", None
    )
    camera_context = getattr(cam_mgr, "camera_context", None)
    if camera_context is not None:
        for strategy in camera_context.cameras.values():
            destroy = getattr(strategy, "destroy", None)
            if destroy is not None:
                destroy()
    # Same for the laser workers: stop each projection thread and the batched
    # lidar-integration thread before the node is torn down
    # (laser_pointcloud_strategy.py / laser_context.py).
    laser_mgr = getattr(
        getattr(node, "config_wrapper_motion", None), "laser_system_manager", None
    )
    laser_context = getattr(laser_mgr, "laser_context", None)
    if laser_context is not None:
        for strategy in laser_context.lasers.values():
            destroy = getattr(strategy, "destroy", None)
            if destroy is not None:
                destroy()
        context_destroy = getattr(laser_context, "destroy", None)
        if context_destroy is not None:
            context_destroy()
    node.gpu.shutdown()
    node.destroy_node()
    # rclpy's own SIGINT handler already shuts the context down, so calling
    # shutdown() unconditionally raises "rcl_shutdown already called" and the
    # process exits 1 on every clean Ctrl-C — which made a normal stop
    # indistinguishable from a crash in the test suites' exit-code checks.
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()

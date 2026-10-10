#!/usr/bin/env python3
"""
IK services for the unified planner node (v2).

Provides a lazy-initialized IK solver that shares the obstacle manager
and robot config of the trajectory planner.

Services exposed (prefixed with the node name):
  /<node>/warmup_ik  (WarmupIK) - init IK solver with given batch size (default 1)
  /<node>/ik         (Ik)       - single pose → joint state
  /<node>/ik_batch   (IkBatch)  - N poses → joint states

v2 notes:
- IKSolver / IKSolverConfig → InverseKinematics / InverseKinematicsCfg.
- CollisionCheckerType is gone; collision is part of InverseKinematicsCfg.create.
- load_from_robot_config(...) replaced by Cfg.create(robot=<yaml_path>, scene_model=<Scene>, ...).
"""

import contextlib
import threading

import torch
import std_msgs.msg
from sensor_msgs.msg import JointState

from curobo.types import Pose as CuroboPose, GoalToolPose, JointState as CuRoboJS
from curobo.scene import Scene
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg

from curobov2_ros_interfaces.srv import Ik, IkBatch, WarmupIK
from curobov2_ros.core.gpu_lanes import PRIO_KIN


class IKServices:
    """
    Manages the IK solver and its ROS services on an existing node.

    Depends on:
    - config_wrapper.robot_config_file   (YAML path — v2 factories accept it directly)
    - config_wrapper.obstacle_manager    (Scene, shared with MotionPlanner)
    - config_wrapper.collision_cache     (single v2 cache integer)

    The solver is created only when warmup_ik is called.
    Obstacle updates are propagated via update_world().
    """

    def __init__(self, node, config_wrapper):
        self._node = node
        self._config = config_wrapper

        self._ik_solver: InverseKinematics | None = None
        self._ik_batch_size: int = 0  # 0 = not yet warmed up
        self._ik_num_seeds: int = 0  # seeds of the current solver (0 = unset)

        # Serializes solves on the single shared solver: /ik, /ik_batch and the
        # reachability service (/generate_rm) can all arrive from different
        # executor threads — a second solve_pose() (or a reinit for a different
        # batch size) racing the first would corrupt the solver's state.
        self._solve_lock = threading.Lock()

        # Device / dtype resolved from config_wrapper (set by RobotModelManager).
        self._device = getattr(config_wrapper, "_device", torch.device("cuda"))
        self._dtype = getattr(config_wrapper, "_ops_dtype", torch.float32)

        name = node.get_name()
        node.create_service(WarmupIK, f"{name}/warmup_ik", self._warmup_ik_callback)
        node.create_service(Ik, f"{name}/ik", self._ik_callback)
        node.create_service(IkBatch, f"{name}/ik_batch", self._ik_batch_callback)

        node.get_logger().info(
            "IKServices registered (solver allocated by the startup warmup, or "
            "on demand via /warmup_ik when warmup_ik:=false)"
        )

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def warmup(self, batch_size: int, num_seeds: int | None = None) -> str:
        """Build (or rebuild) the IK solver for the given batch size/seeds.

        Returns the message the ``/warmup_ik`` service reports. Shared with the
        node's startup default warmup (``unified_planner_node``) so both paths
        take the same ``gpu_lock`` discipline around the CUDA work. Re-invoking
        with a different batch size re-initializes the solver and resets the
        batch the client asked for.
        """
        batch_size = max(1, int(batch_size))
        # Solver construction + warmup run CUDA kernels (solve for a batch
        # of random configs) — must not race a CUDA graph capture.
        with self._gpu_guard():
            self._init(batch_size, num_seeds=num_seeds)
        return f"IK solver ready (batch_size={batch_size})"

    def _warmup_ik_callback(
        self, request: WarmupIK.Request, response: WarmupIK.Response
    ):
        try:
            response.message = self.warmup(request.batch_size)
            response.success = True
        except Exception as e:
            self._node.get_logger().error(f"IK warmup failed: {e}")
            response.success = False
            response.message = str(e)
        return response

    # ------------------------------------------------------------------
    # Services
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _gpu_guard(self):
        """Serialise this callback's CUDA work against CUDA graph captures.

        Every solver/graph capture in the node runs under ``node.gpu_lock``
        (see its docstring in unified_planner_node): a CUDA graph capture is
        process-global, so ANY other CUDA op — including this solver's eager
        solve kernels and the ``.cpu()`` host syncs that extract the result —
        issued from another executor thread while a capture is in progress
        invalidates it (cudaErrorStreamCaptureInvalidated). Acquiring the lock
        blocking (rather than dropping the request) keeps on-demand service
        calls correct: the caller simply waits for the brief capture window.
        """
        gpu_lock = getattr(self._node, "gpu_lock", None)
        if gpu_lock is None:
            yield
        else:
            with gpu_lock:
                yield

    _LANE_TIMEOUT_S = 120.0

    def _on_lane(self, fn):
        """Run fn on the node's 'kin' lane (sole owner of IK/FK solver state)."""
        lane = getattr(self._node, "lane_kin", None)
        if lane is None:                      # standalone / tests
            with self._gpu_guard():
                return fn()
        return lane.call(fn, timeout=self._LANE_TIMEOUT_S, prio=PRIO_KIN)

    def _ik_callback(self, request: Ik.Request, response: Ik.Response):
        if self._ik_solver is None:
            response.success = False
            response.error_msg.data = "IK not initialized. Call warmup_ik first."
            return response

        def _job():
            ok, result = self._solve([request.pose])
            if not ok:
                return None
            js = JointState()
            js.position = result.solution.cpu().numpy()[0][0].tolist()
            valid = std_msgs.msg.Bool()
            valid.data = bool(result.success.cpu().numpy()[0][0])
            return js, valid

        try:
            out = self._on_lane(_job)
        except Exception as exc:
            self._node.get_logger().warn(
                f"IK failed, returning failure (GPU fault?): {exc}",
                throttle_duration_sec=5.0)
            response.success = False
            response.error_msg.data = f"IK failed: {exc}"
            return response
        if out is None:
            response.success = False
            response.error_msg.data = "IK solve failed"
            return response
        response.joint_states, response.joint_states_valid = out
        response.success = True
        return response

    def _ik_batch_callback(self, request: IkBatch.Request, response: IkBatch.Response):
        if self._ik_solver is None:
            response.success = False
            response.error_msg.data = "IK not initialized. Call warmup_ik first."
            return response

        poses = list(request.poses)

        def _job():
            ok, result = self._solve(poses)
            if not ok:
                return None
            sol = result.solution.cpu().numpy()
            suc = result.success.cpu().numpy()
            out = []
            for i in range(len(poses)):            # padded rows (if any) are ignored
                js = JointState()
                js.position = sol[i][0].tolist()
                valid = std_msgs.msg.Bool()
                valid.data = bool(suc[i][0])
                out.append((js, valid))
            return out

        try:
            out = self._on_lane(_job)
        except Exception as exc:
            self._node.get_logger().warn(
                f"IK batch failed, returning failure (GPU fault?): {exc}",
                throttle_duration_sec=5.0)
            response.success = False
            response.error_msg.data = f"IK batch failed: {exc}"
            return response
        if out is None:
            response.success = False
            response.error_msg.data = "IK batch solve failed"
            return response
        for js, valid in out:
            response.joint_states.append(js)
            response.joint_states_valid.append(valid)
        response.success = True
        return response

    # ------------------------------------------------------------------
    # World update (called by the node when obstacles change)
    # ------------------------------------------------------------------

    def update_world(self, scene=None):
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
        built WITH ``collision_cache`` (see ``_init``), so unlike
        ``FKServices`` it can accept the voxel layer.
        """
        if self._ik_solver is None:
            return
        if scene is None:
            # Normalize primitives to solver-supported collision types
            # (sphere/cylinder/capsule -> mesh), or they silently don't collide.
            scene = self._config.obstacle_manager.collision_world_scene()
        self._ik_solver.update_world(scene)

    def rebuild(self):
        """Recreate the IK solver after a collision-cache change.

        The collision cache is fixed at solver creation, so a cache change
        requires a rebuild (reusing the last batch size; seeds re-resolve from
        the current ``num_ik_seeds`` param). No-op if the solver was never
        initialized.
        """
        if self._ik_solver is None:
            return
        self._init(max(1, self._ik_batch_size), num_seeds=None)
        self._node.get_logger().info("IKServices: solver rebuilt after cache change")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_num_seeds(self) -> int:
        """Shared per-pose IK seed budget, read from the node's ``num_ik_seeds``.

        Mirrors the planner's internal IK (``ConfigWrapperMotion``): the
        standalone ``/ik`` / ``/ik_batch`` services resolve a goal pose with
        the SAME seed count the MotionPlanner's IK uses for pose/goalset
        goals, so both IK paths behave identically. ``num_seeds`` / VRAM scale
        as batch x seeds. Overridable per call via ``_solve(..., num_seeds=)``.
        """
        if self._node.has_parameter("num_ik_seeds"):
            return max(
                1,
                int(
                    self._node.get_parameter("num_ik_seeds")
                    .get_parameter_value()
                    .integer_value
                ),
            )
        return 32

    def _init(self, batch_size: int, num_seeds: int | None = None):
        """Create (or recreate) the IK solver for the given batch size/seeds."""
        if num_seeds is None:
            num_seeds = self._resolve_num_seeds()
        # Build against an EMPTY scene, never the live one: after a cache
        # shrink the live world may not fit the new cache, and constructing
        # against it raises "Cannot load N cuboids" (observed live, killing a
        # benchmark rebuild). Empty always fits; the real scene is pushed
        # right below (same primitives-only scene this used to embed).
        scene = Scene()

        self._node.get_logger().info(
            f"Initializing IK solver (batch_size={batch_size}, num_seeds={num_seeds})..."
        )

        # Single shared curobo kinematic from RobotModelManager (one URDF
        # parse for the whole node) — do not re-build from the YAML path.
        # num_seeds runs that many parallel optimisation trajectories per
        # pose in the batch, so GPU memory grows with batch x seeds. All
        # callers use the same seed count (see solve_poses default).
        cfg = InverseKinematicsCfg.create(
            robot=self._config.robot_model_manager.robot_cfg,
            scene_model=scene,
            num_seeds=num_seeds,
            position_tolerance=0.005,
            orientation_tolerance=0.05,
            self_collision_check=True,
            collision_cache=self._config.collision_cache,
            use_cuda_graph=False,
            max_batch_size=max(1, batch_size),
        )
        self._ik_solver = InverseKinematics(cfg)
        self._ik_batch_size = batch_size
        self._ik_num_seeds = num_seeds

        # Warmup: solve a batch of random configs to prime CUDA kernels.
        q_sample = self._ik_solver.sample_configs(batch_size)
        js = CuRoboJS.from_position(
            q_sample, joint_names=self._ik_solver.kinematics.joint_names
        )
        kin_state = self._ik_solver.compute_kinematics(js)
        goal = kin_state.tool_poses.as_goal()
        self._ik_solver.solve_pose(goal_tool_poses=goal)
        # CPU/GPU ordering bridge — off by default (torch_sync param), see
        # node.torch_sync_enabled().
        if self._node.torch_sync_enabled():
            torch.cuda.synchronize()

        # Push the real scene after the empty build (same primitives-only
        # scene construction used to embed). Best-effort: after a cache
        # shrink the previous world may not fit the new cache — warn loudly
        # and continue empty; the next world event (or an explicit world set)
        # pushes a fitting world. Same-cache rebuilds (batch-size changes,
        # startup) always fit, so those paths are unchanged.
        try:
            self.update_world(
                self._config.obstacle_manager.primitives_only_scene())
        except Exception as e:
            self._node.get_logger().warn(
                f"IK solver rebuilt empty: current world does not fit the new "
                f"cache ({e}); push a fitting world before planning")

        self._node.get_logger().info("IK solver ready")

    def _solve(self, poses, num_seeds: int | None = None):
        """
        Solve IK for a list of geometry_msgs/Pose.
        Reinitializes the solver if the batch size or seed count has changed.
        Returns (success: bool, result).
        """
        if num_seeds is None:
            num_seeds = self._resolve_num_seeds()
        with self._solve_lock:
            if not poses:
                self._node.get_logger().error("IK: empty pose list")
                return False, None
            return self._solve_locked(poses, num_seeds)

    def _solve_locked(self, poses, num_seeds: int):
        """Must run with ``_solve_lock`` held (see ``_solve``)."""

        n = len(poses)
        if n != self._ik_batch_size or num_seeds != self._ik_num_seeds:
            try:
                self._init(n, num_seeds=num_seeds)
            except Exception as e:
                self._node.get_logger().error(
                    f"IK reinit for batch_size={n}, num_seeds={num_seeds} failed: {e}"
                )
                self._ik_batch_size = 0
                self._ik_num_seeds = 0
                return False, None

        # v2 Pose quaternion is wxyz; ROS geometry_msgs is xyzw.
        positions = [[p.position.x, p.position.y, p.position.z] for p in poses]
        orientations = [
            [p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z]
            for p in poses
        ]

        pose2d = CuroboPose(
            position=torch.tensor(positions, dtype=self._dtype, device=self._device),
            quaternion=torch.tensor(
                orientations, dtype=self._dtype, device=self._device
            ),
        )
        tool_frame = self._ik_solver.kinematics.tool_frames[0]
        goal = GoalToolPose.from_poses({tool_frame: pose2d})

        try:
            result = self._ik_solver.solve_pose(goal_tool_poses=goal)
        except Exception:
            try:
                self._init(n, num_seeds=num_seeds)
                result = self._ik_solver.solve_pose(goal_tool_poses=goal)
            except Exception as e:
                self._node.get_logger().error(f"IK solve failed: {e}")
                self._ik_batch_size = 0
                self._ik_num_seeds = 0
                return False, None

        if self._node.torch_sync_enabled():
            torch.cuda.synchronize()
        return True, result

    def solve_poses(self, poses, num_seeds: int | None = None):
        """Batch IK for a list of geometry_msgs/Pose (programmatic API).

        Convenience wrapper used by the reachability service: returns the
        solved joint positions and per-pose convergence flags without building
        ROS service responses.

        ``num_seeds`` controls the solver's seed count and defaults to
        ``num_ik_seeds`` (the node param shared with the planner's internal IK)
        so the reachability map gets identical solve quality to the interactive
        /ik services. Note it scales VRAM as batch x seeds.

        Returns (ok, positions, ok_flags, joint_names) where ``ok`` is False if
        the solver could not be (re)initialised for this batch size;
        ``positions[i]`` is the first-seed solved joint vector for pose ``i``;
        ``ok_flags[i]`` is True only when pose ``i`` converged (failed cells
        keep their raw value); ``joint_names`` is the active joint list ([] if
        the solver is unavailable).
        """
        n = len(poses)
        if n == 0:
            return False, None, [], []

        def _job():
            ok, result = self._solve(list(poses), num_seeds=num_seeds)
            if not ok:
                return False, None, [False] * n, []
            sol = result.solution.detach().cpu().numpy()
            suc = result.success.detach().cpu().numpy()
            return (True, [sol[i][0].tolist() for i in range(n)],
                    [bool(suc[i][0]) for i in range(n)],
                    list(self._ik_solver.kinematics.joint_names))

        return self._on_lane(_job)

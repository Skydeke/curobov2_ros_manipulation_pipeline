#!/usr/bin/env python3
"""
IK services for the unified planner node (v2).

Thin ROS service layer over ``core/config_wrapper_ik.py`` (ConfigWrapperIK),
which owns the solver's build-time configuration and lifecycle — the same
split as the motion wrapper (ConfigWrapperMotion) and the planners that plan
over it.

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

from curobo.types import Pose as CuroboPose, GoalToolPose

from curobov2_ros_interfaces.srv import Ik, IkBatch, WarmupIK
from curobov2_ros.core.frame_utils import resolve_poses_to_base
from curobov2_ros.core.gpu_lanes import PRIO_KIN


class IKServices:
    """
    ROS service layer for the standalone IK solver on an existing node.

    Depends on:
    - config_wrapper_ik (ConfigWrapperIK): solver lifecycle (build/rebuild/
      world updates) and build-time tunables, sharing the motion wrapper's
      robot/scene/collision-cache context.

    The solver is created only when warmup_ik is called.
    Obstacle updates are propagated via update_world().
    """

    def __init__(self, node, config_wrapper_ik):
        self._node = node
        self._config = config_wrapper_ik

        # Serializes solves on the single shared solver: /ik, /ik_batch and the
        # reachability service (/generate_rm) can all arrive from different
        # executor threads — a second solve_pose() (or a reinit for a different
        # batch size) racing the first would corrupt the solver's state.
        self._solve_lock = threading.Lock()

        # Device / dtype resolved from the wrapper (set by RobotModelManager).
        self._device = getattr(config_wrapper_ik, "_device", torch.device("cuda"))
        self._dtype = getattr(config_wrapper_ik, "_ops_dtype", torch.float32)

        # VRAM telemetry: log allocated vs reserved every N solves so a
        # monotonic nvidia-smi climb can be told apart from a real leak
        # (allocated flat + reserved climbing = allocator caching).
        self._dbg_calls = 0
        self._dbg_prev_shapes = None
        self._dbg_hunted = False

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
        if num_seeds is None:
            num_seeds = self._resolve_num_seeds()
        if (
            self._config.ik_solver is not None
            and not self._config.needs_rebuild(batch_size, num_seeds)
        ):
            return f"IK solver already ready (batch_size={batch_size})"
        # Solver construction + warmup run CUDA kernels (solve for a batch
        # of random configs) — must not race a CUDA graph capture.
        with self._gpu_guard():
            self._config.build(batch_size, num_seeds=num_seeds)
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

    @property
    def _ik_solver(self):
        """The live IK solver, owned by ConfigWrapperIK (compat accessor).

        Kept so existing readers of the solver (e.g. the attached-obstacle
        re-apply in ``ObstacleManager``) keep working unchanged.
        """
        return self._config.ik_solver

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

        # Server owns every transform: resolve the tagged frame ("" == base)
        # into the configured base link before solving.
        try:
            poses = resolve_poses_to_base(
                [request.pose], getattr(request, "frame_id", "") or "",
                self._config.base_link,
                getattr(self._node, "tf_buffer", None))
        except RuntimeError as e:
            response.success = False
            response.error_msg.data = str(e)
            return response

        def _job():
            ok, result = self._solve(poses)
            if not ok:
                return None
            js = JointState()
            js.position = result.solution.detach().cpu().numpy()[0][0].tolist()
            valid = std_msgs.msg.Bool()
            valid.data = bool(result.success.detach().cpu().numpy()[0][0])
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

        # Server owns every transform: resolve the tagged frame ("" == base)
        # into the configured base link before solving.
        try:
            poses = resolve_poses_to_base(
                poses, getattr(request, "frame_id", "") or "",
                self._config.base_link,
                getattr(self._node, "tf_buffer", None))
        except RuntimeError as e:
            response.success = False
            response.error_msg.data = str(e)
            return response

        def _job():
            ok, result = self._solve(poses)
            if not ok:
                return None
            sol = result.solution.detach().cpu().numpy()
            suc = result.success.detach().cpu().numpy()
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
        """Propagate obstacle changes to the IK solver (delegates to the wrapper).

        ``scene`` is the node's already-resolved solver scene from
        ``update_all_solvers_world`` — see ``ConfigWrapperIK.update_world``
        for why it must be passed, not re-derived here.
        """
        self._config.update_world(scene)

    def rebuild(self):
        """Recreate the IK solver after a collision-cache change (delegates)."""
        self._config.rebuild()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_num_seeds(self) -> int:
        """Shared per-pose IK seed budget (delegates to the wrapper)."""
        return self._config._resolve_num_ik_seeds(self._node)

    def _solve(self, poses, num_seeds: int | None = None):
        """
        Solve IK for a list of geometry_msgs/Pose.
        Reinitializes the solver if the batch size, seed count, or CUDA-graph
        mode has changed. Returns (success: bool, result).
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
        if self._config.needs_rebuild(n, num_seeds):
            try:
                self._config.build(n, num_seeds=num_seeds)
            except Exception as e:
                self._node.get_logger().error(
                    f"IK reinit for batch_size={n}, num_seeds={num_seeds} failed: {e}"
                )
                self._config.invalidate()
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
                self._config.build(n, num_seeds=num_seeds)
                result = self._ik_solver.solve_pose(goal_tool_poses=goal)
            except Exception as e:
                self._node.get_logger().error(f"IK solve failed: {e}")
                self._config.invalidate()
                return False, None

        if self._node.torch_sync_enabled():
            torch.cuda.synchronize()
        self._dbg_calls += 1
        if self._dbg_calls % 500 == 0:
            try:
                import gc as _gc
                import warnings as _warnings
                from collections import Counter as _Counter

                # gc.get_objects() touches lazy module aliases (e.g. a
                # deprecated torch.distributed path) that spray FutureWarnings
                # with our source line; silence them for the whole probe.
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore")
                    _objs_all = _gc.get_objects()

                    alloc = torch.cuda.memory_allocated(self._device) / 1048576.0
                    reserved = torch.cuda.memory_reserved(self._device) / 1048576.0

                    shapes = _Counter()
                    tensor_bytes = 0.0
                    for _o in _objs_all:
                        if isinstance(_o, torch.Tensor) and _o.is_cuda:
                            shapes[(tuple(_o.shape), str(_o.dtype))] += 1
                            tensor_bytes += _o.nelement() * _o.element_size()
                    n_tensors = sum(shapes.values())
                    del _objs_all
                    # Decisive experiment: force a full collection and recount.
                    # Drops to baseline => cyclic trash (fix = break the cycle
                    # or periodic collect). Persists => rooted/C-held leak.
                    _gc.collect()
                    _n2 = 0
                    _b2 = 0.0
                    for _o in _gc.get_objects():
                        if isinstance(_o, torch.Tensor) and _o.is_cuda:
                            _n2 += 1
                            _b2 += _o.nelement() * _o.element_size()
                    _post = (
                        f" post_gc_tensors={_n2} post_gc_bytes={_b2 / 1048576.0:.1f}MiB"
                    )
                growth = ""
                if self._dbg_prev_shapes is not None:
                    grown = {
                        k: v - self._dbg_prev_shapes.get(k, 0)
                        for k, v in shapes.items()
                        if v - self._dbg_prev_shapes.get(k, 0) > 0
                    }
                    top = sorted(grown.items(), key=lambda kv: -kv[1])[:5]
                    growth = f" grown_shapes={top}"
                    # One-shot referrer hunt on the fastest-growing shape:
                    # names the object keeping each solve's tensor alive.
                    # NOTE: the gc.get_objects() list itself refers to every
                    # object, so it is excluded by id to avoid self-report.
                    if top and not self._dbg_hunted:
                        self._dbg_hunted = True
                        try:
                            import types as _types

                            _shape = top[0][0][0]
                            with _warnings.catch_warnings():
                                _warnings.simplefilter("ignore")
                                _objs = _gc.get_objects()
                            _self_id = id(_objs)
                            for _o in _objs:
                                if (
                                    isinstance(_o, torch.Tensor)
                                    and _o.is_cuda
                                    and tuple(_o.shape) == _shape
                                ):
                                    _refs = [
                                        _r
                                        for _r in _gc.get_referrers(_o)
                                        if id(_r) != _self_id
                                    ]
                                    # Exclude every probe-created container so
                                    # only real holders are reported. NOTE: no
                                    # slicing here: _refs[:k] would itself be a
                                    # live list during the query below.
                                    _probe_ids = {_self_id, id(_refs)}
                                    _desc = []
                                    _n = 0
                                    for _i in range(len(_refs)):
                                        if _n >= 6:
                                            break
                                        _r = _refs[_i]
                                        _n += 1
                                        if isinstance(_r, list):
                                            _owners = {}
                                            for _o2 in _gc.get_referrers(_r):
                                                if id(_o2) in _probe_ids:
                                                    continue
                                                _k = type(_o2).__name__
                                                _owners[_k] = _owners.get(_k, 0) + 1
                                            _desc.append(
                                                f"list(len={len(_r)},"
                                                f"owners={_owners})"
                                            )
                                        elif isinstance(_r, _types.CellType):
                                            from collections import Counter as _C2

                                            _all = _C2()
                                            for _o2 in _gc.get_referrers(_r):
                                                if id(_o2) in _probe_ids:
                                                    continue
                                                _all[type(_o2).__name__] += 1
                                            _fns = [
                                                getattr(_f, "__qualname__", "?")
                                                for _f in _gc.get_referrers(_r)
                                                if isinstance(
                                                    _f, _types.FunctionType
                                                )
                                                and id(_f) not in _probe_ids
                                            ][:4]
                                            _desc.append(
                                                f"cell(all={dict(_all)},"
                                                f"fns={_fns})"
                                            )
                                        elif isinstance(_r, dict):
                                            _keys = [
                                                k
                                                for k, v in _r.items()
                                                if v is _o
                                            ][:3]
                                            _desc.append(
                                                f"dict(keys={_keys})"
                                            )
                                        else:
                                            _desc.append(type(_r).__name__)
                                    self._node.get_logger().info(
                                        f"IK leak hunt shape={_shape}: "
                                        f"{_desc}"
                                    )
                                    break
                            del _objs
                        except Exception:  # noqa: BLE001
                            pass
                self._dbg_prev_shapes = shapes
                self._node.get_logger().info(
                    f"IK gpu mem: allocated={alloc:.1f}MiB "
                    f"reserved={reserved:.1f}MiB solves={self._dbg_calls} "
                    f"live_cuda_tensors={n_tensors} "
                    f"tensor_bytes={tensor_bytes / 1048576.0:.1f}MiB"
                    f"{growth}{_post}"
                )
            except Exception:  # noqa: BLE001
                pass
        return True, result

    def solve_poses(self, poses, num_seeds: int | None = None):
        """Batch IK for a list of geometry_msgs/Pose (programmatic API).

        ``poses`` must already be expressed in the robot base frame (unlike
        the /ik_batch service, there is no frame tag here): the reachability
        service resolves its plane via TF before building the grid it passes
        in. Convenience wrapper used by the reachability service: returns the
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

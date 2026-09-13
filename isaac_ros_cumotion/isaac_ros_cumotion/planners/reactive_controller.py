#!/usr/bin/env python3
"""
Abstract base class for *reactive* (closed-loop) controllers.

This is the closed-loop sibling of :class:`SinglePlanner` (which is the shared
base for open-loop, MotionPlanner-based planners). It exists because cuRobo
models reactive control and motion generation as **different wrappers over the
same core** (shared robot model + world collision model, see
https://nvlabs.github.io/curobo/latest/concepts/index.html): they differ only by
horizon and update frequency, not by structure. curobo_ros mirrors that: a
reactive controller is a *thin ROS wrapper* around a cuRobo reactive solver
(e.g. ``ModelPredictiveControl``).

Design
------
``ReactiveController`` owns all the ROS / robot / perception plumbing of the
control loop **once**, so a concrete controller only implements the few
cuRobo-specific steps:

    build_solver()        -> create the cuRobo solver from the SHARED context
    setup(state, goal)    -> set the initial goal on the solver
    step(state)           -> one optimization step, returns the next action
    step_paced(state)     -> one full-plan command window (paced mode; MPC)
    apply_live_goal(raw)  -> retarget the goal during execution
    is_converged()        -> stop condition

Adding a new reactive control = subclass this + register it in
``PlannerFactory._PLANNER_CATALOG``. The same ``SetPlanner`` / ``GetPlanners``
switch then works for it, and it automatically shares the node's single context
(robot, obstacles, scene, collision cache).
"""

import threading
import traceback
from abc import abstractmethod
from contextlib import nullcontext
from typing import Any, Optional

import torch
from curobo.types import JointState, Pose, GoalToolPose
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from .trajectory_planner import TrajectoryPlanner, PlannerResult, ExecutionMode
from isaac_ros_cumotion_interfaces.action import SendTrajectory


class ReactiveController(TrajectoryPlanner):
    """Closed-loop controller base. Subclasses wrap a cuRobo reactive solver."""

    def __init__(self, node, config_wrapper):
        super().__init__(node, config_wrapper)

        # The cuRobo reactive solver, built lazily from the shared context.
        self.solver = None

        # Goal / loop state.
        self.start_state: Optional[JointState] = None
        self.goal: Any = None
        self.is_goal_active = False

        # Raw [x, y, z, qw, qx, qy, qz] written from the ROS thread (topic) and
        # consumed on the control-loop thread to avoid racing CUDA graph capture.
        # The lock only guards the pointer swap (see set_live_goal/_take_live_goal)
        # — apply_live_goal() itself runs outside it, since it does CUDA work.
        self._live_goal_lock = threading.Lock()
        self._latest_goal = None
        self._latest_goal_fresh = False

        # Tunables (overwritten from the per-call config in plan()).
        self.convergence_threshold = 0.01      # meters
        self.max_iterations = 1000
        # Refresh the perception ESDF every N steps (0 disables, 1 = every step).
        # At 20 (~3s at 7Hz) the MPC collision world lagged behind a moving
        # obstacle (a hand) -- the arm made contact before the update landed.
        # 2 is roughly camera rate (5Hz), affordable since voxelization moved to
        # the GPU (no more ~10s CPU fallback). See debug 2026-07-15.
        self.perception_refresh_period = 2

        # Latest scalar position error, written by step(), read by is_converged().
        self._last_position_error = float('inf')
        # Cartesian target (xyz tensor), set by _set_target, read by FK error.
        self._target_position = None
        self._step_times = []
        # ROS-clock time of the last status log (throttled, rate-independent).
        self._last_log_time = 0.0

        # Fixed-interval command pacing (used only when a subclass sets
        # self._command_interval > 0, e.g. MPCController via the
        # mpc_command_interval ROS param). Paced mode is a SEQUENTIAL
        # solve-and-shoot loop: solve a single-plan command window, send it as
        # one FollowJointTrajectory goal, wait for it to fully execute, then
        # re-solve from the FRESH post-execution robot state.
        # The old producer/consumer split (free-running optimize_next_action
        # pops accumulated into a buffer drained by a timer) mixed several
        # re-plans into one window and anchored them on fast-forwarded
        # states — the robot executed kinked trajectories and looked jumpy.
        # cf. debug 2026-09-13.
        # Poll interval while waiting for the sent window to execute
        # (_wait_for_execution) — keeps the wait responsive to cancel without
        # busy-polling.
        self._wait_poll_seconds = 0.02
        # Dedicated callback group for the execution-wait timer, so its poll
        # callbacks never serialize with the executor's default group (a long
        # open-loop plan in the default group must not stall the
        # solve-and-shoot cadence).
        self._wait_cb_group = MutuallyExclusiveCallbackGroup()

        # Device/dtype for building tensors on the hot path.
        self._device = getattr(config_wrapper, '_device', torch.device('cuda'))
        self._dtype = getattr(config_wrapper, '_ops_dtype', torch.float32)

    def _get_execution_mode(self) -> ExecutionMode:
        return ExecutionMode.CLOSED_LOOP

    # ------------------------------------------------------------------
    # Solver lifecycle
    # ------------------------------------------------------------------

    def ensure_solver(self):
        """Build the cuRobo solver once, lazily, from the shared context."""
        if self.solver is None:
            self.solver = self.build_solver()
        return self.solver

    def rebuild_solver(self):
        """Recreate the solver from scratch (e.g. after a collision-cache change).

        The collision cache size is fixed at solver creation, so a change
        requires a full rebuild rather than a world update.
        """
        self.solver = None
        return self.ensure_solver()

    # ------------------------------------------------------------------
    # cuRobo-specific hooks (implemented by concrete controllers)
    # ------------------------------------------------------------------

    @abstractmethod
    def build_solver(self):
        """Create and return the cuRobo reactive solver from ``self.config_wrapper``.

        The shared context exposes everything needed: ``robot_config_file``,
        ``obstacle_manager.get_scene()``, ``collision_cache``, ``_device`` /
        ``_ops_dtype``. Implementations should also publish the solver where the
        node expects it (e.g. ``self.node.mpc``) so world/cache updates reach it.
        """
        raise NotImplementedError

    @abstractmethod
    def setup(self, start_state: JointState, goal_request: Any) -> bool:
        """Set the initial goal on the solver. Return True on success."""
        raise NotImplementedError

    @abstractmethod
    def step(self, current_state: JointState) -> JointState:
        """Run one optimization step and return the next action JointState.

        Implementations must also update ``self._last_position_error``.
        """
        raise NotImplementedError

    def step_paced(self, current_state: JointState) -> JointState:
        """Solve ONE command WINDOW from a single plan (paced mode only).

        ``_execute_paced`` sends the returned window as a single
        FollowJointTrajectory goal and only re-plans after the robot has fully
        executed it, so the window must be the output of ONE coherent solve.
        The default returns the single-command ``step()`` (used by subclasses
        that never set a command interval); MPCController overrides this to
        return the whole multi-point command window of one fresh
        ``optimize_action_sequence`` solve.
        """
        return self.step(current_state)

    @abstractmethod
    def apply_live_goal(self, raw_goal, current_js=None) -> bool:
        """Retarget the goal from a raw [x,y,z,qw,qx,qy,qz] list during execution.

        Args:
            raw_goal: 7-element list ``[x, y, z, qw, qx, qy, qz]``.
            current_js: Optional current joint state (for IK-seeded reseeding).
        """
        raise NotImplementedError

    def update_world(self, scene) -> None:
        """Push the shared Scene into this controller's collision model.

        Reactive solvers each hold their own collision checker, so the node
        delegates world updates here (instead of reaching into solver internals).
        Default is a no-op; concrete controllers override (e.g. MPC reloads its
        scene_collision_checker, retarget updates its IK solvers).
        """
        return None

    # ------------------------------------------------------------------
    # Shared cuRobo helpers (target pose + FK error) usable by every
    # reactive controller — solver exposes tool_frames and FK either directly
    # (MPC) or via .kinematics (retargeter).
    # ------------------------------------------------------------------

    def _set_target(self, raw) -> GoalToolPose:
        """Store the target xyz (for FK error) and build the tool-pose goal.

        Passes ordered_tool_frames + num_goalset like the official cuRobo reactive
        example so the goal buffer is shaped exactly as the solver expects.
        """
        self._target_position = torch.tensor(
            raw[0:3], dtype=self._dtype, device=self._device
        )
        return GoalToolPose.from_poses(
            {self.solver.tool_frames[0]: Pose.from_list(list(raw))},
            ordered_tool_frames=self.solver.tool_frames,
            num_goalset=1,
        )

    def _compute_ee_position(self, current_state: JointState):
        """Current end-effector position via the solver's forward kinematics."""
        fk = getattr(self.solver, 'compute_kinematics', None)
        if fk is None:
            fk = self.solver.kinematics.compute_kinematics
        kin = fk(current_state)
        return kin.tool_poses.position.reshape(-1, 3)[0]  # [B,H,L,3] -> first link

    def _fk_position_error(self, current_state: JointState) -> float:
        """Real Cartesian distance (m) between the current EE and the target."""
        if self._target_position is None:
            return float('inf')
        try:
            ee = self._compute_ee_position(current_state)
            return float(torch.linalg.norm(ee - self._target_position).item())
        except Exception:
            return float('inf')

    def is_on_target(self) -> bool:
        """Signal (NOT a stop condition): the arm is within tolerance of the goal.

        Reactive control keeps servoing even when on target, so this only drives
        the `on_target` feedback flag — it never ends the control loop.
        """
        return self._last_position_error < self.convergence_threshold

    def get_position_error(self) -> float:
        """Latest scalar position error (meters)."""
        return self._last_position_error

    def set_live_goal(self, raw_goal) -> None:
        """Deposit a live goal update. Called from the ROS topic thread.

        Only swaps a pointer under the lock — the expensive retargeting
        (apply_live_goal, CUDA work) happens on the control-loop thread after
        _take_live_goal() hands it off, never here.
        """
        with self._live_goal_lock:
            self._latest_goal = list(raw_goal)
            self._latest_goal_fresh = True

    def _take_live_goal(self):
        """Atomically take-and-clear the pending live goal, or None if stale.

        Replaces the previous test-read-clear sequence (latest_goal is not
        None -> read -> set to None), which raced: a goal written by the ROS
        thread between the read and the clear was silently dropped.
        """
        with self._live_goal_lock:
            if not self._latest_goal_fresh:
                return None
            raw = self._latest_goal
            self._latest_goal = None
            self._latest_goal_fresh = False
            return raw

    def _step_guard(self):
        """gpu_lock for a step() that may capture a CUDA graph, else no lock.

        The node flags exactly the step(s) that follow a graph release/rebuild
        (see take_graph_capture_pending) as capture-pending; every other step
        only replays an already-captured graph and doesn't need exclusivity.
        Locking every step would starve the perception thread (its depth
        callback does a non-blocking gpu_lock acquire and drops the frame —
        see camera_depth_map_strategy.py), undoing perception_refresh_period.
        cf. debug 2026-07-28.
        """
        take = getattr(self.node, 'take_graph_capture_pending', None)
        if take is not None and take():
            return self.node.gpu_lock
        return nullcontext()

    # ------------------------------------------------------------------
    # Planning: set up the reactive goal (no full trajectory is produced).
    # ------------------------------------------------------------------

    def plan(self, start_state: JointState, goal_request: Any, config: dict,
             robot_context: Optional[Any] = None) -> PlannerResult:
        self.ensure_solver()
        if self.solver is None:
            return PlannerResult(
                success=False,
                message=f"{self.get_planner_name()} solver not initialized.",
            )

        self.convergence_threshold = config.get('convergence_threshold', 0.01)
        self.max_iterations = config.get('max_iterations', 1000)
        self.start_state = start_state

        try:
            with self.node.gpu_lock:
                setup_ok = self.setup(start_state, goal_request)
            if not setup_ok:
                return PlannerResult(success=False, message="Failed to set reactive goal")
            # Discard any live goal left over from a previous session — it must
            # not be silently applied to this new one. A goal arriving AFTER
            # this point (i.e. after setup) is still honoured normally.
            self._take_live_goal()
            self.is_goal_active = True

            if robot_context is not None:
                self._init_robot_at_start(robot_context, start_state)

            self.node.get_logger().info(
                f"{self.get_planner_name()} goal set: "
                f"convergence={self.convergence_threshold}m, max_iter={self.max_iterations}"
            )
            return PlannerResult(
                success=True,
                message=f"{self.get_planner_name()} goal set",
                trajectory=None,
                metadata={
                    'convergence_threshold': self.convergence_threshold,
                    'max_iterations': self.max_iterations,
                },
            )
        except Exception as e:
            self.node.get_logger().error(f"{self.get_planner_name()} setup error: {e}")
            self.node.get_logger().error(traceback.format_exc())
            return PlannerResult(success=False, message=f"Reactive setup error: {e}")


    def execute(self, robot_context, goal_handle=None) -> bool:
        """Dispatch to the paced (solve-and-shoot) or immediate servo loop.

        Paced mode (self._command_interval > 0, e.g. MPCController via the
        mpc_command_interval ROS param) sends one solved single-plan command
        window per re-plan — see _execute_paced. Every other caller
        (interval 0, the default) gets the original free-running behavior via
        _execute_immediate.
        """
        if not self.is_goal_active or self.solver is None:
            self.node.get_logger().error(
                f"{self.get_planner_name()} not initialized. Call plan() first."
            )
            return False

        interval = getattr(self, '_command_interval', 0.0)
        if interval > 0.0:
            return self._execute_paced(robot_context, goal_handle, interval)
        return self._execute_immediate(robot_context, goal_handle)

    def _execute_immediate(self, robot_context, goal_handle=None) -> bool:
        try:
            tstep = 0
            self._step_times = []

            self._last_log_time = 0.0
            self.node.get_logger().info(f"Starting {self.get_planner_name()} servo loop")

            exec_csv = self._exec_csv_enabled()
            if exec_csv:
                self._exec_csv_init(
                    prefix="exec_servo",
                    columns=["t_s", "tstep", "error_m", "on_target", "step_ms"],
                )
                self._exec_csv_t0 = self._now()

            # Initialize the solver state from the robot once, then advance it
            # from the solver's own prediction each step (see the loop below).
            current_state = self._read_state(robot_context)

            # Reactive control runs CONTINUOUSLY: reaching the target is only a
            # signal (on_target), never a stop condition. The loop ends solely on
            # cancel (or error). Without an action handle, max_iterations is a
            # safety cap for non-action callers.
            while self.is_goal_active:
                if goal_handle is not None and goal_handle.is_cancel_requested:
                    self.node.get_logger().info(f"{self.get_planner_name()} cancel requested")
                    break
                if goal_handle is None and tstep >= self.max_iterations:
                    break

                # Periodically refresh the perception-based collision world so the
                # controller reacts to obstacles seen by the cameras. Throttled
                # (every N steps) since recomputing the ESDF is heavier than a step.
                if (self.perception_refresh_period > 0
                        and tstep % self.perception_refresh_period == 0
                        and hasattr(self.node, 'refresh_perception_world')):
                    self.node.refresh_perception_world()

                # Consume a pending live goal on the loop thread only, under
                # gpu_lock: retargeting runs the solver's IK, which can capture
                # a CUDA graph, and capture is process-global (any CUDA op on
                # any thread during it raises cudaErrorStreamCaptureUnsupported
                # and poisons the context). The lock is how the perception
                # thread knows to skip its frame — same invariant as plan()'s
                # setup() call. A raising apply_live_goal (bad pose, IK error)
                # is narrowed to the goal itself — it must not kill the whole
                # session, since the arm should keep servoing the previous
                # goal instead. cf. debug 2026-07-28.
                raw = self._take_live_goal()
                if raw is not None:
                    try:
                        with self.node.gpu_lock:
                            self.apply_live_goal(raw, current_state)
                    except Exception as e:
                        self.node.get_logger().error(
                            f"{self.get_planner_name()}: live goal rejected "
                            f"({e}) - keeping previous goal",
                            throttle_duration_sec=1.0,
                        )

                st_time = self._now()
                with self._step_guard():
                    action = self.step(current_state)  # step() already syncs (FK .item())
                if tstep > 5:
                    self._step_times.append(self._now() - st_time)

                self._send_command(robot_context, action)

                # Close the loop with the REAL robot position (a fast,
                # non-blocking read of the joint_states subscriber's latest
                # cached value — no wait, so no new lag). Purely trusting the
                # solver's own predicted position (position-only, from
                # _state_from_action) drifts from reality once a horizon takes
                # real wall-clock time to execute (observed on hardware: MPC
                # correcting toward an imagined position -> growing tracking
                # error, unstable motion). Velocity/acceleration stay the
                # solver's own prediction — the driver doesn't give reliable
                # velocity feedback, and the MPC needs SOME dynamic-continuity
                # estimate for warm-starting.
                predicted_state = self._state_from_action(action)
                current_state = self._close_state_loop(robot_context, predicted_state)


                if goal_handle is not None and tstep % 5 == 0:
                    self._publish_feedback(goal_handle, action)

                now = self._now()
                if now - self._last_log_time > 1.0:
                    self._last_log_time = now
                    self.node.get_logger().info(
                        f"{self.get_planner_name()}: error="
                        f"{self.get_position_error():.4f}m on_target={self.is_on_target()}"
                    )

                if exec_csv and tstep % 1 == 0:
                    self._exec_csv_write([
                        f"{self._now() - self._exec_csv_t0:.3f}",
                        f"{tstep}",
                        f"{self.get_position_error():.6f}",
                        str(self.is_on_target()),
                        f"{self._now() - st_time:.4f}",
                    ])

                tstep += 1

            robot_context.stop_robot()
            if exec_csv:
                self._exec_csv_close()

            if self._step_times:
                avg_time = sum(self._step_times) / len(self._step_times)
                self.node.get_logger().info(
                    f"{self.get_planner_name()} stopped: {tstep} steps, "
                    f"avg time={avg_time * 1000:.1f}ms/step"
                )

            # A clean stop (cancel / safety cap) is a successful session end.
            return True

        except Exception as e:
            self.node.get_logger().error(f"{self.get_planner_name()} execution error: {e}")
            self.node.get_logger().error(traceback.format_exc())
            robot_context.stop_robot()
            self._exec_csv_close()
            return False

    def _execute_paced(self, robot_context, goal_handle, interval: float) -> bool:
        """Solve-and-shoot servo loop for fixed command-window pacing.

        Contract (see docs/concepts/mpc-implementation.md): ONE command window
        is solved and sent as a single FollowJointTrajectory goal, the robot
        fully executes it, and only THEN the solver re-plans — anchored on the
        FRESH post-execution robot state. The window comes from ONE solve
        (MPCController.step_paced: one optimize_action_sequence solve, whose
        command window the curobo execution manager returns directly), so the
        robot never executes a trajectory that blends several re-plans, and
        every replacement window starts where the arm actually is.
        _wait_for_execution() (robot progression, with ``interval`` as the
        nominal-duration fallback) makes the re-solve cadence track the
        driver's window cadence.

        This loop differs from _execute_immediate only in the send-then-wait
        sequencing: the cuRobo step() call itself is unchanged. Only this loop
        calls step() (CUDA).
        """
        try:
            tstep = 0
            self._step_times = []

            self._last_log_time = 0.0
            self.node.get_logger().info(
                f"Starting {self.get_planner_name()} servo loop "
                f"(paced, interval={interval}s, single-plan windows)"
            )

            exec_csv = self._exec_csv_enabled()
            if exec_csv:
                self._exec_csv_init(
                    prefix="exec_servo",
                    columns=["t_s", "tstep", "error_m", "on_target", "step_ms"],
                )
                self._exec_csv_t0 = self._now()

            current_state = self._read_state(robot_context)

            while self.is_goal_active:
                if goal_handle is not None and goal_handle.is_cancel_requested:
                    self.node.get_logger().info(f"{self.get_planner_name()} cancel requested")
                    break
                if goal_handle is None and tstep >= self.max_iterations:
                    break

                if (self.perception_refresh_period > 0
                        and tstep % self.perception_refresh_period == 0
                        and hasattr(self.node, 'refresh_perception_world')):
                    self.node.refresh_perception_world()

                # Under gpu_lock, failure narrowed to the goal — see
                # _execute_immediate for why.
                raw = self._take_live_goal()
                if raw is not None:
                    try:
                        with self.node.gpu_lock:
                            self.apply_live_goal(raw, current_state)
                    except Exception as e:
                        self.node.get_logger().error(
                            f"{self.get_planner_name()}: live goal rejected "
                            f"({e}) - keeping previous goal",
                            throttle_duration_sec=1.0,
                        )

                st_time = self._now()
                with self._step_guard():
                    action = self.step_paced(current_state)  # fresh single-plan window
                if tstep > 5:
                    self._step_times.append(self._now() - st_time)

                self._send_command(robot_context, action)
                if goal_handle is not None:
                    self._publish_feedback(goal_handle, action)

                # Let the window fully execute before re-solving: the next
                # plan must be anchored on the fresh, post-execution robot
                # state, or consecutive windows overlap/disagree.
                self._wait_for_execution(robot_context, interval)

                predicted_state = self._state_from_action(action)
                current_state = self._close_state_loop(robot_context, predicted_state)


                now = self._now()
                if now - self._last_log_time > 1.0:
                    self._last_log_time = now
                    self.node.get_logger().info(
                        f"{self.get_planner_name()}: error="
                        f"{self.get_position_error():.4f}m on_target={self.is_on_target()}"
                    )

                if exec_csv and tstep % 1 == 0:
                    self._exec_csv_write([
                        f"{self._now() - self._exec_csv_t0:.3f}",
                        f"{tstep}",
                        f"{self.get_position_error():.6f}",
                        str(self.is_on_target()),
                        f"{self._now() - st_time:.4f}",
                    ])

                tstep += 1

            robot_context.stop_robot()
            if exec_csv:
                self._exec_csv_close()

            if self._step_times:
                avg_time = sum(self._step_times) / len(self._step_times)
                self.node.get_logger().info(
                    f"{self.get_planner_name()} stopped: {tstep} steps, "
                    f"avg time={avg_time * 1000:.1f}ms/step"
                )

            return True

        except Exception as e:
            self.node.get_logger().error(f"{self.get_planner_name()} execution error: {e}")
            self.node.get_logger().error(traceback.format_exc())
            robot_context.stop_robot()
            self._exec_csv_close()
            return False

    def _wait_for_execution(self, robot_context, interval: float) -> None:
        """Block until the sent command window has executed (or a timeout).

        Event-driven wait on the node's EXECUTOR, not a busy loop: a ROS
        timer (node.create_timer, own callback group, driven by the node's
        clock so it tracks /use_sim_time) polls the FollowJointTrajectory
        progression (robot_context.get_progression: 1.0 = completed) and sets
        an Event; this loop thread just waits on it. ``interval`` is the
        nominal window duration (mpc_command_interval) and the fallback: if
        the controller never reports completion we cap the wait at 2x + 0.2 s
        and re-solve anyway — the next solve anchors on the fresh state read
        right after, so a short cadence over/undershoot self-corrects.
        """
        done = threading.Event()
        timeout_s = max(interval * 2.0 + 0.2, 0.5)

        def _poll():
            if not self.is_goal_active or robot_context.get_progression() >= 1.0:
                done.set()

        timer = self.node.create_timer(
            self._wait_poll_seconds, _poll,
            callback_group=self._wait_cb_group,
        )
        try:
            done.wait(timeout_s)
        finally:
            timer.cancel()
            self.node.destroy_timer(timer)

    def _publish_feedback(self, goal_handle, action_state):
        """Publish the reactive status through the action feedback (no status topic)."""
        err = self.get_position_error()
        on_target = self.is_on_target()
        fb = SendTrajectory.Feedback()
        fb.state = "ON_TARGET" if on_target else "TRACKING"
        fb.on_target = bool(on_target)
        fb.position_error = float(err) if err != float('inf') else -1.0
        fb.step_progression = (
            float(1.0 - min(err / 0.1, 1.0)) if err != float('inf') else 0.0
        )
        try:
            pos = action_state.position
            if pos.dim() == 3:
                pos = pos[:, -1, :]  # command window [1, n, dof]: report its end point
            elif pos.dim() == 2 and pos.shape[0] > 1:
                pos = pos[-1, :]     # unbatched window [n, dof]: last (current-target) point
            fb.joint_command.position = (pos[0] if pos.dim() > 1 else pos).cpu().tolist()
            fb.joint_command.name = list(getattr(self.solver, 'joint_names', []))
        except Exception:
            pass
        goal_handle.publish_feedback(fb)

    def cancel(self):
        self.is_goal_active = False
        self.node.get_logger().info(f"{self.get_planner_name()} execution cancelled")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _now(self) -> float:
        """Seconds on the node's ROS clock (float) — never Python wall time.

        All loop pacing, statistics and CSV timestamps use this so behavior
        is identical under /use_sim_time and unaffected by clock drift.
        """
        return self.node.get_clock().now().nanoseconds * 1e-9

    def _read_state(self, robot_context) -> JointState:
        """Initial solver state from the robot's joint positions.

        Matches the official cuRobo reactive example: joint_names are labelled and
        velocity/acceleration are explicitly zeroed so the solver's dynamic state
        is well-defined at start (an unlabelled/vel-less state weakens tracking).
        """
        actual_joint_pose = robot_context.get_joint_pose()
        pos = torch.tensor([actual_joint_pose], dtype=self._dtype, device=self._device)
        js = JointState.from_position(pos, joint_names=self.solver.joint_names)
        js.velocity = torch.zeros_like(pos)
        js.acceleration = torch.zeros_like(pos)
        return js

    @staticmethod
    def _row(t):
        """Return tensor as a 2D [1, D] row (or None)."""
        if t is None:
            return None
        t = t.detach().clone()
        return t if t.dim() > 1 else t.unsqueeze(0)

    def _state_from_action(self, action: JointState) -> JointState:
        """Build the next solver state from a commanded action (pos+vel+acc).

        For a single command from the native MPC loop (next_action, position
        ``[1, dof]``) the state is used as-is, warm-starting the next optimize
        call from where the last window actually ended — matches cuRobo's own
        reactive_control example's state-continuity pattern. (The stacked
        send-window shape ``[1, horizon, dof]`` was removed with the paced
        producer/consumer loop.)
        """
        pos, vel, acc = action.position, getattr(action, 'velocity', None), getattr(action, 'acceleration', None)
        if pos.dim() == 3:
            pos = pos[:, -1, :]
            vel = vel[:, -1, :] if vel is not None else None
            acc = acc[:, -1, :] if acc is not None else None

        state = JointState.from_position(self._row(pos))
        vel = self._row(vel)
        acc = self._row(acc)
        if vel is not None:
            state.velocity = vel
        if acc is not None:
            state.acceleration = acc
        return state

    def _close_state_loop(self, robot_context, predicted_state: JointState) -> JointState:
        """Replace the predicted POSITION with the robot's real, latest joint
        feedback; keep the solver's own PREDICTED velocity/acceleration.

        ``robot_context.get_joint_pose()`` is a plain attribute read of the
        value already cached by the async joint_states subscriber callback —
        no wait, so no new lag. Without this, the MPC corrects toward a
        purely-imagined position that drifts from where the arm actually is
        once a horizon takes real wall-clock time to execute (observed on
        hardware: growing tracking error, unstable motion).

        Velocity is intentionally kept PREDICTED, not real, even though real
        velocity IS available (dsr_hw_interface2.cpp reads actual_joint_velocity
        from the same real-time struct as position — verified in source).
        Feeding the REAL velocity back here was tried and made things worse:
        the outgoing hardware safety clamp (JointSpeedStrategy) deliberately
        throttles commanded velocity below what the solver just planned: real
        velocity always lags. Reporting that throttled reality back as the
        solver's own state told it "you're going much slower than you
        decided", which the warm-started optimizer (only 25 iterations) can't
        reconcile each cycle without oscillating/diverging. The safety clamp
        legitimately uses real velocity (JointSpeedStrategy._clamp_velocities)
        — that's a downstream actuation limit, not the planner's own model of
        its trajectory, and the two must stay decoupled.
        """
        real_pose = robot_context.get_joint_pose()
        pos = torch.tensor([real_pose], dtype=self._dtype, device=self._device)
        state = JointState.from_position(pos, joint_names=self.solver.joint_names)
        state.velocity = predicted_state.velocity
        state.acceleration = predicted_state.acceleration
        return state

    def _init_robot_at_start(self, robot_context, start_state: JointState):
        """Seed the robot/visualization at the start configuration."""
        start_position = start_state.position[0].cpu().tolist()
        n = len(start_position)
        # joint_names=None -> RobotContext resolves them itself, inside the
        # same critical section as the command (see set_command's docstring:
        # reading robot_strategy.get_joint_name() here first would race a
        # concurrent strategy switch).
        robot_context.set_command(None, [[0.0] * n], [[0.0] * n], [start_position])
        self.node.get_logger().info(
            f"{self.get_planner_name()}: robot init'd at "
            f"{[f'{x:.3f}' for x in start_position]}"
        )

    def _send_command(self, robot_context, action_state: JointState):
        """Push a control action to the robot. Two shapes, from one solve each:

          - Single point: position ``[dof]`` or ``[1, dof]`` -> one
            JointTrajectory point (open-loop planners, RetargetController,
            and the free-running MPC loop popping native commands via
            optimize_next_action).
          - Command window: position ``[1, n, dof]`` — the multi-point
            command window of ONE solve (MPCController.step_paced, an
            optimize_action_sequence result; n = 2*interpolation_steps) -> ALL
            n points streamed as one multi-point JointTrajectory so the robot
            plays one full plan segment smoothly between re-plans. Never a
            stack of several solves' outputs.

        The old polluted multi-point shape — commands stacked from several
        solver pops by the paced producer/consumer send tick (_stack_actions)
        combined with the solver's own stacked horizon — was removed together
        with that loop.
        """
        pos_t, vel_t, acc_t = action_state.position, action_state.velocity, action_state.acceleration

        if pos_t.dim() == 3:
            position = pos_t[0].cpu().tolist()
            velocity = vel_t[0].cpu().tolist() if vel_t is not None else [[0.0] * pos_t.shape[-1]] * pos_t.shape[1]
            acceleration = acc_t[0].cpu().tolist() if acc_t is not None else [[0.0] * pos_t.shape[-1]] * pos_t.shape[1]
        else:
            p = pos_t[0] if pos_t.dim() > 1 else pos_t
            position = [p.cpu().tolist()]
            v = vel_t[0] if (vel_t is not None and vel_t.dim() > 1) else vel_t
            velocity = [v.cpu().tolist() if v is not None else [0.0] * len(position[0])]
            a = acc_t[0] if (acc_t is not None and acc_t.dim() > 1) else acc_t
            acceleration = [a.cpu().tolist() if a is not None else [0.0] * len(position[0])]

        # Atomic: no gap between load and send where a concurrent producer
        # (another call site touching the same RobotContext) could overwrite
        # the buffers first — see RobotContext.set_and_send_command.
        robot_context.set_and_send_command(None, velocity, acceleration, position)

from abc import ABC, abstractmethod
import bisect
import threading
import time
from collections import deque


class RobotState():
    '''
    This class is a enum to represent the robot state
    Currently useless but maybe later...
    '''
    RUNNING = 0
    STOPPED = 1
    READY = 2
    ERROR = 3
    IDLE = 4


class JointCommandStrategy(ABC):
    '''
    Base class (Strategy pattern) for a joint-command CONTROL MODE.

    A strategy describes HOW joints are commanded (emulator / joint speed /
    joint pose / ...), independently of WHICH robot is loaded. The robot-specific
    wiring (command/state/joint_states topics, joint names) comes from the
    RobotDescription passed in, so the same strategy works for any robot.
    '''

    def __init__(self, node, dt, description=None):
        self.node = node
        self.dt = dt
        self.description = description

        # Robot driver topics (empty dict-safe): the descriptor is the source.
        self.params = dict(description.strategy_params) if description is not None else {}

        # Joint names: descriptor fallback (cspace) until real joint_states arrive.
        self.joint_names = list(description.joint_names) if description is not None else []
        self.dof = len(self.joint_names)

        # Command buffers (filled by set_command()).
        self.position_command = []
        self.vel_command = []
        self.accel_command = []
        self.command_index = 0

        self.trajectory_progression = 0.0
        self.robot_state = RobotState.IDLE

        # Guards position_command/vel_command/accel_command/command_index/
        # joint_names/trajectory_progression/robot_state, plus any feedback
        # fields a subclass adds (joint_pose/joint_velocity/
        # current_joint_positions) — same callback that rewrites joint_names
        # usually rewrites those together. RLock: the atomic
        # RobotContext.set_and_send_command() holds it across a nested
        # set_command()+send_trajectrory() pair. Rank: below strategy_lock,
        # above nothing (leaf) — see robot_context.py's lock-order docstring.
        self.buffer_lock = threading.RLock()
        # Bumped by every set_command()/stop_robot() call. Lets a producer
        # detect that its buffers were superseded/cleared before it gets to
        # send them (RobotContext.set_and_send_command's expect_epoch), and
        # lets the emulator's playback thread detect preemption mid-playback.
        self._buffer_epoch = 0

        # Open-loop progression fallback (no state_topic configured).
        # trajectory_progression is only advanced by callback_trajectory_state
        # from the optional Float32 state_topic; without one it stays 0.0 and
        # SinglePlanner.execute() would spin forever. Completion is instead
        # derived from joint FEEDBACK by matching the measured joints to the
        # nearest sent waypoint — true progress even when the robot moves at
        # its own velocity limits rather than the stamped dt. Wall-clock is
        # used only as a no-feedback fallback/cap.
        self._progression_feedback_ready = False
        self._exec_start_mono = None
        self._exec_total_s = 0.0
        self._exec_positions = None
        self._reach_tol = 0.02

        # FollowJointTrajectory action tracking (strategy_params.action_topic,
        # e.g. /joint_trajectory_controller/follow_joint_trajectory). The
        # controller — not a waypoint-matching heuristic on joint feedback —
        # is the authority on when execution finished AND on failure: its
        # Result (Status SUCCEEDED / ABORTED / CANCELED + error_code +
        # error_string) drives _get_progression() so SinglePlanner.execute()
        # returns exactly when the path was done, or errors when it was not.
        # The action is the ONLY execution path for the real-robot strategies:
        # there is deliberately no topic-publish or joint-feedback/wall-clock
        # fallback behind it ("use the action, or error").
        self.action_topic = self.params.get('action_topic')
        self._action_client = None
        self._action_callback_group = None
        self._action_goal_handle = None
        self._action_result = None       # None | dict(succeeded, status, error_code, error_string)
        self._action_last_progress = None
        self._action_total_duration_s = 0.0
        # Monotonic tag on every send/cancel. Action callbacks capture the tag
        # of the goal they belong to and drop themselves when it no longer
        # matches (stale: preempted/canceled/superseded), so a late Result from
        # a previous goal can never corrupt the state of a newer one.
        self._action_seq = 0

        # Timestamped joint feedback buffer: (stamp_ns, position_list).
        # Populated by _record_joint_feedback() from the strategy's own
        # callback_joint_pose; consumed by get_joint_pose_at() for
        # time-synchronized queries (e.g. robot segmentation matching the
        # FK'd collision spheres to the depth image capture time).
        self._feedback_buffer = deque(maxlen=600)

    def _record_joint_feedback(self, stamp_ns, position):
        """Append a timestamped joint state sample to the feedback buffer.

        Called by each strategy's callback_joint_pose after name-remap.
        ``stamp_ns`` is the joint_states message header stamp in nanoseconds.
        """
        with self.buffer_lock:
            self._feedback_buffer.append((int(stamp_ns), list(position)))

    def get_joint_pose_at(self, stamp_ns, offset_ns=0):
        """Time-synchronized joint pose query.

        Returns the interpolated joint position at ``stamp_ns + offset_ns``,
        or None when the feedback buffer is still empty (e.g. no joint_states
        received yet). The depth image's capture timestamp is the expected use.

        Implementation: linear interpolation between the two bracketing
        buffer entries; clamps to the earliest or latest when the target
        falls outside the buffer window.
        """
        target = int(stamp_ns + offset_ns)
        with self.buffer_lock:
            buf = list(self._feedback_buffer)
        if not buf:
            return None
        stamps = [t for t, _ in buf]
        i = bisect.bisect_right(stamps, target)
        if i == 0:
            return list(buf[0][1])
        if i >= len(buf):
            return list(buf[-1][1])
        t0, p0 = buf[i - 1]
        t1, p1 = buf[i]
        if t1 <= t0:
            return list(p1)
        w = (target - t0) / (t1 - t0)
        return [p0[j] + w * (p1[j] - p0[j]) for j in range(len(p0))]

    @property
    def buffer_epoch(self) -> int:
        with self.buffer_lock:
            return self._buffer_epoch

    def set_command(self, joint_names, vel_command, accel_command, position_command) -> int:
        """Load new command buffers. Returns the new buffer epoch.

        Stores defensive copies — the caller's lists must not be aliased into
        the strategy (a caller mutating its own list after this call must not
        also mutate what the strategy/robot will execute).
        """
        with self.buffer_lock:
            self.position_command = [list(p) for p in position_command]
            self.vel_command = [list(v) for v in vel_command]
            self.accel_command = [list(a) for a in accel_command]
            self.joint_names = list(joint_names)
            self._buffer_epoch += 1
            return self._buffer_epoch

    def get_joint_name(self):
        with self.buffer_lock:
            return list(self.joint_names)

    def mark_progression_feedback(self, value):
        """Record REAL progress feedback (state_topic) and disable the fallback."""
        with self.buffer_lock:
            self.trajectory_progression = value
            self._progression_feedback_ready = True

    def _dilated_dt(self) -> float:
        """Output time-step (s) of the RE-TIMED trajectory.

        ``time_dilation_factor`` scales the trajectory's time parameterization
        on its way out: the stamped per-point step becomes
        ``self.dt / time_dilation_factor``, so the whole trajectory plays at
        ``1/factor`` of its nominal (interpolation_dt) duration. A factor < 1.0
        slows the motion down, > 1.0 speeds it up — the cuRobo convention
        (``MotionGenPlanConfig.time_dilation_factor``, curobo CHANGELOG). This
        re-stamping is what actually makes the parameter a speed control, which
        it was NOT in v2 (feedback cadence only).
        """
        factor = 1.0
        if self.node.has_parameter('time_dilation_factor'):
            factor = float(
                self.node.get_parameter('time_dilation_factor').get_parameter_value().double_value
            )
        if factor <= 0.0:
            return self.dt
        return self.dt / factor

    def mark_execution_start(self, num_points, positions=None):
        """Record the sent trajectory for REAL progression measurement.

        ``dilated_dt * (num_points - 1)`` is the nominal end only — the
        controller may actually play the path slower (velocity limits), so
        progression must be derived from joint FEEDBACK (nearest sent waypoint)
        rather than wall clock. ``positions`` is the full sent trajectory
        (per-waypoint joint rows) used for that match. The nominal duration and
        start time are kept only as a no-feedback fallback; it uses the SAME
        re-timed dt as the stamping so the fallback tracks the stamped plan."""
        with self.buffer_lock:
            self._exec_total_s = self._dilated_dt() * max(num_points - 1, 0)
            self._exec_positions = (
                [[float(v) for v in row] for row in positions] if positions else None
            )
            self._exec_start_mono = time.monotonic()

    def clear_execution_timer(self):
        with self.buffer_lock:
            self._exec_start_mono = None
            self._exec_total_s = 0.0
            self._exec_positions = None

    # ---- FollowJointTrajectory action execution (see self.action_topic) ----

    def _reset_action_execution(self):
        """Drop any in-flight action state for a new trajectory."""
        with self.buffer_lock:
            self._action_goal_handle = None
            self._action_last_progress = None
            self._action_result = None
            self._action_seq += 1

    def _action_feedback(self, feedback_msg, seq):
        """Track controller progress from the action Feedback's desired time (
        fraction of the nominal stamped duration covered; monotonic). Ignored
        when ``seq`` is not the active send (late feedback from a preempted
        goal)."""
        try:
            fb = feedback_msg.feedback
            t = fb.desired.time_from_start
            ts = float(t.sec) + float(t.nanosec) * 1e-9
        except Exception:
            return
        with self.buffer_lock:
            if seq != self._action_seq:
                return
            if self._action_total_duration_s > 0:
                self._action_last_progress = min(
                    1.0, max(0.0, ts / self._action_total_duration_s))

    def _action_goal_response(self, future, seq):
        if seq != self._action_seq:
            return
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            with self.buffer_lock:
                self._action_result = {
                    "succeeded": False,
                    "status": getattr(goal_handle, "status", None) if goal_handle else None,
                    "error_code": 0,
                    "error_string": "action goal rejected / not accepted",
                }
            return
        self._action_goal_handle = goal_handle
        self._action_result_future = goal_handle.get_result_async()
        self._action_result_future.add_done_callback(
            lambda fut, s=seq: self._action_result_done(fut, s))

    def _action_result_done(self, future, seq):
        if seq != self._action_seq:
            return
        try:
            from action_msgs.msg import GoalStatus
            resp = future.result()
            succeeded = (resp.status == GoalStatus.STATUS_SUCCEEDED)
            result = resp.result
            with self.buffer_lock:
                self._action_result = {
                    "succeeded": succeeded,
                    "status": resp.status,
                    "error_code": getattr(result, "error_code", 0),
                    "error_string": getattr(result, "error_string", "") or "",
                }
                self._action_last_progress = 1.0 if succeeded else None
        except Exception:
            pass

    def _cancel_action(self):
        """Cancel an in-flight action goal (from stop_robot)."""
        with self.buffer_lock:
            gh = self._action_goal_handle
            self._action_goal_handle = None
            self._action_result = {
                "succeeded": False,
                "status": None,
                "error_code": 0,
                "error_string": "canceled",
            }
            self._action_seq += 1
        if gh is not None:
            try:
                gh.cancel_goal()
            except Exception:
                pass

    def action_failure_summary(self) -> str:
        """Human-readable reason when the action finished non-successfully."""
        with self.buffer_lock:
            r = self._action_result
        if r and not r["succeeded"]:
            return (
                f"status={r['status']} error_code={r['error_code']} "
                f"{r['error_string']}".rstrip()
            )
        return ""

    def _ensure_action_client(self):
        """Lazily create the FollowJointTrajectory ActionClient. Health not
        checked here — only construction, which must not throw from a
        misconfiguration."""
        if self._action_client is not None:
            return self._action_client
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
        from rclpy.action import ActionClient
        from control_msgs.action import FollowJointTrajectory
        if self._action_callback_group is None:
            self._action_callback_group = MutuallyExclusiveCallbackGroup()
        self._action_client = ActionClient(
            self.node, FollowJointTrajectory, self.action_topic,
            callback_group=self._action_callback_group,
        )
        return self._action_client

    def _send_as_action(self, msg, positions):
        """Send a built JointTrajectory as a FollowJointTrajectory goal — the
        ONE execution path (no topic-publish fallback: use the action or error).

        The controller decides when execution actually finished: it replies
        Result Succeeded once the joints are within ``goal_tolerance`` by the
        end of the stamped path (plus any ``goal_time_tolerance``), which also
        clears the frozen-progression hang. Its Feedback (desired time from
        start / total stamped duration) provides progress while running, and a
        non-succeeded Result surfaces as a -1.0 progression (SinglePlanner
        aborts and reports ``action_failure_summary()``).

        Raises RuntimeError when the action is not configured or unreachable —
        the caller's error path stops the robot; it never silently publishes
        the trajectory on a raw topic.
        """
        self._reset_action_execution()
        n = len(msg.points)
        # NOTE: the controller may play the path slower than the stamped dt
        # (its own velocity limits / queueing); Feedback's desired time still
        # tracks the stamped parameterization, so the fraction stays valid.
        self._action_total_duration_s = self._dilated_dt() * max(n - 1, 0)

        if not self.action_topic:
            with self.buffer_lock:
                self._action_result = {
                    "succeeded": False,
                    "status": None,
                    "error_code": 0,
                    "error_string":
                        "strategy_params.action_topic not configured — the "
                        "FollowJointTrajectory action is the only execution path",
                }
            raise RuntimeError(
                f"{type(self).__name__}: strategy_params.action_topic must name "
                f"the FollowJointTrajectory action (e.g. "
                f"/joint_trajectory_controller/follow_joint_trajectory) — the "
                f"action is the only execution path"
            )

        client = self._ensure_action_client()
        if not client.server_is_ready():
            # Bounded wait: the controller_manager may still be coming up the
            # first time a trajectory is requested. If it never shows, error
            # instead of falling back to the topic.
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                time.sleep(0.05)
                if client.server_is_ready():
                    break
            if not client.server_is_ready():
                with self.buffer_lock:
                    self._action_result = {
                        "succeeded": False,
                        "status": None,
                        "error_code": 0,
                        "error_string": (
                            f"FollowJointTrajectory action server "
                            f"{self.action_topic} not available"),
                    }
                raise RuntimeError(
                    f"{type(self).__name__}: FollowJointTrajectory action server "
                    f"{self.action_topic} is not available; refusing to fall "
                    f"back to topic publish"
                )

        seq = self._action_seq
        from control_msgs.action import FollowJointTrajectory
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = msg
        self._action_goal_future = client.send_goal_async(
            goal,
            feedback_callback=lambda fb, s=seq: self._action_feedback(fb, s),
        )
        self._action_goal_future.add_done_callback(
            lambda fut, s=seq: self._action_goal_response(fut, s))
        self.mark_execution_start(
            n,
            positions=[pt.positions for pt in msg.points],
        )

    def _get_progression(self):
        """Progression read — the action is the ONLY source.

        With ``action_topic`` configured, the controller's state is the sole
        truth: 1.0 once it reports Succeeded (path completed), -1.0 on a
        non-success Result (so SinglePlanner.execute() stops and reports the
        failure), and the desired-time fraction from its Feedback while it is
        still running. 0.0 while a goal is accepted but no Feedback has arrived
        yet. There is deliberately no joint-feedback nearest-waypoint or
        wall-clock fallback: the action, or an error.
        """
        with self.buffer_lock:
            if self._action_result is not None:
                if self._action_result["succeeded"]:
                    return 1.0
                return -1.0
            if self._action_last_progress is not None:
                return self._action_last_progress
        return 0.0

    def get_joint_velocity(self):
        """Real, measured joint velocity, if this strategy's driver provides
        one (e.g. JointSpeedStrategy reads it from /dsr01/joint_states'
        actual_joint_velocity). Default: zeros — not every strategy has real
        velocity feedback (e.g. EmulatorStrategy has no physical driver).
        """
        return [0.0] * self.dof

    @abstractmethod
    def get_joint_pose(self):
        ...

    @abstractmethod
    def stop_robot(self):
        ...

    @abstractmethod
    def get_progression(self):
        ...

    @abstractmethod
    def send_trajectrory(self):
        ...

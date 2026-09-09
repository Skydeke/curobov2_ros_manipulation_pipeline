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

    def _pose_progress(self, pose):
        """Fraction [0, 1] of the sent path actually covered, from the index of
        the waypoint nearest the MEASURED joints (body frames stay in sync with
        the projected command rows). Velocity-limit-paced motion is captured
        naturally; wall-clock drift cannot race ahead of it."""
        rows = self._exec_positions or []
        if len(rows) < 2 or len(pose) != len(rows[0]):
            return None
        # ARRIVAL: measured joints within tolerance of the FINAL waypoint.
        # The tail points of an interpolated plan are nearly coincident, so a
        # pure nearest-match can stick at N-2 under feedback noise and peg
        # progression at (N-2)/(N-1) — stalling execution forever.
        if all(abs(p - g) <= self._reach_tol for p, g in zip(pose, rows[-1])):
            return 1.0
        best = 0
        best_d = None
        for i, row in enumerate(rows):
            d = sum((p - g) * (p - g) for p, g in zip(pose, row))
            if best_d is None or d < best_d:
                best_d = d
                best = i
            if d < 1e-12:
                break
        return best / (len(rows) - 1)

    def _get_progression(self):
        """Shared progression read: real feedback when present, otherwise the
        fraction of the path the measured joints have actually covered, with a
        wall-clock fallback/cap only when joint feedback is unavailable or
        stale."""
        with self.buffer_lock:
            if self._progression_feedback_ready:
                return self.trajectory_progression
            pose = getattr(self, 'joint_pose', None)
            last_mono = getattr(self, '_joint_states_last_mono', None)
            if pose is not None and last_mono is not None:
                if time.monotonic() - last_mono < 1.0:
                    prog = self._pose_progress(pose)
                    if prog is not None:
                        return prog
            if self._exec_start_mono is None or self._exec_total_s <= 0:
                return 0.0
            elapsed = time.monotonic() - self._exec_start_mono
            if elapsed >= self._exec_total_s + max(0.5, 0.15 * self._exec_total_s):
                return 1.0
            if elapsed < self._exec_total_s:
                return elapsed / self._exec_total_s
            return 0.99

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

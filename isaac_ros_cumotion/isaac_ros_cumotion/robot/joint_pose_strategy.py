from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import time

from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

from isaac_ros_cumotion.robot.joint_control_strategy import JointCommandStrategy, RobotState


class JointPoseStrategy(JointCommandStrategy):
    '''Joint-POSE control: command target POSITIONS (a JointTrajectory whose points
    carry positions only; the driver interpolates to each pose). Same message type
    and topics as joint_speed, but no velocity/acceleration setpoints — for drivers
    that follow position references rather than streamed speeds.

    Descriptor strategy_params: command_topic, state_topic (opt), joint_states_topic (opt).
    '''

    def __init__(self, node, dt, description=None):
        super().__init__(node, dt, description)
        # self.dt (base class) is already the resolved interpolation_dt —
        # curobo_ros is the single authority on trajectory pacing (see
        # resolve_interpolation_dt).

        command_topic = self.params.get('command_topic', '/execute_trajectory')
        self.pub_trajectory = node.create_publisher(JointTrajectory, command_topic, 10)

        state_topic = self.params.get('state_topic')
        if state_topic:
            self.sub_trajectory_state = node.create_subscription(
                Float32, state_topic, self.callback_trajectory_state, 10,
                callback_group=MutuallyExclusiveCallbackGroup())

        joint_states_topic = self.params.get('joint_states_topic')
        if joint_states_topic:
            self.sub_joint_state = node.create_subscription(
                JointState, joint_states_topic, self.callback_joint_pose, 10,
                callback_group=MutuallyExclusiveCallbackGroup())

        self.joint_pose = [0.0] * self.dof

        # Canonical cspace arm-joint names (e.g. joint_1..joint_7 on Kortex).
        # /joint_states may interleave other joints (e.g. a leading
        # finger_joint), so feedback is remapped by NAME below; a positional
        # slice would shift every arm joint and feed the planner a garbled,
        # often in-collision start state. Captured now because self.joint_names
        # gets overwritten by set_command()/callback.
        self._expected_joint_names = list(self.joint_names)

    def send_trajectrory(self):
        with self.buffer_lock:
            self.robot_state = RobotState.RUNNING
            msg = JointTrajectory()
            msg.joint_names = self.joint_names

            if len(self.position_command) == 0:
                self.trajectory_progression = 1.0

            # Re-timed step: time_dilation_factor scales the stamped duration
            # (see JointCommandStrategy._dilated_dt).
            stamp_dt = self._dilated_dt()
            for i in range(len(self.position_command)):
                point = JointTrajectoryPoint()
                point.positions = self.position_command[i]   # positions only
                point.time_from_start = Duration(
                    sec=int(stamp_dt * i), nanosec=int((stamp_dt * i % 1) * 1e9))
                msg.points.append(point)

            self.position_command = []
            self.vel_command = []
            self.accel_command = []
            self.trajectory_progression = 0.0

        self.pub_trajectory.publish(msg)
        self.mark_execution_start(
            len(msg.points),
            positions=[pt.positions for pt in msg.points],
        )

    def get_joint_pose(self):
        with self.buffer_lock:
            return list(self.joint_pose)

    def stop_robot(self):
        with self.buffer_lock:
            self.vel_command = []
            self.position_command = []
            self.accel_command = []
            self.command_index = 0
            self.trajectory_progression = 0.0
            self.robot_state = RobotState.STOPPED
        self.clear_execution_timer()
        self.pub_trajectory.publish(JointTrajectory())

    def callback_trajectory_state(self, msg):
        self.mark_progression_feedback(msg.data)

    def callback_joint_pose(self, msg):
        # Prefer a NAME-based remap against the canonical cspace arm joints when
        # the message names every one of them (order-agnostic; drops interleaved
        # non-arm joints like a leading finger_joint). Falls back to a
        # positional read when the names don't line up — see
        # JointSpeedStrategy.callback_joint_pose for the rationale.
        expected = getattr(self, '_expected_joint_names', None) or []
        remap = None
        if expected and msg.name:
            try:
                idx = {n: i for i, n in enumerate(msg.name)}
                if all(n in idx for n in expected):
                    remap = [idx[n] for n in expected]
            except Exception:
                remap = None

        n = self.dof or len(msg.position)
        with self.buffer_lock:
            self._joint_states_last_mono = time.monotonic()
            if remap is not None:
                self.joint_pose = [msg.position[i] for i in remap]
                self.joint_names = list(expected)
                feedback = self.joint_pose
            else:
                self.joint_pose = list(msg.position[:n])
                if msg.name:
                    self.joint_names = list(msg.name[:n])
                feedback = self.joint_pose
        have_stamp = (
            msg.header.stamp is not None
            and msg.header.stamp.sec != 0
            and msg.header.stamp.nanosec != 0
        )
        if have_stamp:
            self._record_joint_feedback(
                msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec,
                feedback,
            )

    def get_progression(self):
        return self._get_progression()

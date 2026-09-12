from isaac_ros_cumotion.robot.joint_control_strategy import JointCommandStrategy, RobotState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class GhostStrategy(JointCommandStrategy):
    '''
    This class is a strategie to control doosan robot with motion_gen
    '''

    def __init__(self, node, dt, description=None):
        super().__init__(node, dt, description)
        # create a publisher (namespaced under the owning node, e.g.
        # /curobo_server/trajectory) so the preview topic always lives with
        # the planner it tracks -- matching the <node>/planned_path pattern.
        self.pub_command = node.create_publisher(
            JointTrajectory, node.get_name() + '/trajectory', 10)
        # self.dt (base class) is already the resolved interpolation_dt —
        # curobo_ros is the single authority on trajectory pacing (see
        # resolve_interpolation_dt). The RViz preview now plays back at the
        # actual configured rate instead of a fixed 0.02s.



    def send_trajectrory(self):
        """
        Convert CuRobo JointState to ROS2 JointTrajectory message with multiple points.

        Args:
            joint_state (JointState): CuRobo JointState object that may contain multiple time steps.
            time_step (float): Time between each trajectory point in seconds.

        Returns:
            JointTrajectory: A ROS2 JointTrajectory message.
        """
        with self.buffer_lock:
            self.robot_state = RobotState.RUNNING
            joint_trajectory_msg = JointTrajectory()

            # Set joint names
            joint_trajectory_msg.joint_names = self.joint_names

            # Re-timed step: time_dilation_factor scales the stamped duration
            # (see JointCommandStrategy._dilated_dt) so the preview matches the
            # executed trajectory.
            stamp_dt = self._dilated_dt()
            # Create a list of JointTrajectoryPoints for every position in the JointState
            for i in range(len(self.position_command)):
                joint_trajectory_point = JointTrajectoryPoint()

                # Extract the i-th positions, velocities, and accelerations
                joint_trajectory_point.positions = self.position_command[i]
                joint_trajectory_point.velocities = self.vel_command[i]
                joint_trajectory_point.accelerations = self.accel_command[i]

                # Set efforts to an empty list (can be customized later)
                joint_trajectory_point.effort = []

                # Set the time_from_start for this point (incremented by time_step for each point)
                joint_trajectory_point.time_from_start = Duration(sec=int(stamp_dt * i),
                                                                nanosec=int((stamp_dt * i % 1) * 1e9))

                # Add the point to the trajectory message
                joint_trajectory_msg.points.append(joint_trajectory_point)

        self.pub_command.publish(joint_trajectory_msg)

    # GhostStrategy is visualization-only (RViz preview): the rest of the
    # JointCommandStrategy contract is inert.
    def get_joint_pose(self):
        return []

    def stop_robot(self):
        self.robot_state = RobotState.STOPPED

    def get_progression(self):
        return self.trajectory_progression



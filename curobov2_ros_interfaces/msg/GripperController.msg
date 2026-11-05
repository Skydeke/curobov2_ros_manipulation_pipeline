# One entry of the robot descriptor's `gripper_controllers:` map.
#
# A gripper is never a planning DOF, so its open/close positions cannot be
# reached by a joint-space plan when the finger joints are locked out of the
# cspace. This message carries everything a UI needs to send the descriptor's
# own open/close command to the controller instead.
string name
# "GripperCommand", "ParallelGripperCommand", "JointTrajectory" or
# "FollowJointTrajectory" (the last two are the same FollowJointTrajectory
# controller; the descriptor uses both spellings).
string type
# ros2_control controller name, e.g. "left_gripper_controller".
string controller
# Action namespace for a GripperCommand controller (default "gripper_cmd").
# JointTrajectory controllers use /<controller>/follow_joint_trajectory.
string action_ns
# Every joint the controller owns (ParallelGripperCommand / JointTrajectory).
string[] joints
# The single joint a GripperCommand's position applies to.
string command_joint
float64 open
float64 close
float64 max_effort
float64 time_from_start

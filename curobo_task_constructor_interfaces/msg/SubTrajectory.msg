# generic solution information
SolutionInfo info

# trajectory execution information, like controller configuration
TrajectoryExecutionInfo execution_info

# trajectory
moveit_msgs/RobotTrajectory trajectory

# planning scene of end state as diff w.r.t. start state
moveit_msgs/PlanningScene scene_diff

# curobo backend necessity (no MTC counterpart): ``trajectory`` above now
# carries the full validated plan and is replayed verbatim (MTC
# ExecuteTaskSolution parity), but the adapter still needs the segment's
# planning goal for the PLANNER switch (SetPlanner) and its allowed-collision
# links — neither of which the waypoints express. This YAML carries that
# planning goal (start_pose, goalsets, planner key, allowed-collision links)
# plus its scene ops.
string replay_goal_yaml

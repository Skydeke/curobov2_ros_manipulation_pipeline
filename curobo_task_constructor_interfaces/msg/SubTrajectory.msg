# generic solution information
SolutionInfo info

# trajectory execution information, like controller configuration
TrajectoryExecutionInfo execution_info

# trajectory
moveit_msgs/RobotTrajectory trajectory

# planning scene of end state as diff w.r.t. start state
moveit_msgs/PlanningScene scene_diff

# curobo backend necessity (no MTC counterpart): the validated plan is
# replayed from the curobo_server trajectory cache, whose key is the
# segment's planning goal (start joints + goalset poses/joint targets).
# Waypoints alone cannot reproduce that key, and re-solving instead would
# let a multi-candidate goal pick another IK branch mid-chain. This YAML
# carries the segment's planning goal (start_pose, goalsets, planner key,
# allowed-collision links) plus its scene ops; a cache miss fails the
# execution (force_cached) instead of re-solving.
string replay_goal_yaml

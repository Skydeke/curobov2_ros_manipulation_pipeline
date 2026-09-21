# Copyright (c) 2026
#
# Core planning request. Embedded by TrajectoryGeneration.srv,
# TrajectoryGenerationBatch.srv, and SendTrajectory.action (DRY — one schema,
# three surfaces).
#
# `goalsets` is the whole-task waypoint path — ONE entry (Goalset) per segment,
# exactly as before:
#   len(poses)==1                 -> fixed Cartesian waypoint
#   len(poses)==N                 -> candidate set; planner picks best (goalset)
#   target_joint_positions set    -> joint-space segment (plan_cspace)
# The first entry's waypoint is planned FROM `start_pose`; each subsequent
# segment starts where its predecessor left off (whole-task, single solve).
sensor_msgs/JointState start_pose

# The whole-task waypoint path (one Goalset per segment, in request order).
Goalset[] goalsets

# Per-request planner options. Defaults are honored; non-default values are
# validated per-planner (classic planners reject non-defaults).
PlanningOptions options

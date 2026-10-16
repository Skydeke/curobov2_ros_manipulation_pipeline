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

# CartesianPath solver options (MTC `solvers::CartesianPath` parity). These are
# the LIN q properties of the Cartesian planner: the planner is selected with
# SetPlanner CARTESIAN, and these three values tune the linear interpolation it
# generates. They describe the whole request (one task leg = one request) and
# are ignored by every other planner.
#
# WHY ON THE REQUEST and not the Goalset: the Cartesian planner plans ONE
# straight segment per stage/leg, and the client emits one request per leg, so
# the leg's interpolation parameters belong on the request, exactly where MTC
# keeps them (on the solver, not the waypoint).
#
# Sentinel: the defaults below mean "use the planner's own default", so a
# request from a non-Cartesian stage (which never sets them) is inert.
#
#   cartesian_step_size     max EEF translation between interpolated waypoints
#                           (m). <= 0 -> planner default (0.01).
#   cartesian_min_fraction  required feasible fraction of the linear path in
#                           [0, 1]. < 0 -> planner default (1.0). 0 accepts
#                           any path (free-space opt-out).
#   cartesian_precision     translational tolerance (m) under which an
#                           interpolated waypoint counts as ON the line.
#                           <= 0 -> planner default (0.001, MTC
#                           CartesianPrecision.translational).
float64 cartesian_step_size
float64 cartesian_min_fraction
float64 cartesian_precision

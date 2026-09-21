# Copyright (c) 2026
#
# Core planning result. Embedded by TrajectoryGeneration.srv,
# TrajectoryGenerationBatch.srv, and SendTrajectory.action (DRY — one schema,
# three surfaces).
bool success
string message                 # compact; structured detail in metadata/stats
sensor_msgs/JointState[] trajectory   # whole-task trajectory waypoints
float64 dt                     # time step between waypoints (s)

# Winner candidate index per `goalsets` entry; -1 when a segment failed or
# `goalsets` is empty.
int16[] selected_goal_index
# Winner restart (seed) index per `goalsets` entry; -1 on failure.
int16[] selected_seed_index
# Per-waypoint FK verification: 1 when this waypoint was reached within
# waypoint_tolerance, 0 when missed (per-waypoint failure insight).
int16[] waypoint_status

# Collision diagnostics (populated on failure)
StateCollisions start_state_collisions
StateCollisions end_state_collisions

# Planning stats — ALWAYS populated (structured), log optional.
PlanningStats stats

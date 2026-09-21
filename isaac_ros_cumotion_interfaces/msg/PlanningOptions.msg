# Copyright (c) 2026
#
# Per-request planning options. Embedded by TrajectoryGoal.msg (single core
# schema, three surfaces). Left at defaults for classic (single goal) planning;
# the node rejects non-default options on classic planners.
#
# num_seeds: parallel solver restarts per (problem, goal). 0 -> node/planner
#            default. The ONLY search knob (max_attempts removed).
uint16 num_seeds

# Max allowed per-waypoint tool-pose FK error (m) for success. A waypoint is
# reached when its FK error is within this tolerance.
float64 waypoint_tolerance

# Joint names pinned verbatim across their segment(s) — never freed, perturbed,
# or optimized (e.g. "finger_joint"). Carried value is honored exactly.
string[] exact_joints

# Emit the considered-trajectories detail block (node param is the default).
bool log_considered_trajectories

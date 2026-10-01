# Copyright (c) 2026
#
# ALWAYS populated (structured) on every plan — including classic (1 problem)
# and batch. Logging the detail block is optional (log_considered_trajectories);
# the structured stats are always present.
uint32 problems                 # batch members considered (1 for a single request)

# Candidate accounting (search-shape summary, never quality-clamped):
uint32 candidates_generated     # total IK candidates IK'd by the generator
uint32 candidates_solved        # candidate solves that produced a feasible state
uint32 candidates_pruned        # surplus candidates spilled / capped by caps
uint32 waypoints_planned        # total waypoints solved across all problems

# One row per (problem, segment, candidate, seed) — for ALL considered rows.
ConsideredTrajectory[] considered

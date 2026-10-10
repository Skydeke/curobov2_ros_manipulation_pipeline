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

# cuRobo's self-reported solve time for this problem (result.solve_time, s) —
# the SAME metric the native leg records as "Solve Time", so the ROS motion
# tables can carry a like-for-like row instead of only client wall time.
# (result.total_time, which includes additional solver overhead, is what the
# planner-phases log line reports as solver_total for the wrapper split.)
# NaN when no solve ran (request validation failure, exception path, cache
# hit, cross-problem rollup).
float64 solver_time_s

# Solver-reported position error for the winning candidate (mm) — the SAME
# metric the native leg records (winner seed's residual × 1000). Read off the
# planner's considered rows, so it needs no request flag and is always
# present. NaN when unavailable (no solve, winner unattributable).
float64 position_error_mm

# Motion duration the way the native leg reports it: measured on the KNOT
# trajectory (js_solution.knot_dt * (H_knots - offset), offset = 2 *
# interpolation_steps for bspline, else 1) — NOT on the interpolated
# waypoints, whose (n - 1) * dt counts boundary knots and overstates by
# offset * dt (0.5 s at dt=0.025). NaN when unavailable (runner omits row).
float64 motion_time_s

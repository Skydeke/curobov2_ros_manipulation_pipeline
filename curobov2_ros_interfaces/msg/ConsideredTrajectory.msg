# Copyright (c) 2026
#
# One considered row: a single (problem, segment, goalset candidate, seed)
# solver restart that was part of the whole-task search. Populated for ALL
# considered trajectories — not just the winner — so callers get full per-row
# insight (success, cost, per-waypoint error, attribution).
uint32 problem                # batch index (0 for a single request)
uint32 segment                # waypoint segment index (0 for classic/batch)
uint32 goalset_candidate      # candidate goal index within the segment (0 if N/A)
uint32 seed                   # solver restart (seed) index
bool   success                # this (problem, candidate, seed) restart succeeded
float64 cost                  # total winner cost for this row's trajectory
float64 waypoint_cost         # per-segment waypoint cost term contribution
float64 path_length           # path length of the considered trajectory (m)
float64 clearance             # min scene clearance over the trajectory (m)
float64 max_waypoint_error    # worst per-waypoint FK error over the trajectory (m)
float64 solve_time            # wall time for this row's solve (s)

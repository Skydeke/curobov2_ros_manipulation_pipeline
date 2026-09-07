# One segment of a Cartesian goal path: 1..N candidate poses.
#   length 1  -> fixed waypoint (degenerates to a single-goal plan)
#   length N  -> candidate set, the planner picks the best (goalset)
geometry_msgs/Pose[] poses
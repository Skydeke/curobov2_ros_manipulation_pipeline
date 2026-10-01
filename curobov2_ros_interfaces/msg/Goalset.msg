# One segment of a goal path: 1..N candidate poses (Cartesian) OR a
# joint-space target (joint-space planning).
#   len(poses)==1  -> fixed waypoint (degenerates to a single-goal plan)
#   len(poses)==N  -> candidate set, the planner picks the best (goalset)
geometry_msgs/Pose[] poses

# Target joint configuration for joint-space planning. When set (non-empty),
# this segment is planned in joint space (plan_cspace) with these joint
# values in cuRobo's active-DOF order.
float64[] target_joint_positions

# Cartesian axis hold flags for this segment:
# [theta_x, theta_y, theta_z, x, y, z] (1 = lock that axis along the path).
# Applied via ToolPoseCriteria (held along the length of this segment).
int8[] trajectory_constraints

# Flattened per-waypoint constraints for THIS segment:
# [theta_x0, theta_y0, theta_z0, x0, y0, z0, theta_x1, theta_y1, theta_z1,
#  x1, y1, z1, ...]
# (Note: historical field-name typo is preserved.)
int8[] trajectories_contraints
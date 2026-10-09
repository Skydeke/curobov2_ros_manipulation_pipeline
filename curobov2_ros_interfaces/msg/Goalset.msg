# One segment of a goal path: 1..N candidate poses (Cartesian) OR a
# joint-space target (joint-space planning).
#   len(poses)==1  -> fixed waypoint (degenerates to a single-goal plan)
#   len(poses)==N  -> candidate set, the planner picks the best (goalset)
geometry_msgs/Pose[] poses

# Target joint configuration for joint-space planning. When set (non-empty),
# this segment is planned in joint space (plan_cspace) with these joint
# values in cuRobo's active-DOF order.
float64[] target_joint_positions

# Joint names for `target_joint_positions`, parallel to it (optional).
#
# WHY THIS EXISTS
#   `target_joint_positions` has always been a bare float list that the node
#   resolves VERBATIM in the robot's active cspace order. A client therefore
#   had to know that order out-of-band — RoboBrain/curobo had to ship a robot
#   descriptor and reorder every goal itself, and a client that got it wrong
#   produced a start/target mismatch that read as a limit violation or a pile
#   of phantom self-contacts rather than as an ordering bug.
#
#   When this field is non-empty AND matches `target_joint_positions` in
#   length, the node maps the values onto the names and reorders them into
#   the robot's active cspace order itself. Empty (or length-mismatched) is
#   the legacy behaviour: the positions are taken to be already in
#   active-DOF order, as a name-less value must be.
#
#   Unknown or duplicated names are rejected with an error naming them, so a
#   typo fails at the request instead of planning toward the wrong joint.
string[] target_joint_names

# Cartesian axis hold flags for this segment:
# [theta_x, theta_y, theta_z, x, y, z] (1 = lock that axis along the path).
# Applied via ToolPoseCriteria (held along the length of this segment).
int8[] trajectory_constraints

# Flattened per-waypoint constraints for THIS segment:
# [theta_x0, theta_y0, theta_z0, x0, y0, z0, theta_x1, theta_y1, theta_z1,
#  x1, y1, z1, ...]
# (Note: historical field-name typo is preserved.)
int8[] trajectories_contraints
# One segment of a goal path: 1..N candidate poses (Cartesian) OR a
# joint-space target (joint-space planning).
#   len(poses)==1  -> fixed waypoint (degenerates to a single-goal plan)
#   len(poses)==N  -> candidate set, the planner picks the best (goalset)
geometry_msgs/Pose[] poses

# Simultaneous per-tool targets (bimanual): one pose per named tool frame,
# planned in the same solve. Parallel to `tool_poses`. When this is non-empty
# the segment is a multi-tool goal and `poses` is ignored.
string[] tool_frames
geometry_msgs/Pose[] tool_poses

# Target joint configuration for joint-space planning. When set (non-empty),
# this segment is planned in joint space (plan_cspace) with these joint
# values in cuRobo's active-DOF order.
float64[] target_joint_positions

# Cartesian axis hold flags for this segment:
# [theta_x, theta_y, theta_z, x, y, z] (1 = lock that axis along the path).
# Applied via ToolPoseCriteria (held along the length of this segment).
int8[] trajectory_constraints

# Per-tool Cartesian axis holds for a multi-tool segment (bimanual):
# flat 6 * N, parallel to `tool_frames` / `tool_poses`, groups in wire order
# [theta_x, theta_y, theta_z, x, y, z] (1 = lock that axis along the path).
# Empty means legacy behaviour: the single `trajectory_constraints` applies to
# the first named tool. When non-empty it must be exactly 6 * len(tool_frames).
int8[] tool_trajectory_constraints

# TF frame `poses` are expressed in; "" (or the robot base frame) means no
# transform is needed. The server resolves every other frame into the
# configured base link via TF before planning (SetMask.srv convention).
# Joint-space segments (target_joint_positions) are frame-free and ignore it.
string frame_id ""

# ---------------------------------------------------------------------------
# Ordered Cartesian waypoint path (MULTIPOINT / Cartesian Waypoint planner).
#
# A waypoint path is an ordered list of tool poses the end effector must pass
# through, in order, starting AFTER the segment start (the start pose itself is
# excluded; the planner prepends the start FK). Unlike `poses` (a candidate set
# where the planner picks the best one), EVERY waypoint below is an ordered
# target along one continuous trajectory. When `waypoints` or `tool_waypoints`
# is non-empty this segment is a waypoint path and `poses` is ignored.
# Waypoint poses are absolute poses in the robot base frame (`frame_id` is the
# legacy candidate-pose selector and does not apply to them).
#
# `waypoints` is a single-tool path: K poses, one per waypoint. When
# `tool_frames` is empty the first configured tool frame is used; otherwise it
# must name exactly that one tool.
geometry_msgs/Pose[] waypoints

# Simultaneous multi-tool waypoint path: K * N poses in WAYPOINT-MAJOR order
# (wp0 tool0, wp0 tool1, ..., wp1 tool0, ...), matching the native layout.
# Parallel to `tool_frames` (N frames); all N tools are tracked at every
# waypoint. Cannot be combined with `waypoints` or `target_joint_positions`.
geometry_msgs/Pose[] tool_waypoints

# Number of ordered waypoints (K) in `waypoints` / `tool_waypoints`. 0 means
# "infer from the pose array" (K = len(waypoints) or
# len(tool_waypoints) / len(tool_frames)); a non-zero value is validated
# against the arrays.
uint32 waypoint_count 0

# Tolerances for an ordered waypoint path. Every sampled point along the path
# must stay within the PATH tolerances; the final waypoint must additionally
# satisfy the tighter FINAL tolerances. Position is metres, orientation is
# radians (0, pi].
float64 path_position_tolerance 0.05
float64 path_orientation_tolerance 0.261799388
float64 final_position_tolerance 0.005
float64 final_orientation_tolerance 0.05

# Track orientation along a waypoint path. false ignores rotations (including
# the final waypoint) and plans a position-only path.
bool track_orientation true

# Maximum number of initial solve/check passes for a waypoint path. Must be
# >= 1; each pass uses the solver's default iteration budget.
int32 outer_iterations 4

# Results of a reachability-map solve: a uniform grid of IK goals on a plane.
#
# The server solves one IK per grid cell (all cells share the plane's
# orientation; only the position varies) and reports, for each cell, whether
# a collision-free joint configuration was found within tolerance, plus the
# joint configuration itself so a viewer can render the full robot at the
# selected cells.
#
# Per-cell fields reuse the same message types as the Ik / IkBatch services:
# joint_states / joint_states_valid mirror IkBatch's response exactly.
std_msgs/Header header

# Monotonically increasing solve counter, reset at node start.
uint32 map_id

# Plane the grid was solved against (header.frame_id). grid_size_x cells span
# plane_size_x along the plane's local X axis, grid_size_y cells span
# plane_size_y along the plane's local Y axis; the grid is centred on the
# plane pose.
geometry_msgs/Pose plane_pose
float64 plane_size_x
float64 plane_size_y
uint32 grid_size_x
uint32 grid_size_y

# One entry per grid cell, row-major with X fastest
# (index = iy * grid_size_x + ix). goals[i] is the requested tool pose in
# header.frame_id.
geometry_msgs/Pose[] goals

# One JointState per grid cell, same order as goals. Each carries the joint
# names and the solved positions for that cell; a cell whose IK did not
# converge has an empty position array (joint_states_valid[i].data == false).
sensor_msgs/JointState[] joint_states

# Per-cell IK convergence (IK succeeded AND is collision-free), same order as
# joint_states.
std_msgs/Bool[] joint_states_valid

# Convenience counts / timing.
uint32 n_total
uint32 n_solved
float64 solve_time_ms
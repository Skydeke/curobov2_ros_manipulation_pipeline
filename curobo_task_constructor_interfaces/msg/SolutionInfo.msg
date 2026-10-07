# One considered (attempted) solution — success AND failure — of a stage.
# The executor publishes one of these per stage attempt as compute() runs,
# matching MTC's choice to keep failed attempts visible for debugging.
string task_id
uint32 stage_id
string stage_name
uint32 solution_id
float64 cost
bool success
string comment               # failure hint / human note
string planner_id            # planner backing this stage (e.g. "MULTIPOINT")
bool ranked true             # ranked (stored, lifted, executable) vs a
                             # best-of-N seed that succeeded but lost: shown
                             # and counted, never ranked or executed
visualization_msgs/MarkerArray markers   # optional debug markers (may be empty)
# Full-clone trajectory playback (MTC TaskDisplay equivalent): waypoints of
# this attempt (empty on failure) plus the FK tool pose per waypoint.
sensor_msgs/JointState[] trajectory
geometry_msgs/Pose[] tool_poses
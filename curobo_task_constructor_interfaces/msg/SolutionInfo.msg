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
visualization_msgs/MarkerArray markers   # optional debug markers (may be empty)
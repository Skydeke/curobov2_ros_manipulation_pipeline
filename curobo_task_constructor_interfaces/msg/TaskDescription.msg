# The built task graph, published once per task so a panel can render
# structure before/without a solve (MTC TaskDescription analog).
#
# Carry the same flat StageSpec[] representation as the goal (root = the
# stage with parent_id == id).
string task_id
StageSpec[] stages
uint32 stage_count
bool valid              # init() succeeded; false + comment when it failed
string comment
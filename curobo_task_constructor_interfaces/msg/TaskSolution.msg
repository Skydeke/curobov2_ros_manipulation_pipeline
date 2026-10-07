# One complete task solution (a full chain, MTC Solution equivalent).
#
# The executor ranks every complete root solution by cost; the node publishes
# one of these per solution right after a solve and keeps the solution itself
# (segment plan requests + scene ops) in a bounded registry. Sending the
# solution back via ExecuteSolution drives exactly the stored plan — no
# replanning, like MTC's ExecuteTaskSolution — so any ranked alternative can
# be executed, not just the winner.
string task_id
uint32 solution_index       # rank order at solve time (0 = cheapest)
float64 cost
string[] stage_names        # leaf stage names in chain order
uint32[] stage_ids          # wire stage id per leaf
uint32[] solution_ids       # winning attempt id per leaf

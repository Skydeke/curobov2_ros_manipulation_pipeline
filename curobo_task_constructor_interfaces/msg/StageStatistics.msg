# Dynamically changing information for a stage, MTC StageStatistics shape.
uint32 id                    # unique stage id within task
uint32[] solved              # successful solution ids, increasing cost
uint32[] failed              # failed attempt ids
uint32 num_failed            # number of failed attempts
float64 total_compute_time   # total computation time in seconds
# --- curobo extensions (per-stage rollup the panel shows) ---
string task_id
string stage_name
string stage_type            # registered stage type
uint32 attempt_count         # planner calls + records from call-less computes
uint32 success_count         # ranked solutions stored
float64 last_cost            # cheapest ranked solution cost (inf if none)

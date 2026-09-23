# Rollup statistics for one stage over the last task solve (MTC StageStatistics
# analog). Published once per stage after the solve completes.
string task_id
uint32 stage_id
string stage_name
string stage_type            # registered stage type (for panel icon/grouping)
uint32 attempt_count
uint32 success_count
float64 last_cost
float64 total_compute_time
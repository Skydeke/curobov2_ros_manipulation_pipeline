# Rolled-up statistics for a whole task, MTC TaskStatistics shape:
# one entry per stage, including the task stage itself.
string task_id               # unique id of generating task
StageStatistics[] stages     # list of all stages, including the task stage

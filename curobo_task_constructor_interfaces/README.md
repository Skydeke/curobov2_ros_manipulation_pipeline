# curobo_task_constructor_interfaces

msg/srv/action interfaces for the in-process task constructor:
`StageDescription`, `TaskDescription`, `TaskStatistics`, `Solution`,
`SubSolution`, `SubTrajectory` and the `ExecuteTaskSolution` action.

Planning is local; only execution crosses the wire. Mirrors
`moveit_task_constructor_msgs`.

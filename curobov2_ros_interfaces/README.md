# curobov2_ros_interfaces

msg/srv/action interfaces for `curobov2_ros`: `TrajectoryGoal`/`Goalset`,
`TrajectoryGeneration[Batch]`, `Ik`/`Fk[Batch]`, the scene services
(`AddObject`, `AttachObject`, `GetSceneObjects`, ...), `SetPlanner` and the
`SendTrajectory` action.

`Goalset` supports an ordered Cartesian waypoint path as well as the legacy
candidate `poses`: `waypoints` (single tool, K poses) or `tool_waypoints`
(waypoint-major, K * N poses parallel to `tool_frames`), plus
`waypoint_count` and the path/final tolerances, `track_orientation` and
`outer_iterations`. These are consumed by the `MULTIPOINT` (4, `Cartesian
Waypoint`) planner; a waypoint path never mixes with `target_joint_positions`.

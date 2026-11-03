# curobo_task_constructor_rviz

RViz2 panel for the cuRobo task constructor. It renders the stage tree and
per-stage statistics, browses the planned solutions and publishes/drives a
selected solution through `ExecuteTaskSolution`.

The shipped config in `rviz/task_constructor.rviz` also adds the shared
`curobov2_ros_rviz/CuroboPlanningDisplay`, subscribed to
`/curobo_task_constructor/solution_trajectory`. Trajectory link poses are
fetched from `/curobo_server/fk_batch`, and attached-object spheres from
`/curobo_server/collision_spheres` (`Show Attached`).

The target gizmo and goal-state ghost are disabled in this config
(`Gizmo Visible: false`, `Show Goal State: false`): the task constructor has no
Curobo panel to push tool frames, so a target marker would otherwise sit at the
display's default pose. The display still shows the live robot, the selected
solution with trail/scrubber, and the attached object.

Change `Planner Node` and `Collision Spheres Topic` together if the task
constructor is pointed at a differently named server. The server must expose
`fk_batch` with `link_names`/`link_poses` (the current curobov2_ros build).

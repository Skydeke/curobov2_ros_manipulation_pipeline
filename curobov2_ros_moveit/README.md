# curobov2_ros_moveit

MoveIt 2 planning plugin that exposes the cuRobo server as a MoveIt planning
pipeline (`curobov2_ros/CumotionPlanner`). It translates a `MotionPlanRequest`
into a cuRobo solve and offers the `cuMotion/JointSpace` planner id.

## Planning-scene synchronisation

The plugin keeps the MoveIt world and the cuRobo collision scene in step. Both
directions run before every plan and are best-effort: a failed sync is logged
and the plan still proceeds.

**MoveIt → cuRobo (`syncPlanningScene`)**

- World collision objects become cuRobo objects, one object per shape
  (`{id}` for a single-shape object, `{id}__{i}` for multi-shape ones).
- Boxes, spheres, cylinders, capsules and in-memory `shapes::Mesh` geometry are
  supported; meshes travel inline through `add_object`, so no temporary file is
  written.
- Attached bodies are kept registered on the server and attached/detached via
  `attach_object`/`detach_object`.
- An unchanged object is left alone, so a static scene does not churn the
  server every plan.

**cuRobo → MoveIt (`syncObstaclesFromServer`)**

- The plugin asks `get_scene_objects` for the solver's *full* scene — type,
  pose, size, colour and attached flag — and mirrors every server-only object
  into the MoveIt world as a `CollisionObject`.
- That includes objects the plugin never sent: perception/ESDF obstacles, task
  constructor or grasp-orchestrator objects, and plain `ros2 service call
  add_object` writes.
- Capsules (which `moveit_msgs` cannot express) are mirrored as the equivalent
  cylinder + two end spheres; file-backed meshes are loaded with
  `geometric_shapes`, and inline meshes use the geometry the plugin itself
  forwarded.
- A mirror removed on the server is removed from MoveIt; a mirror deleted from
  the MoveIt world is removed on the server, so the two directions do not
  resurrect old objects.
- Server-attached objects are deliberately **not** mirrored into the MoveIt
  world: `get_scene_objects` does not report the attach link, and a free-floating
  copy of a held payload would make MoveIt collide with it twice. The forward
  direction remains responsible for MoveIt `AttachedBody` state.

Both directions are part of the planning plugin itself; no extra node or
external synchronisation script is needed.

The plugin discovers the server's services under
`<cumotion_service_namespace>/...` (default `curobo_server`); set that MoveIt
parameter to `unified_planner` when using `curobov2_ros/gen_traj.launch.py`,
whose node is named `unified_planner`. `cumotion_service_timeout` bounds each
service round-trip (default 60 s, sized for the first cold-start plan).

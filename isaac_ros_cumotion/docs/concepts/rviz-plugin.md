# RViz Plugin

The graphical interface is provided by the companion package [`curobo_rviz`](https://github.com/Lab-CORO/curobo_rviz) (pulled in by `my.repos`). It is a set of RViz 2 panels and displays that talk to the `unified_planner` node — everything the panels do goes through the same public services and action documented in [ROS Interfaces](ros-interfaces.md), so the GUI and the CLI are always interchangeable.

**Status**: a functional debugging and development tool; some conveniences (object preview, persistence) are still on the package's roadmap.

## Launching

The default launch starts RViz with the plugin configuration:

```bash
ros2 launch curobo_ros gen_traj.launch.py          # gui:=true is the default
```

Expect RViz to appear before the planner finishes its warmup (roughly 25–35 s): the panels stay inert until the `node_is_available` parameter flips to `true` — the panel polls it for you.

![RViz at startup](img/init_rviz.png)

## Components

The plugin registers three components (`rviz2_plugin.xml`):

| Component | Kind | Role |
|---|---|---|
| `CuroboPanel` | Panel | The whole planner UI in one dock: Context, Planning, Joints, Manipulation, Scene Objects |
| `AddObjectsPanel` | Panel | Add and remove collision obstacles (also a tab of the Curobo panel) |
| `TargetDisplay` | Display | Generic 6-DOF target pose (planner/MPC logic lives in the Curobo panel) |

### Target pose — the interactive marker

`TargetDisplay` shows a draggable 6-DOF target in the 3D view. The Planning
tab's pose spin boxes (X/Y/Z, roll/pitch/yaw) stay synchronized
with the target in both directions: drag the marker or type coordinates,
whichever is easier. The pose is expressed in the robot base frame.

The display is planner-agnostic — all planner/MPC logic lives in the panel.

### Main panel (`CuroboPanel`)

![Control panel](img/control_panel.png)

One dock, five tabs, laid out as MoveIt's Motion Planning panel: **Context**,
**Planning**, **Joints**, **Manipulation**, **Scene Objects**. Every group title
is MoveIt's own wording (`Planning Library`, `Query`, `Options`, `Commands`,
`Current Scene Objects`, `Add/Remove scene object(s)`, `Object status`), and each
MoveIt group with no honest curobo equivalent is absent rather than stubbed —
each omission is named in the source next to the code it would have gone in.

| Tab | What it holds |
|---|---|
| **Context** | "Planning Library": **Planning Group:** (the planner node — the only editable copy in the panel), trajectory type, control strategy. Plus one ungrouped collision-sphere checkbox. |
| **Planning** | "Query": the goal pose as position and orientation-RPY triples. "Options": time dilation, and a read-only echo of the planner node. "Commands": Clear / Compute / Execute / Compute + Execute / Cancel, then the status line. |
| **Joints** | MoveIt's Joints widget: a bare "Group joints:" label over a `Joint │ Current │ Target` table, where **Target is a draggable progress bar** spanning that joint's own limits. Then a "Commands" group with the request strip (`Set from current` / `Zero` / `Check` / `Clear locks`) and the run strip (`Plan` / `Plan + send` / `Stop`). |
| **Manipulation** | The same Joints widget, scoped to the gripper joints only, plus a "Commands" group with `Gripper:` (Open / Close / Reset to current) and `Execute:` (Plan / Plan + send / Stop). Open and Close command the upper and lower **limits** — nothing in the cspace says which way a gripper travels — so the bar stays draggable for an exact opening. |
| **Scene Objects** | "Add/Remove scene object(s)": the object form. "Current Scene Objects": a live, flat list from `get_obstacles` with Refresh / Remove / Clear / Attach / Detach on the selection. "Object status": one status line. |

**The draggable joint bar.** The `Target` cell is not a spin box. It is painted
as a progress bar spanning the joint's reported limits and is edited by dragging
it — a port of MoveIt's `ProgressBarDelegate` /
`ProgressBarEventFilter`, because that bar is the single most recognisable thing
about MoveIt's joint view. Click a cell and drag to set the target. It is shown in
radians, and a joint the model reports no limit for is left as plain text rather
than being given an invented range. `Check` runs the server's own FK/collision
validation on the target configuration before you plan.

**Planner node selection.** The Context tab's "Planning Group:" dropdown is the
only place it can be changed. It binds every tab's service/action clients —
parameters, `set_planner`, `generate_trajectory`, `execute_trajectory`,
`mpc_goal`, `get_obstacles`, `add_object`, `remove_object`, `attach_object`,
`get_joint_info` — and re-probes readiness immediately. Its list is the live ROS
node graph, refreshed every 2 s, and it is editable so a planner that is not up
yet can still be named. The Planning tab's "Planner node:" row is a read-only
echo of it. Saved to the RViz config as `planner_node_name` under the `Context`
child.

**Trajectory type.** Also on the Context tab, wired to `set_planner`: Classic,
MPC. The indexes map 1:1 to the `SetPlanner` enums (0 = Classic, 1 = MPC). The
combo is put back if the server refuses, so what it shows is always what the
server is doing.

**Speed (Time dilatation).** In the Planning tab's "Options" group. Sets the
`time_dilation_factor` parameter; a real speed control (stamped dt =
`interpolation_dt / tdf`, so the field speeds up / slows down every sent
trajectory).

**Two object lists.** The "Add/Remove scene object(s)" form keeps its own list of
what *it* added, and the "Current Scene Objects" group below it is the
server's truth from `get_obstacles` — whoever created the object, including the
task constructor and any `ros2 service call`. Remove and Clear act on the
authoritative list. The form's own list is the older path and its tooltip says
so.

### MPC live tracking

When the trajectory-type combo is on **MPC**, "Generate and Send" hands the
reactive loop to the panel (which owns it): the panel switches the planner to
MPC via `set_planner`, sends the `execute_trajectory` action goal with the
target as the goal set and starts streaming the target pose to
`/<planner>/mpc_goal` at 10 Hz. Dragging the target then retargets the robot
**live** while it moves — this is the quickest way to feel what closed-loop
control does. Stopping the robot cancels the goal. See [MPC
Implementation](mpc-implementation.md).

### Multiple target displays

The panel scans every `TargetDisplay` in the display tree (including ones
nested inside display Groups) so the primary (first) display can be found
after RViz reloads a config; classic and MPC plan only toward that primary
display. The old Multipoint planner that turned each extra display into a
waypoint is removed.

### Objects panel (`AddObjectsPanel`)

![Object manager](img/object_manager.png)

Adds obstacles through `add_object` and removes them through `remove_object`. The type combo (Cube, Sphere, Capsule, Cylinder, Mesh) maps to the service constants — `CUBOID=0`, `SPHERE=1`, `CAPSULE=2`, `CYLINDER=3`, `MESH=4` — and the dimension fields follow the same semantics as the service (see [ROS Interfaces](ros-interfaces.md)): full extents for a cuboid, radius for a sphere, radius + length for capsule/cylinder, scale for a mesh. A mesh object needs a mesh path.

Added objects appear immediately in the Scene Objects tab's "Current Scene
Objects" list, which asks the server via `get_obstacles` rather than tracking the
form's own additions — so objects added by the task constructor or from the CLI
show up too, and are removable from the UI. Select one there and use Remove, or
Clear to empty the scene of every added object.

Known limitations: no preview before adding, no edit-after-add (MoveIt's "Change
object pose/scale" group is absent — curobo has no object pose/scale service),
and colors are unreliable.

### Trajectory preview

Planned trajectories are replayed by a translucent "preview" robot (namespace `preview/`), fed by the ghost strategy's namespaced topic (`<node>/trajectory`, e.g. `/curobo_server/trajectory`). This happens for every plan, regardless of whether you execute it.

![Trajectory preview](img/trajectory_preview.png)

## Typical workflow

1. Launch, wait for warmup to finish.
2. Drag the arrow to a reachable pose.
3. **Generate Trajectory** — check the preview robot's path.
4. Adjust obstacles or the pose as needed; regenerate.
5. **Send Trajectory** to execute (on the emulator first — see [Tutorial 4](../tutorials/04-robot-execution.md)).

![Organized RViz layout](img/rviz_organized.png)

## Troubleshooting

- **Panels stay grey / buttons do nothing** — the planner is still warming up, or the node crashed; check `ros2 param get /unified_planner node_is_available` and the launch terminal.
- **No target in the 3D view** — add or enable the `TargetDisplay` display (Displays → Add → By display type); the panel logs a warning until it finds one.
- **"Trajectory type" switch fails** — only the Classic and MPC options are wired to a real planner; the node's other planners (joint-space, retarget) are reachable from the CLI only.
- **Wrong planner being driven** — every tab binds to the node selected in the **Context** tab's "Planning Group:" dropdown; make sure it matches the running planner. The Planning tab's "Planner node:" row is a read-only echo of it.

## Related pages

- [ROS Interfaces](ros-interfaces.md) — the services behind every button
- [Tutorial 1: First Trajectory](../tutorials/01-first-trajectory.md) — the same workflow from the CLI
- [MPC Implementation](mpc-implementation.md) — what live tracking does underneath

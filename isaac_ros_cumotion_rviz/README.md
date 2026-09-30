# isaac_ros_cumotion_rviz

RViz 2 companion package for `isaac_ros_cumotion`: panels and displays that drive
the planner through the same public services/actions as the CLI, plus displays
for reachability maps, the voxel grid and full-robot trajectory playback.

**Status:** working and used for daily debugging/development. See the notes at
the end of each component for known limitations and roadmap items.

## Launching

```bash
ros2 launch isaac_ros_cumotion_rviz isaac_ros_cumotion_rviz.launch.py
```

The launch forwards `planner_node_name` and `base_link` as node parameters.
Note that `planner_node_name` only takes effect if the running RViz node
exposes it as a parameter; the value saved in the RViz config (the Curobo
panel's Context tab) is what the panel actually uses, and it wins when both are
present. Time dilation is no longer a launch argument — it is a live control on
the panel's Planning tab.

## Registered plugins (`rviz2_plugin.xml`)

| Component | Kind | Role |
|---|---|---|
| `isaac_ros_cumotion_rviz/CuroboPanel` | Panel | The whole planner UI in one dock: Context, Planning, Joints, Manipulation, Scene Objects |
| `add_objects_panel/AddObjectsPanel` | Panel | Add/remove scene obstacles (also a tab of the Curobo panel) |
| `isaac_ros_cumotion_rviz/TargetDisplay` | Display | Generic 6-DOF target pose (planner/MPC logic lives in the Curobo panel) |
| `isaac_ros_cumotion_rviz/SparseVoxelGridDisplay` | Display | Renders the mapper's occupied voxels |
| `isaac_ros_cumotion_rviz/CuroboTrajectoryDisplay` | Display | Animates a full robot body along a `JointTrajectory` |
| `isaac_ros_cumotion_rviz/ReachabilityMapDisplay` | Display | Solves + visualises a reachability map on a plane |

## CuroboPanel

### Current state
One dock with five tabs, laid out like MoveIt's Motion Planning panel. This is
the only curobo planner panel — the planner-args panel it reuses as its Planning
tab (`PlanningTab`, formerly the standalone `RvizArgsPanel`) is deliberately not
registered separately, so it cannot be added as a dock of its own.

Every group title below is MoveIt's **own** wording, transcribed from
`motion_planning_rviz_plugin_frame.ui`. MoveIt's tabs are Context, Planning,
Joints (inserted programmatically at index 2), Scene Objects, Stored Scenes,
Stored States, Status, Manipulation; the five kept here are the ones curobo can
fill honestly, in MoveIt's order. A group whose control would be permanently
dead is left out rather than stubbed, and each omission is named in the source
next to the code it would have gone in.

- **Context** — MoveIt's "Planning Library" group: "Planning Group:" (the planner
  node), trajectory type, and the control strategy. Below it, one ungrouped
  "Publish collision spheres" checkbox. MoveIt's other two Context groups,
  "Warehouse" (the planning-database host/port) and "Workspace" (scene centre
  and size), are absent — curobo has neither a planning database nor a service
  that reports a workspace.
- **Planning** — "Query" (the goal pose as `Goal State:` position + orientation
  RPY triples), "Options" (time dilation, read-only echo of the planner node) and
  "Commands" (Clear / Compute / Execute / Compute + Execute / Cancel, then the
  status line). MoveIt's "Path Constraints" group is absent: `TrajectoryGoal` has
  no path-constraint field, and only one of MoveIt's five "Options" numbers
  exists on the curobo path — planning time and planning attempts have no
  equivalent.
- **Joints** — MoveIt's Joints widget exactly: a bare "Group joints:" label over
  a `Joint | Current | Target` table, with **no** group box, because MoveIt's
  Joints tab has none either. The `Target` cell is a **draggable progress bar**,
  not a spin box — see below. Under it, a "Commands" group holding the request
  strip (`Set from current` / `Zero` / `Check` / `Clear locks`) and the run strip
  (`Plan` / `Plan + send` / `Stop`). MoveIt's "Nullspace exploration:" sliders are
  absent: curobo's `generate_trajectory` takes a goal configuration, not a
  nullspace goal.
- **Manipulation** — the gripper, built from the same MoveIt widget as the Joints
  tab: a "Group joints:" table holding only the *gripper* joints (name contains
  finger/gripper/claw), each with a draggable bar, plus a "Commands" group with
  `Gripper:` (Open / Close / Reset to current) and `Execute:` (Plan / Plan + send
  / Stop). Open and Close command the upper and lower **limits** — nothing in the
  cspace says which way a gripper travels, so the buttons say what they do
  numerically and the bar stays draggable for an exact opening.
  MoveIt's own Manipulation tab is *object detection* (`Detected Objects`,
  `Support Surfaces`, `ROI`, firing `moveit_ros_perception`), which this server
  cannot support at all; the joint widget is the closest honest thing it owns.
  The tab no longer persists anything: the gripper joint set is re-derived from
  the cspace every 2 s, so an old config's `Manipulation: finger_joint` entry is
  ignored.
- **Scene Objects** — MoveIt's "Current Scene Objects" group over a live,
  flat list from the server's `get_obstacles` service, with a strip that acts on
  the selection (Refresh / Remove / Clear / Attach / Detach), then the
  "Object status" group. Above it is MoveIt's "Add/Remove scene object(s)"
  group — the `AddObjectsPanel` form. MoveIt's "Change object pose/scale" and
  "Scene Geometry" (`[Publish]`/`[Export]`/`[Import]`) groups are absent: curobo
  has no object pose/scale edit service and no `PlanningSceneMonitor` to publish,
  export or import. The list is flat rather than MoveIt's three-way split
  (present/attached/disabled) because `get_obstacles` answers with newline-joined
  names and no state, and because `attach` only flags an object out of the
  rasterisation — an attached object still comes back from `get_obstacles`.

**The draggable joint bar.** `ProgressBarDelegate` / `ProgressBarEditor` in
`progress_bar_delegate.{hpp,cpp}` is a faithful port of MoveIt's
`ProgressBarDelegate`, `ProgressBarEditor` and `ProgressBarEventFilter` from
`motion_planning_frame_joints_widget.cpp` (BSD-3, provenance retained in the
header). The cell is painted with `QStyle::CE_ProgressBar` at
`1000*(value-min)/(max-min)` with the value drawn on top of the fill, and a click
opens a drag editor that sets the value from the mouse x position. This is why
there is no `QSlider` on the Joints tab: MoveIt has no slider in its joint table,
and a slider is what made the tab unrecognisable before.

Five deliberate deviations, each documented at its call site and listed in full in
the delegate's header. All of them follow from one difference: MoveIt's widget
reads a `RobotState`, where a joint's bounds, live value and target are three
views of one object, while a `QTreeWidgetItem` stores each role independently and
nothing keeps them in step.

1. **Radians only.** MoveIt converts `REVOLUTE` joints to degrees for display.
   curobo's `GetJointInfo.srv` reports no joint type — only cspace names and
   position limits — so there is nothing to switch on, and every other number in
   the panel and in the server API is radians.
2. **Reads `Qt::EditRole`, not `DisplayRole`,** for both the fill and the caption.
   A committed drag writes `Qt::EditRole` via `model->setData` and
   `QTreeWidgetItem` stores it verbatim, leaving `DisplayRole` stale; a delegate
   reading `DisplayRole` would paint a bar that never moves. MoveIt can get away
   with `DisplayRole` because its two roles are projections of the same state.
3. **`createEditor` returns `nullptr` off-column.** MoveIt returns the base-class
   editor, which for a `QTreeView` is a line edit — so a double-click on a joint
   *name* in MoveIt would offer to rename it. `Qt::ItemIsEditable` is an item-level
   flag and cannot confine editing to one column, so `nullptr` is what confines it,
   together with the column test in `ProgressBarEventFilter`.
4. **No bar without bounds.** A joint the model reports no limit for gets no
   `VariableBoundsRole`, so the delegate leaves the cell as plain, non-draggable
   text rather than inventing a range (±1e6 was dropped: it let a ±inf joint be
   dragged to a number nobody can act on).
5. **The fill is clamped, and the editor divides by its width only when that
   width is non-zero.** Neither is reachable from MoveIt's data (its bounds always
   contain its value, and a laid-out cell has a width).

> **Watch the argument order.** `QTreeWidgetItem::data` is `(column, role)` but
> `QTreeWidgetItem::setData` is `(column, role, value)` — the value goes *last*.
> `setData(column, value, role)` is the shape neither Qt API has, and it does not
> merely fail to compile: the target column is a `double`, `double` converts
> implicitly to `int`, so it binds as `setData(column, role=(int)value,
> value=QVariant(role))` and writes the role into the cell *as its data*, under a
> nonsense role — with no diagnostic at all, and a bar that never moves again.
> `check_panel.py` rule 12c checks the order at every call site.

**Planner node selection:** the Context tab's dropdown — the first row of the
"Planning Library" group, labelled "Planning Group:" — binds every tab's
service/action clients to a named planner node (default `curobo_server`). It is
the **only** editable copy in the panel; the Planning tab's "Planner node:" row is
a read-only echo, and the other tabs have no control for it at all. Changing the
selection immediately rebinds everything and re-probes readiness. The list is the
live ROS node graph, refreshed every 2 s, and is editable so a planner that is not
up yet can still be named. The stored selection is persisted via
`planner_node_name` in the saved config.

**Trajectory type:** chosen on the Context tab, which makes the `set_planner`
call and puts the combo back if the server refuses — so what the dropdown shows is
always what the server is doing.

**MPC mode:** with "MPC (Real-time)" selected, "Compute + Execute" on the
Planning tab switches the planner to MPC, sends an `execute_trajectory` goal, and
streams the target pose to `/<planner>/mpc_goal` at 10 Hz while the gizmo is
dragged.

**Multiple target displays:** the Planning tab scans every `TargetDisplay` in the
display tree — including ones nested inside display Groups — so the primary
(first) display is found after an RViz config reload. Classic and MPC act on
that primary display only; the old Multipoint mode that turned every extra
display into a waypoint is removed.

### Future development
- [ ] Save and load the system's state

## TargetDisplay

Generic 6-DOF target marker, planner-agnostic. All planner/MPC logic lives in
the Curobo panel's Planning tab; this display only owns the gizmo and the
draggable pose.

Self-contained: the gizmo is rendered in-place by the display (no separate
"Interactive Markers" display needed). The Planning tab's pose spin boxes
stay in sync with the target in both directions (the tab finds the display
automatically and talks to it via `getPose`/`setPose`).

## AddObjectsPanel

### Current state
Manages objects in the scene through the `add_object`/`remove_object` services.
It publishes nothing: the objects it adds appear in the Curobo panel's Scene
Objects tab, which lists them by asking the server via `get_obstacles`.

Mounted inside the Curobo panel's Scene Objects tab under MoveIt's own heading
for this control — an "Add/Remove scene object(s)" group — above the live object
list. The form is a `QFormLayout`, so it reflows to the dock width; it used to be
a fixed 465x327 form with absolutely-positioned widgets and needed a scroll area
to stop being clipped, and it no longer does. The buttons are MoveIt's `[+ Add]`
/ `[- Remove]` pair.

**Known wart:** the form keeps its own `listWidgetObjects` alongside the Scene
Objects tab's authoritative tree. That list is written on a successful add and
never re-read from the server, so it can disagree with the tree — an object
removed from the tree is still in the form's list, and anything the task
constructor or a `ros2 service call` adds is in neither. Its tooltip says it
lists only objects added through this form. Deleting it and letting the
authoritative tree own Remove/Clear is the correct fix and is still open.

The former `AddObjectsDisplay` (which drew the added obstacles as RViz scene
nodes) was removed — its shape/colour mapping to cuRobo's obstacle types was
unreliable, and the server's own obstacle markers are authoritative. Objects
added through this panel still affect planning; they are simply not drawn as
extra RViz geometry.

### Future development
- [ ] Disable boxes when the parameter is not needed
- [ ] Persist objects across RViz restarts and pre-opened objects
- [ ] Save and load the system's state
- [ ] Previsualise a moving object marker before adding
- [ ] Show selected-object parameters in the boxes
- [ ] Select a mesh path with a file explorer

## SparseVoxelGridDisplay

Renders the cuRobo mapper's occupied voxels from the `SparseVoxelGrid` topic
(the same data the U-Net consumer and reachability/obstacle logic see).

## CuroboTrajectoryDisplay

Animates a **full robot body** (every link, every joint) through a
`trajectory_msgs/JointTrajectory` — e.g. the ghost preview topic `<node>/trajectory`
or the MPC's `<node>/mpc_predicted_path`.

Properties: Trajectory Topic, Alpha, Show Trail (+ Trail Step Size),
Loop Animation, Speed.

### FK is computed in the display, for display only

The server is a *planner*, not a visualiser: it publishes only the joint-space
trajectory (joint names + positions + velocities + timestamps). All forward
kinematics used to render the robot is computed **inside this plugin**, from the
URDF, purely for visualisation:

- `CuroboFK` parses the URDF once (`urdf::Model`, Eigen only) and walks the
  kinematic tree per waypoint to produce link transforms.
- `CuroboLinkUpdater` bridges those Eigen transforms into RViz's `Robot`
  renderer (same `LinkUpdater` abstraction MoveIt's displays use).
- The URDF is read from the latched `/robot_description` topic (fallback: the
  RViz node's `robot_description` parameter, then `/tmp/kortex.urdf`).

This FK is **not** cuRobo's FK: no cuRobo kinematics, CUDA kernel, GPU buffers
or planner config are touched for visualisation. cuRobo's FK remains the source
of truth for *planning*; the display only needs rendering-accurate geometry.
The plugin also never consults TF — link transforms come from the message + URDF
alone, so it works with no TF tree. No MoveIt dependency.

## ReachabilityMapDisplay

Solves a reachability map on a configurable plane via the
`/curobo_server/generate_rm` service and shows the solved cells (arrows, or the
robot at each cell). Displays the embedded IK solve result per cell; see the
display's properties (grid size, cell style, plane pose, solved/failed colours,
"Show Solutions", "Gizmo Visible") for configuration. The embedded "gizmo"
toggles the reachability display on/off.

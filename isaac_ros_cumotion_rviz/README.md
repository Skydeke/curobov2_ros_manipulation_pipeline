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

The launch forwards `max_attempts`, `timeout`, `time_dilation_factor`,
`collision_activation_distance` and `base_link` as node parameters. Point the
displays at a running planner node (default `unified_planner`).

## Registered plugins (`rviz2_plugin.xml`)

| Component | Kind | Role |
|---|---|---|
| `isaac_ros_cumotion_rviz/RvizArgsPanel` | Panel | Trajectory/planner parameters, planner-node selection and plan/send controls |
| `add_objects_panel/AddObjectsPanel` | Panel | Add/remove scene obstacles |
| `add_objects_display/AddObjectsDisplay` | Display | Renders the obstacles added from the panel |
| `isaac_ros_cumotion_rviz/TargetDisplay` | Display | Generic 6-DOF target pose (planner/MPC logic lives in the panel) |
| `isaac_ros_cumotion_rviz/SparseVoxelGridDisplay` | Display | Renders the mapper's occupied voxels |
| `isaac_ros_cumotion_rviz/CuroboTrajectoryDisplay` | Display | Animates a full robot body along a `JointTrajectory` |
| `isaac_ros_cumotion_rviz/ReachabilityMapDisplay` | Display | Solves + visualises a reachability map on a plane |

## RvizArgsPanel

### Current state
Control panel for trajectory planning. Retrieves parameters at launch; time
dilation is applied live. Exposes trajectory type (Classic / MPC / Multipoint),
planner node selection, the target pose (synced with the
`TargetDisplay` gizmo), and the plan / send / stop actions.

**Planner node selection:** the dropdown binds the panel's service/action
clients to a named planner node (default `unified_planner`). Changing the
selection immediately rebinds all clients and re-probes readiness. The stored
selection is persisted across RViz restarts via `planner_node_name` in the
saved config.

**MPC mode:** selecting "MPC (Real-time)" and pressing "Generate and send"
switches the planner to MPC, sends an `execute_trajectory` goal, and streams
the target pose to `/<planner>/mpc_goal` at 10 Hz while the gizmo is dragged.

**Multipoint mode:** selecting "Multipoint (Multiple traj)" makes every
`TargetDisplay` in the display tree a waypoint, in display-tree order (reorder
displays to change the path). Displays are found automatically — including
ones nested inside display Groups — and the panel keeps scanning, so a display
added later is used without reloading. Classic and MPC act on the first
(primary) display only.

### Future development
- [ ] Save and load the system's state

## TargetDisplay

Generic 6-DOF target marker, planner-agnostic. All planner/MPC logic lives in
`RvizArgsPanel`; this display only owns the gizmo and the draggable pose.

Self-contained: the gizmo is rendered in-place by the display (no separate
"Interactive Markers" display needed). The `RvizArgsPanel` pose spin boxes
stay in sync with the target in both directions (the panel finds the display
automatically and talks to it via `getPose`/`setPose`).

## AddObjectsPanel / AddObjectsDisplay

### Current state
Manages objects in the scene through the `add_object`/`remove_object` services;
the display renders them. The scene-node part needs revisiting: some cuRobo
shapes have no RViz `Shape` equivalent and vice-versa, and colours are
unreliable.

### Future development
- [ ] Disable boxes when the parameter is not needed
- [ ] Refactor so the Display owns the add/remove service calls
- [ ] Persist object display across RViz restarts and pre-opened objects
- [ ] Save and load the system's state
- [ ] Previsualise a moving object marker before adding
- [ ] Show selected-object parameters in the boxes
- [ ] Merge all panels into one with tabs
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
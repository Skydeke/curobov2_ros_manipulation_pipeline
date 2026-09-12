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
`voxel_size`, `collision_activation_distance` and `base_link` as node
parameters. Point the displays at a running `curobo_server` (e.g. as started by
the kortex deployment's `moveit_cumotion.launch.py`).

## Registered plugins (`rviz2_plugin.xml`)

| Component | Kind | Role |
|---|---|---|
| `isaac_ros_cumotion_rviz/RvizArgsPanel` | Panel | Trajectory/planner parameters and plan/send controls |
| `add_objects_panel/AddObjectsPanel` | Panel | Add/remove scene obstacles |
| `add_objects_display/AddObjectsDisplay` | Display | Renders the obstacles added from the panel |
| `isaac_ros_cumotion_rviz/ArrowInteractionDisplay` | Display | Interactive 6-DOF arrow for the target pose |
| `isaac_ros_cumotion_rviz/SparseVoxelGridDisplay` | Display | Renders the mapper's occupied voxels |
| `isaac_ros_cumotion_rviz/CuroboTrajectoryDisplay` | Display | Animates a full robot body along a `JointTrajectory` |
| `isaac_ros_cumotion_rviz/ReachabilityMapDisplay` | Display | Solves + visualises a reachability map on a plane |

## RvizArgsPanel

### Current state
Control panel for trajectory planning. Retrieves parameters at launch; some are
updated live, others need the "Confirm Changes" button. Exposes planner and
control-strategy selection, the target pose (synced with the arrow), voxel
size, time dilation, and the plan / send / stop actions.

### Future development
- [ ] Multithread the "Confirm Changes" press (disable button while loading)
- [ ] Save and load the system's state

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

## ArrowInteractionDisplay

Interactive 6-DOF arrow marker for the target pose, expressed in the robot base
frame. The `RvizArgsPanel` pose spin boxes stay in sync with the arrow in both
directions.

## SparseVoxelGridDisplay

Renders the cuRobo mapper's occupied voxels from the `SparseVoxelGrid` topic
(the same data the U-Net consumer and reachability/obstacle logic see).

## CuroboTrajectoryDisplay

Animates a **full robot body** (every link, every joint) through a
`trajectory_msgs/JointTrajectory` — e.g. the planner's `<node>/planned_path`,
`<node>/mpc_predicted_path`, or the ghost preview topic `<node>/trajectory`.

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
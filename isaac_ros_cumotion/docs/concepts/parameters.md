# Parameters Guide

Every parameter of the `unified_planner` node, with its real default, what it does, and when a change takes effect. Defaults below are taken directly from the code (`declare_parameter` calls); if this page and the code ever disagree, the code wins.

## How parameters take effect

There are three kinds of knobs, and they differ in *when* a change is applied:

| Kind | How to change | When it takes effect |
|---|---|---|
| **Plan-time parameters** | `ros2 param set /unified_planner <name> <value>` | Next planning request — no rebuild needed |
| **Build-time parameters** | `ros2 param set …` then call `/unified_planner/update_motion_gen_config` | After the solver rebuild (blocking, ~20 s) |
| **Robot YAML configuration** | Edit the cuRobo robot config file | Restart, or `update_motion_gen_config` |

`update_motion_gen_config` (`std_srvs/srv/Trigger`) rebuilds the motion-planning solver from the current parameters. Exception: `set_collision_cache` rebuilds the solvers by itself — no extra call needed (see [ROS Interfaces](ros-interfaces.md)).

## Launch arguments (`gen_traj.launch.py`)

These are the launch arguments that actually configure the system:

| Argument | Default | Description |
|---|---|---|
| `robot` | `emulator` | Robot descriptor name (loads `robots/<name>.yaml`) or path. Robot-abstract: no robot is hardcoded; concrete robots (e.g. Kortex) provide their own descriptor and model at launch |
| `robot_config_file` | `''` (auto) | cuRobo robot YAML; empty = resolved from the robot descriptor |
| `urdf_path` | `''` (auto) | URDF for `robot_state_publisher`; empty = read from the robot descriptor |
| `camera_topic` | `[]` | Raw depth streams for the perception Mapper and the robot segmentation (see [Tutorial 7](../tutorials/07-pointcloud-detection.md)); array launch arg (Python repr), empty = no cameras |
| `camera_info_topic` | `[]` | Camera intrinsics topics (`sensor_msgs/CameraInfo`), array launch arg |
| `world_file` | `''` | Static world YAML (e.g. `config/floor_world.yml`); empty = empty world — **including no floor** |
| `gui` | `true` | Start RViz |
| `voxel_size` | `0.05` | Perception/collision voxel size (m) |
| `mapper_extent_xyz` | `[2.56, 2.56, 2.56]` | Perception volume extent (m), centred on the robot base |
| `max_attempts` | `1` | Planning retries per request |
| `time_dilation_factor` | `1.0` | Trajectory re-timing: 1.0 = nominal interpolation_dt pacing; <1.0 slows the motion, >1.0 speeds it up. Also gates execute() feedback re-reads |
| `collision_activation_distance` | `0.025` | Distance (m) at which the collision cost activates |

The last four are forwarded straight to the node, so their defaults above are also the node's defaults — see the tables below for what each one does.

There is no world floor added automatically at startup: if you want a ground plane, pass a `world_file` that contains one (the shipped `config/floor_world.yml` does).

## Node parameters — planning

| Parameter | Default | Effect | Kind |
|---|---|---|---|
| `planner_type` | `'classic'` | Planner selected at startup (`classic`, `mpc`, `multi_point`, `joint_space`, `retarget`) — switch at runtime with `set_planner` | Startup |
| `max_attempts` | `1` | Planning retries per request | Plan-time |
| `interpolation_dt` | `0.025` | Time step (s) of the interpolated output trajectory | Build-time |
| `voxel_size` | `0.05` | Voxel size (m) shared by the perception ESDF, the collision cache, and `get_voxel_grid` | Build-time |
| `collision_activation_distance` | `0.025` | Distance (m) at which collision cost activates | Build-time |
| `use_cuda_graph` | `true` | Capture/replay CUDA graphs in the solvers (faster). Env var `CUROBO_USE_CUDA_GRAPH=0` overrides for A/B testing | Startup |
| `time_dilation_factor` | `1.0` | Trajectory RE-TIMING: the stamped per-point dt of every sent trajectory becomes `interpolation_dt / tdf` — 1.0 = nominal, <1.0 slower, >1.0 faster (cuRobo convention). Also gates how often execute() re-reads progression | Plan-time |
| `trajectory_cache_ttl` | `30.0` | Lifetime (s) of a trajectory cached by `generate_trajectory` and reusable by the execute action (`allow_cached`) | Plan-time |
| `sparse_voxel_publish_rate` | `7.0` | Publish rate (Hz) of `/unified_planner/voxel_grid_sparse`; `<= 0` disables | Startup |
| `push_esdf_to_solvers` | `true` | Diagnostic toggle — `false` withholds the camera ESDF from the solvers and disables camera-based avoidance | Runtime |

## Node parameters — perception (Mapper)

Declared by the obstacle manager; all are startup/build-time.

| Parameter | Default | Effect |
|---|---|---|
| `use_mapper` | `true` | Enable the cuRobo v2 `Mapper` (ESDF perception) |
| `mapper_extent_xyz` | `[2.0, 2.0, 2.0]` | Perception volume (m) around `mapper_grid_center` |
| `mapper_grid_center` | `[0, 0, 0]` | Centre of the perception volume (robot base frame) |
| `mapper_voxel_size` | `0.02` | Internal TSDF voxel size (m) |
| `mapper_image_width` / `mapper_image_height` | `640` / `480` (launch file passes `1280`/`720`) | Depth-projection resolution; every depth frame is resized to this before integration |
| `mapper_depth_min` / `mapper_depth_max` | `0.1` / `5.0` | Depth clipping range (m) |
| `decay_half_life_s` | `0.7` | Half-life (s) of stale voxels. Replaces the removed `decay_factor` parameter |

## Node parameters — MPC (reactive control)

Read by the MPC controller when it is built (first switch to `mpc`); change them before switching, or switch away and back. See [MPC Implementation](mpc-implementation.md).

| Parameter | Default | Effect |
|---|---|---|
| `mpc_solver_type` | `'mppi_acceleration'` | Solver recipe: `mppi_acceleration` (MPPI in acceleration space, validated on a real deployment robot) or `lbfgs_bspline` (cuRobo's L-BFGS/B-spline config) |
| `mpc_step_dt` | `0.03` | Optimization time step (s) |
| `mpc_horizon_steps` | `30` | Receding horizon length |
| `mpc_warm_start_iters` | `5` | Iterations per solve after the first (L-BFGS values must be multiples of 25) |
| `mpc_cold_start_iters` | `10` | Iterations on the first solve |
| `mpc_mppi_num_particles` | `400` | MPPI particle count (`mppi_acceleration` only) |
| `mpc_vel_feedback_alpha` | `1.0` | Velocity feedback blend when reading robot state |
| `mpc_command_interval` | `0.24` | Fixed command pacing (s): each command window executes fully before re-solving. `0` = re-solve as fast as possible |
| `convergence_threshold` | `0.01` | End-effector error (m) considered "on target" |
| `max_mpc_iterations` | `1000` | Safety cap on servo steps per goal |
| `mpc_debug` | `false` | Write per-step diagnostic CSVs to the ROS log directory |

## Node parameters — retargeting (teleoperation)

| Parameter | Default | Effect |
|---|---|---|
| `retarget_position_weight` | `1.0` | Position tracking weight |
| `retarget_orientation_weight` | `1.0` | Orientation tracking weight |
| `retarget_use_mpc` | `false` | Route retarget commands through the MPC solver instead of direct IK |

## Node parameters — robot selection

| Parameter | Default | Effect |
|---|---|---|
| `robot` | `'emulator'` | Robot descriptor (`robots/<name>.yaml`) or path |
| `control_strategy` | from descriptor | How commands reach the robot: `emulator`, `joint_speed`, `joint_pose` — switch at runtime with `set_robot_strategy` |
| `base_link` | from descriptor | Robot base frame (defaults to `base_0` if the descriptor/model does not specify one) |
| `world_file` | `''` | Static world YAML |
| `robot_config_file` | from descriptor | Resolved cuRobo robot YAML; required when the descriptor ships model-less (e.g. the default `emulator`) |
| `camera_topic` | `['']` | Raw depth streams (`sensor_msgs/Image`), one array entry per camera; empty = no cameras |
| `node_is_available` | `false` | Read-only status: flips to `true` once warmup completes |

## Node parameters — cameras (`PerceptionCameraCfg`)

One array per parameter, one entry per camera (indexed together; `camera_index`
pairs a camera with the same-position entry of every other array). Declared by
`CameraSystemManager`, consumed by the mapper's strategies AND the in-server
`RobotSegmentation` — no `cameras.yaml`.

| Parameter | Default | Effect |
|---|---|---|
| `camera_topic` | `['']` | RAW depth stream per camera (`sensor_msgs/Image`, `16UC1` or `32FC1`) |
| `camera_info_topic` | `['']` | Camera intrinsics (`sensor_msgs/CameraInfo`) per camera |
| `camera_frame` | `['']` | Map/mask frame per camera: the frame BOTH the mapper integrates the depth into AND the segmenter masks in (TF fallback = raw `frame_id` of the depth msg) |
| `camera_intrinsics` | `['']` | `''` = read once from `camera_info_topic` (5 s wait), or a comma-separated row-major K `'fx,0,cx,0,fy,cy,0,0,1'` |
| `camera_extrinsics` | `['']` | `''` = per-frame TF `base frame → camera_frame`, or `'x,y,z,qw,qx,qy,qz'` camera pose in the base frame |
| `camera_frame_rate_hz` | `[0.0]` (→ 30.0) | Declared publication rate per camera; drives the TSDF decay normalisation |
| `camera_purpose` | `['all']` | What each camera feeds: `all` (default, = everything) / `esdf` / `segmentation` |

## Node parameters — robot segmentation (in-server component)

`RobotSegmentation` runs inside the planner node (no standalone executable). Its input —
raw depth topics, camera-info topics, camera/base frame — comes from the shared
`camera_*` arrays; each segmented camera gets one `RobotSegmentationCameraStrategy`
registered through the same `CameraContext` the mapper uses, so only its tuning knobs
are listed here.

| Parameter | Default | Effect |
|---|---|---|
| `enable_robot_segmentation` | `true` | Mask the robot (collision spheres) out of each `segmentation`-purpose camera's stream; those cameras' mapper input switches to their masked output |
| `robot_segmentation_mask_margin` | `0.0` | Extra margin (m) around the robot mask |
| `robot_segmentation_distance_threshold` | `0.05` | Distance (m) to a collision sphere below which a pixel is masked |

The masked output topic is **derived per camera** from that camera's own
`camera_topic` — the leaf segment is stripped and replaced by `masked_depth`
(e.g. raw `/kortex_vision/depth/image` → `/kortex_vision/depth/masked_depth`).
There is no output-topic parameter; each segmented camera automatically gets a
unique stream.

## Robot YAML configuration

The cuRobo robot configuration (provided by each robot's own repo, e.g. a Kortex package supplies `config/kortex.curobo.yml`) defines kinematics (`urdf_path`, `base_link`, `ee_link`), the cspace with joint limits (position/velocity/acceleration/jerk), and the collision spheres. This is also where the robot's *speed* is set — scale the cspace velocity/acceleration/jerk limits and rebuild with `update_motion_gen_config`.

See [Tutorial 2: Adding Your Robot](../tutorials/02-adding-your-robot.md) for the full anatomy.

## Tuning guidance

**`voxel_size`** is the main precision/VRAM trade-off. `0.05` is a good default; `0.02–0.03` for cluttered scenes on GPUs with headroom; `0.1` for large sparse environments. Smaller voxels cube the memory cost. To measure the cost on your own GPU before committing, run the shipped benchmark against a live planner:

```bash
ros2 run curobo_ros benchmark_voxel_grid --ros-args -p voxel_size:=0.02 -p n_runs:=10
```

**`collision_activation_distance`** adds safety margin at the cost of a smaller reachable workspace in clutter. `0.025` default; raise toward `0.05–0.1` around humans, lower toward `0.01` for tight picking.

**Collision caches** (`set_collision_cache` service, fields `obb`/`mesh`/`blox`) bound how many obstacles of each kind the solvers can hold. Raise `obb` if adding objects fails with a cache error; set `blox: 0` only if you deliberately want to disable camera perception. Every change triggers an automatic ~20 s solver rebuild and is refused while an execution goal is active.

**MPC**: keep the `mppi_acceleration` recipe unless you have a reason not to — the `mpc_warm_start_iters`/`mpc_cold_start_iters` defaults (5/10) are tuned for it. If you switch to `lbfgs_bspline`, use multiples of 25 (e.g. 25/100).

## Worked example

```bash
# Tighten the voxel grid, then rebuild the solver
ros2 param set /unified_planner voxel_size 0.03
ros2 service call /unified_planner/update_motion_gen_config std_srvs/srv/Trigger

# Give planning more retries (no rebuild needed)
ros2 param set /unified_planner max_attempts 3

# Grow the OBB cache (rebuilds by itself, ~20 s)
ros2 service call /unified_planner/set_collision_cache curobo_msgs/srv/SetCollisionCache \
  "{obb: 300, mesh: -1, blox: -1}"
```

## Related pages

- [ROS Interfaces](ros-interfaces.md) — the services referenced above
- [MPC Implementation](mpc-implementation.md) — what the `mpc_*` family controls
- [Tutorial 1: First Trajectory](../tutorials/01-first-trajectory.md)

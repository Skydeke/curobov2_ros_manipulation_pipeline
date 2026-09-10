# Manager Architecture

How the `unified_planner` node's configuration layer is composed. This is a contributor-oriented page: read it before modifying `curobo_ros/core/`. For the bird's-eye view see [Architecture](architecture.md).

## The composition

`ConfigWrapper` (`curobo_ros/core/config_wrapper.py`) is the orchestrator. It builds five specialized managers, **in this order** (each depends on the previous ones), then wires perception and observers:

```{mermaid}
graph LR
    CM[1. ConfigManager] --> RMM[2. RobotModelManager]
    RMM --> OM[3. ObstacleManager]
    OM --> CSM[4. CameraSystemManager]
    CSM --> RSM[5. RosServiceManager]
    RSM --> P[setup_perception + observers]
```

The concrete subclass used by the node is `ConfigWrapperMotion` (`curobo_ros/core/config_wrapper_motion.py`): it additionally builds the shared cuRobo `MotionPlanner` and registers the `update_motion_gen_config` service.

### 1. `ConfigManager` (`config_manager.py`)

Resolves *what robot and world we are working with*: reads the `robot` parameter, loads the robot descriptor (`robots/<name>.yaml`), resolves the cuRobo robot YAML (rewriting `package://` and relative asset paths into a temp copy — see `robot_description.py`), exposes `base_link` and the initial `Scene` from `world_file`.

### 2. `RobotModelManager` (`robot_model_manager.py`)

Owns the cuRobo kinematics: `Kinematics(KinematicsCfg.from_robot_yaml_file(...))`, the robot collision spheres, and per-link collision toggling (`set_link_collision`). All solvers share this kinematics configuration.

### 3. `ObstacleManager` (`obstacle_manager.py`)

Owns the **single `Scene`** and everything that mutates the collision world:

- obstacle add/remove/list (the `add_object` family of services delegates here),
- the collision caches (v2 format: `{"cuboid": N, "mesh": N, "voxel": {"layers": 1, "dims": …, "voxel_size": …}}` — the `SetCollisionCache` service's `obb`/`mesh`/`blox` fields map onto it),
- GPU voxel rasterization for `get_voxel_grid` and the sparse voxel topic,
- the cuRobo v2 `Mapper` (ESDF perception) and its parameters (`mapper_*`, `decay_half_life_s`),
- observer callbacks so the node can propagate world changes and cache rebuilds to every solver.

One subtlety worth knowing: solvers are constructed from `primitives_only_scene()` — a copy of the scene *without* the live perception voxel layer. The collision cache allocates the voxel storage and `update_world` fills it by copy. Passing the live ESDF layer at construction would alias the solver's buffer onto the Mapper tensor and corrupt it on the first update.

### 4. `CameraSystemManager` (`camera_system_manager.py`)

Reads the shared `camera_*` array params (`PerceptionCameraCfg`, no YAML file),
builds a `CameraContext` and instantiates one `DepthMapCameraStrategy` per camera
whose `camera_purpose` includes `esdf`. Perception is **push-based with a
 worker-thread handoff**: the depth subscription callback stays CPU-only (decode,
 TF lookup, enqueue) and the Mapper integration runs on a dedicated daemon thread —
`<camera>_integrate` — never on the executor. That thread pops the freshest queued
 frame (single-slot deque, drops stale frames), integrates under a *non-blocking*
 `gpu_lock` (frames are dropped during CUDA-graph capture or on TF failure),
 integrates every handed-off frame, and never runs on the executor. Keeping the
 integration off the executor entirely takes the read path off the collect and
 keeps viz timers off the integrate critical path. The same
params feed the in-server `RobotSegmentation`, which registers one
`RobotSegmentationCameraStrategy` per `segmentation`-purpose camera through the
SAME `CameraContext` — a segmented camera is just another camera strategy, and its
masked output topic is derived from that camera's own raw `camera_topic` (leaf
segment → `masked_depth`). Both consumers stay aligned because they read the same
arrays. See [Tutorial 7](../tutorials/07-pointcloud-detection.md).

### 5. `RosServiceManager` (`ros_service_manager.py`)

Registers the obstacle/introspection services and the visualization publishers (scene markers, collision spheres, sparse voxel grid) with their timers. It guards GPU-touching timers with a non-blocking `gpu_lock` acquire so visualization never stalls planning, and assigns its timers to the node's dedicated `_viz_callback_group` / `_marker_callback_group` so they can run in parallel with the perception callbacks (see below).

## Self-registering service groups

Kinematics and attachment services live outside the five managers as small self-registering groups instantiated by the node: `IKServices`, `FKServices` (`ik_services.py`, `fk_services.py` — lazy solvers, built on `warmup_*`), and `AttachmentServices` (`attachment_services.py`).

## Concurrency invariants

These are the rules the code is written against — keep them when contributing:

- **Callback groups** (`unified_planner_node.py`): the `MultiThreadedExecutor(num_threads=8)` only runs callbacks in parallel if they live in *different* groups — everything ungrouped shares the node's default `MutuallyExclusiveCallbackGroup`. The node therefore creates dedicated groups: `_perception_callback_group` (depth handoff + robot-segmentation subscriptions; depth's Mapper integration itself runs off the executor on its own worker thread), `_viz_callback_group` (collision-spheres, sparse-voxel, mapper-workspace, mapper-stats timers), `_marker_callback_group` (scene-obstacle markers, CPU-only). New subscriptions/timers that do GPU or rate-sensitive work MUST be assigned one of these (never left in the default group, or they serialize against the camera callbacks).
- **Lock order**: `gpu_lock > strategy_lock > buffer_lock` (documented in `robot_context.py`). Never acquire in the other direction.
- **`gpu_lock`** (an RLock on the node) serializes CUDA-graph capture against camera integration. Capture is process-global — no other thread may issue ANY CUDA op while it's in progress — but replaying a captured graph is not. So the lock is held **only around calls that may capture**: reactive steps use `_step_guard`, open-loop plans use `_plan_lock` (both gated on `take_graph_capture_pending()`). Camera callbacks and viz/mapper-stats use a *non-blocking* acquire and skip their work if a capture holds the lock — they must never block on it.
- **One live CUDA graph**: the node tracks graph ownership across solvers (`_ensure_exclusive_graph`); a planner switch releases the previous solver's captured graph and sets `_graph_capture_pending` so the next plan/step in the new owner captures under the lock.
- **Single goal admission**: `execute_trajectory` accepts one goal at a time; `set_planner` and `set_collision_cache` are refused while a goal is active.
- **Buffer epochs**: `RobotContext` stamps trajectories with an epoch and refuses commands from a superseded plan.

## Observer wiring

`ConfigWrapper` connects `ObstacleManager` callbacks to the node so that:

- any world mutation (`add_object`, mesh attach, Mapper ESDF update) triggers `update_all_solvers_world()` — every built solver receives the new scene;
- a collision-cache change triggers a full solver rebuild (`rebuild_solvers_for_cache_change`, blocking ~20 s).

This is why there is no per-solver world synchronization anywhere else in the code: mutate the world through `ObstacleManager`, and the observers do the rest.

## Startup sequence

1. Node declares its parameters (`unified_planner_node.py`).
2. `RobotContext` is created (control strategy, ghost preview).
3. `ConfigWrapperMotion` builds the five managers in order, then the shared `MotionPlanner`, then warms it up (this is the 25–35 s startup phase).
4. `AttachmentServices`, `IKServices`, `FKServices`, `PlannerManager` are attached.
5. The `node_is_available` parameter flips to `true`; `generate_trajectory` and the action are ready.

## Backward-compat aliases

`node.motion_gen` survives only as an alias to `node.motion_planner` (`config_wrapper_motion.py`) for old call sites. New code must use v2 names — there is no `MotionGen`, `WorldConfig`, or nvblox anywhere in the package. See [Migration to cuRobo v2](../MIGRATION_V2.md).

## Related pages

- [Architecture](architecture.md) — overall structure
- [ROS Interfaces](ros-interfaces.md) — the services these managers register
- [Parameters](parameters.md) — the parameters these managers declare

# Isaac ROS cuMotion

> **Fork notice:** This repository is a fork of NVIDIA's `isaac_ros_cumotion`
> and related packages, forked because NVIDIA did not ship a ROS-ready version
> of cuRobo v2. It retains NVIDIA's original licensing terms (see `LICENSE`
> files in each submodule). The code merged here is built on top of NVIDIA's
> cuRobo library and Isaac ROS framework — respect the original copyrights and
> license obligations when distributing or modifying.
>
> Much of the ROS wrapping of cuRobo in this repository is derived from
> <https://github.com/Lab-CORO/curobo_ros>; see that project's license for the
> terms of that portion.

NVIDIA cuRobo v2 wrapped as a single GPU-accelerated ROS 2 node (`curobo_trajectory_planner`)
for arm motion planning, IK/FK, collision checking, world management,
depth-to-ESDF mapping, robot segmentation, and trajectory optimization.

## Packages

| Package | Purpose |
|---|---|
| `curobo_core` | cuRobo v2 library (vendored) |
| `isaac_ros_cumotion_interfaces` | ROS actions/services/messages (see `USED_INTERFACES.txt` for what is wired in) |
| `isaac_ros_cumotion` | The unified node `curobo_trajectory_planner` and supporting services |
| `isaac_ros_cumotion_extra` | Viser/visualization nodes and tools (e.g. `build_curobo_config`) |
| `isaac_ros_cumotion_moveit` | MoveIt 2 planning plugin |
| `isaac_ros_cumotion_rviz` | RViz plugin and visualizations |

## Build

```bash
colcon build --symlink-install \
  --packages-select curobo_core isaac_ros_cumotion_interfaces \
  isaac_ros_cumotion isaac_ros_cumotion_moveit isaac_ros_cumotion_rviz \
  isaac_ros_cumotion_extra
```

## Run

```bash
# Terminal 1: the unified node
ros2 run isaac_ros_cumotion curobo_trajectory_planner

# Or launch with a pre-built trajectory scene
ros2 launch isaac_ros_cumotion gen_traj.launch.py
```

Callers send trajectory requests to the `TrajectoryGeneration` service
(`/<name>/generate_trajectory`) and stream resulting joint trajectories through
the `SendTrajectory` action (`/<name>/execute_trajectory`). IK/FK and world
management are exposed via the `Ik`/`IkBatch`, `Fk`/`FkBatch`,
`AddObject`/`AttachObject`/`RemoveObject`, `GetVoxelGrid`, and
`GetCollisionDistance` services.

## Documentation

- `isaac_ros_cumotion/docs/` — user guide (concepts, getting started, tutorials)
  and `MIGRATION_V2.md` for the v1 → v2 transition.
- `isaac_ros_cumotion_interfaces/USED_INTERFACES.txt` — authoritative audit of
  which interfaces are actually referenced by code.
# Isaac ROS cuMotion

> **Fork notice:** This repository is a fork of NVIDIA's `isaac_ros_cumotion`
> and related packages, forked because NVIDIA did not ship a ROS-ready version
> of cuRobo v2.
>
> Licensing:
> - `isaac_ros_cumotion`, `isaac_ros_cumotion_interfaces`,
>   `isaac_ros_cumotion_extra`, and `isaac_ros_cumotion_rviz` are
>   **Apache License 2.0** (see `LICENSE` in each package).
> - `isaac_ros_cumotion_moveit` is from NVIDIA and retains the
>   **NVIDIA Isaac ROS Software License**.
> - `curobo_core` vendors NVIDIA's cuRobo library, which carries its own
>   license terms.

NVIDIA cuRobo v2 wrapped as a single GPU-accelerated ROS 2 node (`curobo_trajectory_planner`)
for arm motion planning, IK/FK, collision checking, world management,
depth-to-ESDF mapping, robot segmentation, and trajectory optimization.

## Packages

| Package | Purpose |
|---|---|
| `curobo_core` | cuRobo v2 library (vendored) |
| `isaac_ros_cumotion_interfaces` | ROS actions/services/messages |
| `isaac_ros_cumotion` | The unified node `curobo_trajectory_planner` and supporting services |
| `isaac_ros_cumotion_extra` | Viser/visualization nodes, the `build_curobo_config` robot-config generator, and the cuRobo benchmarks reproduction (`curobo_benchmark`) |
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

## Quickstart (Docker)

The fastest way to try the fork is the interactive compose sessions: RViz +
cuRobo planner, with an emulated robot and no physical driver. Both robots use
the same pipeline-built image (`ghcr.io/skydeke/isaac_ros_cumotion/isaac-ros-cumotion:latest`),
selected by `robot:=...` at launch; `--build` instead rebuilds locally from
`docker/Dockerfile.cumotion`.

```bash
# On the host: let the container's root user reach your X server (for RViz)
xhost +local:root

# Franka Emika Panda
docker compose -f docker/franka.yaml pull
docker compose -f docker/franka.yaml up

# Universal Robots UR10e
docker compose -f docker/ur10e.yaml pull
docker compose -f docker/ur10e.yaml up
```

Pull the freshest pipeline image first (`pull`), then start the session (`up`).
Requires the NVIDIA container runtime, X11 forwarding via `xhost +local:root`,
and a working `ROS_DOMAIN_ID`/`DISPLAY`.

## Benchmarks

The fork reproduces the [cuRobo benchmarks
page](https://nvlabs.github.io/curobo/latest/reference/benchmarks.html) —
motion generation (with and without torque limits), inverse kinematics, and
kinematics & collision — running every problem **twice**: through curobo_core
natively and through the ROS-wrapped planner (`/unified_planner/...`), then
printing all metrics and a parity verdict. Neither leg is ever skipped.

```bash
# One command: server + full reproduction (details in
# isaac_ros_cumotion_extra/README.md, "cuRobo benchmarks reproduction").
docker compose -f docker/compose_benchmark.yaml up
# CUROBO_RUN_PARITY=1 additionally runs the compare-verdict suite (all/ik/cost).
```

- Runs the page's ~2600-problem `full` dataset (motion_benchmaker + mpinets)
  at the page's 100-attempt budget by default — `CUROBO_MAX_ATTEMPTS` is one
  knob for both legs (the runner pins the server's plan-time `max_attempts`
  to the run's budget before each motion leg).
- Prints all results in the page's order under a final `ALL RESULTS` banner;
  JSONs land in `/tmp/benchmark_webpage.*.json`.
- Envelope knobs: `CUROBO_MAX_ATTEMPTS`, `CUROBO_LOAD_DYNAMICS` /
  `CUROBO_PAYLOAD_MASS` (torque limits), `CUROBO_DATASET`,
  `CUROBO_NUM_TRAJOPT_SEEDS`, `CUROBO_USE_CUDA_GRAPH`,
  `CUROBO_COLLISION_MODE`.
- Pure-Python benchmark tests: `docker compose -f docker/compose_tests.yaml up`.

## Documentation

- `isaac_ros_cumotion/docs/` — user guide (concepts, getting started,
  tutorials) and `MIGRATION_V2.md` for the v1 → v2 transition; the tunable
  node parameters (including the plan-time `max_attempts` the benchmark
  re-pins) are in `docs/concepts/parameters.md`.
- `isaac_ros_cumotion_extra/README.md` — the cuRobo benchmarks reproduction in
  depth: one-shot compose run, the full `curobo_benchmark` CLI, solver
  envelope, timing attribution, and the honest receipt.

## Acknowledgements

This project builds on the work of:

- **[NVIDIA cuRobo](https://github.com/NVlabs/curobo)** — the motion-planning
  library at the core of this node (vendored under `curobo_core/`).
- **[Isaac ROS](https://github.com/isaac-ros/isaac_ros_common)** — the ROS 2
  framework the package integrates with.
- **[curobo_ros](https://github.com/Lab-CORO/curobo_ros)** — the ROS wrapping of
  cuRobo that much of this repository's ROS-side integration is derived from.
- **[MoveIt Task Constructor](https://github.com/moveit/moveit_task_constructor)
  ** — the stage/container architecture used by this repo's
  task constructor (`curobo_task_constructor/`): generators/propagators/
  connectors, serial/alternatives/fallbacks/independent containers,
  interface-adjacency validation, and the plan/rank/execute lifecycle all
  reimplement that design for the cuRobo planning stack (workspace
  `documentation/curobo_task_constructor_plan.md`).
- **[moveit_task_constructor_visualization](https://github.com/moveit/moveit_task_constructor_visualization)** —
  specifically the pluginlib-registered `rviz_common::Panel` structure of
  `curobo_task_constructor_rviz` (Sec. 7 of that plan) mirrors the panel/
  plugin-registration shape of this visualization package. The panel only uses
  `curobo_task_constructor_interfaces` topics and does not reuse MoveIt's
  introspection messages or code.

# Isaac ROS cuMotion

> **Fork notice:** This repository is a fork of NVIDIA's `isaac_ros_cumotion`
> and related packages, forked because NVIDIA did not ship a ROS-ready version
> of cuRobo v2.
>
> Licensing:
> - `curobov2_ros`, `curobov2_ros_interfaces`,
>   `curobov2_ros_extra`, and `curobov2_ros_rviz` are
>   **Apache License 2.0** (see `LICENSE` in each package).
>   The code from `curobov2_ros` wasa forked from curobo_ros (see Acknowledgements section).
> - `curobov2_ros_moveit` is from NVIDIA and retains the
>   **NVIDIA Isaac ROS Software License**.
> - `curobo` vendors NVIDIA's cuRobo library, which carries its own
>   license terms.

NVIDIA cuRobo v2 wrapped as a single GPU-accelerated ROS 2 node (`curobo_trajectory_planner`)
for arm motion planning, IK/FK, collision checking, world management,
depth-to-ESDF mapping, robot segmentation, and trajectory optimization.

## Packages

| Package | Purpose |
|---|---|
| `curobo` | cuRobo v2 library (vendored) |
| `curobo_task_constructor` | Task-constructor stages and containers over the cuRobo server — see its [README](curobo_task_constructor/README.md) |
| `curobov2_ros_interfaces` | ROS actions/services/messages |
| `curobov2_ros` | The unified node `curobo_trajectory_planner` and supporting services |
| `curobov2_ros_extra` | Viser/visualization nodes, the `build_curobo_config` robot-config generator, and the cuRobo benchmarks reproduction (`curobo_benchmark`) |
| `curobov2_ros_moveit` | MoveIt 2 planning plugin |
| `curobov2_ros_rviz` | RViz plugin and visualizations |

## Quickstart (Docker)

The fastest way to try the fork is the interactive compose sessions: RViz +
cuRobo planner, with an emulated robot and no physical driver. Both robots use
the same pipeline-built image (`ghcr.io/skydeke/curobov2_ros_manipulation_pipeline/curobov2_ros:latest`),
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
kinematics & collision — running every problem **twice**: through curobo
natively and through the ROS-wrapped planner (`/unified_planner/...`), then
printing all metrics and a parity verdict.

```bash
# One command: server + full reproduction (details in
# curobov2_ros_extra/README.md, "cuRobo benchmarks reproduction").
docker compose -f docker/compose_benchmark.yaml up
```

- Runs the page's ~2600-problem `full` dataset (motion_benchmaker + mpinets)
  at the page's 100-attempt budget by default — `CUROBO_MAX_ATTEMPTS` is one
  knob for both legs (the runner pins the server's plan-time `max_attempts`
  to the run's budget before each motion leg).
- Prints all results in the page's order under a final `ALL RESULTS` banner;
  JSONs land in `/tmp/benchmark_webpage.*.json`.

## Documentation

- `curobov2_ros/docs/` — user guide (concepts, getting started, tutorials); the tunable node parameters
  (including the plan-time `max_attempts` the benchmark re-pins) are in `docs/concepts/parameters.md`.
- `curobov2_ros_extra/README.md` — the cuRobo benchmarks reproduction in
  depth: one-shot compose run, the full `curobo_benchmark` CLI, solver
  envelope, timing attribution, and the honest receipt.

## Acknowledgements

This project builds on the work of:

- **[NVIDIA cuRobo](https://github.com/NVlabs/curobo)** — the motion-planning
  library at the core of this node (vendored under `curobo/`).
- **[Isaac ROS](https://github.com/isaac-ros/isaac_ros_common)** — the ROS 2
  framework the package integrates with.
- **[curobo_ros](https://github.com/Lab-CORO/curobo_ros)** — the ROS wrapping of
  cuRobo that much of this repository's ROS-side integration is derived from.
- **[MoveIt Task Constructor](https://github.com/moveit/moveit_task_constructor)** — the stage/container
  architecture used by this repo's task constructor package (`curobo_task_constructor/`): generators/
  propagators/connectors, serial/alternatives/fallbacks/independent containers, interface-adjacency
  validation, and the plan/rank/execute lifecycle all reimplement that design for the cuRobo planning stack.
- **[moveit_task_constructor_visualization](https://github.com/moveit/moveit_task_constructor_visualization)**
  —
  inspiration for UIs
- **[Moveit2](https://github.com/moveit/moveit2/tree/main)**
  —
  inspiration for UIs and ideas on how to tackle some of the problems I encountered while working on this

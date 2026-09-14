# Testing

`isaac_ros_cumotion` ships 14 integration suites (`config/config_test/test_*_{franka,ur10e}.yaml` → generated `test/test_test_*_{franka,ur10e}.py`, one per demo robot) plus the standard `ament_copyright` / `ament_flake8` / `ament_pep257` linters. This page covers how to run them and how the generator that produces them works.

Each suite pinpoints its robot through a dedicated launch wrapper (`gen_traj_test_{franka,ur10e}.launch.py`): `robot:=<robot>` resolves the cuRobo model from the package's own robot descriptor (`robots/<robot>.yaml` → `package://isaac_ros_cumotion/config/<robot>.curobo.yml`). No external / private robot repo is involved — the models, URDFs and meshes ship inside this package.

## Running the tests

### The normal way — `colcon test`

```bash
colcon test --packages-select isaac_ros_cumotion
colcon test-result --all --verbose
```

This runs all 17 tests (14 launch suites + 3 linters) and reports a JUnit summary. No special flags are needed.

A full run takes several minutes: each suite launches the real `unified_planner` node, waits for it to build its cuRobo solvers, then exercises services and topics against it.

### One suite at a time — `launch_test`

Useful while iterating on a single YAML, since you see live output instead of waiting for the whole package:

```bash
launch_test test/test_test_kinematics_franka.py
```

Note: the mesh-obstacle suites reference `config/config_test/Rubber_Duck.stl` by absolute path (`/root/ros2_ws/src/isaac_ros_cumotion/...`), matching the layout baked into `docker/Dockerfile.cumotion` (`ROS_WS=/root/ros2_ws`). The generated `test_test_mesh_obstacle_{franka,ur10e}.py` fail on file-not-found if the workspace lives elsewhere.

### All suites, outside colcon — `run_all_tests.sh`

```bash
test/run_all_tests.sh
```

Generated alongside the suites; loops `launch_test` over all 14 with per-suite timeouts. Same tests as `colcon test`, without the linters and without a JUnit report.

### In a container — Docker Compose

The `docker/` dir ships one per-robot GUI compose file (RViz + planner) and one per-robot headless test compose file. All four build this fork's own `docker/Dockerfile.cumotion` (public default base image, nothing private) and bind-mount the fork root at the ROS workspace src so edits apply without rebuilding.

```bash
# Franka: RViz + unified_planner (interactive)
docker compose -f docker/franka.yaml up --build
# UR10e: same
docker compose -f docker/ur10e.yaml up --build

# Franka: the 7 generated test_test_*_franka.py suites, headless
docker compose -f docker/franka_tests.yaml up --build
# rerun without rebuilding
docker compose -f docker/franka_tests.yaml run --rm tests
# UR10e: same
docker compose -f docker/ur10e_tests.yaml up --build
```

The test services run exactly the robot's slice of the generated `test/run_all_tests.sh`, reading each suite's timeout from it (single source of truth) and exiting non-zero on any failure. The GUI services launch `gen_traj.launch.py robot:=<robot> gui:=true`; the planner runs on the GPU, so all four need the nvidia runtime.

## The generator — `ros2_test_compose`

Nothing under `test/` is hand-written except `test_flake8.py` / `test_copyright.py` / `test_pep257.py`. Everything else — the `test_test_*.py` suites and `run_all_tests.sh` — is generated from `config/config_test/*.yaml` and carries a `DO NOT EDIT` banner. **Never edit a generated file by hand**: the fix belongs in the YAML, or in the generator itself if the YAML can't express it. A hand-patched generated file silently regresses the next time anyone reruns the generator.

Regenerate after touching any `config/config_test/*.yaml`:

```bash
ros2 run ros2_test_compose test_generator --ros-args -p package_name:=isaac_ros_cumotion
```

To check the generator itself hasn't drifted (e.g. after changing `ros2_test_compose`), regenerate into a scratch copy and diff instead of overwriting in place — see the "Checking the generator hasn't drifted" section of `ros2_test_compose/README.md`.

### Key YAML settings (`environment:` block)

- **`ready_service` / `ready_wait`** — how the suite knows the node is up. `setUpClass` polls `get_service_names_and_types()` until the named service is *advertised* (it is never called — observing the graph can't hang even if a service callback is broken). `unified_planner` builds its solvers synchronously in `__init__` (25–35 s), so all suites set:
  ```yaml
  ready_service: "/unified_planner/generate_trajectory"
  ready_wait: 180.0
  startup_delay: 0.0
  ```
  There is no readiness *service* to call: the node exposes its state as the `node_is_available` parameter, and `ready_service` deliberately watches the graph instead.

- **`allowable_exit_codes`** *(optional, suite-wide, not per-test)* — exit codes accepted besides `0`. The suites accept `[0, -2, -9]`: under the headless launcher the planner occasionally exits `-2` (unhandled SIGINT during solver teardown) or `-9` (the generator kills a suite that overstays its per-suite `timeout`), and a suite has already verified what it came to verify by then.

- **`wait_for_timeout`** *(default 10 s)* — per-test timeout for topic/service checks; a per-test `timeout:` key overrides it. Actions use a separate, larger `action_timeout` (default 60 s) since they run a whole trajectory.

- **Budget a solver rebuild generously.** `set_collision_cache`, `update_motion_gen_config`, the first `add_object` and `attach_object` all rebuild and re-warm the solvers *synchronously*. Measured on an idle Jetson Orin that is ~25 s — but after ~30 minutes of sustained GPU load the SoC throttles and the same rebuild takes ~37 s. Every such call therefore uses `timeout: 90.0`. A budget sitting just above the idle cost does not detect a hang, it just turns the whole suite red on a warm or shared machine.

Full reference for every key: `ros2_test_compose/README.md`.

## Known flakes

Two suites fail intermittently for reasons unrelated to what they assert. Both reproduce on an unchanged tree, so a single red run is not a regression signal — rerun the suite before investigating.

`test_test_object_franka.py` / `test_test_object_ur10e.py`: test `05 Attach object` occasionally times out (`Service call to '/unified_planner/attach_object' timed out`). Root cause: MorphIt's stochastic sphere-fit retries (`morphit_sphere_fit: attempt N returned too few spheres`), which can take longer than the test's timeout on a slow attempt.

`test_test_trajectory_franka.py` / `test_test_trajectory_ur10e.py` occasionally fail on `test_exit_codes` alone: every service assertion passes, then the planner process aborts (`SIGABRT`, exit `-6`) a few seconds after the shutdown SIGINT, during CUDA/solver teardown. Recognise it by the log — all numbered tests green, and `process has died [... exit code -6]` *after* `sending signal 'SIGINT'`. Nothing is wrong with the trajectories it just planned.

## Next steps

- [Troubleshooting](troubleshooting.md)
- [Tutorial 1: Your First Trajectory](../tutorials/01-first-trajectory.md)

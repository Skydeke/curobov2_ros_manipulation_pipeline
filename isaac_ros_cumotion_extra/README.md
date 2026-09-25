# isaac_ros_cumotion_extra

Extra tools for isaac_ros_cumotion: the **cuRobo benchmarks reproduction**
(`curobo_benchmark` — replays the reference-page benchmarks natively and
through the ROS-wrapped planner; the main content of this README), ESDF/Viser
visualization nodes, and the cuRobo robot-config generator.

## Contents

- [CLI: `build_curobo_config`](#cli-build_curobo_config) — cuRobo robot-config generator from a URDF
- [Viser nodes](#viser-nodes) — interactive FK/IK/MP/MPC visualization nodes
- [cuRobo benchmarks reproduction: `curobo_benchmark`](#curobo-benchmarks-reproduction-curobo_benchmark) — the reference-page benchmark, native + ROS legs (primary)
  - [One-shot reproduction (`docker compose`)](#one-shot-reproduction-docker-compose) — the full page in one command
- [IK and kinematics & collision parity legs](#ik-and-kinematics--collision-parity-legs) — the page's other two benchmark families

## CLI: `build_curobo_config`

Build a cuRobo v2 YAML config (collision spheres + self-collision matrix) from a URDF.

### Kortex (Kinova Gen3 + Robotiq 2F-140)

```bash
# 1. Generate URDF from xacro
xacro $(ros2 pkg prefix iki_kortex_description)/share/iki_kortex_description/urdf_xacro/kortex_standalone.urdf.xacro \
  name:=gen3 arm:=gen3 dof:=7 gripper:=robotiq_2f_140 sim_gazebo:=true \
  > /tmp/kortex.urdf

# 2. Build cuRobo config
ros2 run isaac_ros_cumotion_extra build_curobo_config \
  --urdf /tmp/kortex.urdf \
  --asset-path ~/ros2_ws \
  --output /tmp/kortex_curobo.yml \
  --tool-frame grasping_frame
```

### Any other robot

```bash
ros2 run isaac_ros_cumotion_extra build_curobo_config \
  --urdf /path/to/robot.urdf \
  --asset-path /path/to/mesh/parent \
  --output robot.yml \
  --tool-frame tool0
```

## Arguments

| Argument | Description |
|---|---|
| `--urdf` | Path to URDF file |
| `--asset-path` | Parent directory containing mesh dirs |
| `--output` | Output YAML path (default: `curobo_config.yml`) |
| `--tool-frame` | Tool frame name (default: `grasping_frame`) |
| `--base-link` | Base link (auto-detected if omitted) |
| `--sphere-density` | Sphere density multiplier (default: 1.0) |
| `--num-collision-samples` | Collision matrix samples (default: 1000) |
| `--visualize` | Start Viser viewer |

## Viser nodes

The package provides viser visualization nodes with interactive GUIs. They keep
the ROS2 service architecture — motion is computed on `curobo_server` while the
viser view adds interactive controls:

| Node | Interactive UI |
|---|---|
| `fk_viser_node` | per-joint sliders + a "Sweep" toggle to animate the arm |
| `ik_viser_node` | a draggable 6-DOF goal gizmo; IK re-solves live as you drag it |
| `mp_viser_node` | draggable goal gizmo + upstream "Move"/"Grasp" buttons (Classic; Grasp chains approach→grasp→lift waypoints) and a joint-trajectory plot |
| `mpc_viser_node` | draggable target gizmo driving a live MPC closed loop via the execution action + live-goal topic |

```bash
ros2 launch isaac_ros_cumotion_extra getting_started_viser.launch.py \
  content_path:=/path/to/robot.curobo.yml \
  urdf_path:=/tmp/robot.urdf \
  server_node:=curobo_server \
  nodes:=mp_viser_node
```

Leave `nodes` empty to start all eight demo nodes (each on its own viser port).

## cuRobo benchmarks reproduction: `curobo_benchmark`

Re-created replacement for the deleted `isaac_ros_cumotion_benchmark` package.
It runs **curobo_core's native planning benchmark** (the same solver/machinery
the [cuRobo benchmarks page](https://nvlabs.github.io/curobo/latest/reference/benchmarks.html)
is generated with) and replays the **same problems through the ROS-wrapped
planner** (`/unified_planner/generate_trajectory` on the node started by
`gen_traj.launch.py`), then compares them so wrapping curobo in ROS can be
verified to preserve planning outcomes.

Problems come from the same robometrics datasets the upstream
`motion_plan_benchmark` uses (`demo`, `motion_benchmaker`, `mpinets`).

### One-shot reproduction (`docker compose`)

The primary reproduction is one command: the compose brings up the server with
the reference solver envelope and runs the page's suites in order (motion
generation → IK → kinematics & collision), each **native + ROS**, then prints
ALL results again in the page's order — no leg is ever skipped.

```bash
docker compose -f docker/compose_benchmark.yaml up
# CUROBO_RUN_PARITY=1 additionally runs the compare-verdict suite (all/ik/cost).
```

Motion figures use the full ~2600-problem `full` dataset at the page's
100-attempt budget (`CUROBO_MAX_ATTEMPTS` — one knob for both legs; the runner
re-pins the server's plan-time `max_attempts` before each motion leg and
switches the server's torque mode at runtime, so one plain launch fills both
motion ROS rows). `CUROBO_DATASET=demo` / `--max-attempts 1` is a fast smoke
pass. Result JSONs: `/tmp/benchmark_webpage.*.json`. The full CLI and the
deep-dive (solver envelope, timing attribution, honest receipt) follow below.

The **native leg reproduces the reference benchmark**: it reuses the upstream
machinery read-only (`check_problems` / `load_curobo` from
`curobo/benchmark/motion_plan_benchmark.py` — upstream ships that directory
without `__init__.py`, so it is not an importable package and the leg loads
the script by path next to the installed `curobo` package instead of importing
it) and replays the same per-scene,
per-problem loop the reference table is generated with — curobo's bundled
`franka.yml` (tool frame `panda_hand`, joint limits expanded by ±0.2 rad),
particle + LBFGS ik/trajopt optimizers,
`optimizer_collision_activation_distance=0.0025`, `{obb: n_cubes}` collision
cache, `num_ik_seeds=32`, `num_trajopt_seeds=4`, fixed seeds, CUDA-graph warmup,
one planner per scene and the real solve at `max_attempts=100` by default —
the page's real-solve budget, for the WHOLE benchmark (every subcommand
defaults both legs to 100; the ROS runner pins the server's plan-time
`max_attempts` to the run's `--max-attempts`, so native and ROS always share
one retry envelope; lower it for a fast smoke pass). Per-problem
worlds are OBB conversions of the problem obstacles (`--mesh` switches to
meshes). Its `Metric`/`Value` table should therefore match the reference page.

The **ROS leg** replays the same problems through the ROS-wrapped planner. It
inherits the server's solver envelope (see below), so parity verdicts measure
how well the wrapper preserves the reference benchmark's outcomes.

The compare treats planning **outcomes** as the parity signal (success, path
length, motion time, waypoint count, all computed on the interpolated
trajectory the server returns) and treats **timing** as informational. Both
timing numbers come from the same fields on both legs:

- `time_s` — core: the solver's own `result.total_time`; ros: client-side wall
  around the `generate_trajectory` call.
- `solve_time_s` — curobo's `result.solve_time` on BOTH legs: the
  optimizer-iteration CUDA-event time, **accumulated across every attempt of
  the `plan_pose` retry loop** (`MotionPlanner._plan_pose_single`).

They are NOT "apples-to-apples solver speed": on the box the ROS leg's
solve_time used to track its wall (~2 s/problem) instead of native's ~0.06 s,
because the per-request work that made the wall slow — the object-set churn
(`remove_all_objects` + `add_object` reloading the solver collision model) plus
the server's drifting, unseeded RNG — was charged *inside* curobo's own timers.
So the pre-fix ~2 s was solver-loop cost in the server environment, **not**
serialization/RTT around the call (the `solve_tracks_wall_pct` summary field,
near 100% on the box, makes this visible per run). With the geometry, retry,
seed and collision-cache divergences closed (below), the ROS leg's solve_time
now reports the same calm single-attempt optimizer cost as native.

The report therefore prints the solver-reported **Solve Time** row and an
explicit `solve time (info)` line with that attribution — solve time is
informational, never a parity signal — plus `solveC`/`solveR` columns in the
per-problem delta table.

**Diagnosing the gap.** The measurements in this retrofit are pre-fix
history; the fixes they motivated are described below and now apply at the
envelope defaults. Server `[plan-perf]` logs on the box pinned the wall to
the solver: `setup ~1 ms`, `world refresh 0.0 ms`, and `plan()` (the whole
`plan_pose` call) at nearly the ros wall, with `solve ≈ plan()` (≈96-98% of
the wall). curobo's `result.solve_time` accumulates the optimizer-iteration
time across *every attempt of the `plan_pose` retry loop*
(`_plan_pose_single`), so the per-request cost sat inside curobo's
retry/optimizer loop — not serialization/RTT.

CUDA-graph re-capture is **not** the driver: relaunching the server with
`CUROBO_USE_CUDA_GRAPH=false` (eager solvers; the toggle is forwarded to the
node and honoured — the startup `MotionPlanner solver envelope:` log reports
the resolved flags on every run) left `plan()` at ~2.0 s (pre-fix),
indistinguishable from the graphs-on run. Eager warmup is only ~1.4x slower
than graph warmup (15 s vs 11 s) when nothing overlaps it — the earlier
"4.4x" reading was the concurrent core leg contaminating the warmup wall, not
a solver-mode effect. (The `gpu_lock busy (CUDA graph capture in progress)`
warning in the logs is static text fired on any failed non-blocking lock
acquire — it does *not* prove a capture was in progress.)

The retry budget WAS part of it, and the box measurement resolved it: with the
envelope at `max_attempts:=100`, `plan()` was uniform ~2.0 ± 0.1 s per problem;
capping it at `max_attempts:=1` dropped `plan()` to **~0.70 s with 5/5
first-attempt success**. So the retry loop contributed ~1.35 s/request
(≈14 ms of unseeded churn per extra attempt) while native exits the loop in
one or two calm attempts. The page's success rate still requires the budget —
native is ~99.7 % at 100 attempts but only ~90 % at 1, and the ROS legs sat at
the old one-attempt server cap for the same reason — so the **whole benchmark**
now defaults `max_attempts=100` for **both** legs (`core`/`ros`/`all` included,
not just the webpage reproduction), and the ROS runner pins the server's
plan-time `max_attempts` to the run's own `--max-attempts` before each motion
leg (one `set_parameters`, no rebuild). Both legs therefore retry like the
page: solve-time numbers stay calm for the ~99 % of problems that solve on the
first attempt and accumulate the same retry cost as native on the rare
failures. `CUROBO_MAX_ATTEMPTS=10` shows the cost-scaling curve at a shorter
wall time. (The process-level explanations for
the churn — unseeded server RNG and the denser voxel/ESDF world — remain
candidates for *why* the server's later attempts keep being explored, but they
no longer cost the reproduction anything.)

**Root cause of the 12x single-attempt gap (resolved).** One capped server
attempt used to cost ~0.67 s of `solve_time` vs native's ~0.055 s for its
entire plan. The cause is the server's obstacle representation, not retries,
CUDA-graph replay, RTT or the seed budget: `ObstacleManager._collision_supported`
converted every sphere/cylinder/capsule obstacle into a **trimesh** (the
conversion was originally introduced so the primitives would show up in the
voxel-map rasterization, which only reads cuboid/mesh/voxel buckets), so the
solvers collided the whole scene through the mesh-SDF path. The native leg's
`get_obb_world()` keeps the very same obstacles as **OBB cuboid primitives**
and collides them with the fast primitive-cuboid kernels — the per-iteration
kernel cost differs by roughly the measured 12x. (Warmup-primed CUDA-graph
replay, `max_attempts:=1` and the seed/tolerance recipe were already ruled
out; `gpu_lock busy (CUDA graph capture in progress)` fires once at the
warmup probe and never again while solves stay ~0.68 s.)

The fix: `obstacle_collision_mode:=cuboid` (new default) converts
sphere/cylinder/capsule to OBB cuboids via `get_cuboid()` — the exact same
approximation native's `create_obb_world()` uses — for every solver-bound /
rasterized scene. The voxel map still shows the obstacles (as boxes), so the
original motivation for the mesh conversion holds without the mesh cost.
`obstacle_collision_mode:=mesh` restores the
legacy exact-trimesh geometry (`CUROBO_COLLISION_MODE=mesh` in compose) and
brings the ~12x per-attempt cost back, for deployments that need it.

**Second root cause: the solver's collision cache padding (resolved).** With
the mesh path gone, one capped server attempt still cost ~0.40 s vs native's
~0.055 s for its whole plan (~7x). The remaining difference was not geometry
— the analytic-obstacle worlds now matched native's exactly — but the
*capacity* the server pre-allocates per obstacle type. curobo's Warp
collision kernels launch **one thread per (robot sphere, padded obstacle
slot) per obstacle type** (`wp_collision_kernel.py`: `dim = b*h*n*max_n`),
and the server's former deployment default
`collision_cache = {cuboid: 100, mesh: 100, voxel: ...}` padded every solver
iteration's kernel grids to 100 cuboid slots + 100 mesh slots (even with
zero meshes registered) + 1 voxel layer — **201 threads per sphere per
collision query** — while native's `{obb: n_cubes}` cache runs a single
16-slot cuboid grid. That ~12.6x thread-grid inflation (empty slots
early-return cheaply, landing at the measured ~7x) is why the server's one
attempt stayed slow even though it collided identical geometry. The server
now ships **32/4 defaults** (`collision_cache_cuboid` /
`collision_cache_mesh` launch params; exceeding the capacity raises loudly,
so deployments that need more slots raise the params instead of padding
every plan), and the ROS
benchmark leg additionally **sizes the server's collision cache to the
dataset's actual per-type obstacle counts** — the same computation as native's
`check_problems` (max over scenes of `get_obb_world().get_cache_dict()["obb"]`,
i.e. cuboids + converted sphere/cylinder/capsule) — and disables the empty
no-camera voxel layer (`blox=0`) via `SetCollisionCache` before the timed
run. One synchronous solver rebuild (~25-37 s) is paid once at startup; after
it the server runs native-equivalent kernel grids (`{obb: 16}` for the demo
dataset, mesh off, voxel off). `curobo_benchmark ros --no-size-cache` keeps
the padded deployment default for A/B.

Two recipe divergences (now aligned) compound the per-attempt cost: the server
node's `num_trajopt_seeds` default is **12**, triple the native reference's 4
(`core_runner` default), and the compose reference envelope previously did not
forward the param at all — so every server attempt solved 12 seed trajectories
instead of 4 (~3x per-attempt trajopt cost, and a *different winner*: more
seeds → a different optimum, which is exactly the path/motion/waypoint deltas
the report shows). The envelope now pins `num_trajopt_seeds:=4`
(`CUROBO_NUM_TRAJOPT_SEEDS` to override; 12 = the deployment default) so the
ROS leg runs the same seed budget as native. The envelope log line
(`MotionPlanner solver envelope: use_cuda_graph=…, num_ik_seeds=…,
num_trajopt_seeds=…, collision_activation_distance=…`) self-verifies both
divergences are resolved on every run.

The native-leg seed knobs are secondary (they reproduce the server's RNG
drift; useful for ruling it out cheaply): `--no-reset-seed` skips
`mg.reset_seed()` before each solve, `--unseeded` also skips the upstream
fixed seeds (`seed_globals`, `np/random/torch.manual_seed(2)`).
`curobo_benchmark core --unseeded` should stay at ~0.06 s.

Each leg is also printed as a `Metric`/`Value` grid table in the same layout
as the upstream `curobo/benchmark/motion_plan_benchmark.py` report ("native
(curobo_core)" and "ros (unified_planner)"), so the two can be compared at a
glance against the numbers on the
[cuRobo benchmarks page](https://nvlabs.github.io/curobo/latest/reference/benchmarks.html).

### Solver envelope and scene selection

The two legs now run the **same solver recipe by default**: both use
particle + LBFGS for IK and trajopt — the server's MotionPlanner builds the
same optimizer set as the native curobo reference benchmark in every startup
(no flag) — so the envelope only has to align the scene frame and the
retry/seed caps:

- `robot_config_file:=config/franka.curobo.reference.yml` — same panda_hand
  tool frame as the reference franka.yml (the robometrics goal poses are
  defined for `panda_hand`; the product default `grasping_frame` sits a
  finger-length further out and makes those goals unreachable);
- `max_attempts` (plan-time — the compose launches the server at
  `${CUROBO_MAX_ATTEMPTS:-100}`, the page's budget) and
  `collision_activation_distance:=0.0025`, with `num_trajopt_seeds:=4` — the
  reference recipe's retry budget and seed count. Every benchmark subcommand
  defaults both legs to 100: the ROS runner pins the server's plan-time
  `max_attempts` to the run's own `--max-attempts` before each motion leg
  (stock `set_parameters`, no rebuild — one knob, identical envelopes).
  (back when the server defaulted to an LBFGS-only single-attempt solver, this
  envelope was what made the ROS leg pass the hard problems; the solver recipe
  itself is now common by default). `max_attempts` is plumbed from the node
  parameter (declared, default 1) into `_get_planner_config` → `plan_pose`, so
  the launch argument actually takes effect.
- `obstacle_collision_mode:=cuboid` (default) — sphere/cylinder/capsule
  obstacles are converted to the same OBB cuboids native's `get_obb_world()`
  produces, instead of the legacy trimesh conversion that made the mesh-SDF
  path ~12x slower per solver attempt (see "timing attribution" above).
  `:=mesh` restores the exact-trimesh behaviour.
- The ROS leg sizes the server's collision cache to the dataset before the
  timed run (one synchronous rebuild at startup): the Warp collision kernels
  launch one thread per (sphere, padded obstacle slot) per type, and even the
  server's lean 32/4 defaults (`collision_cache_cuboid` /
  `collision_cache_mesh`) pad grids larger than native's exact `{obb: n_cubes}`
  (plus the empty no-camera voxel layer). `SetCollisionCache` sets
  `obb=cuboids+converted` (the same count as native's `check_problems`),
  `mesh=0` for the demo dataset, `blox=0` (no cameras — no voxel layer).
  `curobo_benchmark ros --no-size-cache` keeps the padded default for A/B.

`docker/compose_benchmark.yaml` launches the server with exactly this
envelope, so `curobo_benchmark all` produces comparable metrics from both legs
on the hard `demo`/`bookshelf_*` problems. Both legs share the particle +
LBFGS solver recipe by default, so a bare `gen_traj.launch.py` startup (no
envelope) runs the same solvers as the benchmark.

The ROS leg computes path length / motion time / max|jerk| client-side from
the interpolated trajectory the server returns (jerk via a third finite
difference — the native leg reports the planner's control-point jerk), so both
tables show the same rows; `Position Error (mm)` is reported on both legs —
the ROS leg reads the winner's solver-reported `position_error` from its
`PlanningStats.considered` row (the runner sets `log_considered_trajectories`
— a reporting-only flag, accepted on the classic planner; it gates the detail
block, never the search — so the solver's per-seed residual rides out of the
server as `max_waypoint_error`, meters, ×1000 here). That is the same
convergence metric the native leg records: max-over-links position error at
the last timestep of the optimized trajectory, which the optimizer stops as
soon as it is within tolerance — so values range from ~0 up to ~mm. (The
returned interpolated trajectory's final waypoint is pinned exactly to the
goal joint state by implicit-goal interpolation, so FK'ing it — as an earlier
revision of this harness did — reports ~0 mm *by construction*, not solver
accuracy; that is why the row is not FK-derived.) The ROS **Solve Time (s)**
row is read from the same
winner's `PlanningStats.considered` row, so the same
`result.solve_time` the native leg records rides out of the server.

With `obstacle_collision_mode:=cuboid` (default) the analytic-obstacle
geometry matches native's `{obb: n_cubes}` cache exactly — both legs convert
sphere/cylinder/capsule obstacles to the same OBB cuboids via `get_cuboid()`
— and the ROS leg additionally sizes the server's collision cache to the
dataset's actual per-type counts with the (empty, no-camera) voxel layer
disabled, so the solver kernel grids match native's too. The remaining
difference between the two legs is RNG: the server process does not seed its
RNG the way the native leg's `seed(2)` does, so its seed candidate selection
can pick a different (equally valid) trajectory. Residual path/motion deltas
are reported by the parity verdict, not hidden.

Scene keys are printed at the start of each leg's run. Restrict a run with
`--scene` (also on `core`/`ros`); `box_panda` goals sit under the table and are
unreachable from the floor-mounted base, so skip that scene.

```bash
ros2 run isaac_ros_cumotion_extra curobo_benchmark all \
  --dataset mpinets --scene dresser_neutral_start
ros2 run isaac_ros_cumotion_extra curobo_benchmark all \
  --dataset motion_benchmaker --scene table_pick_panda
```

### Reproducing the reference page's tables (with / without torque limits)

The page's "Latest Motion Generation Results" **both tables** come from the
same run over the combined dataset — `--dataset full`
(motion_benchmaker + mpinets, the page's 2600 problems), fixed seeds
(`np`/`random`/`torch` = 2) and per-problem `reset_seed`, exactly like the
upstream script's `file_paths` loop; the only difference between the two
tables is `--use-dynamics --mass 3.0` (the full 3 kg payload). "full" keeps
upstream's **per-dataset** `mpinets_data` classification (benchmaker scenes
never gain the mpinets finger locks), and `--scene` filtering works on it
like any other dataset.

- `Energy (J)` / `Torque (N·m)` rows are computed for **every** successful
  plan in *both* tables via the upstream Pinocchio recipe
  (`load_robot_model_for_dynamics("franka", attached_object_mass=mass)` +
  `compute_trajectory_energy` on the B-spline solution) — the reference page
  reports both columns in both tables. `--use-dynamics` only toggles the
  solver's torque-limited mode (`robot_cfg["load_dynamics"]` + the mass
  payload on the planner); the energy columns are not gated on it.
- The torque-limits section flows through the ROS server leg too: the
  *webpage* run switches the running server's torque mode itself — stock
  `set_parameters` (`load_dynamics`, `robot_payload_mass`) + the
  `update_motion_gen_config` trigger rebuilds the MotionPlanner from them
  (see `docs/concepts/parameters.md`; the launch-time flags below are only
  the initial state). Standalone legs can still do it explicitly:
  `curobo_benchmark ros --use-dynamics --mass 3.0 ...` after launch-time
  `load_dynamics:=true robot_payload_mass:=3.0`. Per the accepted "server
  plans; client computes" split the server needs **no response-schema
  change**; `ros_runner.py` reconstructs Energy/Torque client-side from the
  returned trajectory. The wire carries positions + velocity + dt but **no
  acceleration**, so the ROS-leg numbers are an approximation (`qdd`
  finite-differenced from velocity) and may not match the native Pinocchio
  values.
- Honest receipt notes: the whole benchmark defaults `max_attempts=100` for
  BOTH legs — every subcommand (`webpage`, `reference`, `all`, `core`, `ros`)
  and the compose-launched server share the page's 100-attempt budget, and the
  runner pins the server's plan-time `max_attempts` to the run's
  `--max-attempts` before each motion ROS leg, so a plain
  `docker compose -f docker/compose_benchmark.yaml up` reproduces the page's
  99.73 % success on native AND ROS alike (~90 % is what either leg scores at
  `--max-attempts 1`, the old harness default). And
  the box's GPU differs from the page's RTX 6000 Ada, so the timing columns
  stay informational; the quality metrics are deterministic given the same
  config + seeds + dataset.
- Server robot-config parity: the ROS motion-gen solver runs the same franka
  kinematics, solver recipe (particle+LBFGS, same transitions), seeds (32/4),
  `collision_activation_distance:=0.0025`, and the `load_dynamics` /
  `robot_payload_mass` envelope knobs as the page, on
  `config/franka.curobo.reference.yml` (`tool_frames: [panda_hand]`,
  `collision_sphere_buffer: 0.0` — the benchmark forces 0.0, not the 0.004
  product default). Two **accepted** deltas vs upstream `load_curobo` remain:
  (1) the page widens the solver's joint limits by ±0.2 rad; the server
  plans with the raw URDF limits (the shared `robot_cfg` also feeds the
  IK/FK services, whose native legs use raw limits — widening motion-gen
  only would require a solver-local cfg copy); (2) dresser (mpinets)
  scenes lock the finger joints at 0.025 in the page vs the yml's 0.04 on
  the server (affects finger-sphere geometry only; the `panda_hand`
  tool-frame result is unaffected).

The webpage order, both legs — `webpage` runs the page's suites in the page's
order (motion generation → IK → kinematics & collision), each once with the
native curobo leg and once via the ROS server, then prints ALL results again
grouped exactly like the page (the docker compose default; `reference` is the
motion-generation-only variant of the same envelope). With `--run-ros`,
nothing is ever skipped — neither a native nor a ROS leg. The two motion ROS
rows differ only in the server's torque mode, and the runner ensures it per
leg via the runtime switch (`set_parameters` + `update_motion_gen_config`:
the without-torque leg asks for plain, the with-torque leg asks for
torque-limits at `--mass`), so **one server life fills both motion ROS rows**
— the launch-time `load_dynamics` / `robot_payload_mass` are just the initial
state the runner overrides:

```bash
ros2 run isaac_ros_cumotion_extra curobo_benchmark webpage \
  --dataset full -o /tmp/benchmark_webpage.json
# native-only (no server needed): omit --run-ros
#   -> runs, in order: motion gen without torque -> with torque (3 kg) ->
#      IK -> kinematics & collision, each native + ROS (when the server is
#      up), then prints everything again in that order, plus the
#      published-page comparison grids; JSONs:
#      /tmp/benchmark_webpage.{motion-plain-core,motion-plain-ros,
#      motion-torque-core,motion-torque-ros,ik-core,ik-ros,cost-core,
#      cost-ros}.json
#   CUROBO_DATASET=demo / --max-attempts 1  -> fast smoke pass
```

```bash
# Rebuild only if you want the `ros2 run` entry point (not required to run):
#   colcon build --symlink-install --packages-select isaac_ros_cumotion_extra
ros2 run isaac_ros_cumotion_extra curobo_benchmark --help

# Run everything against a running server (core + ROS + compare):
ros2 run isaac_ros_cumotion_extra curobo_benchmark all \
  --dataset demo --show-all -o /tmp/benchmark_report.json

# Or run the legs independently:
ros2 run isaac_ros_cumotion_extra curobo_benchmark core --dataset demo -o /tmp/core.json
ros2 run isaac_ros_cumotion_extra curobo_benchmark ros --dataset demo -o /tmp/ros.json
ros2 run isaac_ros_cumotion_extra curobo_benchmark compare \
  /tmp/core.json /tmp/ros.json --show-all
```

From the source tree without a rebuild:

```bash
export PYTHONPATH=/root/ros2_ws/src/isaac_ros_cumotion_extra:$PYTHONPATH
python3 -m isaac_ros_cumotion_extra.benchmark.run all --dataset demo
```

Inspect a result JSON inside the compose network (the benchmark service writes
them to the shared `/tmp`):

```bash
docker compose -f docker/compose_benchmark.yaml exec curobo_benchmark \
  cat /tmp/benchmark_webpage.motion-plain-core.json
```

## IK and kinematics & collision parity legs

Besides motion planning, the [cuRobo benchmarks
page](https://nvlabs.github.io/curobo/latest/reference/benchmarks.html) also
lists an **inverse-kinematics** family (upstream
`benchmark/ik_benchmark.py`) and a **kinematics & collision** family
(upstream `benchmark/cost_gradient_benchmark.py`). The compose's parity suite
(`CUROBO_RUN_PARITY=1`; `all` motion planning, then `ik`, then `cost`) runs
all three, and the CLI exposes each leg:

```bash
# native IK leg                -> report.json + report.ik-core.json
curobo_benchmark ik-core -o /tmp/ik_report.json
# ROS IK leg (server up)       -> report.json + report.ik-ros.json
curobo_benchmark ik-ros  -o /tmp/ik_report.json
# both + compare (verdict = exit code)
curobo_benchmark ik -o /tmp/ik_report.json

# native FK + collision leg    -> report.json + report.cost-core.json
curobo_benchmark cost-core -o /tmp/cost_report.json
# ROS FK/collision leg         -> report.json + report.cost-ros.json
curobo_benchmark cost-ros  -o /tmp/cost_report.json
curobo_benchmark cost -o /tmp/cost_report.json

# compare existing JSONs (planning | ik | cost; default planning)
curobo_benchmark compare core.json ros.json --capability ik
```

Both new families solve/evaluate **shared, deterministically reseeded
inputs** (`benchmark/synthetic.py`): the ROS leg can't read the upstream
benchmark's internal RNG, so the harness regenerates identical goal poses
(IK) and joint configurations (cost) for both legs from `--seed` (default 2),
mirroring the upstream benchmarks' own sample-config input style.

**IK parity (`ik`)** solves collision-free IK goals through the server's
`/ik_batch` (per-goal `joint_states_valid` = the solver's best-seed `success`)
and natively with the server's exact `InverseKinematicsCfg.create` recipe
(`--robot-config` defaults to the envelope's
`config/franka.curobo.reference.yml`; 32 seeds, position tolerance 0.005 m,
orientation tolerance 0.05 rad, `{cuboid: 1}` cache — the table world). The
ROS response carries no solver residual, so **both** legs score the same
FK-verified `position_error_mm` / `orientation_error_deg` of the returned
first-seed solution versus the goal. Parity is success agreement plus those
two errors within `--position-tolerance-mm` (default 0.5) and
`--orientation-tolerance-deg` (default 0.5); per-goal `time_ms` is
informational. The native leg also emits the upstream-style `plain`
(kinematics-only, no scene) variant as a reference row — the server has no
collision-free-off IK mode, so `plain` is **never** compared (the parity
verdict only counts `cfree` goals).

**Kinematics & collision parity (`cost`)** evaluates the shared config batches
through the server's `/fk_batch` (per-config `poses_valid` = joint-limits +
self + scene collision via `RobotCollisionChecker.validate`) and natively with
the same model + `RobotCollisionCheckerCfg.load_from_config(
collision_activation_distance=0.001)` on the cost world (table + tall
cuboid). Parity is `valid` agreement (exact) plus FK tool-pose agreement
within `--position-tolerance-mm` / `--orientation-tolerance-deg` (defaults
1e-3 — the expected deltas are float32 reproducibility, ~0).

Both ROS legs size the server's collision cache to their synthetic world
(cuboid=1 for IK, cuboid=2 for cost, mesh/voxel off) before the timed runs,
so the server's collision kernels match the native side. The setup order is
`remove_all_objects` → `SetCollisionCache` → `add_object` → `warmup_ik` /
`warmup_fk`: cuRobo requires the registered scene to never hold more cuboids
than the active cache capacity — a cache change rebuilds every solver from
the then-current scene (raising if it exceeds the new cap), and an add raises
if it would exceed the current cap. Clearing first empties the scene (so
sizing down over planning-suite leftovers, up to 8 cuboids, is safe) and
sizing *before* the adds guarantees the parity world can never overflow the
previous leg's shrunk cache (the IK leg shrinks it to 1, so a `clear → add →
size` order would crash the node on the cost world's second cuboid). The
sized solver then captures the registered world at construction
(`--no-size-cache` keeps the padded defaults for A/B).

Every cuRobo construction in these legs — native and the server's `FkBatch`
collision validator — runs on an explicitly indexed CUDA device (`cuda:0`),
canonicalized from a bare `"cuda"` by `synthetic.canonical_device`. The
upstream reference scripts do the same (`DeviceCfg` defaults to
`torch.device("cuda", 0)`; `cost_gradient_benchmark.py` passes `"cuda:0"`):
Warp's `wp.device_from_torch` requires a device index, so an index-less
`torch.device("cuda")` raises `TypeError` whenever curobo builds collision
`MeshData`. This also fixes the server validator, which previously swallowed
that crash and silently reported every `poses_valid` as True.

Pure-Python tests have their own compose (`curobo_test` — the benchmark
package's tests — plus `curobo_task_constructor_test`, the
task-constructor core suite driven through its `MockCuroboServer` double):

```bash
docker compose -f docker/compose_tests.yaml up
```

Module layout (`isaac_ros_cumotion_extra/benchmark/`):

| Module | What it does |
|---|---|
| `problems.py` | robometrics loaders (import-guarded; needs `curobo[benchmark]`) |
| `core_runner.py` | native leg: replays upstream `motion_plan_benchmark` (reference benchmark config) |
| `ros_runner.py` | ROS leg: `generate_trajectory` + `add_object`/`remove_all_objects` |
| `obstacle_convert.py` | cuRobo obstacle dicts -> AddObject payloads (pure Python) |
| `compare.py` | parity report (success/path/motion/waypoints; time informational) |
| `run.py` | `curobo_benchmark` CLI (`core`/`ros`/`webpage`/`reference`/`all`/`compare`, `ik`/`ik-core`/`ik-ros`, `cost`/`cost-core`/`cost-ros`) |

Pure-Python tests live in `benchmark/tests/` and run without curobo/torch/ROS
(robometrics-gated tests skip when robometrics is absent).

# isaac_ros_cumotion_benchmark

**Local benchmarking/unit-testing tool** that runs curobo_core's own benchmark
problems two ways and reports where they disagree:

1. **`core`** — directly against cuRobo's `MotionPlanner`, exactly the way
   `curobo_core`'s own `benchmark/motion_plan_benchmark.py` does it.
2. **`ros`** — the *same* problems through the running `curobo_server_node`
   over ROS (`cumotion/update_world` + `cumotion/plan_motion`).

The disagreement is the deliverable: it surfaces bugs in the ROS wrapping
layer that a single "does it plan?" smoke test would never catch.

**This is not a CI job, not a regression gate.** It runs when a human (or a
future agent) runs it, the same way `motion_plan_benchmark.py` already does
today. No pipeline runs this automatically.

---

## Prerequisites

1. **`curobo_core` benchmark extras** — install `robometrics` and other
   benchmark dependencies:
   ```bash
   cd submodules/isaac_ros_cumotion/curobo_core/curobo
   pip install -e ".[benchmark]"
   ```

2. **colcon workspace** — this package must be built in the same colcon
   workspace as `isaac_ros_cumotion`, `isaac_ros_cumotion_interfaces`, and
   their ROS message packages:
   ```bash
   cd <your_ws>
   colcon build --packages-select isaac_ros_cumotion_benchmark \
       isaac_ros_cumotion isaac_ros_cumotion_interfaces
   source install/setup.bash
   ```

3. **CUDA-capable GPU** — both the core runner and the `curobo_server_node`
   require a CUDA-capable GPU with cuRobo's torch dependencies.

---

## How to bring up the ROS side

Start a Franka `curobo_server_node` in a separate terminal:

```bash
ros2 run isaac_ros_cumotion curobo_server_node \
    --ros-args -p robot:=`ros2 pkg prefix isaac_ros_cumotion_robot_description`/share/isaac_ros_cumotion_robot_description/xrdf/franka.xrdf \
    -p urdf_path:=`ros2 pkg prefix moveit_resources_panda_description`/share/urdf/panda.urdf
```

**Confirm it's up before running `ros` / `all`:**

```bash
ros2 action list          # should show cumotion/plan_motion
ros2 service list | grep cumotion  # should show cumotion/update_world
```

---

## CLI usage

All subcommands are accessed via the single `curobo_benchmark` entry point:

```bash
ros2 run isaac_ros_cumotion_benchmark curobo_benchmark <subcommand> [options]
```

Or directly from the source tree:

```bash
python3 -m isaac_ros_cumotion_benchmark.run <subcommand> [options]
```

### `core` — direct cuRobo benchmark

```bash
ros2 run isaac_ros_cumotion_benchmark curobo_benchmark core --dataset demo
```

Optional: `--output results.json` to save per-problem results.

Datasets: `demo` (5 problems, fast), `motion_benchmaker` (800 problems),
`mpinets` (1800 problems).

### `ros` — benchmark through curobo_server_node

```bash
ros2 run isaac_ros_cumotion_benchmark curobo_benchmark ros --dataset demo
```

Make sure a `curobo_server_node` is running first (see above).

Optional flags: `--time_dilation_factor 1.0` (default, matches core's
implicit scaling; set to `0.0` to trigger the node's fallback to 0.1).

### `compare` — diff core vs ros results

```bash
ros2 run isaac_ros_cumotion_benchmark curobo_benchmark compare \
    core_results.json ros_results.json [--show-all]
```

Shows per-problem mismatches. Use `--show-all` to see matching entries too.
Use `--output report.json` to save the full diff as JSON.

### `all` — core + ros + compare in one command

```bash
ros2 run isaac_ros_cumotion_benchmark curobo_benchmark all --dataset demo \
    --output report.json --save-all
```

Runs both runners sequentially, then prints the comparison report.
With `--save-all`, saves `*_core.json`, `*_ros.json`, and `report.json`.

---

## Running as a unit test

```bash
colcon test --packages-select isaac_ros_cumotion_benchmark
```

Or plain pytest (from the package root):

```bash
python3 -m pytest test/
```

**What a pass looks like:** the parity test (`test_benchmark_parity.py`) is
currently skipped with a clear reason because it requires both a CUDA-capable
GPU and a live `curobo_server_node`. When those are available, remove the
`@pytest.mark.skip` decorator and run again. A pass means zero
`core.success != ros.success` mismatches across all demo problems.

---

## Reading a comparison report

A mismatch line looks like:

```
scene_5: core.success=True != ros.success=False  (core=0.342s, ros=0.000s)
```

This means problem `bookshelf_small_panda_5` was solved by the direct cuRobo
call but *not* by the ROS-wrapped call — indicating a bug in the ROS wrapping
layer for that problem's specific obstacle/goal combination.

Common divergence classes to check first (see §7 of AGENTS.md):

- **Tool frame default.** Does `ros_runner`'s empty `tool_frame` resolve to
  the same frame `core_runner` uses?
- **Time dilation factor.** The ROS node treats `time_dilation_factor == 0.0`
  as `0.1`; if you don't send an explicit value, timing numbers are
  meaningless.
- **World staleness.** If `ros_runner` doesn't fully replace the world per
  problem, leftover obstacles from previous problems leak into the current
  one. This runner uses `CLEAR_ALL + REPLACE` to guarantee isolation.
- **Locking/serialization.** The server funnels all requests through a
  `MutuallyExclusiveCallbackGroup` plus a lock; run problems sequentially
  (which this runner does) for comparable timing.

---

## Package structure

```
isaac_ros_cumotion_benchmark/
├── package.xml
├── setup.py / setup.cfg
├── resource/isaac_ros_cumotion_benchmark
├── isaac_ros_cumotion_benchmark/
│   ├── __init__.py
│   ├── problems.py          # loads robometrics problem sets
│   ├── core_runner.py       # direct cuRobo benchmark
│   ├── ros_runner.py        # through-ROS benchmark
│   ├── obstacle_convert.py  # cuRobo scene → CollisionObject
│   ├── compare.py           # diff core vs ros results
│   └── run.py               # CLI entry point
├── test/
│   ├── test_copyright.py, test_flake8.py, test_pep257.py  # lint
│   └── test_benchmark_parity.py   # the actual parity test
└── README.md
```

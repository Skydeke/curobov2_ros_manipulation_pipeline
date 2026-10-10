# cuRobo Benchmark — NVIDIA GeForce RTX 4060 Laptop GPU

Benchmark run: **2026-10-10**  

This report records a cuRobo benchmark run performed on an NVIDIA GeForce RTX 4060 Laptop GPU. The benchmark completed with exit code `0`.

## Hardware

- GPU: **NVIDIA GeForce RTX 4060 Laptop GPU**
- VRAM: **8 GiB**

## Software / Benchmark Configuration

- CUDA version: **13.2.0**
- Robot: **Franka Emika Panda**
- cuRobo configuration: `franka.yml` (reference: `franka.curobo.reference.yml`)
- Solver: **particle + LBFGS**
- IK seeds: **32**
- Trajectory optimization seeds: **4**
- Maximum planning attempts: **100**
- CUDA graphs: **enabled**
- Dataset: **full cuRobo benchmark scene set** (2590 motion problems)
- Attached payload used for the benchmark: **3.0 kg**
- Motion interpolation timestep: **0.025 s**
- Torque-limited run: `use_dynamics=true`
- Non-torque-limited run: `use_dynamics=false`

The benchmark exercised both native cuRobo and the ROS server interface.

Notable since the previous report (2026-09-28): the ROS leg now sizes the
server collision cache per scene (native-equivalent kernel grids) instead of
once per leg, and reports solver time, position error, and motion time from
the same cuRobo-native definitions (motion time excludes interpolated
boundary knots, as the reference does).

## Results

### Motion Generation

#### Without torque limits — native cuRobo

| Metric | Mean ± Std. Dev. | Median | 75% | 98% |
|---|---:|---:|---:|---:|
| Success (%) | 99.69 | — | — | — |
| Plan Time (s) | 0.108 ± 0.055 | 0.102 | 0.114 | 0.197 |
| Solve Time (s) | 0.097 ± 0.053 | 0.092 | 0.102 | 0.175 |
| Position Error (mm) | 0.043 ± 0.332 | 0.000 | 0.000 | 0.407 |
| Path Length (rad.) | 3.125 ± 1.048 | 3.256 | 3.818 | 5.049 |
| Motion Time (s) | 1.244 ± 0.344 | 1.241 | 1.485 | 2.066 |
| Jerk | 227.919 ± 83.350 | 214.263 | 267.672 | 465.281 |
| Energy (J) | 89.302 ± 49.197 | 79.402 | 116.757 | 201.817 |
| Torque (N·m) | 70.923 ± 25.181 | 67.514 | 86.202 | 131.350 |

#### Without torque limits — ROS server

| Metric | Mean ± Std. Dev. | Median | 75% | 98% |
|---|---:|---:|---:|---:|
| Success (%) | 99.65 | — | — | — |
| Plan Time (s) | 0.156 ± 0.151 | 0.149 | 0.196 | 0.350 |
| Solve Time (s) | 0.109 ± 0.054 | 0.114 | 0.140 | 0.189 |
| Position Error (mm) | 0.042 ± 0.327 | 0.000 | 0.000 | 0.395 |
| Path Length (rad.) | 3.123 ± 1.045 | 3.253 | 3.809 | 5.056 |
| Motion Time (s) | 1.243 ± 0.340 | 1.240 | 1.487 | 2.065 |
| Jerk | 125.825 ± 42.091 | 114.700 | 154.068 | 232.524 |
| Energy (J) | 86.790 ± 48.825 | 76.569 | 112.738 | 203.485 |
| Torque (N·m) | 63.526 ± 20.761 | 60.856 | 77.445 | 112.750 |

#### With torque limits — native cuRobo

| Metric | Mean ± Std. Dev. | Median | 75% | 98% |
|---|---:|---:|---:|---:|
| Success (%) | 99.69 | — | — | — |
| Plan Time (s) | 0.137 ± 0.091 | 0.116 | 0.186 | 0.472 |
| Solve Time (s) | 0.122 ± 0.086 | 0.103 | 0.170 | 0.439 |
| Position Error (mm) | 0.042 ± 0.340 | 0.000 | 0.000 | 0.348 |
| Path Length (rad.) | 3.233 ± 1.133 | 3.321 | 3.898 | 5.653 |
| Motion Time (s) | 1.329 ± 0.476 | 1.302 | 1.533 | 2.645 |
| Jerk | 218.889 ± 81.606 | 205.162 | 253.982 | 445.907 |
| Energy (J) | 81.294 ± 41.588 | 73.220 | 105.012 | 175.174 |
| Torque (N·m) | 62.260 ± 13.721 | 65.327 | 73.298 | 82.741 |

#### With torque limits — ROS server

| Metric | Mean ± Std. Dev. | Median | 75% | 98% |
|---|---:|---:|---:|---:|
| Success (%) | 99.69 | — | — | — |
| Plan Time (s) | 0.196 ± 0.183 | 0.170 | 0.213 | 0.680 |
| Solve Time (s) | 0.133 ± 0.088 | 0.128 | 0.172 | 0.437 |
| Position Error (mm) | 0.035 ± 0.294 | 0.000 | 0.000 | 0.265 |
| Path Length (rad.) | 3.235 ± 1.145 | 3.314 | 3.903 | 5.557 |
| Motion Time (s) | 1.328 ± 0.467 | 1.304 | 1.534 | 2.611 |
| Jerk | 125.574 ± 41.806 | 117.027 | 152.294 | 230.117 |
| Energy (J) | 79.700 ± 41.733 | 71.769 | 103.048 | 175.266 |
| Torque (N·m) | 58.388 ± 13.891 | 60.087 | 69.668 | 80.999 |

### Inverse Kinematics

| Interface | Success (%) | IK Time mean ± Std. Dev. (ms) | Median (ms) | 75% (ms) | 98% (ms) |
|---|---:|---:|---:|---:|---:|
| Native cuRobo | 100.00 | 38.422 ± 1.565 | 37.899 | 38.603 | 41.345 |
| ROS `/ik_batch` | 100.00 | 50.293 ± 3.473 | 48.645 | 49.220 | 57.193 |

Position error was **0.001 ± 0.009 mm** (median 0.000 mm, 98th percentile 0.010 mm) for both interfaces. Orientation error was **0.000 ± 0.000°** for both interfaces.

### Kinematics & Collision

| Interface | Valid (%) | FK Time mean ± Std. Dev. (ms) | Median (ms) | 75% (ms) | 98% (ms) |
|---|---:|---:|---:|---:|---:|
| Native cuRobo | 72.80 | 1.608 ± 0.111 | 1.556 | 1.560 | 1.829 |
| ROS `/fk_batch` | 72.80 | 7.176 ± 0.638 | 6.967 | 7.325 | 8.316 |

Per-sample FK time:

| Interface | Mean ± Std. Dev. (ms) | Median (ms) | 75% (ms) | 98% (ms) |
|---|---:|---:|---:|---:|
| Native cuRobo | 0.016 ± 0.001 | 0.016 | 0.016 | 0.018 |
| ROS `/fk_batch` | 0.072 ± 0.006 | 0.070 | 0.073 | 0.083 |

## Comparison with Published RTX 6000 Ada Benchmark

The benchmark log also compares motion-generation results with the published cuRobo benchmark using an **RTX 6000 Ada**.

The "Published RTX 6000 Ada" figures in the two tables below are transcribed from the cuRobo documentation, section
**"Latest Motion Generation Results"**:
<https://nvlabs.github.io/curobo/latest/reference/benchmarks.html>

All 18 published means and the 4 published medians quoted here were checked against that page. The same values are
hardcoded in `curobov2_ros_extra/benchmark/compare.py` (`PUBLISHED_REFERENCE_PAGE`) and printed with their source
URL, so a reader of the raw log can check them without trusting this report.

### Without torque limits

| Metric | Published RTX 6000 Ada | This RTX 4060 Laptop GPU | Published Median | This Median |
|---|---:|---:|---:|---:|
| Success (%) | 99.73 | 99.691 | — | — |
| Plan Time (s) | 0.038 | 0.108 | — | 0.102 |
| Solve Time (s) | 0.031 | 0.097 | — | 0.092 |
| Position Error (mm) | 0.041 | 0.043 | — | 0.000 |
| Path Length (rad.) | 3.126 | 3.125 | — | 3.256 |
| Motion Time (s) | 1.250 | 1.244 | — | 1.241 |
| Jerk | 227.365 | 227.919 | — | 214.263 |
| Energy (J) | 89.270 | 89.302 | 78.959 | 79.402 |
| Torque (N·m) | 71.028 | 70.923 | 67.328 | 67.514 |

### With torque limits — 3 kg payload

| Metric | Published RTX 6000 Ada | This RTX 4060 Laptop GPU | Published Median | This Median |
|---|---:|---:|---:|---:|
| Success (%) | 99.73 | 99.691 | — | — |
| Plan Time (s) | 0.052 | 0.137 | — | 0.116 |
| Solve Time (s) | 0.042 | 0.122 | — | 0.103 |
| Position Error (mm) | 0.042 | 0.042 | — | 0.000 |
| Path Length (rad.) | 3.234 | 3.233 | — | 3.321 |
| Motion Time (s) | 1.336 | 1.329 | — | 1.302 |
| Jerk | 217.786 | 218.889 | — | 205.162 |
| Energy (J) | 81.409 | 81.294 | 72.707 | 73.220 |
| Torque (N·m) | 62.270 | 62.260 | 65.345 | 65.327 |

> The benchmark log notes that timing columns are informational when comparing different GPUs. The quality metrics (success, position error, path/motion, jerk, energy, and torque) are deterministic given the same configuration, seeds, and dataset.
>
> Two caveats on that determinism claim, which the comparison above does not resolve:
>
> - **The dataset is not established by the comparison itself.** The run used `--dataset full`, the 2600 problems the page
>   aggregates, but the grid prints identically for any other dataset or a `--scene`-restricted subset, and `--dataset`
>   defaults to `demo`.
> - **"Same configuration" is unverifiable from the source.** The page publishes no solver configuration (IK seed count,
>   trajectory-optimization seed count, attempt budget, interpolation timestep). This run used 32 / 4 / 100 / 0.025 s;
>   whether the published run used the same is not knowable from the page.
>
> Only the mean and median columns are compared. The page also publishes the 75th and 98th percentiles, and the 98th
> percentile is where a dataset or configuration divergence would actually show up — mean path length agreeing to
> 3.126 vs 3.124 rad is consistent with an identical dataset but does not demonstrate one.

## Run Notes

- The benchmark used a **100-attempt budget** for both motion-generation legs.
- The benchmark runner switched the server's torque mode between the two motion-generation passes.
- ROS-leg energy and torque values are reconstructed client-side from returned trajectories using velocity finite differences because the wire format does not carry acceleration.
- ROS motion Solve Time, Position Error, and Motion Time rows now use the same cuRobo-native definitions as the reference leg (solver self-reported solve time, winner residual × 1000, knot-based duration excluding interpolated boundary knots).
- Perception was inactive during the benchmark: no cameras or lasers were configured.
- The benchmark used an empty initial scene; obstacles were added and removed as part of the benchmark dataset.
- Wall time per leg: motion plain native 8m06s / ROS 10m16s, motion torque native 9m20s / ROS 11m50s, IK 5s / 5s, cost native 0s / ROS 3s (39m47s total).
- Final reported exit code: **0**.

## Source

This Markdown report was generated from the raw benchmark output captured on 2026-10-10.

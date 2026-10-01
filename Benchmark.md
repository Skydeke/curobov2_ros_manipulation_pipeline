# cuRobo Benchmark — NVIDIA GeForce RTX 4060 Laptop GPU

Benchmark run: **2026-09-28**  

This report records a cuRobo benchmark run performed on an NVIDIA GeForce RTX 4060 Laptop GPU. The benchmark completed with exit code `0`.

## Hardware

- GPU: **NVIDIA GeForce RTX 4060 Laptop GPU**
- VRAM: **8 GiB**

## Software / Benchmark Configuration

- CUDA version: **13.2.0**
- Robot: **Franka Emika Panda**
- cuRobo configuration: `franka.yml`
- Solver: **particle + LBFGS**
- IK seeds: **32**
- Trajectory optimization seeds: **4**
- Maximum planning attempts: **100**
- CUDA graphs: **enabled**
- Dataset: **full cuRobo benchmark scene set**
- Attached payload used for the benchmark: **3.0 kg**
- Motion interpolation timestep: **0.025 s**
- Torque-limited run: `use_dynamics=true`
- Non-torque-limited run: `use_dynamics=false`

The benchmark exercised both native cuRobo and the ROS server interface.

## Results

### Motion Generation

#### Without torque limits — native cuRobo

| Metric | Mean ± Std. Dev. | Median | 75% | 98% |
|---|---:|---:|---:|---:|
| Success (%) | 99.69 | — | — | — |
| Plan Time (s) | 0.111 ± 0.056 | 0.102 | 0.125 | 0.208 |
| Solve Time (s) | 0.099 ± 0.055 | 0.092 | 0.113 | 0.192 |
| Position Error (mm) | 0.043 ± 0.332 | 0.000 | 0.000 | 0.395 |
| Path Length (rad.) | 3.124 ± 1.046 | 3.255 | 3.818 | 5.034 |
| Motion Time (s) | 1.244 ± 0.344 | 1.241 | 1.485 | 2.066 |
| Jerk | 227.877 ± 83.367 | 213.936 | 267.658 | 465.281 |
| Energy (J) | 89.297 ± 49.242 | 79.390 | 116.932 | 201.817 |
| Torque (N·m) | 70.896 ± 25.171 | 67.514 | 86.202 | 131.350 |

#### Without torque limits — ROS server

| Metric | Mean ± Std. Dev. | Median | 75% | 98% |
|---|---:|---:|---:|---:|
| Success (%) | 99.65 | — | — | — |
| Plan Time (s) | 0.220 ± 0.205 | 0.197 | 0.204 | 0.612 |
| Solve Time (s) | 0.185 ± 0.045 | 0.176 | 0.181 | 0.427 |
| Position Error (mm) | 0.044 ± 0.337 | 0.000 | 0.000 | 0.405 |
| Path Length (rad.) | 3.124 ± 1.046 | 3.254 | 3.812 | 5.058 |
| Motion Time (s) | 1.625 ± 0.400 | 1.500 | 2.000 | 2.500 |
| Jerk | 126.004 ± 42.297 | 114.700 | 154.175 | 232.898 |
| Energy (J) | 86.944 ± 48.789 | 76.688 | 113.597 | 203.485 |
| Torque (N·m) | 63.543 ± 20.788 | 60.948 | 77.427 | 113.047 |

#### With torque limits — native cuRobo

| Metric | Mean ± Std. Dev. | Median | 75% | 98% |
|---|---:|---:|---:|---:|
| Success (%) | 99.69 | — | — | — |
| Plan Time (s) | 0.143 ± 0.094 | 0.119 | 0.191 | 0.498 |
| Solve Time (s) | 0.126 ± 0.088 | 0.106 | 0.176 | 0.461 |
| Position Error (mm) | 0.041 ± 0.339 | 0.000 | 0.000 | 0.321 |
| Path Length (rad.) | 3.231 ± 1.134 | 3.319 | 3.898 | 5.653 |
| Motion Time (s) | 1.329 ± 0.476 | 1.301 | 1.533 | 2.635 |
| Jerk | 218.928 ± 81.610 | 205.239 | 253.516 | 445.907 |
| Energy (J) | 81.343 ± 41.593 | 73.057 | 105.268 | 175.174 |
| Torque (N·m) | 62.255 ± 13.722 | 65.300 | 73.265 | 82.729 |

#### With torque limits — ROS server

| Metric | Mean ± Std. Dev. | Median | 75% | 98% |
|---|---:|---:|---:|---:|
| Success (%) | 99.69 | — | — | — |
| Plan Time (s) | 0.285 ± 0.303 | 0.210 | 0.220 | 0.958 |
| Solve Time (s) | 0.221 ± 0.104 | 0.187 | 0.194 | 0.511 |
| Position Error (mm) | 0.034 ± 0.287 | 0.000 | 0.000 | 0.307 |
| Path Length (rad.) | 3.232 ± 1.138 | 3.314 | 3.903 | 5.753 |
| Motion Time (s) | 1.706 ± 0.515 | 1.500 | 2.000 | 3.000 |
| Jerk | 125.879 ± 41.876 | 117.142 | 153.131 | 231.632 |
| Energy (J) | 79.665 ± 41.437 | 71.962 | 102.478 | 175.582 |
| Torque (N·m) | 58.398 ± 13.920 | 60.166 | 69.754 | 81.057 |

### Inverse Kinematics

| Interface | Success (%) | IK Time mean ± Std. Dev. (ms) | Median (ms) | 75% (ms) | 98% (ms) |
|---|---:|---:|---:|---:|---:|
| Native cuRobo | 100.00 | 35.521 ± 2.722 | 36.010 | 37.401 | 39.299 |
| ROS `/ik_batch` | 100.00 | 47.596 ± 5.020 | 45.970 | 47.135 | 57.266 |

Position error was **0.001 ± 0.009 mm** (median 0.000 mm, 98th percentile 0.010 mm) for both interfaces. Orientation error was **0.000 ± 0.000°** for both interfaces.

### Kinematics & Collision

| Interface | Valid (%) | FK Time mean ± Std. Dev. (ms) | Median (ms) | 75% (ms) | 98% (ms) |
|---|---:|---:|---:|---:|---:|
| Native cuRobo | 72.80 | 1.609 ± 0.147 | 1.545 | 1.601 | 1.889 |
| ROS `/fk_batch` | 72.80 | 6.571 ± 1.183 | 6.021 | 6.813 | 8.768 |

Per-sample FK time:

| Interface | Mean ± Std. Dev. (ms) | Median (ms) | 75% (ms) | 98% (ms) |
|---|---:|---:|---:|---:|
| Native cuRobo | 0.016 ± 0.001 | 0.015 | 0.016 | 0.019 |
| ROS `/fk_batch` | 0.066 ± 0.012 | 0.060 | 0.068 | 0.088 |

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
| Plan Time (s) | 0.038 | 0.111 | — | 0.102 |
| Solve Time (s) | 0.031 | 0.099 | — | 0.092 |
| Position Error (mm) | 0.041 | 0.043 | — | 0.000 |
| Path Length (rad.) | 3.126 | 3.124 | — | 3.255 |
| Motion Time (s) | 1.250 | 1.244 | — | 1.241 |
| Jerk | 227.365 | 227.877 | — | 213.936 |
| Energy (J) | 89.270 | 89.297 | 78.959 | 79.390 |
| Torque (N·m) | 71.028 | 70.896 | 67.328 | 67.514 |

### With torque limits — 3 kg payload

| Metric | Published RTX 6000 Ada | This RTX 4060 Laptop GPU | Published Median | This Median |
|---|---:|---:|---:|---:|
| Success (%) | 99.73 | 99.691 | — | — |
| Plan Time (s) | 0.052 | 0.143 | — | 0.119 |
| Solve Time (s) | 0.042 | 0.126 | — | 0.106 |
| Position Error (mm) | 0.042 | 0.041 | — | 0.000 |
| Path Length (rad.) | 3.234 | 3.231 | — | 3.319 |
| Motion Time (s) | 1.336 | 1.329 | — | 1.301 |
| Jerk | 217.786 | 218.928 | — | 205.239 |
| Energy (J) | 81.409 | 81.343 | 72.707 | 73.057 |
| Torque (N·m) | 62.270 | 62.255 | 65.345 | 65.300 |

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
- Perception was inactive during the benchmark: no cameras or lasers were configured.
- The benchmark used an empty initial scene; obstacles were added and removed as part of the benchmark dataset.
- Final reported exit code: **0**.

## Source

This Markdown report was generated from the raw benchmark output captured on 2026-09-28.

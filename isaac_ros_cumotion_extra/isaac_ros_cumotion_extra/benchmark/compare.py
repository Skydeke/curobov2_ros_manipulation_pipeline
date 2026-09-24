# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare the native (curobo_core) and ROS-wrapped benchmark results.

The interesting parity fields are planning *outcomes*: success, trajectory
path length, motion time and waypoint count. Timing is informational and is
reported with attribution rather than as a parity signal.

- ``time_s`` — core: the solver's own ``result.total_time`` (CUDA-event wall
  of the ``plan_pose`` retry loop, seed prep + metrics included); ros:
  client-side wall around the ``generate_trajectory`` service call (server
  plan + per-request world churn + serialization/RTT).
- ``solve_time_s`` — curobo's ``result.solve_time``: the optimizer-iteration
  CUDA-event time of ``ik_result.solve_time + trajopt_result.solve_time``
  **accumulated across every attempt of the ``plan_pose`` retry loop**
  (``MotionPlanner._plan_pose_single``). Both legs read the SAME field, so a
  native-vs-ROS delta is real solver-observed behaviour, not measurement skew.

On the server (box logs `[plan-perf]`:
`setup ~1 ms`, `world refresh ~0 ms`, so the wall sits inside `plan()`), the
ROS leg's solve_time used to track its wall (~2 s per problem) rather than
native's ~0.06 s, identically with CUDA graphs on and off. The per-request
cost was inside curobo's `plan_pose` retry/optimizer loop, driven by three
server-side divergences from the native recipe, all now closed in the
reference envelope:
  1. `obstacle_collision_mode:=mesh` (old default) — the legacy trimesh
     conversion sent every sphere/cylinder/capsule obstacle through the
     mesh-SDF path, ~12x slower per solver iteration than native's OBB
     `get_obb_world()` geometries (fixed: default `cuboid`);
  2. `max_attempts` (every benchmark subcommand now defaults both legs to 100 —
     the page's real-solve budget — and the ROS runner pins the server's
     plan-time `max_attempts` to the run's own `--max-attempts`, so native
     and ROS always share the same retry envelope; at 1 attempt both legs sit
     at their ~90 % 1-attempt success rate, which is what the old harness
     measured) and `num_trajopt_seeds` (12 vs the native recipe's 4; the
     envelope pins 4 via `CUROBO_NUM_TRAJOPT_SEEDS`) — the retry loop
     contributed ~1.35 s/request;
  3. the collision cache: curobo's Warp kernels launch one thread per (sphere,
     padded obstacle slot) per obstacle type, and the server's (former)
     deployment default `{cuboid: 100, mesh: 100, voxel: ...}` padded grids
     that native's `{obb: n_cubes}` (16-cuboid demo) never had — ~7x of the
     remaining single-attempt cost. The server now ships 32/4 defaults
     (`collision_cache_cuboid` / `collision_cache_mesh` launch params), and
     the ROS leg sizes the cache to the dataset's actual per-type counts and
     disables the (empty, no-camera) voxel layer via `SetCollisionCache`
     before the timed run (`--no-size-cache` keeps the padded behaviour for
     A/B).
Both legs now solve the same problems with the same OBB geometry, seed
budget, retry cap and kernel grids; see the README "timing attribution"
section.

Result entries (both legs) look like::

    {
        'problem_name': 'dresser_task_oriented_1',
        'scene_key': 'dresser_task_oriented',
        'capability': 'planning',
        'success': True,
        'time_s': 0.3,            # core: solver total_time; ros: client wall-clock
        'n_waypoints': 200,
        'path_length': 3.2,       # sum of ||d(q)|| over interpolated waypoints
        'motion_time_s': 4.975,   # dt * (n_waypoints - 1)
        'solve_time_s': 0.28,     # optional; result.solve_time when available
    }

On failure: ``success=False`` with ``n_waypoints=0`` and
``path_length``/``motion_time_s`` = None.
"""

# Standard Library
import json
import math
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Shared trajectory metrics (identical arithmetic on both legs)
# ---------------------------------------------------------------------------


def trajectory_metrics(
    waypoints: Sequence[Sequence[float]], dt: float
) -> tuple:
    """Compute (n_waypoints, path_length, motion_time_s) from waypoint rows.

    Args:
        waypoints: joint-position rows, e.g. the interpolated trajectory
            [[q1_1..q1_7], [q2_1..q2_7], ...] (ROS leg) or the equivalent
            native tensor converted to lists.
        dt: time step between consecutive waypoints (s).

    Returns:
        ``(n, path_length, motion_time_s)`` where path_length is the sum of
        Euclidean joint-space distances between consecutive waypoints and
        motion_time_s is ``dt * (n - 1)`` (the interpolated plan's span).
    """
    n = len(waypoints)
    if n < 2:
        return n, 0.0, 0.0
    path_length = 0.0
    prev = waypoints[0]
    for cur in waypoints[1:]:
        dist = math.sqrt(
            sum((float(a) - float(b)) ** 2 for a, b in zip(cur, prev))
        )
        path_length += dist
        prev = cur
    return n, path_length, dt * (n - 1)


def position_error_mm(
    achieved_xyz: Sequence[float], goal_xyz: Sequence[float]
) -> float:
    """Euclidean distance (mm) between an achieved tool position and its goal.

    General distance-in-mm helper (the FK-verified value of a trajectory
    endpoint against a goal pose, in the same mm units as the native leg's
    ``result.position_error * 1000``). The ROS leg's report row is fed by
    ``winner_position_error_mm`` (the solver's own residual); this helper
    remains for ad-hoc FK checks. NOTE: the returned interpolated trajectory
    ends exactly on the goal joint state (implicit-goal interpolation pins the
    final waypoint), so FK'ing that final waypoint reproduces the goal pose to
    float32 precision — ~0 mm by construction, not by solver accuracy.
    """
    return (
        math.sqrt(
            sum(
                (float(a) - float(g)) ** 2 for a, g in zip(achieved_xyz, goal_xyz)
            )
        )
        * 1000.0
    )


def orientation_error_deg(
    achieved_quat_wxyz: Sequence[float], goal_quat_wxyz: Sequence[float]
) -> float:
    """Angular distance (deg) between two quaternions (wxyz order).

    ``2 * atan2(||q_misalign||_vec, |w|)`` — the shortest-arc rotation between
    ``achieved`` and ``goal``, in degrees. Used by both IK legs to score the FK
    of the solved config against the goal pose (the ROS server reports no
    solver residual, so this replaces it on both sides with identical math).
    """
    w1, x1, y1, z1 = (float(v) for v in achieved_quat_wxyz)
    w2, x2, y2, z2 = (float(v) for v in goal_quat_wxyz)
    # Relative rotation: q_goal^-1 ⊗ q_achieved (use conjugate of the goal).
    w = w2 * w1 + x2 * x1 + y2 * y1 + z2 * z1
    x = w2 * x1 - x2 * w1 - y2 * z1 + z2 * y1
    y = w2 * y1 + x2 * z1 - y2 * w1 - z2 * x1
    z = w2 * z1 - x2 * y1 + y2 * x1 - z2 * w1
    vec_norm = math.sqrt(x * x + y * y + z * z)
    return math.degrees(2.0 * math.atan2(vec_norm, abs(w)))


def trajectory_jerk(waypoints: Sequence[Sequence[float]], dt: float) -> Optional[float]:
    """Max |jerk| (rad/s^3) over an interpolated joint trajectory.

    The curobo-core leg reports the planner's own max|jerk| (control-point
    spline); the ROS leg replays the dense interpolated plan the server returns,
    so jerk is estimated client-side with a third finite difference:

        jerk_k = (q_{k+3} - 3 q_{k+2} + 3 q_{k+1} - q_k) / dt^3

    Maximum absolute value across joints and samples (same units/interpretation
    as the reference row). Returns ``None`` for trajectories too short to
    difference.
    """
    rows = [list(map(float, w)) for w in waypoints]
    n = len(rows)
    dof = len(rows[0]) if n else 0
    if n < 4 or dof == 0 or dt <= 0.0:
        return None
    worst = 0.0
    for k in range(n - 3):
        for j in range(dof):
            jerk = (
                rows[k + 3][j] - 3.0 * rows[k + 2][j]
                + 3.0 * rows[k + 1][j] - rows[k][j]
            ) / (dt ** 3)
            worst = max(worst, abs(jerk))
    return worst


def _rel_delta(core_value, ros_value):
    """Relative delta |c-r|/|c|, or None when not comparable."""
    if core_value is None or ros_value is None:
        return None
    core_value = float(core_value)
    ros_value = float(ros_value)
    if core_value == 0.0 and ros_value == 0.0:
        return 0.0
    if core_value == 0.0:
        return math.inf if ros_value != 0.0 else 0.0
    return abs(ros_value - core_value) / abs(core_value)


def _row_value(row, key, default=None):
    """Read a field off a considered-row entry (dict or attribute object)."""
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def winner_solve_time(rows, goal_index, seed_index) -> Optional[float]:
    """Solver-reported solve time (s) of the winning candidate, or None.

    ``rows`` are the considered-trajectory entries the server returns for the
    request (each carrying ``problem``/``segment``/``goalset_candidate``/
    ``seed``/``solve_time``); ``goal_index`` / ``seed_index`` are the winner's
    per-segment candidate and restart indices. The whole-task solve rides
    every considered row, so the first row whose (problem, segment) attribution
    matches the winner is the solver time — the same ``result.solve_time`` the
    native leg records (curobo's accumulated retry-loop optimizer time, NOT the
    client-side wall clock; see the module docstring for how that behaves on
    the ROS leg).
    """
    rows = [r for r in (rows or []) if r is not None]
    goal_index = [int(v) for v in (goal_index or [])]
    seed_index = [int(v) for v in (seed_index or [])]
    if not rows or not goal_index or len(seed_index) != len(goal_index):
        return None
    for row in rows:
        seg = int(_row_value(row, 'segment', 0))
        if int(_row_value(row, 'problem', 0)) != 0 or seg >= len(goal_index):
            continue
        if (
            int(_row_value(row, 'goalset_candidate', 0)) == goal_index[seg]
            and int(_row_value(row, 'seed', 0)) == seed_index[seg]
        ):
            try:
                return float(_row_value(row, 'solve_time', None))
            except (TypeError, ValueError):
                return None
    return None


def winner_position_error_mm(rows, goal_index, seed_index) -> Optional[float]:
    """Solver-reported position error (mm) of the winning candidate, or None.

    The considered rows the ROS server returns for the request carry the
    solver's per-seed ``position_error`` as ``max_waypoint_error`` (meters,
    same convergence metric the native leg records as ``result.position_error``
    — max over tracked links at the last rollout timestep of the optimized
    trajectory). The winner's row, identified exactly as in
    ``winner_solve_time``, is the seed the report ranks first; its field times
    1000 is the mm-scale analog of the native leg's
    ``result.position_error * 1000``.
    """
    rows = [r for r in (rows or []) if r is not None]
    goal_index = [int(v) for v in (goal_index or [])]
    seed_index = [int(v) for v in (seed_index or [])]
    if not rows or not goal_index or len(seed_index) != len(goal_index):
        return None
    for row in rows:
        seg = int(_row_value(row, 'segment', 0))
        if int(_row_value(row, 'problem', 0)) != 0 or seg >= len(goal_index):
            continue
        if (
            int(_row_value(row, 'goalset_candidate', 0)) == goal_index[seg]
            and int(_row_value(row, 'seed', 0)) == seed_index[seg]
        ):
            try:
                err = float(_row_value(row, 'max_waypoint_error', None))
            except (TypeError, ValueError):
                return None
            return err * 1000.0 if err is not None else None
    return None


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


def compare(
    core_results: List[Dict[str, Any]],
    ros_results: List[Dict[str, Any]],
    path_tolerance: float = 0.02,
    motion_tolerance: float = 0.02,
) -> Dict[str, Any]:
    """Compare native vs ROS results; return a structured report.

    Rows are matched by ``problem_name``. A problem matches when:

    - success agrees, AND (when both succeed)
    - relative path-length delta <= ``path_tolerance``, AND
    - relative motion-time delta <= ``motion_tolerance``, AND
    - waypoint counts are equal.

    Both-sides failure is an agreement. Wall-clock ``time_s`` never counts as a
    mismatch (gathered in ``summary`` instead).
    """
    core_by_name = {r["problem_name"]: r for r in core_results}
    ros_by_name = {r["problem_name"]: r for r in ros_results}

    common = sorted(set(core_by_name) & set(ros_by_name))
    core_only = sorted(set(core_by_name) - set(ros_by_name))
    ros_only = sorted(set(ros_by_name) - set(core_by_name))

    matches = 0
    success_mismatches = 0
    mismatches: List[Dict[str, Any]] = []
    deltas: List[Dict[str, Any]] = []
    path_deltas: List[float] = []
    motion_deltas: List[float] = []

    for name in common:
        c, r = core_by_name[name], ros_by_name[name]
        ok = True

        if bool(c["success"]) != bool(r["success"]):
            ok = False
            success_mismatches += 1
            mismatches.append(
                {
                    "problem_name": name,
                    "diff_type": "success_mismatch",
                    "core": bool(c["success"]),
                    "ros": bool(r["success"]),
                }
            )
            continue

        if not c["success"]:
            # Both failed: agreement.
            if ok:
                matches += 1
            continue

        path_delta = _rel_delta(c.get("path_length"), r.get("path_length"))
        motion_delta = _rel_delta(c.get("motion_time_s"), r.get("motion_time_s"))
        wp_core = c.get("n_waypoints")
        wp_ros = r.get("n_waypoints")

        if path_delta is not None and path_delta > path_tolerance:
            ok = False
            mismatches.append(
                {
                    "problem_name": name,
                    "diff_type": "path_length",
                    "core": c.get("path_length"),
                    "ros": r.get("path_length"),
                    "rel_delta": path_delta,
                }
            )
        if motion_delta is not None and motion_delta > motion_tolerance:
            ok = False
            mismatches.append(
                {
                    "problem_name": name,
                    "diff_type": "motion_time",
                    "core": c.get("motion_time_s"),
                    "ros": r.get("motion_time_s"),
                    "rel_delta": motion_delta,
                }
            )
        if wp_core != wp_ros:
            ok = False
            mismatches.append(
                {
                    "problem_name": name,
                    "diff_type": "waypoints",
                    "core": wp_core,
                    "ros": wp_ros,
                }
            )

        if path_delta is not None:
            path_deltas.append(path_delta)
        if motion_delta is not None:
            motion_deltas.append(motion_delta)
        deltas.append(
            {
                "problem_name": name,
                "path_delta": path_delta,
                "motion_delta": motion_delta,
                "time_core": c.get("time_s"),
                "time_ros": r.get("time_s"),
                "solve_core": c.get("solve_time_s"),
                "solve_ros": r.get("solve_time_s"),
            }
        )

        if ok:
            matches += 1

    total = len(common)
    n_mismatches = total - matches

    summary = {
        "success_rate_core": _success_rate(core_by_name, common),
        "success_rate_ros": _success_rate(ros_by_name, common),
        "avg_path_delta": _mean(path_deltas),
        "max_path_delta": _max(path_deltas),
        "avg_motion_delta": _mean(motion_deltas),
        "max_motion_delta": _max(motion_deltas),
        "avg_time_core": _mean(
            [core_by_name[n].get("time_s") for n in common]
        ),
        "avg_time_ros": _mean([ros_by_name[n].get("time_s") for n in common]),
        "time_overhead_pct": (
            _time_overhead_pct(
                _mean([core_by_name[n].get("time_s") for n in common]),
                _mean([ros_by_name[n].get("time_s") for n in common]),
            )
        ),
        # Solver-reported solve times are curobo's accumulated plan_pose
        # retry-loop optimizer time on BOTH legs (see the module docstring):
        # the ROS leg sizes the server's collision cache to the dataset
        # (native-equivalent kernel grids) and the benchmark envelope pins the
        # server's max_attempts to the run's own budget (100 by default), so
        # both legs' solve_time reflects the same calm-attempt,
        # same-geometry optimizer cost on the ~99 % of problems that solve at
        # attempt 1 and should track each other (and the ROS wall -
        # solve ≈ plan() ≈ wall on the server).
        "avg_solve_time_core": _mean(
            [core_by_name[n].get("solve_time_s") for n in common]
        ),
        "avg_solve_time_ros": _mean(
            [ros_by_name[n].get("solve_time_s") for n in common]
        ),
        # How much of the ROS wall the solver's own timers claim — ~100% when
        # per-request churn work is charged inside them (the box behaviour),
        # much less when the wall is dominated by RTT/serialization instead.
        "solve_tracks_wall_pct": _solve_tracks_wall_pct(
            _mean([ros_by_name[n].get("time_s") for n in common]),
            _mean([ros_by_name[n].get("solve_time_s") for n in common]),
        ),
    }

    return {
        "total": total,
        "matches": matches,
        "mismatches": n_mismatches,
        "success_mismatches": success_mismatches,
        "core_only": core_only,
        "ros_only": ros_only,
        "details": {"mismatches": mismatches, "deltas": deltas},
        "summary": summary,
        "verdict": (
            "PARITY-OK"
            if n_mismatches == 0 and not core_only and not ros_only
            else "PARITY-DELTA"
        ),
    }


def _success_rate(results, names, key: str = "success"):
    """Fraction of named rows with a truthy field (``success`` by default)."""
    if not names:
        return None
    ok = sum(1 for n in names if bool(results[n][key]))
    return ok / len(names)


def _mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _max(values):
    values = [v for v in values if v is not None]
    return max(values) if values else None


def _time_overhead_pct(core_avg, ros_avg):
    if core_avg is None or ros_avg is None or core_avg <= 0.0:
        return None
    return (ros_avg - core_avg) / core_avg * 100.0


def _solve_tracks_wall_pct(ros_wall_avg, ros_solve_avg):
    """Ros solver-charged share of the ros wall (``solve_avg / wall_avg``).

    Near 100 means per-request churn/retry work is charged inside curobo's own
    timers (the box behaviour); far below 100 means the wall is dominated by
    serialization/RTT on top of a fast solver.
    """
    if ros_wall_avg is None or ros_solve_avg is None or ros_wall_avg <= 0.0:
        return None
    return ros_solve_avg / ros_wall_avg * 100.0


# ---------------------------------------------------------------------------
# IK / kinematics-collision parity (the `ik` and `cost` benchmark capabilities)
# ---------------------------------------------------------------------------


def _rows_for(results: List[Dict[str, Any]], capability: str) -> List[Dict[str, Any]]:
    """Filter result entries to one capability (and, for ``ik``, the ``cfree``
    parity variant — the ``plain`` row family is native-only reference data)."""
    rows = [r for r in results if r.get("capability") == capability]
    if capability == "ik":
        rows = [r for r in rows if r.get("variant") == "cfree"]
    return rows


def compare_ik(
    core_results: List[Dict[str, Any]],
    ros_results: List[Dict[str, Any]],
    position_tolerance_mm: float = 0.5,
    orientation_tolerance_deg: float = 0.5,
) -> Dict[str, Any]:
    """Compare IK parity (native vs ROS, both legs' ``cfree`` rows).

    Rows are matched by ``problem_name`` (``ik_cfree_bNN_gNNN``, identical on
    both legs from the shared goal generator). A goal matches when:

    - success agrees, AND (when both succeed)
    - FK-verified ``position_error_mm`` delta <= ``position_tolerance_mm``,
      AND
    - FK-verified ``orientation_error_deg`` delta <=
      ``orientation_tolerance_deg``.

    Both-sides failure is an agreement. Timing (``time_ms``, a per-batch wall)
    never counts as a mismatch — it is gathered in ``summary`` informational
    fields. The native leg's ``plain`` variant rows are excluded by design
    (the ROS server has no collision-free-off IK mode; they are reference
    rows within the native results only).
    """
    core_by_name = {r["problem_name"]: r for r in _rows_for(core_results, "ik")}
    ros_by_name = {r["problem_name"]: r for r in _rows_for(ros_results, "ik")}

    common = sorted(set(core_by_name) & set(ros_by_name))
    core_only = sorted(set(core_by_name) - set(ros_by_name))
    ros_only = sorted(set(ros_by_name) - set(core_by_name))

    matches = 0
    success_mismatches = 0
    mismatches: List[Dict[str, Any]] = []
    pos_deltas: List[float] = []
    ori_deltas: List[float] = []
    deltas: List[Dict[str, Any]] = []

    for name in common:
        c, r = core_by_name[name], ros_by_name[name]
        ok = True

        if bool(c["success"]) != bool(r["success"]):
            ok = False
            success_mismatches += 1
            mismatches.append(
                {
                    "problem_name": name,
                    "diff_type": "success_mismatch",
                    "core": bool(c["success"]),
                    "ros": bool(r["success"]),
                }
            )
            continue

        if not c["success"]:
            matches += 1
            continue

        pos_delta = _abs_delta(c.get("position_error_mm"), r.get("position_error_mm"))
        ori_delta = _abs_delta(
            c.get("orientation_error_deg"), r.get("orientation_error_deg")
        )
        if pos_delta is not None and pos_delta > position_tolerance_mm:
            ok = False
            mismatches.append(
                {
                    "problem_name": name,
                    "diff_type": "position_error",
                    "core": c.get("position_error_mm"),
                    "ros": r.get("position_error_mm"),
                    "delta_mm": pos_delta,
                }
            )
        if ori_delta is not None and ori_delta > orientation_tolerance_deg:
            ok = False
            mismatches.append(
                {
                    "problem_name": name,
                    "diff_type": "orientation_error",
                    "core": c.get("orientation_error_deg"),
                    "ros": r.get("orientation_error_deg"),
                    "delta_deg": ori_delta,
                }
            )

        if pos_delta is not None:
            pos_deltas.append(pos_delta)
        if ori_delta is not None:
            ori_deltas.append(ori_delta)
        deltas.append(
            {
                "problem_name": name,
                "pos_delta_mm": pos_delta,
                "ori_delta_deg": ori_delta,
                "time_core_ms": c.get("time_ms"),
                "time_ros_ms": r.get("time_ms"),
            }
        )
        if ok:
            matches += 1

    total = len(common)
    n_mismatches = total - matches

    summary = {
        "success_rate_core": _success_rate(core_by_name, common),
        "success_rate_ros": _success_rate(ros_by_name, common),
        "avg_pos_delta_mm": _mean(pos_deltas),
        "max_pos_delta_mm": _max(pos_deltas),
        "avg_ori_delta_deg": _mean(ori_deltas),
        "max_ori_delta_deg": _max(ori_deltas),
        "avg_time_core_ms": _mean(
            [core_by_name[n].get("time_ms") for n in common]
        ),
        "avg_time_ros_ms": _mean([ros_by_name[n].get("time_ms") for n in common]),
    }

    return {
        "capability": "ik",
        "total": total,
        "matches": matches,
        "mismatches": n_mismatches,
        "success_mismatches": success_mismatches,
        "core_only": core_only,
        "ros_only": ros_only,
        "details": {"mismatches": mismatches, "deltas": deltas},
        "summary": summary,
        "verdict": (
            "PARITY-OK"
            if n_mismatches == 0 and not core_only and not ros_only
            else "PARITY-DELTA"
        ),
    }


def compare_cost(
    core_results: List[Dict[str, Any]],
    ros_results: List[Dict[str, Any]],
    position_tolerance_mm: float = 1e-3,
    orientation_tolerance_deg: float = 1e-3,
) -> Dict[str, Any]:
    """Compare kinematics & collision parity (native FK/validate vs ``/fk_batch``).

    Rows are matched by ``problem_name`` (``cost_bNN_gNNN``). Per config:

    - ``valid`` agreement is exact (both sides run the same
      ``RobotCollisionChecker.validate`` on the same world, so a flipped bit is
      a real parity fault), and
    - the FK tool poses agree within ``position_tolerance_mm`` /
      ``orientation_tolerance_deg`` (float32 reproducibility guards — the
      expected deltas are ~0).

    Both mismatches are recorded independently for the same config. Timing is
    informational (``summary`` only).
    """
    core_by_name = {r["problem_name"]: r for r in _rows_for(core_results, "cost")}
    ros_by_name = {r["problem_name"]: r for r in _rows_for(ros_results, "cost")}

    common = sorted(set(core_by_name) & set(ros_by_name))
    core_only = sorted(set(core_by_name) - set(ros_by_name))
    ros_only = sorted(set(ros_by_name) - set(core_by_name))

    matches = 0
    validity_mismatches = 0
    mismatches: List[Dict[str, Any]] = []
    pos_deltas: List[float] = []
    ori_deltas: List[float] = []
    deltas: List[Dict[str, Any]] = []

    for name in common:
        c, r = core_by_name[name], ros_by_name[name]
        ok = True

        if bool(c["valid"]) != bool(r["valid"]):
            ok = False
            validity_mismatches += 1
            mismatches.append(
                {
                    "problem_name": name,
                    "diff_type": "validity",
                    "core": bool(c["valid"]),
                    "ros": bool(r["valid"]),
                }
            )

        pos_delta = _abs_delta(c.get("position_xyz"), r.get("position_xyz"), _vec_mm)
        ori_delta = _abs_delta(
            c.get("quaternion_wxyz"), r.get("quaternion_wxyz"), orientation_error_deg
        )
        if pos_delta is not None and pos_delta > position_tolerance_mm:
            ok = False
            mismatches.append(
                {
                    "problem_name": name,
                    "diff_type": "pose",
                    "core": c.get("position_xyz"),
                    "ros": r.get("position_xyz"),
                    "delta_mm": pos_delta,
                }
            )
        if ori_delta is not None and ori_delta > orientation_tolerance_deg:
            ok = False
            mismatches.append(
                {
                    "problem_name": name,
                    "diff_type": "pose_orientation",
                    "core": c.get("quaternion_wxyz"),
                    "ros": r.get("quaternion_wxyz"),
                    "delta_deg": ori_delta,
                }
            )

        if pos_delta is not None:
            pos_deltas.append(pos_delta)
        if ori_delta is not None:
            ori_deltas.append(ori_delta)
        deltas.append(
            {
                "problem_name": name,
                "pos_delta_mm": pos_delta,
                "ori_delta_deg": ori_delta,
                "time_core_ms": c.get("time_ms"),
                "time_ros_ms": r.get("time_ms"),
            }
        )
        if ok:
            matches += 1

    total = len(common)
    n_mismatches = total - matches

    summary = {
        "valid_rate_core": _success_rate(core_by_name, common, key="valid"),
        "valid_rate_ros": _success_rate(ros_by_name, common, key="valid"),
        "validity_mismatches": validity_mismatches,
        "avg_pos_delta_mm": _mean(pos_deltas),
        "max_pos_delta_mm": _max(pos_deltas),
        "avg_ori_delta_deg": _mean(ori_deltas),
        "max_ori_delta_deg": _max(ori_deltas),
        "avg_time_core_ms": _mean(
            [core_by_name[n].get("time_ms") for n in common]
        ),
        "avg_time_ros_ms": _mean([ros_by_name[n].get("time_ms") for n in common]),
    }

    return {
        "capability": "cost",
        "total": total,
        "matches": matches,
        "mismatches": n_mismatches,
        "success_mismatches": validity_mismatches,
        "core_only": core_only,
        "ros_only": ros_only,
        "details": {"mismatches": mismatches, "deltas": deltas},
        "summary": summary,
        "verdict": (
            "PARITY-OK"
            if n_mismatches == 0 and not core_only and not ros_only
            else "PARITY-DELTA"
        ),
    }


def _abs_delta(core_value, ros_value, metric=None) -> Optional[float]:
    """Absolute delta between two comparable values (None-safe)."""
    if core_value is None or ros_value is None:
        return None
    if metric is not None:
        return float(metric(core_value, ros_value))
    return abs(float(core_value) - float(ros_value))


def _vec_mm(core_vec: Sequence[float], ros_vec: Sequence[float]) -> float:
    """Euclidean distance (mm) between two 3-vectors (posed positions)."""
    return math.sqrt(
        sum(
            (float(a) - float(b)) ** 2
            for a, b in zip(core_vec, ros_vec)
        )
    ) * 1000.0


# ---------------------------------------------------------------------------
# curobo-style reporting (mirrors curobo benchmark/motion_plan_benchmark.py)
# ---------------------------------------------------------------------------


def _percentile(values: Sequence[float], p: float) -> float:
    """Numpy-style linear percentile interpolation over sorted numeric values.

    Mirrors curobo's ``Statistic`` percentile aggregation so the new IK / cost
    tables aggregate identically to the planning table. ``values`` must be
    non-empty (callers filter first).
    """
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    rank = (n - 1) * p / 100.0
    lo = int(math.floor(rank))
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _stat_str(values: Sequence[Optional[float]], ndigits: int = 3) -> str:
    """Render values like curobo's ``Statistic.__str__`` for the grid-table cell.

    ``mean: X ± Y  median: Z  75%: W  98%: V`` using a population std and
    numpy-style linear percentile interpolation. ``-`` when no values remain.
    """
    values = [float(v) for v in values if v is not None and v < math.inf]
    if not values:
        return "-"
    n = len(values)
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
    median = _percentile(values, 50.0)
    return (
        f"mean: {mean:2.{ndigits}f} \u00b1 {std:2.{ndigits}f}"
        f"  median: {median:2.{ndigits}f}"
        f"  75%: {_percentile(values, 75.0):2.{ndigits}f}"
        f"  98%: {_percentile(values, 98.0):2.{ndigits}f}"
    )


def _grid_table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> str:
    """Render a ``tabulate(table, headers, tablefmt='grid')``-style table.

    Standard-library implementation so the parity output matches the curobo
    benchmark layout without requiring ``tabulate`` to be installed.
    """
    rendered = [list(headers)] + [[str(c) for c in row] for row in rows]
    widths = [max(len(r[i]) for r in rendered) for i in range(len(headers))]

    def _sep(dash: str = "-") -> str:
        return "+" + "+".join(dash * (w + 2) for w in widths) + "+"

    def _row(cells: Sequence[str]) -> str:
        return "|" + "|".join(f" {str(c).ljust(w)} " for c, w in zip(cells, widths)) + "|"

    lines = [_sep(), _row(headers), _sep("=")]
    for row in rows:
        lines.append(_row([str(c) for c in row]))
        lines.append(_sep())
    return "\n".join(lines)


def curobo_style_rows(results: List[Dict[str, Any]]) -> List[List[str]]:
    """Build the ``Metric``/``Value`` body rows for one leg (curobo layout).

    Row order follows the reference benchmark table (Success %, Plan Time,
    Solve Time, Position Error, Path Length, Motion Time, Jerk, Energy (J),
    Torque (N·m)). Success % covers all problems; the statistic rows cover
    successful problems only (curobo filters ``inf``/failed timings the same
    way).

    Path Length / Motion Time prefer the upstream control-point values
    (``path_length_curobo`` / ``motion_time_curobo``) when present so the
    native leg reproduces the reference page; the ROS leg falls back to its
    dense-interpolated values. Solve Time / Position Error / Jerk / Energy /
    Torque rows are emitted only when their data exists.
    """
    ok = [r for r in results if bool(r.get("success"))]
    success_pct = 100.0 * len(ok) / len(results) if results else 0.0
    rows: List[List[str]] = [
        ["Success %", f"{success_pct:2.2f}"],
        ["Plan Time (s)", _stat_str([r.get("time_s") for r in ok])],
    ]
    solve = [r.get("solve_time_s") for r in ok]
    if any(v is not None for v in solve):
        rows.append(["Solve Time (s)", _stat_str(solve)])
    pos_err = [r.get("position_error_mm") for r in ok]
    if any(v is not None for v in pos_err):
        rows.append(["Position Error (mm)", _stat_str(pos_err)])
    path = [r.get("path_length_curobo") for r in ok]
    if not any(v is not None for v in path):
        path = [r.get("path_length") for r in ok]
    rows.append(["Path Length (rad.)", _stat_str(path)])
    motion = [r.get("motion_time_curobo") for r in ok]
    if not any(v is not None for v in motion):
        motion = [r.get("motion_time_s") for r in ok]
    rows.append(["Motion Time(s)", _stat_str(motion)])
    jerk = [r.get("jerk") for r in ok]
    if any(v is not None for v in jerk):
        rows.append(["Jerk", _stat_str(jerk)])
    energy = [r.get("energy_j") for r in ok]
    if any(v is not None for v in energy):
        rows.append(["Energy (J)", _stat_str(energy)])
    torque = [r.get("torque_nm") for r in ok]
    if any(v is not None for v in torque):
        rows.append(["Torque (N·m)", _stat_str(torque)])
    return rows


def ik_style_rows(results: List[Dict[str, Any]]) -> List[List[str]]:
    """Build the ``Metric``/``Value`` rows for one IK leg (curobo-layout).

    Follows the reference ``ik_benchmark`` table: Success % over all goals;
    IK Time over successful goals (``time_ms`` is the per-batch solve wall the
    goals belong to — the upstream table also aggregates per-inference times);
    then the FK-verified pose errors, which are the parity fields both legs
    report identically. Statistics rows cover successful goals only.
    """
    cfree = _rows_for(results, "ik")
    ok = [r for r in cfree if bool(r.get("success"))]
    success_pct = 100.0 * len(ok) / len(cfree) if cfree else 0.0
    rows: List[List[str]] = [
        ["Success %", f"{success_pct:2.2f}"],
        ["IK Time (ms)", _stat_str([r.get("time_ms") for r in ok])],
    ]
    pos_err = [r.get("position_error_mm") for r in ok]
    if any(v is not None for v in pos_err):
        rows.append(["Position Error (mm)", _stat_str(pos_err)])
    ori_err = [r.get("orientation_error_deg") for r in ok]
    if any(v is not None for v in ori_err):
        rows.append(["Orientation Error (deg)", _stat_str(ori_err)])
    return rows


def cost_style_rows(results: List[Dict[str, Any]]) -> List[List[str]]:
    """Build the ``Metric``/``Value`` rows for one kinematics & collision leg.

    Valid % over all configs (the parity field); per-batch FK+validate wall
    and the per-sample marginal (``time_ms`` is per batch of ``n_configs``).
    """
    cost = _rows_for(results, "cost")
    valid = [r for r in cost if bool(r.get("valid"))]
    valid_pct = 100.0 * len(valid) / len(cost) if cost else 0.0
    rows: List[List[str]] = [
        ["Valid %", f"{valid_pct:2.2f}"],
        ["FK Time (ms)", _stat_str([r.get("time_ms") for r in cost])],
        [
            "FK Time / Sample (ms)",
            _stat_str(
                [
                    r.get("time_ms", 0.0) / r.get("n_configs", 1)
                    if r.get("n_configs")
                    else None
                    for r in cost
                ]
            ),
        ],
    ]
    return rows


def print_curobo_table(title: str, results: List[Dict[str, Any]]) -> None:
    """Print one leg's results as a curobo benchmark-style grid table."""
    print(f"== {title} ==")
    print(_grid_table(curobo_style_rows(results), ["Metric", "Value"]))


# ---------------------------------------------------------------------------
# Reference-page reproduction (the `reference` subcommand)
# ---------------------------------------------------------------------------

# Metrics as published on the cuRobo benchmarks page, "Latest Motion
# Generation Results"
# (https://nvlabs.github.io/curobo/latest/reference/benchmarks.html, RTX 6000
# Ada, 2600 problems, mean / median). Informational printout only — never
# used for a verdict; timing columns on a different GPU are not comparable.
PUBLISHED_REFERENCE_PAGE: Dict[str, Dict[str, tuple]] = {
    "without_torque": {
        "Success %": ("99.73", None),
        "Plan Time (s)": ("0.038", None),
        "Solve Time (s)": ("0.031", None),
        "Position Error (mm)": ("0.041", None),
        "Path Length (rad.)": ("3.126", None),
        "Motion Time(s)": ("1.250", None),
        "Jerk": ("227.365", None),
        "Energy (J)": ("89.270", "78.959"),
        "Torque (N·m)": ("71.028", "67.328"),
    },
    "with_torque": {
        "Success %": ("99.73", None),
        "Plan Time (s)": ("0.052", None),
        "Solve Time (s)": ("0.042", None),
        "Position Error (mm)": ("0.042", None),
        "Path Length (rad.)": ("3.234", None),
        "Motion Time(s)": ("1.336", None),
        "Jerk": ("217.786", None),
        "Energy (J)": ("81.409", "72.707"),
        "Torque (N·m)": ("62.270", "65.345"),
    },
}

# (result key, table label) for the statistic rows, in page order.
_REFERENCE_METRIC_KEYS = (
    ("time_s", "Plan Time (s)"),
    ("solve_time_s", "Solve Time (s)"),
    ("position_error_mm", "Position Error (mm)"),
    ("jerk", "Jerk"),
    ("energy_j", "Energy (J)"),
    ("torque_nm", "Torque (N·m)"),
)


def _metric_vals(results, key, fallback=None):
    """Successful-solve values for one result key (fallback key if empty).

    Filters ``inf`` like ``_stat_str`` (failed/unrun timings are ``inf`` in
    the native results).
    """
    vals = []
    for r in results:
        if not bool(r.get("success")):
            continue
        v = r.get(key)
        if v is None and fallback is not None:
            v = r.get(fallback)
        if v is not None and float(v) < math.inf:
            vals.append(float(v))
    return vals


def reference_stats(results: List[Dict[str, Any]]) -> Dict[str, tuple]:
    """Aggregate one reference table into {metric label: (mean, median)}.

    Statistic rows cover successful solves only (curobo filters failed
    timings the same way); ``Success %`` is over all problems. Path Length
    and Motion Time prefer the upstream control-point fields when present,
    else the dense-interpolated values. A label is omitted (``None`` mean) when
    it has no data — matching ``curobo_style_rows`` row emission.
    """
    ok = [r for r in results if bool(r.get("success"))]
    out: Dict[str, tuple] = {
        "Success %": (100.0 * len(ok) / len(results) if results else 0.0, None)
    }
    for key, label in _REFERENCE_METRIC_KEYS:
        vals = _metric_vals(results, key)
        if vals:
            out[label] = (_mean(vals), _percentile(vals, 50.0))
    path = _metric_vals(results, "path_length_curobo", "path_length")
    if path:
        out["Path Length (rad.)"] = (_mean(path), _percentile(path, 50.0))
    motion = _metric_vals(results, "motion_time_curobo", "motion_time_s")
    if motion:
        out["Motion Time(s)"] = (_mean(motion), _percentile(motion, 50.0))
    return out


def _fmt_stat(v) -> str:
    return "-" if v is None else f"{v:.3f}"


def print_published_comparison(
    plain: List[Dict[str, Any]],
    torque: List[Dict[str, Any]],
    mass: float = 3.0,
) -> None:
    """Print the side-by-side mean/median grids: published page vs this box.

    One grid per motion-generation table (without / with torque limits at
    ``mass`` kg). Pure printout — see ``reference_stats`` for the aggregation.
    """
    for key, (results, label) in (
        ("without_torque", (plain, "without torque limits")),
        ("with_torque", (torque, f"with torque limits ({mass:g} kg)")),
    ):
        stats = reference_stats(results)
        published = PUBLISHED_REFERENCE_PAGE[key]
        grid: List[List[str]] = []
        for metric, (p_mean, p_median) in published.items():
            ours = stats.get(metric)
            ours_mean = ours[0] if ours else None
            ours_median = ours[1] if ours else None
            grid.append(
                [
                    metric,
                    p_mean if p_mean is not None else "-",
                    _fmt_stat(ours_mean),
                    p_median if p_median is not None else "-",
                    _fmt_stat(ours_median),
                ]
            )
        abs_deltas = [
            abs(float(p) - float(s[0]))
            for metric, (p, _m) in published.items()
            if (s := stats.get(metric)) and s[0] is not None and p is not None
        ]
        note = ""
        # Only claim a comparison when the run actually produced data
        # (an empty / all-failed run still shows the grid, just without the
        # delta note — its `.0` Success % vs the page is not a signal).
        if abs_deltas and any(m != "Success %" for m in stats):
            note = (
                f"\n  worst mean delta vs the page: {max(abs_deltas):.3f} "
                f"(page units — each metric keeps its own unit)"
            )
        print(
            f"\n== published page (RTX 6000 Ada) vs this box — {label} =="
            f"{note}"
        )
        print(
            _grid_table(
                grid,
                ["Metric", "Docs mean", "Ours mean", "Docs median", "Ours median"],
            )
        )
        print(
            "  note: timing columns are informational on a different GPU; the "
            "quality metrics (success, position error, path/motion, jerk, "
            "energy, torque) are deterministic given the same config, seeds "
            "and dataset."
        )


def print_reference_comparison(
    plain: List[Dict[str, Any]],
    torque: List[Dict[str, Any]],
    mass: float = 3.0,
) -> None:
    """Print the reference-page reproduction: both curobo-style tables, then a
    side-by-side mean/median grid against the values published on the page.

    ``plain`` / ``torque`` are the native result lists for the two passes
    (solver without / with torque limits at ``mass`` kg). Printed after the
    runs so the box numbers can be compared against the page without digging
    through the logs.
    """
    print_curobo_table("without torque limits (native)", plain)
    print_curobo_table(f"with torque limits ({mass:g} kg) (native)", torque)
    print_published_comparison(plain, torque, mass=mass)


def _print_leg(label: str, results: Optional[List[Dict[str, Any]]],
               table_printer) -> None:
    """Print one leg's table, or a short placeholder when the leg didn't run."""
    if results is None:
        print(f"  {label}: (not run — see the run log for why)")
        return
    table_printer(label, results)


def print_webpage_summary(
    *,
    mass: float = 3.0,
    motion_plain_native: Optional[List[Dict[str, Any]]] = None,
    motion_plain_ros: Optional[List[Dict[str, Any]]] = None,
    motion_torque_native: Optional[List[Dict[str, Any]]] = None,
    motion_torque_ros: Optional[List[Dict[str, Any]]] = None,
    ik_native: Optional[List[Dict[str, Any]]] = None,
    ik_ros: Optional[List[Dict[str, Any]]] = None,
    cost_native: Optional[List[Dict[str, Any]]] = None,
    cost_ros: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Print ALL results again, grouped like the cuRobo benchmarks page.

    The page's suite order is motion generation (without torque limits, then
    with torque limits at the full ``mass`` kg payload) → inverse kinematics →
    kinematics & collision. Each suite is shown once for the native curobo leg
    and once for the ROS leg (``None`` = that leg was not run in this
    invocation). The printed output is the "end of ``compose up``" summary a
    box run is compared against.
    """
    print("=" * 72)
    print("cuRobo benchmarks page reproduction — all results again, in the")
    print("page's suite order (motion generation -> IK -> kinematics & collision)")
    print("=" * 72)

    print("\n[1/3] MOTION GENERATION  (the page's \"Latest Motion Generation")
    print("      Results\" — two tables, same dataset, same seeds)")
    print(f"  -- without torque limits --  (mass {mass:g} kg payload model)")
    _print_leg("without torque limits — native curobo", motion_plain_native,
               print_curobo_table)
    _print_leg("without torque limits — ROS (server)", motion_plain_ros,
               print_curobo_table)
    print(f"  -- with torque limits (full {mass:g} kg payload) --")
    _print_leg(f"with torque limits ({mass:g} kg) — native curobo",
               motion_torque_native, print_curobo_table)
    _print_leg(f"with torque limits ({mass:g} kg) — ROS (server)",
               motion_torque_ros, print_curobo_table)

    print("\n[2/3] INVERSE KINEMATICS  (the page's \"Inverse Kinematics\")")
    _print_leg("inverse kinematics — native curobo", ik_native, print_ik_table)
    _print_leg("inverse kinematics — ROS (/ik_batch)", ik_ros, print_ik_table)

    print("\n[3/3] KINEMATICS & COLLISION  (the page's \"Kinematics & Collision\")")
    _print_leg("kinematics & collision — native curobo", cost_native,
               print_cost_table)
    _print_leg("kinematics & collision — ROS (/fk_batch)", cost_ros,
               print_cost_table)

    if motion_plain_native is not None and motion_torque_native is not None:
        print(
            "\n[APPENDIX] PUBLISHED PAGE vs THIS BOX — motion-generation means/"
            "medians"
        )
        print_published_comparison(motion_plain_native, motion_torque_native,
                                   mass=mass)

    print("\n" + "=" * 72)
    print("Receipt notes:")
    print("  - timing columns are informational: this box's GPU differs from the")
    print("    page's RTX 6000 Ada.")
    print("  - the quality metrics are deterministic given the same config, seeds")
    print("    and dataset.")
    print("  - the page's 99.73% success assumes the 100-attempt budget; this")
    print("    run defaults to 100 for BOTH legs (the runner pins the server's")
    print("    plan-time max_attempts to --max-attempts before each motion ROS")
    print("    leg). At 1 attempt only a ~1-attempt success-rate is expected.")
    print("  - ROS-leg Energy (J) / Torque (N·m) are the client-side velocity")
    print("    finite-difference approximation (the wire carries no acceleration).")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(v, ndigits=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{ndigits}g}"
    return str(v)


def _pct(v):
    """Scale a relative delta to a percentage; None stays None (0-success runs).

    Relative deltas are only defined over problems where both legs succeeded,
    so an all-failed run leaves every delta None — without the guard,
    ``avg_path_delta * 100`` would crash the report on a valid 0% success run.
    """
    return None if v is None else float(v) * 100.0


def print_report(
    report: Dict[str, Any],
    show_all: bool = False,
    core_results: Optional[List[Dict[str, Any]]] = None,
    ros_results: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Print the parity report to stdout.

    When ``core_results``/``ros_results`` are provided, both legs are first
    printed as curobo-style ``Metric``/``Value`` grid tables (the same layout
    ``curobo/benchmark/motion_plan_benchmark.py`` prints), followed by the
    parity details and verdict.
    """
    s = report["summary"]

    print("=" * 78)
    print("cuRobo planner parity report: native core vs ROS wrapper")
    print("=" * 78)
    if core_results is not None:
        print_curobo_table("native (curobo_core)", core_results)
    if ros_results is not None:
        print_curobo_table("ros (unified_planner)", ros_results)
    print(
        f"problems: {report['total']} common, "
        f"{len(report['core_only'])} core-only, "
        f"{len(report['ros_only'])} ros-only"
    )
    print(
        f"success rate (core/ros): {_fmt(s['success_rate_core'])} / "
        f"{_fmt(s['success_rate_ros'])}"
    )
    if (
        report["total"] > 0
        and s["success_rate_core"] == 0.0
        and s["success_rate_ros"] == 0.0
    ):
        scenes = sorted(
            {r.get("scene_key") for r in (core_results or [])}
            | {r.get("scene_key") for r in (ros_results or [])}
        )
        scene_txt = ", ".join(s for s in scenes if s) or "?"
        print(
            "note: both legs solved 0 problems on this run "
            f"(scenes: {scene_txt}).",
            "Both legs run the same particle+LBFGS solver recipe at the same",
            "retry budget (the whole benchmark defaults both to max_attempts=100",
            "— the runner pins the server's plan-time max_attempts to the",
            "run's own --max-attempts). A 0/0",
            "outcome means that solver envelope could",
            "not crack these scenes on this box. Path/motion cells are empty "
            "and the parity",
            "deltas are vacuous on an all-failed run.",
            "To compare metrics, restrict the run to scenes the server's "
            "solver can reach, e.g.:",
            "  curobo_benchmark all --dataset mpinets "
            "--scene dresser_neutral_start",
            "  curobo_benchmark all --dataset motion_benchmaker "
            "--scene table_pick_panda",
            "The parity verdict above still holds for every problem both legs "
            "ran.",
            sep="\n",
        )
    print(
        f"parity: {report['matches']}/{report['total']} match, "
        f"{report['mismatches']} mismatch "
        f"({report['success_mismatches']} success mismatches)"
    )
    print(
        f"path delta: avg {_fmt(_pct(s['avg_path_delta']), 2)}% "
        f"max {_fmt(_pct(s['max_path_delta']), 2)}%"
    )
    print(
        f"motion delta: avg {_fmt(_pct(s['avg_motion_delta']), 2)}% "
        f"max {_fmt(_pct(s['max_motion_delta']), 2)}%"
    )
    print(
        f"time (info): core avg {_fmt(s['avg_time_core'])}s, "
        f"ros wall avg {_fmt(s['avg_time_ros'])}s, "
        f"overhead {_fmt(s['time_overhead_pct'], 2)}%"
    )
    if (
        s.get("avg_solve_time_core") is not None
        or s.get("avg_solve_time_ros") is not None
    ):
        tracks = s.get("solve_tracks_wall_pct")
        note = (
            "solver-reported (curobo's accumulated plan_pose retry-loop "
            "optimizer time on both legs). The benchmark envelope runs both "
            "legs at the same budget — the runner pins the server's plan-time "
            "max_attempts to the run's --max-attempts (CUROBO_MAX_ATTEMPTS "
            "sets one knob for both, default 100) — and the ROS leg sizes the "
            "server's collision cache to the dataset's "
            "actual obstacle counts with the voxel layer off before the timed "
            "run, so one server attempt runs native-equivalent solver kernels "
            "(see the README timing-attribution section); ros solve tracks "
            "the ros wall (server solve ≈ plan()) and reads the same field as "
            "native's calm-attempt solve time"
        )
        if tracks is not None:
            note += (
                f" (ros solve = {tracks:.1f}% of the ros wall, "
                f"i.e. the per-request cost is solver-loop cost, not RTT)"
            )
        print(
            f"solve time (info): core avg {_fmt(s.get('avg_solve_time_core'))}s, "
            f"ros avg {_fmt(s.get('avg_solve_time_ros'))}s — {note}"
        )

    mismatches = report["details"]["mismatches"]
    deltas = report["details"]["deltas"]

    if report["core_only"]:
        print(f"core-only problems: {', '.join(report['core_only'])}")
    if report["ros_only"]:
        print(f"ros-only problems: {', '.join(report['ros_only'])}")

    if mismatches:
        print("\n--- mismatches ---")
        for m in mismatches:
            print(
                f"  {m['problem_name']}: {m['diff_type']} "
                f"(core={m.get('core')}, ros={m.get('ros')})"
            )

    # Per-problem delta table: always shown for mismatched problems; every
    # problem when --show-all.
    mismatch_names = {m["problem_name"] for m in mismatches}
    rows = deltas
    if not show_all:
        rows = [d for d in deltas if d["problem_name"] in mismatch_names]
    if rows:
        print("\n--- per-problem deltas ---")
        header = (
            f"  {'problem':38s} {'path%':>8s} {'motion%':>8s} "
            f"{'timeC':>9s} {'timeR':>9s} {'solveC':>9s} {'solveR':>9s}"
        )
        print(header)
        print(
            "  (timeC/timeR: core total_time / ros client wall; "
            "solveC/solveR: curobo-accumulated retry-loop solver time)"
        )
        for d in rows:
            path_pct = None if d.get("path_delta") is None else d["path_delta"] * 100
            motion_pct = None if d.get("motion_delta") is None else d["motion_delta"] * 100
            print(
                f"  {d['problem_name']:38s} "
                f"{_fmt(path_pct, 2):>8s} "
                f"{_fmt(motion_pct, 2):>8s} "
                f"{_fmt(d.get('time_core')):>9s} "
                f"{_fmt(d.get('time_ros')):>9s} "
                f"{_fmt(d.get('solve_core')):>9s} "
                f"{_fmt(d.get('solve_ros')):>9s}"
            )

    verdict = report["verdict"]
    print("\n" + "=" * 78)
    print(f"VERDICT: {verdict}")
    print("=" * 78)


def _print_pose_parity_report(
    title: str,
    report: Dict[str, Any],
    show_all: bool = False,
    core_results: Optional[List[Dict[str, Any]]] = None,
    ros_results: Optional[List[Dict[str, Any]]] = None,
    nag_ok: bool = True,
) -> None:
    """Shared body of the IK and cost parity reports (mirrors print_report)."""
    s = report["summary"]

    print("=" * 78)
    print(title)
    print("=" * 78)
    if core_results is not None and report["capability"] == "ik":
        print_ik_table("native (curobo_core)", core_results)
    elif core_results is not None:
        print_cost_table("native (curobo_core)", core_results)
    if ros_results is not None and report["capability"] == "ik":
        print_ik_table("ros (unified_planner)", ros_results)
    elif ros_results is not None:
        print_cost_table("ros (unified_planner)", ros_results)

    n_problems = report["total"]
    per_leg_label = (
        "success rate (core/ros)"
        if report["capability"] == "ik"
        else "valid rate (core/ros)"
    )
    rate_core = s.get("success_rate_core", s.get("valid_rate_core"))
    rate_ros = s.get("success_rate_ros", s.get("valid_rate_ros"))
    print(
        f"goals/configs: {n_problems} common, "
        f"{len(report['core_only'])} core-only, "
        f"{len(report['ros_only'])} ros-only"
    )
    print(f"{per_leg_label}: {_fmt(rate_core)} / {_fmt(rate_ros)}")
    print(
        f"parity: {report['matches']}/{n_problems} match, "
        f"{report['mismatches']} mismatch "
        f"({report['success_mismatches']} success/validity mismatches)"
    )
    if n_problems == 0 and nag_ok:
        print(
            "note: no common rows to compare — check both legs ran the same "
            "goal/config seed and batch sizes (synthetic goals are "
            "deterministic on --seed).",
            sep="\n",
        )
    print(
        f"position delta (mm): avg {_fmt(s.get('avg_pos_delta_mm'))} "
        f"max {_fmt(s.get('max_pos_delta_mm'))}"
    )
    print(
        f"orientation delta (deg): avg {_fmt(s.get('avg_ori_delta_deg'))} "
        f"max {_fmt(s.get('max_ori_delta_deg'))}"
    )
    print(
        f"time (info): core avg {_fmt(s.get('avg_time_core_ms'))}ms, "
        f"ros avg {_fmt(s.get('avg_time_ros_ms'))}ms"
    )

    mismatches = report["details"]["mismatches"]
    deltas = report["details"]["deltas"]

    if report["core_only"]:
        print(f"core-only problems: {', '.join(report['core_only'])}")
    if report["ros_only"]:
        print(f"ros-only problems: {', '.join(report['ros_only'])}")

    if mismatches:
        print("\n--- mismatches ---")
        for m in mismatches:
            print(
                f"  {m['problem_name']}: {m['diff_type']} "
                f"(core={m.get('core')}, ros={m.get('ros')})"
            )

    mismatch_names = {m["problem_name"] for m in mismatches}
    rows = deltas
    if not show_all:
        rows = [d for d in deltas if d["problem_name"] in mismatch_names]
    if rows:
        print("\n--- per-goal deltas ---")
        print(
            f"  {'goal':38s} {'posD':>9s} {'oriD':>9s} "
            f"{'timeC':>9s} {'timeR':>9s}"
        )
        print("  (posD/oriD: absolute mm / deg deltas; timeC/timeR: ms)")
        for d in rows:
            print(
                f"  {d['problem_name']:38s} "
                f"{_fmt(d.get('pos_delta_mm')):>9s} "
                f"{_fmt(d.get('ori_delta_deg')):>9s} "
                f"{_fmt(d.get('time_core_ms')):>9s} "
                f"{_fmt(d.get('time_ros_ms')):>9s}"
            )

    verdict = report["verdict"]
    print("\n" + "=" * 78)
    print(f"VERDICT: {verdict}")
    print("=" * 78)


def print_report_ik(
    report: Dict[str, Any],
    show_all: bool = False,
    core_results: Optional[List[Dict[str, Any]]] = None,
    ros_results: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Print the IK parity report (native ``ik-core`` + ROS ``ik-ros``)."""
    _print_pose_parity_report(
        "IK parity report: native curobo core vs ROS /ik_batch",
        report,
        show_all=show_all,
        core_results=core_results,
        ros_results=ros_results,
    )


def print_report_cost(
    report: Dict[str, Any],
    show_all: bool = False,
    core_results: Optional[List[Dict[str, Any]]] = None,
    ros_results: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Print the kinematics & collision parity report (``cost-core`` + ``cost-ros``)."""
    _print_pose_parity_report(
        "Kinematics & collision parity report: native curobo core vs ROS /fk_batch",
        report,
        show_all=show_all,
        core_results=core_results,
        ros_results=ros_results,
    )


def print_ik_table(title: str, results: List[Dict[str, Any]]) -> None:
    """Print one IK leg's results as a curobo benchmark-style grid table."""
    print(f"== {title} ==")
    print(_grid_table(ik_style_rows(results), ["Metric", "Value"]))


def print_cost_table(title: str, results: List[Dict[str, Any]]) -> None:
    """Print one FK/collision leg's results as a curobo benchmark-style grid table."""
    print(f"== {title} ==")
    print(_grid_table(cost_style_rows(results), ["Metric", "Value"]))


def save_report(report: Dict[str, Any], path: str) -> None:
    """Write the report dict as pretty JSON."""
    with open(path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=False)
        f.write("\n")
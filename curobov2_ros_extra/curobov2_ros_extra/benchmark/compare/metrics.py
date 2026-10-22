# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare the native (curobo) and ROS-wrapped benchmark results. The interesting parity fields are planning *outcomes*: success, trajectory
path length, motion time and waypoint count. Timing is informational and is
reported with attribution rather than as a parity signal."""

# Standard Library
import json
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

def trajectory_metrics(
    waypoints: Sequence[Sequence[float]], dt: float
) -> tuple:
    """Compute (n_waypoints, path_length, motion_time_s) from waypoint rows. Args:
        waypoints: joint-position rows, e.g. the interpolated trajectory
            [[q1_1..q1_7], [q2_1..q2_7], ...] (ROS leg) or the equivalent
            native tensor converted to lists. dt: time step between consecutive waypoints (s)."""
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
    """Euclidean distance (mm) between an achieved tool position and its goal. General distance-in-mm helper (the FK-verified value of a trajectory
    endpoint against a goal pose, in the same mm units as the native leg's
    ``result.position_error * 1000``). The ROS leg's report row is fed by
    ``winner_position_error_mm`` (the solver's own residual); this helper
    remains for ad-hoc FK checks."""
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
def _rows_for(results: List[Dict[str, Any]], capability: str) -> List[Dict[str, Any]]:
    """Filter result entries to one capability (and, for ``ik``, the ``cfree``
    parity variant — the ``plain`` row family is native-only reference data)."""
    rows = [r for r in results if r.get("capability") == capability]
    if capability == "ik":
        rows = [r for r in rows if r.get("variant") == "cfree"]
    return rows
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

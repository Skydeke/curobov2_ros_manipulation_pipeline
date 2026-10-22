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

from .metrics import _rel_delta, _rows_for, orientation_error_deg
from .tables import _abs_delta, _vec_mm

def compare(
    core_results: List[Dict[str, Any]],
    ros_results: List[Dict[str, Any]],
    path_tolerance: float = 0.02,
    motion_tolerance: float = 0.02,
) -> Dict[str, Any]:
    """Compare native vs ROS results; return a structured report. Rows are matched by ``problem_name``. A problem matches when:

    - success agrees, AND (when both succeed)
    - relative path-length delta <= ``path_tolerance``, AND
    - relative motion-time delta <= ``motion_tolerance``, AND
    - waypoint counts are equal."""
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
def compare_ik(
    core_results: List[Dict[str, Any]],
    ros_results: List[Dict[str, Any]],
    position_tolerance_mm: float = 0.5,
    orientation_tolerance_deg: float = 0.5,
) -> Dict[str, Any]:
    """Compare IK parity (native vs ROS, both legs' ``cfree`` rows). Rows are matched by ``problem_name`` (``ik_cfree_bNN_gNNN``, identical on
    both legs from the shared goal generator). A goal matches when:

    - success agrees, AND (when both succeed)
    - FK-verified ``position_error_mm`` delta <= ``position_tolerance_mm``,
      AND
    - FK-verified ``orientation_error_deg`` delta <=
      ``orientation_tolerance_deg``."""
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
    """Compare kinematics & collision parity (native FK/validate vs ``/fk_batch``). Rows are matched by ``problem_name`` (``cost_bNN_gNNN``). Per config:

    - ``valid`` agreement is exact (both sides run the same
      ``RobotCollisionChecker.validate`` on the same world, so a flipped bit is
      a real parity fault), and
    - the FK tool poses agree within ``position_tolerance_mm`` /
      ``orientation_tolerance_deg`` (float32 reproducibility guards — the
      expected deltas are ~0)."""
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

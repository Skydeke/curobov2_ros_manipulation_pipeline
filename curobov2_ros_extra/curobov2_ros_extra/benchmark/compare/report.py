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

from .metrics import PUBLISHED_REFERENCE_PAGE
from .parity import _mean
from .tables import _grid_table, _percentile, cost_style_rows, curobo_style_rows, ik_style_rows

def print_curobo_table(title: str, results: List[Dict[str, Any]]) -> None:
    """Print one leg's results as a curobo benchmark-style grid table."""
    print(f"== {title} ==")
    print(_grid_table(curobo_style_rows(results), ["Metric", "Value"]))
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
    """Aggregate one reference table into {metric label: (mean, median)}. Statistic rows cover successful solves only (curobo filters failed
    timings the same way); ``Success %`` is over all problems. Path Length
    and Motion Time prefer the upstream control-point fields when present,
    else the dense-interpolated values."""
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
    """Print ALL results again, grouped like the cuRobo benchmarks page. The page's suite order is motion generation (without torque limits, then
    with torque limits at the full ``mass`` kg payload) → inverse kinematics →
    kinematics & collision. Each suite is shown once for the native curobo leg
    and once for the ROS leg (``None`` = that leg was not run in this
    invocation)."""
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
def print_ik_table(title: str, results: List[Dict[str, Any]]) -> None:
    """Print one IK leg's results as a curobo benchmark-style grid table."""
    print(f"== {title} ==")
    print(_grid_table(ik_style_rows(results), ["Metric", "Value"]))
def print_cost_table(title: str, results: List[Dict[str, Any]]) -> None:
    """Print one FK/collision leg's results as a curobo benchmark-style grid table."""
    print(f"== {title} ==")
    print(_grid_table(cost_style_rows(results), ["Metric", "Value"]))

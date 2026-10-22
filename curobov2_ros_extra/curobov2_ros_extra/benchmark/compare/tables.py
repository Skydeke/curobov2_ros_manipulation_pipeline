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

from .metrics import _rows_for

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
    """Build the ``Metric``/``Value`` body rows for one leg (curobo layout). Row order follows the reference benchmark table (Success %, Plan Time,
    Solve Time, Position Error, Path Length, Motion Time, Jerk, Energy (J),
    Torque (N·m)). Success % covers all problems; the statistic rows cover
    successful problems only (curobo filters ``inf``/failed timings the same
    way)."""
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
def _amortized_ik_ms(entry: Dict[str, Any]) -> Optional[float]:
    """Per-goal IK cost: batch wall divided by the entry's batch size.

    Entries carry the whole batch's solve wall in ``time_ms`` (one parallel
    GPU/batch call per batch); ``n_goals`` is that batch's size (1 for
    per-goal callers, who are unaffected). Missing/empty values stay None
    so ``_stat_str`` skips them like any other absent field.
    """
    wall = entry.get("time_ms")
    if wall is None:
        return None
    n = entry.get("n_goals") or 1
    if not n:
        return None
    return float(wall) / float(n)
def _batch_ik_ms(entry: Dict[str, Any]) -> Optional[float]:
    """Batch IK wall for one entry (batch duration, not per-goal). Batch callers (native/core and ROS ``ik_batch``) stamp the whole
    batch's solve wall in ``time_ms``; per-goal callers (one ``compute_ik``
    per goal) stamp the true per-goal call wall in ``time_ms`` and keep
    the batch wall separately in ``batch_time_ms``. Preferring
    ``batch_time_ms`` when present therefore reads as batch duration on
    both schemas."""
    wall = entry.get("batch_time_ms")
    if wall is None:
        wall = entry.get("time_ms")
    if wall is None:
        return None
    return float(wall)
def _per_ik_ms(entry: Dict[str, Any], batch_sizes: Dict[Any, int]) -> Optional[float]:
    """Amortized per-goal IK cost (batch wall / batch size). Prefers the entry's ``batch_time_ms`` batch wall divided by the number
    of entries sharing its ``batch``: per-goal callers stamp the batch wall
    separately while ``time_ms`` holds only the pure per-call servo wall,
    which excludes queueing/transport overhead (dividing that by
    ``n_goals=1`` reads ~100x too fast next to the batch row). Batch
    callers stamp the batch wall in ``time_ms`` and divide by ``n_goals``
    via ``_amortized_ik_ms``."""
    wall = entry.get("batch_time_ms")
    if wall is not None and entry.get("batch") is not None:
        n = batch_sizes.get(entry.get("batch")) or entry.get("n_goals") or 1
        if not n:
            return None
        return float(wall) / float(n)
    return _amortized_ik_ms(entry)
def ik_style_rows(results: List[Dict[str, Any]]) -> List[List[str]]:
    """Build the ``Metric``/``Value`` rows for one IK leg (curobo-layout). Follows the reference ``ik_benchmark`` table: Success % over all goals;
    IK Time is the per-batch solve wall (batch duration), plus the amortized
    per-goal companion (batch wall / batch size — the upstream table
    likewise reports per-inference times); then the FK-verified pose errors,
    which are the parity fields both legs report identically. Statistics
    rows cover successful goals only."""
    cfree = _rows_for(results, "ik")
    ok = [r for r in cfree if bool(r.get("success"))]
    success_pct = 100.0 * len(ok) / len(cfree) if cfree else 0.0
    batch_sizes = Counter(r.get("batch") for r in cfree)
    rows: List[List[str]] = [
        ["Success %", f"{success_pct:2.2f}"],
        ["IK Time (ms)", _stat_str([_batch_ik_ms(r) for r in ok])],
        ["IK Time (ms) per IK", _stat_str(
            [_per_ik_ms(r, batch_sizes) for r in ok])],
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
    (batch duration) and the per-FK marginal (``time_ms`` is per batch of
    ``n_configs``).
    """
    cost = _rows_for(results, "cost")
    valid = [r for r in cost if bool(r.get("valid"))]
    valid_pct = 100.0 * len(valid) / len(cost) if cost else 0.0
    rows: List[List[str]] = [
        ["Valid %", f"{valid_pct:2.2f}"],
        ["FK Time (ms)", _stat_str([r.get("time_ms") for r in cost])],
        [
            "FK Time (ms) per FK",
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

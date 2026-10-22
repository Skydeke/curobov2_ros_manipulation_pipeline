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

from .report import print_cost_table, print_curobo_table, print_ik_table

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
        print_curobo_table("native (curobo)", core_results)
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
        print_ik_table("native (curobo)", core_results)
    elif core_results is not None:
        print_cost_table("native (curobo)", core_results)
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
def save_report(report: Dict[str, Any], path: str) -> None:
    """Write the report dict as pretty JSON."""
    with open(path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=False)
        f.write("\n")

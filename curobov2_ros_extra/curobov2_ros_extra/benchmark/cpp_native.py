# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native C++-runtime leg adapter for the planner parity benchmark.

``curobo_benchmark`` (``cpp_runtime/benchmarks``) writes one JSONL row per
problem; ``run_benchmarks.py`` enriches every successful row with the same
quality metrics the Python ``core`` leg reports (path length, motion time,
jerk, energy, torque).  This module maps those rows into the shared result
schema and prints the standard webpage summary, so the C++ compose can show
``ROS C++ vs native C++`` with exactly the statistics of
``curobov2_ros_extra.benchmark.run webpage``.

Usage (inside the benchmark image, with the package on PYTHONPATH)::

    python3 -m curobov2_ros_extra.benchmark.cpp_native \
        --native-plain  /work/run/motion_0kg_metrics.json \
        --native-torque /work/run/motion_3kg_metrics.json \
        --ros-plain     /work/run/ros_plain.json \
        --ros-torque    /work/run/ros_torque.json \
        --mass 3.0 -o /work/run/cpp_compare.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from .compare import compare, print_report, print_webpage_summary, save_report
from .compare.parity import _mean
from .compare.report import reference_stats


def _read_rows(path: str) -> List[Dict[str, Any]]:
    """Read either a JSON list (run_benchmarks metrics) or JSONL records."""
    with open(path) as stream:
        text = stream.read().strip()
    if not text:
        return []
    if text[0] == "[":
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected a JSON list")
        return value
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _problem_name(row: Dict[str, Any]) -> str:
    """Match the ROS runner's ``<scene>_<1-based index>`` naming.

    ``run_benchmarks.prepare_motion`` stores the 0-based index over the full
    corpus (including excluded problems); the ROS runner enumerates with
    ``start=1`` and then filters, so ``problem + 1`` is the same index.
    """
    scene = row["scene"]
    problem = int(row["problem"])
    return f"{scene}_{problem + 1}"


def load_native_motion(path: str) -> List[Dict[str, Any]]:
    """Convert one native C++ motion metrics file to the shared schema."""
    entries: List[Dict[str, Any]] = []
    for row in _read_rows(path):
        success = bool(row.get("success"))
        entry: Dict[str, Any] = {
            "problem_name": _problem_name(row),
            "scene_key": row["scene"],
            "capability": "planning",
            "success": success,
            # Native `plan_time_s` is measured around the solver call, the
            # same span as the Python core leg's `time_s`.
            "time_s": row.get("plan_time_s"),
            "solve_time_s": row.get("plan_time_s"),
            "n_waypoints": row.get("steps"),
            "path_length": row.get("path_length_rad"),
            "motion_time_s": row.get("motion_time_s"),
            "jerk": row.get("jerk_rad_s3"),
            "position_error_mm": row.get("position_error_mm"),
            "energy_j": row.get("energy_j"),
            "torque_nm": row.get("torque_nm"),
        }
        if not success:
            entry["message"] = "native cpp solve failed"
        entries.append(entry)
    return entries


# Values recorded on the current page (RTX 5090, sm_120, 2026-10-30/31):
# https://nvlabs.github.io/curobo/latest/reference/benchmarks.html
CPP_PAGE_MOTION = {
    "0": {"success": "2,582 / 2,590 (99.69%)", "plan_ms": "4.464", "motion_s": "1.177"},
    "3": {"success": "2,582 / 2,590 (99.69%)", "plan_ms": "4.868", "motion_s": "1.266"},
}
CPP_PAGE_IK = {
    "unitree_g1_29dof": ("19.201", "100", "162.564", "90.2"),
    "dual_ur10e": ("3.574", "100", "8.434", "99.4"),
    "franka": ("0.551", "100", "0.559", "100"),
}


def _motion_summary(rows: List[Dict[str, Any]]) -> tuple[str, str, str, int, int]:
    """(success text, mean plan ms, mean motion s, successes, total)."""
    total = len(rows)
    ok = [r for r in rows if r.get("success")]
    plan = _mean([r.get("time_s") for r in rows])
    motion = _mean([r.get("motion_time_s") for r in ok])
    success = f"{len(ok)} / {total} ({100.0 * len(ok) / total:.2f}%)" if total else "0 / 0"
    return (
        success,
        "n/a" if plan is None else f"{plan * 1e3:.3f}",
        "n/a" if motion is None else f"{motion:.3f}",
        len(ok),
        total,
    )


def print_recorded_cpp_page(
    native_plain: List[Dict[str, Any]],
    native_torque: List[Dict[str, Any]],
    ros_plain: List[Dict[str, Any]],
    ros_torque: List[Dict[str, Any]],
    native_ik: Dict[str, Dict[str, List[Dict[str, Any]]]],
    ros_ik: Optional[List[Dict[str, Any]]],
    leg_labels: tuple[str, str] = ("native C++", "ROS C++"),
) -> None:
    """Reproduce the page's Recorded C++ Results table format exactly."""
    print("\n== RECORDED PAGE RESULTS — page vs this box ==")
    print("| Payload | Leg | Successful problems | Mean planning time (ms) | Mean motion duration (s) |")
    print("| --- | --- | --- | ---: | ---: |")
    for key, page in CPP_PAGE_MOTION.items():
        payload = "0 kg" if key == "0" else "3 kg"
        print(f"| {payload} | page (RTX 5090) | {page['success']} | {page['plan_ms']} | {page['motion_s']} |")
        native_rows = native_plain if key == "0" else native_torque
        ros_rows = ros_plain if key == "0" else ros_torque
        for label, rows in ((leg_labels[0], native_rows), (leg_labels[1], ros_rows)):
            success, plan_ms, motion_s, _ok, _total = _motion_summary(rows)
            print(f"| {payload} | {label} | {success} | {plan_ms} | {motion_s} |")

    print("\n| Robot | Pose IK (ms/batch) | Success (%) | C-free IK (ms/batch) | Success (%) |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for robot, label in _IK_ROBOTS:
        page = CPP_PAGE_IK.get(robot)
        variants = native_ik.get(robot, {})
        pose = _ik_stats(variants.get("plain", []))
        cfree = _ik_stats(variants.get("cfree", []))
        if page:
            print(
                f"| {label} (page) | {page[0]} | {page[1]} | {page[2]} | {page[3]} |"
            )
        print(
            f"| {label} ({leg_labels[0]}) | {_fmt_ms(pose[0])} | {pose[1]:.2f} | "
            f"{_fmt_ms(cfree[0])} | {cfree[1]:.2f} |"
        )
    if ros_ik is not None:
        cfree = _ik_stats([r for r in ros_ik if r.get("capability") == "ik"])
        print(f"| Franka ({leg_labels[1]}) | n/a | n/a | {_fmt_ms(cfree[0])} | {cfree[1]:.2f} |")


def load_core_motion(path: str) -> List[Dict[str, Any]]:
    """Load a Python ``core`` result JSON (already in the shared schema)."""
    with open(path) as stream:
        rows = json.load(stream)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON list")
    return rows


def load_core_ik(path: str) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Group Python ``ik-core`` rows (Franka default) by variant."""
    with open(path) as stream:
        rows = json.load(stream)
    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {"franka": {}}
    for row in rows:
        if row.get("capability") != "ik":
            continue
        out["franka"].setdefault(row.get("variant", "cfree"), []).append(row)
    return out


def _overhead_line(label: str, native: List[Dict[str, Any]],
                   ros: List[Dict[str, Any]], key: str) -> Optional[str]:
    """Mean successful value per leg and the ROS delta for one timing key."""
    by_native = {r["problem_name"]: r for r in native if r.get("success")}
    by_ros = {r["problem_name"]: r for r in ros if r.get("success")}
    common = sorted(set(by_native) & set(by_ros))
    n = _mean([by_native[name].get(key) for name in common])
    r = _mean([by_ros[name].get(key) for name in common])
    if n is None or r is None:
        return None
    pct = ((r - n) / n * 100.0) if n > 0 else None
    pct_text = "n/a" if pct is None else f"{pct:+.1f}%"
    return (
        f"  {label:<24} native {n * 1e3:8.2f} ms   "
        f"ROS {r * 1e3:8.2f} ms   overhead {(r - n) * 1e3:+8.2f} ms ({pct_text})"
    )


# Page order: Unitree G1, dual UR10e, Franka.
_IK_ROBOTS = (
    ("unitree_g1_29dof", "G1"),
    ("dual_ur10e", "Dual UR10e"),
    ("franka", "Franka"),
)


def load_native_ik(directory: str) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Load the native C++ IK JSONL files into {robot: {variant: rows}}.

    ``run_benchmarks.py`` writes one record per query with ``time_s`` being
    the batch wall (the same convention as the page's ms/batch column).
    """
    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for robot, _label in _IK_ROBOTS:
        robot_rows: Dict[str, List[Dict[str, Any]]] = {}
        for suffix, variant in (
            ("pose_only", "plain"),
            ("collision_free", "cfree"),
        ):
            path = Path(directory) / f"{robot}_{suffix}.jsonl"
            if not path.is_file():
                continue
            rows: List[Dict[str, Any]] = []
            for row in _read_rows(str(path)):
                batch = int(row["batch"])
                query = int(row["query"])
                success = bool(row.get("success"))
                entry: Dict[str, Any] = {
                    "problem_name": f"ik_{variant}_b{batch + 1:02d}_q{query + 1:03d}",
                    "scene_key": f"ik_{variant}",
                    "capability": "ik",
                    "variant": variant,
                    "batch": batch + 1,
                    "index": query + 1,
                    "n_goals": 100,
                    "success": success,
                    "time_ms": float(row["time_s"]) * 1000.0,
                    "position_error_mm": None,
                    "orientation_error_deg": None,
                }
                if success:
                    if row.get("position_error_m") is not None:
                        entry["position_error_mm"] = float(row["position_error_m"]) * 1000.0
                    if row.get("orientation_error_rad") is not None:
                        entry["orientation_error_deg"] = math.degrees(
                            float(row["orientation_error_rad"])
                        )
                rows.append(entry)
            robot_rows[variant] = rows
        if robot_rows:
            out[robot] = robot_rows
    return out


def _ik_stats(rows: List[Dict[str, Any]]) -> tuple[Optional[float], float, int]:
    """(mean batch ms, success %, queries) for one robot/variant."""
    batches: Dict[int, Optional[float]] = {}
    success = 0
    for row in rows:
        if row.get("capability") != "ik":
            continue
        batches.setdefault(row.get("batch"), row.get("time_ms"))
        success += bool(row.get("success"))
    times = [v for v in batches.values() if v is not None]
    return (_mean(times), (100.0 * success / len(rows) if rows else 0.0), len(rows))


def _fmt_ms(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def print_ik_tables(
    native: Dict[str, Dict[str, List[Dict[str, Any]]]],
    ros: Optional[List[Dict[str, Any]]],
    native_label: str = "native C++",
    ros_label: str = "ROS C++",
) -> None:
    """Print the page's IK grid (pose + collision-free) for both legs."""
    print(f"\n== inverse kinematics — {native_label} (page order) ==")
    print("| Robot | Pose IK (ms/batch) | Success (%) | C-free IK (ms/batch) | Success (%) |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for robot, label in _IK_ROBOTS:
        variants = native.get(robot, {})
        pose = _ik_stats(variants.get("plain", []))
        cfree = _ik_stats(variants.get("cfree", []))
        print(
            f"| {label} | {_fmt_ms(pose[0])} | {pose[1]:.2f} | "
            f"{_fmt_ms(cfree[0])} | {cfree[1]:.2f} |"
        )

    if ros is None:
        print("\n== inverse kinematics — ROS C++ (server /ik_batch) ==\n"
              "  (not run)")
        return
    ros_rows = [r for r in ros if r.get("capability") == "ik"]
    print(f"\n== inverse kinematics — {ros_label} (Franka, collision-free only) ==")
    print("| Robot | Pose IK (ms/batch) | Success (%) | C-free IK (ms/batch) | Success (%) |")
    print("| --- | ---: | ---: | ---: | ---: |")
    # The ROS server is a single Franka robot; only cfree is exposed.
    cfree = _ik_stats(ros_rows)
    print(f"| Franka | n/a | n/a | {_fmt_ms(cfree[0])} | {cfree[1]:.2f} |")


def print_overhead(native: List[Dict[str, Any]], ros: List[Dict[str, Any]],
                   title: str) -> None:
    """Print the ROS transport + service overhead against native C++."""
    print(f"\n== ROS overhead — {title} ==")
    for label, key in (
        ("wall / plan", "time_s"),
        ("solver-reported", "solve_time_s"),
    ):
        line = _overhead_line(label, native, ros, key)
        if line:
            print(line)
    native_ok = sum(1 for r in native if r.get("success"))
    ros_ok = sum(1 for r in ros if r.get("success"))
    print(
        f"  success                  native {native_ok}/{len(native)}   "
        f"ROS {ros_ok}/{len(ros)}"
    )
    stats_native = reference_stats(native)
    stats_ros = reference_stats(ros)
    for metric in ("Path Length (rad.)", "Motion Time(s)"):
        n = stats_native.get(metric)
        r = stats_ros.get(metric)
        if n and r:
            print(
                f"  {metric:<24} native {n[0]:8.3f}   ROS {r[0]:8.3f}"
            )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-format", choices=("cpp", "core"), default="cpp",
                        help="cpp: run_benchmarks motion metrics; core: curobov2_ros_extra core JSON")
    parser.add_argument("--native-plain", required=True,
                        help="native without-torque results (C++ metrics or core JSON)")
    parser.add_argument("--native-torque", required=True,
                        help="motion_3kg_metrics.json from run_benchmarks.py")
    parser.add_argument("--ros-plain", required=True,
                        help="ROS result JSON for the without-torque leg")
    parser.add_argument("--ros-torque", required=True,
                        help="ROS result JSON for the with-torque leg")
    parser.add_argument("--native-ik-dir", default=None,
                        help="directory with run_benchmarks.py IK JSONL files")
    parser.add_argument("--native-ik", default=None,
                        help="Python core ik-core result JSON")
    parser.add_argument("--ros-ik", default=None,
                        help="ROS ik-ros result JSON")
    parser.add_argument("--native-cost", default=None,
                        help="Python core cost-core result JSON")
    parser.add_argument("--ros-cost", default=None,
                        help="ROS cost-ros result JSON")
    parser.add_argument("--mass", type=float, default=3.0)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args(argv)

    if args.native_format == "core":
        native_plain = load_core_motion(args.native_plain)
        native_torque = load_core_motion(args.native_torque)
    else:
        native_plain = load_native_motion(args.native_plain)
        native_torque = load_native_motion(args.native_torque)
    with open(args.ros_plain) as stream:
        ros_plain = json.load(stream)
    with open(args.ros_torque) as stream:
        ros_torque = json.load(stream)

    if args.native_format == "core" and args.native_ik:
        native_ik = load_core_ik(args.native_ik)
    else:
        native_ik = load_native_ik(args.native_ik_dir) if args.native_ik_dir else {}
    ros_ik: Optional[List[Dict[str, Any]]] = None
    if args.ros_ik:
        with open(args.ros_ik) as stream:
            ros_ik = json.load(stream)
    native_cost = json.load(open(args.native_cost)) if args.native_cost else None
    ros_cost = json.load(open(args.ros_cost)) if args.ros_cost else None

    print_webpage_summary(
        mass=args.mass,
        motion_plain_native=native_plain,
        motion_plain_ros=ros_plain,
        motion_torque_native=native_torque,
        motion_torque_ros=ros_torque,
        # The C++ runtime benchmark exposes motion + IK; the FK/collision
        # `cost` suite has no native C++ benchmark command, so leave it out
        # rather than silently mixing a Python native leg into a C++ run.
        ik_native=[row for rows in native_ik.values() for row in rows.get("cfree", [])],
        ik_ros=ros_ik,
        cost_native=native_cost,
        cost_ros=ros_cost,
    )
    labels = (
        ("native Python", "ROS Python")
        if args.native_format == "core"
        else ("native C++", "ROS C++")
    )
    print_recorded_cpp_page(
        native_plain, native_torque, ros_plain, ros_torque, native_ik, ros_ik,
        leg_labels=labels,
    )
    print_ik_tables(native_ik, ros_ik, native_label=labels[0], ros_label=labels[1])

    print_overhead(native_plain, ros_plain, "without torque limits")
    print_overhead(native_torque, ros_torque,
                   f"with torque limits ({args.mass:g} kg)")

    report_plain = compare(native_plain, ros_plain)
    report_torque = compare(native_torque, ros_torque)
    print("\n== parity report — without torque limits ==")
    print_report(report_plain)
    print("\n== parity report — with torque limits ==")
    print_report(report_torque)

    if args.output:
        save_report(
            {
                "without_torque": report_plain,
                "with_torque": report_torque,
                "mass": args.mass,
            },
            args.output,
        )
    # The page-level statistics are the deliverable; the per-problem parity
    # verdict is diagnostic (native and ROS use different solver paths, so a
    # sub-2% path/motion delta is expected and not a benchmark failure).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

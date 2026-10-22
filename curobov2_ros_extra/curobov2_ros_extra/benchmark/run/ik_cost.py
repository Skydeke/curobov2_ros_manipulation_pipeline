# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI for the planner parity benchmark.

Subcommands:

    curobo_benchmark core [options]          native curobo leg (planning)
    curobo_benchmark ros  [options]          ROS-wrapped planning leg (server up)
    curobo_benchmark compare C R [opts]      compare two result JSONs
    curobo_benchmark all [options]           core + ros + compare in one go
    curobo_benchmark reference [options]     reproduce BOTH reference-page
                                             tables (with/without torque
                                             limits) and print them for
                                             comparison
    curobo_benchmark webpage [options]       run the page's suites in order
                                             (motion gen -> IK -> cost),
                                             once native / once ROS, then
                                             print everything again (the
                                             docker compose default)

    curobo_benchmark ik-core [options]       native IK leg (cfree + plain rows)
    curobo_benchmark ik-ros [options]        ROS IK leg via /ik_batch (server up)
    curobo_benchmark ik [options]            ik-core + ik-ros + compare_ik

    curobo_benchmark cost-core [options]     native FK + collision leg
    curobo_benchmark cost-ros [options]      ROS FK/collision leg via /fk_batch
    curobo_benchmark cost [options]          cost-core + cost-ros + compare_cost

Run as a ROS package entry point (``ros2 run curobov2_ros_extra
curobo_benchmark ...`` after a rebuild) or directly::

    python3 -m curobov2_ros_extra.benchmark.run all --dataset demo
"""

# Standard Library
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from ..compare import (
    compare,
    compare_cost,
    compare_ik,
    print_report,
    print_report_cost,
    print_report_ik,
    save_report,
)

from .common import _dump, _print_run_summary, _resolve_tol, _sidecar

def cmd_ik_core(args) -> int:
    from ..core_ik import run_ik_core

    results = run_ik_core(
        batch=args.batch,
        n_batches=args.num_batches,
        num_seeds=args.num_seeds,
        robot_config=args.robot_config,
        seed=args.seed,
        variants=args.variants,
    )
    if args.output:
        _dump(results, args.output)
    _print_ik_core_summary(results)
    return 0
def cmd_ik_ros(args) -> int:
    from ..ros_ik import run_ik_ros

    results = run_ik_ros(
        batch=args.batch,
        n_batches=args.num_batches,
        robot_config=args.robot_config,
        seed=args.seed,
        service_timeout=args.service_timeout,
        call_timeout=args.call_timeout,
        size_collision_cache=not args.no_size_cache,
    )
    if args.output:
        _dump(results, args.output)
    _print_run_summary(results, "ik-ros")
    return 0
def cmd_cost_core(args) -> int:
    from ..core_cost import run_cost_core

    results = run_cost_core(
        batch=args.batch,
        n_batches=args.num_batches,
        robot_config=args.robot_config,
        seed=args.seed,
    )
    if args.output:
        _dump(results, args.output)
    _print_cost_core_summary(results)
    return 0
def cmd_cost_ros(args) -> int:
    from ..ros_cost import run_cost_ros

    results = run_cost_ros(
        batch=args.batch,
        n_batches=args.num_batches,
        robot_config=args.robot_config,
        seed=args.seed,
        service_timeout=args.service_timeout,
        call_timeout=args.call_timeout,
        size_collision_cache=not args.no_size_cache,
    )
    if args.output:
        _dump(results, args.output)
    _print_run_summary(results, "cost-ros", key="valid")
    return 0
def cmd_ik(args) -> int:
    """ik-core + ik-ros + compare_ik, with the parity verdict as exit code."""
    from ..core_ik import run_ik_core
    from ..ros_ik import run_ik_ros

    print("== ik-core leg (native curobo IK) ==", flush=True)
    core_results = run_ik_core(
        batch=args.batch,
        n_batches=args.num_batches,
        num_seeds=args.num_seeds,
        robot_config=args.robot_config,
        seed=args.seed,
        variants=args.variants,
    )
    core_path = _sidecar(args.output, "ik-core")
    if core_path:
        _dump(core_results, core_path)
    _print_ik_core_summary(core_results)

    print("== ik-ros leg (ROS-wrapped IK server) ==", flush=True)
    ros_results = run_ik_ros(
        batch=args.batch,
        n_batches=args.num_batches,
        robot_config=args.robot_config,
        seed=args.seed,
        service_timeout=args.service_timeout,
        call_timeout=args.call_timeout,
        size_collision_cache=not args.no_size_cache,
    )
    ros_path = _sidecar(args.output, "ik-ros")
    if ros_path:
        _dump(ros_results, ros_path)
    _print_run_summary(ros_results, "ik-ros")

    print("== comparing IK parity ==", flush=True)
    report = compare_ik(
        core_results,
        ros_results,
        position_tolerance_mm=_resolve_tol(args.position_tolerance_mm, 0.5),
        orientation_tolerance_deg=_resolve_tol(
            args.orientation_tolerance_deg, 0.5
        ),
    )
    print_report_ik(
        report,
        show_all=args.show_all,
        core_results=core_results,
        ros_results=ros_results,
    )
    if args.output:
        save_report(report, args.output)
    return 0 if report["verdict"] == "PARITY-OK" else 1
def cmd_cost(args) -> int:
    """cost-core + cost-ros + compare_cost, with the parity verdict as exit code."""
    from ..core_cost import run_cost_core
    from ..ros_cost import run_cost_ros

    print("== cost-core leg (native FK + collision) ==", flush=True)
    core_results = run_cost_core(
        batch=args.batch,
        n_batches=args.num_batches,
        robot_config=args.robot_config,
        seed=args.seed,
    )
    core_path = _sidecar(args.output, "cost-core")
    if core_path:
        _dump(core_results, core_path)
    _print_cost_core_summary(core_results)

    print("== cost-ros leg (ROS-wrapped FK server) ==", flush=True)
    ros_results = run_cost_ros(
        batch=args.batch,
        n_batches=args.num_batches,
        robot_config=args.robot_config,
        seed=args.seed,
        service_timeout=args.service_timeout,
        call_timeout=args.call_timeout,
        size_collision_cache=not args.no_size_cache,
    )
    ros_path = _sidecar(args.output, "cost-ros")
    if ros_path:
        _dump(ros_results, ros_path)
    _print_run_summary(ros_results, "cost-ros", key="valid")

    print("== comparing kinematics & collision parity ==", flush=True)
    report = compare_cost(
        core_results,
        ros_results,
        position_tolerance_mm=_resolve_tol(args.position_tolerance_mm, 1e-3),
        orientation_tolerance_deg=_resolve_tol(
            args.orientation_tolerance_deg, 1e-3
        ),
    )
    print_report_cost(
        report,
        show_all=args.show_all,
        core_results=core_results,
        ros_results=ros_results,
    )
    if args.output:
        save_report(report, args.output)
    return 0 if report["verdict"] == "PARITY-OK" else 1
def _print_ik_core_summary(results) -> None:
    n_ok = sum(1 for r in results if r.get("success"))
    n_cfree = sum(1 for r in results if r.get("variant") == "cfree")
    n_plain = sum(1 for r in results if r.get("variant") == "plain")
    print(
        f"ik-core: {len(results)} goals "
        f"({n_cfree} cfree, {n_plain} plain), {n_ok} succeeded",
        flush=True,
    )
def _print_cost_core_summary(results) -> None:
    n_valid = sum(1 for r in results if r.get("valid"))
    print(
        f"cost-core: {len(results)} configs, {n_valid} valid",
        flush=True,
    )

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

from .common import _dump, _print_run_summary, _print_wall_time, _resolve_tol, _seed_knobs, _sidecar

def cmd_core(args) -> int:
    from ..core_runner import run_core

    results = run_core(
        dataset=args.dataset,
        scene=args.scene,
        warmup_iters=args.warmup_iters,
        max_attempts=args.max_attempts,
        num_ik_seeds=args.num_ik_seeds,
        num_trajopt_seeds=args.num_trajopt_seeds,
        use_cuda_graph=not args.no_cuda_graph,
        mesh=args.mesh,
        use_dynamics=args.use_dynamics,
        mass=args.mass,
        **_seed_knobs(args),
    )
    if args.output:
        _dump(results, args.output)
    _print_run_summary(results, "core")
    return 0
def cmd_ros(args) -> int:
    from ..ros_runner import run_ros

    results = run_ros(
        dataset=args.dataset,
        scene=args.scene,
        service_timeout=args.service_timeout,
        call_timeout=args.call_timeout,
        size_collision_cache=not args.no_size_cache,
        apply_mpinets_locks=not args.no_mpinets_locks,
        use_dynamics=args.use_dynamics,
        mass=args.mass,
        # Whole-benchmark envelope: pin the server's plan-time max_attempts
        # to this run's own --max-attempts (default 100 = the page's budget),
        # so a standalone ROS leg retries exactly like every other leg.
        server_max_attempts=args.max_attempts,
        # …and its per-plan RNG rewind to this run's own seed diagnostic, so
        # --no-reset-seed A/Bs both legs instead of one.
        server_reset_seed=_seed_knobs(args)["reset_seed_per_problem"],
    )
    if args.output:
        _dump(results, args.output)
    _print_run_summary(results, "ros")
    return 0
def cmd_reference(args) -> int:
    """Reproduce BOTH reference-page tables (native leg) and print them again. Runs the same native benchmark twice over the dataset (default ``full`` —
    the page's 2600 problems), once with the plain solver and once with
    torque-limited planning at ``--mass`` (default 3.0 kg = full payload),
    then prints both curobo-style tables again plus a side-by-side mean /
    median comparison against the values published on the cuRobo benchmarks
    page (see ``compare.print_reference_comparison``), so the box numbers can
    be eyeballed against the webpage without digging through the logs. ``--max-attempts`` defaults to 100 — the upstream script's real-solve
    budget, whose retry loop produces the page's 99.73 % success rate — for
    every benchmark subcommand (the ROS runner pins the server's plan-time
    ``max_attempts`` to whichever budget the run uses, so native and ROS
    always share the same retry envelope)."""
    from ..compare import print_reference_comparison
    from ..core_runner import run_core

    common = dict(
        dataset=args.dataset,
        scene=args.scene,
        warmup_iters=args.warmup_iters,
        max_attempts=args.max_attempts,
        num_ik_seeds=args.num_ik_seeds,
        num_trajopt_seeds=args.num_trajopt_seeds,
        use_cuda_graph=not args.no_cuda_graph,
        mesh=args.mesh,
        mass=args.mass,
        **_seed_knobs(args),
    )
    print("== reference pass 1/2: without torque limits ==", flush=True)
    plain = run_core(**common, use_dynamics=False)
    print(
        f"== reference pass 2/2: with torque limits ({args.mass:g} kg) ==",
        flush=True,
    )
    torque = run_core(**common, use_dynamics=True)

    if args.output:
        _dump(plain, _sidecar(args.output, "plain"))
        _dump(torque, _sidecar(args.output, "torque"))
    _print_run_summary(plain, "without torque limits")
    _print_run_summary(torque, f"with torque limits ({args.mass:g} kg)")

    print(
        "\n== reference-page reproduction — motion generation results ==",
        flush=True,
    )
    print_reference_comparison(plain, torque, mass=args.mass)
    return 0
def cmd_webpage(args) -> int:
    """Run the page's benchmark suites in order, once native / once ROS, and
    print everything again at the end (webpage order, both legs). Suites follow the cuRobo benchmarks page: motion generation (without
    torque limits, then with torque limits at the full ``--mass`` payload),
    inverse kinematics, and kinematics & collision. The native legs always
    run; the ROS legs run with ``--run-ros`` (the composed server must be
    up)."""
    from ..compare import print_webpage_summary
    from ..core_cost import run_cost_core
    from ..core_ik import run_ik_core
    from ..core_runner import run_core
    from ..ros_cost import run_cost_ros
    from ..ros_ik import run_ik_ros
    from ..ros_runner import run_ros

    common = dict(
        dataset=args.dataset,
        scene=args.scene,
        warmup_iters=args.warmup_iters,
        max_attempts=args.max_attempts,
        num_ik_seeds=args.num_ik_seeds,
        num_trajopt_seeds=args.num_trajopt_seeds,
        use_cuda_graph=not args.no_cuda_graph,
        mesh=args.mesh,
        mass=args.mass,
        verbose=False,  # quiet legs: the FINAL SUMMARY below is the deliverable
        **_seed_knobs(args),
    )
    service = dict(
        service_timeout=args.service_timeout,
        call_timeout=args.call_timeout,
        size_collision_cache=not args.no_size_cache,
    )
    synthetic = dict(
        batch=args.batch,
        n_batches=args.num_batches,
        seed=args.seed,
        robot_config=args.robot_config,
    )

    import time as _time

    run_t0 = _time.perf_counter()
    wall: Dict[str, float] = {}

    def _timed(label: str, func, *f_args, **f_kwargs):
        t0 = _time.perf_counter()
        try:
            return func(*f_args, **f_kwargs)
        finally:
            wall[label] = _time.perf_counter() - t0

    print("== [1/3] motion generation: pass 1/2 (without torque limits) ==",
          flush=True)
    motion_plain_native = _timed(
        "motion plain — native", run_core, **common, use_dynamics=False)
    motion_plain_ros = None
    if args.run_ros:
        # Neither ROS leg is ever skipped: the runner ensures the server's
        # torque mode itself (set_server_torque_mode) — plain for this leg,
        # torque-limited at --mass for the pass below. No --server-torque gate.
        print("   ROS leg (server torque mode off)", flush=True)
        motion_plain_ros = _timed(
            "motion plain — ROS", run_ros,
            dataset=args.dataset, scene=args.scene,
            use_dynamics=False, mass=args.mass, verbose=False,
            server_dynamics=False, server_payload_mass=0.0,
            server_max_attempts=args.max_attempts,
            server_reset_seed=_seed_knobs(args)["reset_seed_per_problem"],
            apply_mpinets_locks=not args.no_mpinets_locks, **service,
        )

    print("== [1/3] motion generation: pass 2/2 (with torque limits) ==",
          flush=True)
    motion_torque_native = _timed(
        "motion torque — native", run_core, **common, use_dynamics=True)
    motion_torque_ros = None
    if args.run_ros:
        print("   ROS leg (server torque mode on, payload from --mass)",
              flush=True)
        motion_torque_ros = _timed(
            "motion torque — ROS", run_ros,
            dataset=args.dataset, scene=args.scene,
            use_dynamics=True, mass=args.mass, verbose=False,
            server_dynamics=True, server_payload_mass=args.mass,
            server_max_attempts=args.max_attempts,
            server_reset_seed=_seed_knobs(args)["reset_seed_per_problem"],
            apply_mpinets_locks=not args.no_mpinets_locks, **service,
        )

    print("== [2/3] inverse kinematics ==", flush=True)
    ik_native = _timed("ik — native", run_ik_core,
                         num_seeds=args.num_seeds, **synthetic)
    ik_ros = _timed("ik — ROS", run_ik_ros,
                    **synthetic, **service) if args.run_ros else None

    print("== [3/3] kinematics & collision ==", flush=True)
    cost_native = _timed("cost — native", run_cost_core, **synthetic)
    cost_ros = _timed("cost — ROS", run_cost_ros,
                      **synthetic, **service) if args.run_ros else None

    written: List[str] = []
    if args.output:
        for label, data in (
            ("motion-plain-core", motion_plain_native),
            ("motion-plain-ros", motion_plain_ros),
            ("motion-torque-core", motion_torque_native),
            ("motion-torque-ros", motion_torque_ros),
            ("ik-core", ik_native),
            ("ik-ros", ik_ros),
            ("cost-core", cost_native),
            ("cost-ros", cost_ros),
        ):
            if data is not None:
                path = _sidecar(args.output, label)
                _dump(data, path)
                written.append(path)

    print("\n" + "#" * 72, flush=True)
    print("# ALL RESULTS — final printout (webpage order, native + ROS legs)", flush=True)
    print("#", flush=True)
    print("# Everything below this banner is the complete results summary; the", flush=True)
    print("# per-problem progress lines above were verbose-run diagnostics.", flush=True)
    print("#" * 72, flush=True)
    print_webpage_summary(
        mass=args.mass,
        motion_plain_native=motion_plain_native,
        motion_plain_ros=motion_plain_ros,
        motion_torque_native=motion_torque_native,
        motion_torque_ros=motion_torque_ros,
        ik_native=ik_native,
        ik_ros=ik_ros,
        cost_native=cost_native,
        cost_ros=cost_ros,
    )
    print("ALL RESULTS END — ^ above is the complete final printout.", flush=True)
    if written:
        print("Result JSONs:", flush=True)
        for path in written:
            print(f"  {path}", flush=True)
    _print_wall_time(wall, _time.perf_counter() - run_t0)
    return 0
def cmd_compare(args) -> int:
    with open(args.core_results) as f:
        core_results = json.load(f)
    with open(args.ros_results) as f:
        ros_results = json.load(f)

    capability = args.capability
    if capability == "ik":
        report = compare_ik(
            core_results,
            ros_results,
            position_tolerance_mm=_resolve_tol(
                args.position_tolerance_mm, 0.5
            ),
            orientation_tolerance_deg=_resolve_tol(
                args.orientation_tolerance_deg, 0.5
            ),
        )
        printer = print_report_ik
    elif capability == "cost":
        report = compare_cost(
            core_results,
            ros_results,
            position_tolerance_mm=_resolve_tol(
                args.position_tolerance_mm, 1e-3
            ),
            orientation_tolerance_deg=_resolve_tol(
                args.orientation_tolerance_deg, 1e-3
            ),
        )
        printer = print_report_cost
    else:
        report = compare(
            core_results,
            ros_results,
            path_tolerance=args.path_tolerance,
            motion_tolerance=args.motion_tolerance,
        )
        printer = print_report
    printer(
        report,
        show_all=args.show_all,
        core_results=core_results,
        ros_results=ros_results,
    )
    if args.output:
        save_report(report, args.output)
    return 0 if report["verdict"] == "PARITY-OK" else 1
def cmd_all(args) -> int:
    from ..core_runner import run_core
    from ..ros_runner import run_ros

    print("== core leg (native curobo) ==", flush=True)
    core_results = run_core(
        dataset=args.dataset,
        scene=args.scene,
        warmup_iters=args.warmup_iters,
        max_attempts=args.max_attempts,
        num_ik_seeds=args.num_ik_seeds,
        num_trajopt_seeds=args.num_trajopt_seeds,
        use_cuda_graph=not args.no_cuda_graph,
        mesh=args.mesh,
        use_dynamics=args.use_dynamics,
        mass=args.mass,
        **_seed_knobs(args),
    )
    core_path = _sidecar(args.output, "core")
    if core_path:
        _dump(core_results, core_path)
    _print_run_summary(core_results, "core")

    print("== ros leg (ROS-wrapped planner) ==", flush=True)
    ros_results = run_ros(
        dataset=args.dataset,
        scene=args.scene,
        service_timeout=args.service_timeout,
        call_timeout=args.call_timeout,
        size_collision_cache=not args.no_size_cache,
        use_dynamics=args.use_dynamics,
        mass=args.mass,
        # Keep the server's plan_pose retry budget identical to the native
        # leg's: the ROS runner pins the server's plan-time max_attempts to
        # this run's own --max-attempts (default 100 = the page's budget for
        # the whole benchmark), so both legs share one envelope regardless of
        # how the server was launched.
        server_max_attempts=args.max_attempts,
        # Same for the per-plan RNG rewind, so --no-reset-seed A/Bs both legs
        # whatever the server was launched with.
        server_reset_seed=_seed_knobs(args)["reset_seed_per_problem"],
    )
    ros_path = _sidecar(args.output, "ros")
    if ros_path:
        _dump(ros_results, ros_path)
    _print_run_summary(ros_results, "ros")

    print("== comparing ==", flush=True)
    report = compare(
        core_results,
        ros_results,
        path_tolerance=args.path_tolerance,
        motion_tolerance=args.motion_tolerance,
    )
    print_report(
        report,
        show_all=args.show_all,
        core_results=core_results,
        ros_results=ros_results,
    )
    if args.output:
        save_report(report, args.output)
    return 0 if report["verdict"] == "PARITY-OK" else 1

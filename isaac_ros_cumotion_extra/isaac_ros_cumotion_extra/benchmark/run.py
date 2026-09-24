# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI for the planner parity benchmark.

Subcommands:

    curobo_benchmark core [options]          native curobo_core leg (planning)
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

Run as a ROS package entry point (``ros2 run isaac_ros_cumotion_extra
curobo_benchmark ...`` after a rebuild) or directly::

    python3 -m isaac_ros_cumotion_extra.benchmark.run all --dataset demo
"""

# Standard Library
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from .compare import (
    compare,
    compare_cost,
    compare_ik,
    print_report,
    print_report_cost,
    print_report_ik,
    save_report,
)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _dump(results: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"Wrote {len(results)} results to {path}", flush=True)


def _sidecar(output: Optional[str], label: str) -> Optional[str]:
    """``report.json`` + ``core`` -> ``report.core.json`` ('' output -> None)."""
    if not output:
        return None
    root, ext = os.path.splitext(output)
    return f"{root}.{label}{ext or '.json'}"


def _print_run_summary(results, label: str, key: str = "success") -> None:
    n_ok = sum(1 for r in results if r.get(key, False))
    print(f"{label}: {len(results)} problems, {n_ok} succeeded", flush=True)


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------


def _seed_knobs(args) -> Dict[str, bool]:
    """Core-leg RNG diagnostic knobs from CLI args.

    ``--unseeded`` implies ``--no-reset-seed``: the server reseeds neither the
    global generators nor the per-solve solver RNG, so both flags together
    reproduce its drifting-RNG condition.
    """
    reset_seed_per_problem = not (args.no_reset_seed or args.unseeded)
    seed_globals = not args.unseeded
    return {
        "seed_globals": seed_globals,
        "reset_seed_per_problem": reset_seed_per_problem,
    }


def cmd_core(args) -> int:
    from .core_runner import run_core

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
    from .ros_runner import run_ros

    results = run_ros(
        dataset=args.dataset,
        scene=args.scene,
        service_timeout=args.service_timeout,
        call_timeout=args.call_timeout,
        size_collision_cache=not args.no_size_cache,
        use_dynamics=args.use_dynamics,
        mass=args.mass,
        # Whole-benchmark envelope: pin the server's plan-time max_attempts
        # to this run's own --max-attempts (default 100 = the page's budget),
        # so a standalone ROS leg retries exactly like every other leg.
        server_max_attempts=args.max_attempts,
    )
    if args.output:
        _dump(results, args.output)
    _print_run_summary(results, "ros")
    return 0


def cmd_reference(args) -> int:
    """Reproduce BOTH reference-page tables (native leg) and print them again.

    Runs the same native benchmark twice over the dataset (default ``full`` —
    the page's 2600 problems), once with the plain solver and once with
    torque-limited planning at ``--mass`` (default 3.0 kg = full payload),
    then prints both curobo-style tables again plus a side-by-side mean /
    median comparison against the values published on the cuRobo benchmarks
    page (see ``compare.print_reference_comparison``), so the box numbers can
    be eyeballed against the webpage without digging through the logs.

    ``--max-attempts`` defaults to 100 — the upstream script's real-solve
    budget, whose retry loop produces the page's 99.73 % success rate — for
    every benchmark subcommand (the ROS runner pins the server's plan-time
    ``max_attempts`` to whichever budget the run uses, so native and ROS
    always share the same retry envelope). Lower it for a fast smoke pass.
    """
    from .compare import print_reference_comparison
    from .core_runner import run_core

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
    print everything again at the end (webpage order, both legs).

    Suites follow the cuRobo benchmarks page: motion generation (without
    torque limits, then with torque limits at the full ``--mass`` payload),
    inverse kinematics, and kinematics & collision. The native legs always
    run; the ROS legs run with ``--run-ros`` (the composed server must be
    up). Neither is ever skipped: each motion ROS leg first ensures the
    running server's torque mode via the runtime switch (stock
    ``set_parameters`` + ``update_motion_gen_config`` — a server's
    launch-time ``load_dynamics``/``robot_payload_mass`` are only the INITIAL
    state), so one server life fills BOTH motion ROS rows. IK and kinematics
    & collision use ``/ik_batch`` / ``/fk_batch`` and run either way.

    ``--max-attempts`` defaults to 100 (the page's real-solve budget, whose
    retry loop produces 99.73 % success) — lower it for a smoke pass. The
    same budget is pinned onto the running server's plan-time ``max_attempts``
    for both motion ROS legs (one ``set_parameters``, no rebuild), so native
    and ROS always share the same retry envelope. Result lists are dumped as
    ``<output>.<suite>-<leg>`` sidecar JSONs and printed again by
    ``compare.print_webpage_summary`` in the page's suite order.
    """
    from .compare import print_webpage_summary
    from .core_cost import run_cost_core
    from .core_ik import run_ik_core
    from .core_runner import run_core
    from .ros_cost import run_cost_ros
    from .ros_ik import run_ik_ros
    from .ros_runner import run_ros

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

    print("== [1/3] motion generation: pass 1/2 (without torque limits) ==",
          flush=True)
    motion_plain_native = run_core(**common, use_dynamics=False)
    motion_plain_ros = None
    if args.run_ros:
        # Neither ROS leg is ever skipped: the runner ensures the server's
        # torque mode itself (set_server_torque_mode) — plain for this leg,
        # torque-limited at --mass for the pass below. No --server-torque gate.
        print("   ROS leg (server torque mode off)", flush=True)
        motion_plain_ros = run_ros(
            dataset=args.dataset, scene=args.scene,
            use_dynamics=False, mass=args.mass, verbose=False,
            server_dynamics=False, server_payload_mass=0.0,
            server_max_attempts=args.max_attempts, **service,
        )

    print("== [1/3] motion generation: pass 2/2 (with torque limits) ==",
          flush=True)
    motion_torque_native = run_core(**common, use_dynamics=True)
    motion_torque_ros = None
    if args.run_ros:
        print("   ROS leg (server torque mode on, payload from --mass)",
              flush=True)
        motion_torque_ros = run_ros(
            dataset=args.dataset, scene=args.scene,
            use_dynamics=True, mass=args.mass, verbose=False,
            server_dynamics=True, server_payload_mass=args.mass,
            server_max_attempts=args.max_attempts, **service,
        )

    print("== [2/3] inverse kinematics ==", flush=True)
    ik_native = run_ik_core(num_seeds=args.num_seeds, **synthetic)
    ik_ros = run_ik_ros(**synthetic, **service) if args.run_ros else None

    print("== [3/3] kinematics & collision ==", flush=True)
    cost_native = run_cost_core(**synthetic)
    cost_ros = run_cost_ros(**synthetic, **service) if args.run_ros else None

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


def _resolve_tol(value: Optional[float], default: float) -> float:
    """Resolve capability-dependent tolerance defaults (None = capability default)."""
    return default if value is None else float(value)


def cmd_all(args) -> int:
    from .core_runner import run_core
    from .ros_runner import run_ros

    print("== core leg (native curobo_core) ==", flush=True)
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


# ---------------------------------------------------------------------------
# IK / kinematics-collision subcommands
# ---------------------------------------------------------------------------


def cmd_ik_core(args) -> int:
    from .core_ik import run_ik_core

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
    from .ros_ik import run_ik_ros

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
    from .core_cost import run_cost_core

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
    from .ros_cost import run_cost_ros

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
    from .core_ik import run_ik_core
    from .ros_ik import run_ik_ros

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
    from .core_cost import run_cost_core
    from .ros_cost import run_cost_ros

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


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="curobo_benchmark",
        description=(
            "Planner parity benchmark: curobo_core native vs the ROS-wrapped "
            "planner on the same robometrics problems."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_core = subparsers.add_parser("core", help="native curobo_core leg")
    _add_dataset(p_core)
    _add_scene(p_core)
    _add_solver_opts(p_core)
    p_core.add_argument("--warmup-iters", type=int, default=3)
    p_core.add_argument("--output", "-o", default=None)
    p_core.set_defaults(func=cmd_core)

    p_ros = subparsers.add_parser("ros", help="ROS-wrapped leg (server must be running)")
    _add_dataset(p_ros)
    _add_scene(p_ros)
    p_ros.add_argument("--service-timeout", type=float, default=30.0,
                       help="seconds to wait for the planner services")
    p_ros.add_argument("--call-timeout", type=float, default=120.0,
                       help="seconds per generate_trajectory call")
    p_ros.add_argument("--no-size-cache", action="store_true",
                       help="DIAGNOSTIC: leave the server's default collision "
                            "cache (cuboid=32, mesh=4 + voxel) in place instead "
                            "of sizing it to the dataset (the residual "
                            "padding inflates per-attempt solve vs native's "
                            "exact {obb: n_cubes} — A/B with the default "
                            "sized cache)")
    p_ros.add_argument("--use-dynamics", action="store_true",
                       help="record the run as the torque-limited variant "
                            "(server must be launched with the matching "
                            "load_dynamics robot config; Energy/Torque are "
                            "reconstructed client-side from the returned "
                            "trajectory)")
    p_ros.add_argument("--mass", type=float, default=3.0,
                       help="attached_object payload mass in kg (reference "
                            "default 3.0 = full payload) used client-side for "
                            "the Energy (J) / Torque (N·m) reconstruction")
    p_ros.add_argument("--output", "-o", default=None)
    p_ros.set_defaults(func=cmd_ros)

    p_cmp = subparsers.add_parser("compare", help="compare two result JSONs")
    p_cmp.add_argument("core_results")
    p_cmp.add_argument("ros_results")
    _add_compare_opts(p_cmp)
    p_cmp.add_argument(
        "--capability", default="planning",
        choices=["planning", "ik", "cost"],
        help="which parity family the two result JSONs belong to (default "
             "planning; `ik`/`cost` use their own fields and tolerances — "
             "see --position-tolerance-mm / --orientation-tolerance-deg)",
    )
    p_cmp.add_argument(
        "--position-tolerance-mm", type=float, default=None,
        help="IK: absolute FK position-error parity tolerance in mm (default "
             "0.5); cost: absolute FK pose tolerance in mm (default 1e-3)",
    )
    p_cmp.add_argument(
        "--orientation-tolerance-deg", type=float, default=None,
        help="IK: absolute orientation-error parity tolerance in deg (default "
             "0.5); cost: absolute quaternion parity tolerance in deg "
             "(default 1e-3)",
    )
    p_cmp.add_argument("--output", "-o", default=None,
                       help="write the report dict as JSON")
    p_cmp.set_defaults(func=cmd_compare)

    p_all = subparsers.add_parser("all", help="run core + ros + compare")
    _add_dataset(p_all)
    _add_scene(p_all)
    _add_solver_opts(p_all)
    p_all.add_argument("--warmup-iters", type=int, default=3)
    p_all.add_argument("--service-timeout", type=float, default=30.0)
    p_all.add_argument("--call-timeout", type=float, default=120.0)
    p_all.add_argument("--no-size-cache", action="store_true",
                       help="DIAGNOSTIC: leave the server's default collision "
                            "cache in place (see `ros --no-size-cache`)")
    _add_compare_opts(p_all)
    p_all.add_argument("--output", "-o", default=None)
    p_all.set_defaults(func=cmd_all)

    p_reference = subparsers.add_parser(
        "reference",
        help="reproduce BOTH reference-page tables (with/without torque "
             "limits) and print them again for comparison",
    )
    _add_dataset(p_reference, default="full")
    _add_scene(p_reference)
    # max_attempts defaults to 100 (the upstream real-solve budget, whose
    # retry loop produces the page's 99.73 % success); --use-dynamics is
    # implicit — reference runs BOTH modes (mass feeds both passes' Energy and
    # Torque reconstruction).
    _add_solver_opts(p_reference, max_attempts_default=100, add_use_dynamics=False)
    p_reference.add_argument("--warmup-iters", type=int, default=3)
    p_reference.add_argument("--output", "-o", default=None)
    p_reference.set_defaults(func=cmd_reference)

    # --- the docker compose default: the page's suites in order, both legs ---
    p_webpage = subparsers.add_parser(
        "webpage",
        help="run the page's suites in order (motion gen -> IK -> cost), once "
             "native / once ROS, then print all results again",
    )
    _add_dataset(p_webpage, default="full")
    _add_scene(p_webpage)
    # both torque modes are implicit (like `reference`); max_attempts defaults
    # to the page's 100-attempt real-solve budget.
    _add_solver_opts(p_webpage, max_attempts_default=100, add_use_dynamics=False)
    p_webpage.add_argument("--warmup-iters", type=int, default=3)
    p_webpage.add_argument("--run-ros", action="store_true",
                           help="also run the ROS legs (composed server must be up)")
    p_webpage.add_argument("--server-torque", action="store_true",
                           help="DEPRECATED — accepted for compatibility and "
                                "ignored: the webpage run now switches the "
                                "running server's torque-limited mode itself "
                                "(set_parameters + update_motion_gen_config), "
                                "so both motion ROS rows always run",
                           )
    # ik / kinematics & collision knobs (--output is declared below — the
    # shared _add_ik_cost_opts would conflict on the option name)
    from .synthetic import DEFAULT_ROBOT_CONFIG

    p_webpage.add_argument("--batch", type=int, default=100)
    p_webpage.add_argument("--num-batches", type=int, default=5)
    p_webpage.add_argument("--num-seeds", type=int, default=32)
    p_webpage.add_argument("--seed", type=int, default=2)
    p_webpage.add_argument("--robot-config", type=str, default=DEFAULT_ROBOT_CONFIG)
    p_webpage.add_argument("--service-timeout", type=float, default=30.0)
    p_webpage.add_argument("--call-timeout", type=float, default=120.0)
    p_webpage.add_argument("--no-size-cache", action="store_true")
    p_webpage.add_argument("--output", "-o", default=None)
    p_webpage.set_defaults(func=cmd_webpage)

    # --- IK benchmark (native IK vs the server's /ik_batch) ---
    p_ik_core = subparsers.add_parser(
        "ik-core", help="native IK leg (cfree + plain rows)"
    )
    _add_ik_cost_opts(p_ik_core, include_num_seeds=True)
    p_ik_core.add_argument(
        "--variants", default="both", choices=["both", "cfree", "plain"],
        help="which IK variants to solve natively (default: both)",
    )
    p_ik_core.set_defaults(func=cmd_ik_core)

    p_ik_ros = subparsers.add_parser(
        "ik-ros", help="ROS IK leg via /ik_batch (server must be running)"
    )
    _add_ik_cost_opts(p_ik_ros)
    _add_ros_service_opts(p_ik_ros)
    p_ik_ros.set_defaults(func=cmd_ik_ros)

    p_ik = subparsers.add_parser(
        "ik", help="ik-core + ik-ros + compare (parity verdict = exit code)"
    )
    _add_ik_cost_opts(p_ik, include_num_seeds=True)
    p_ik.add_argument(
        "--variants", default="both", choices=["both", "cfree", "plain"],
        help="which IK variants to solve natively (default: both)",
    )
    _add_ros_service_opts(p_ik)
    _add_pose_compare_opts(p_ik)
    p_ik.set_defaults(func=cmd_ik)

    # --- kinematics & collision benchmark (native FK/validate vs /fk_batch) ---
    p_cost_core = subparsers.add_parser(
        "cost-core", help="native FK + collision validity leg"
    )
    _add_ik_cost_opts(p_cost_core)
    p_cost_core.set_defaults(func=cmd_cost_core)

    p_cost_ros = subparsers.add_parser(
        "cost-ros",
        help="ROS FK/collision leg via /fk_batch (server must be running)",
    )
    _add_ik_cost_opts(p_cost_ros)
    _add_ros_service_opts(p_cost_ros)
    p_cost_ros.set_defaults(func=cmd_cost_ros)

    p_cost = subparsers.add_parser(
        "cost",
        help="cost-core + cost-ros + compare (parity verdict = exit code)",
    )
    _add_ik_cost_opts(p_cost)
    _add_ros_service_opts(p_cost)
    _add_pose_compare_opts(p_cost)
    p_cost.set_defaults(func=cmd_cost)

    return parser


def _add_dataset(parser, default: str = "demo") -> None:
    parser.add_argument(
        "--dataset", default=default,
        choices=["demo", "motion_benchmaker", "mpinets", "full"],
        help="robometrics dataset: demo, motion_benchmaker, mpinets, or full "
             "(motion_benchmaker + mpinets combined — the 2600 problems the "
             "reference page aggregates)",
    )


def _add_scene(parser) -> None:
    parser.add_argument(
        "--scene", default=None,
        help="restrict the run to one scene key within the dataset "
             "(run without it to list the dataset's scenes first)",
    )


def _add_solver_opts(
    parser, max_attempts_default: int = 100, add_use_dynamics: bool = True,
) -> None:
    parser.add_argument("--num-ik-seeds", type=int, default=32,
                        help="IK seeds (curobo reference benchmark default: 32)")
    parser.add_argument("--num-trajopt-seeds", type=int, default=4,
                        help="trajopt seeds (reference default: 4)")
    max_attempts_help = (
        "plan_pose retry budget per problem (default 100 — the page's "
        "real-solve budget, for the WHOLE benchmark: its retry loop produces "
        "the 99.73 % success rate, and the ROS runner pins the server's "
        "plan-time max_attempts to this value so native and ROS always share "
        "the same envelope; lower it for a fast smoke pass)"
    )
    parser.add_argument("--max-attempts", type=int, default=max_attempts_default,
                        help=max_attempts_help)
    parser.add_argument("--mesh", action="store_true",
                        help="convert obstacles to meshes instead of OBBs "
                             "(reference --mesh; default: OBB worlds)")
    if add_use_dynamics:
        parser.add_argument("--use-dynamics", action="store_true",
                            help="solver torque-limited mode (sets the reference "
                                 "robot_cfg['load_dynamics'] and applies the "
                                 "--mass payload — the page's 'with torque "
                                 "limits' table). The Energy (J) / Torque (N·m) "
                                 "columns are computed in both modes at --mass.")
    parser.add_argument("--mass", type=float, default=3.0,
                        help="attached_object payload mass in kg (reference "
                             "default 3.0 = full payload) for both the solver "
                             "payload (with --use-dynamics) and the Pinocchio "
                             "energy/torque model")
    parser.add_argument("--no-cuda-graph", action="store_true",
                        help="disable CUDA graphs (debug only)")
    parser.add_argument("--no-reset-seed", action="store_true",
                        help="DIAGNOSTIC: skip mg.reset_seed() before each "
                             "solve (native leg mimics the ROS server's "
                             "drifting RNG; see README 'timing attribution')")
    parser.add_argument("--unseeded", action="store_true",
                        help="DIAGNOSTIC: skip the upstream fixed seeds AND "
                             "per-problem reset_seed() — full server-like RNG "
                             "drift (implies --no-reset-seed)")


def _add_compare_opts(parser) -> None:
    parser.add_argument("--path-tolerance", type=float, default=0.02,
                        help="relative path-length parity tolerance (default 0.02)")
    parser.add_argument("--motion-tolerance", type=float, default=0.02,
                        help="relative motion-time parity tolerance (default 0.02)")
    parser.add_argument("--show-all", action="store_true",
                        help="print the per-problem delta table for every problem")


def _add_ik_cost_opts(parser, include_num_seeds: bool = False) -> None:
    """Shared synthetic-input knobs for the ik/cost legs (both run the same
    seeded input so the two legs' JSONs line up 1:1 for comparison)."""
    from .synthetic import DEFAULT_ROBOT_CONFIG

    parser.add_argument("--batch", type=int, default=100,
                        help="goals/configs per batch (default 100)")
    parser.add_argument("--num-batches", type=int, default=5,
                        help="number of batches (default 5)")
    parser.add_argument("--seed", type=int, default=2,
                        help="input-generation seed, shared by both legs "
                             "(default 2)")
    parser.add_argument("--robot-config", type=str, default=DEFAULT_ROBOT_CONFIG,
                        help="robot YAML the ROS server is launched with "
                             "(default: the benchmark envelope's "
                             "franka.curobo.reference.yml)")
    if include_num_seeds:
        parser.add_argument("--num-seeds", type=int, default=32,
                            help="IK seed restarts (default 32 = the server's "
                                 "num_ik_seeds node param)")
    parser.add_argument("--output", "-o", default=None)


def _add_ros_service_opts(parser) -> None:
    parser.add_argument("--service-timeout", type=float, default=30.0,
                        help="seconds to wait for the planner services")
    parser.add_argument("--call-timeout", type=float, default=120.0,
                        help="seconds per service call (warmup rebuilds the "
                             "solver collision cache, so keep this > ~60 s)")
    parser.add_argument("--no-size-cache", action="store_true",
                        help="DIAGNOSTIC: leave the server's default collision "
                             "cache (cuboid=32, mesh=4 + voxel) in place "
                             "instead of sizing it to the synthetic world "
                             "(padded kernel grids — A/B diagnostic)")


def _add_pose_compare_opts(parser) -> None:
    parser.add_argument("--position-tolerance-mm", type=float, default=None,
                        help="IK: FK position-error parity tolerance in mm "
                             "(default 0.5); cost: FK pose tolerance in mm "
                             "(default 1e-3)")
    parser.add_argument("--orientation-tolerance-deg", type=float, default=None,
                        help="IK: FK orientation-error parity tolerance in deg "
                             "(default 0.5); cost: quaternion parity tolerance "
                             "in deg (default 1e-3)")
    parser.add_argument("--show-all", action="store_true",
                        help="print the per-goal delta table for every goal")


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


def main(argv: Optional[List[str]] = None) -> int:
    # Docker logs attach a pipe: make stdout line-buffered so progress and the
    # result tables show up live instead of being block-buffered.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
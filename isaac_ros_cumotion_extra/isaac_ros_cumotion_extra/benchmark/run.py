# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI for the planner parity benchmark.

Subcommands:

    curobo_benchmark core [options]      native curobo_core leg
    curobo_benchmark ros  [options]      ROS-wrapped leg (server must be up)
    curobo_benchmark compare C R [opts]  compare two result JSONs
    curobo_benchmark all [options]       core + ros + compare in one go

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

from .compare import compare, print_report, save_report


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


def _print_run_summary(results, label: str) -> None:
    n_ok = sum(1 for r in results if r["success"])
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
    )
    if args.output:
        _dump(results, args.output)
    _print_run_summary(results, "ros")
    return 0


def cmd_compare(args) -> int:
    with open(args.core_results) as f:
        core_results = json.load(f)
    with open(args.ros_results) as f:
        ros_results = json.load(f)

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
    p_ros.add_argument("--output", "-o", default=None)
    p_ros.set_defaults(func=cmd_ros)

    p_cmp = subparsers.add_parser("compare", help="compare two result JSONs")
    p_cmp.add_argument("core_results")
    p_cmp.add_argument("ros_results")
    _add_compare_opts(p_cmp)
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
    _add_compare_opts(p_all)
    p_all.add_argument("--output", "-o", default=None)
    p_all.set_defaults(func=cmd_all)

    return parser


def _add_dataset(parser) -> None:
    parser.add_argument(
        "--dataset", default="demo",
        choices=["demo", "motion_benchmaker", "mpinets"],
    )


def _add_scene(parser) -> None:
    parser.add_argument(
        "--scene", default=None,
        help="restrict the run to one scene key within the dataset "
             "(run without it to list the dataset's scenes first)",
    )


def _add_solver_opts(parser) -> None:
    parser.add_argument("--num-ik-seeds", type=int, default=32,
                        help="IK seeds (curobo reference benchmark default: 32)")
    parser.add_argument("--num-trajopt-seeds", type=int, default=4,
                        help="trajopt seeds (reference default: 4)")
    parser.add_argument("--max-attempts", type=int, default=100,
                        help="plan_pose retry budget (reference default: 100)")
    parser.add_argument("--mesh", action="store_true",
                        help="convert obstacles to meshes instead of OBBs "
                             "(reference --mesh; default: OBB worlds)")
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
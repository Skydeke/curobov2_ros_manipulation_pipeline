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

from .motion import cmd_all, cmd_compare, cmd_core, cmd_reference, cmd_ros, cmd_webpage
from .ik_cost import cmd_cost, cmd_cost_core, cmd_cost_ros, cmd_ik, cmd_ik_core, cmd_ik_ros

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="curobo_benchmark",
        description=(
            "Planner parity benchmark: curobo native vs the ROS-wrapped "
            "planner on the same robometrics problems."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_core = subparsers.add_parser("core", help="native curobo leg")
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
    _add_max_attempts_opt(p_ros)
    _add_reset_seed_opt(p_ros)
    _add_motion_lock_opt(p_ros)
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
    _add_motion_lock_opt(p_all)
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
    from ..synthetic import DEFAULT_ROBOT_CONFIG

    p_webpage.add_argument("--batch", type=int, default=100)
    p_webpage.add_argument("--num-batches", type=int, default=5)
    p_webpage.add_argument("--num-seeds", type=int, default=32)
    p_webpage.add_argument("--seed", type=int, default=2)
    p_webpage.add_argument("--robot-config", type=str, default=DEFAULT_ROBOT_CONFIG)
    p_webpage.add_argument("--service-timeout", type=float, default=30.0)
    p_webpage.add_argument("--call-timeout", type=float, default=120.0)
    p_webpage.add_argument("--no-size-cache", action="store_true",
                           help="DIAGNOSTIC: leave the server's default collision "
                                "cache in place (see `ros --no-size-cache`)")
    _add_motion_lock_opt(p_webpage)
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
    _add_max_attempts_opt(parser, max_attempts_default)
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
    _add_reset_seed_opt(parser)
    parser.add_argument("--unseeded", action="store_true",
                        help="DIAGNOSTIC: skip the upstream fixed seeds AND "
                             "per-problem reset_seed() — full server-like RNG "
                             "drift (implies --no-reset-seed)")
def _add_max_attempts_opt(parser, max_attempts_default: int = 100) -> None:
    """``--max-attempts``: the retry budget the ROS leg pins onto the server.

    Split out of ``_add_solver_opts`` for the same reason as
    ``_add_reset_seed_opt``: the ROS-only ``ros`` subcommand declares its own
    ``--use-dynamics``/``--mass`` help text, so it cannot call
    ``_add_solver_opts``, yet ``cmd_ros`` reads ``args.max_attempts`` to pin
    the server's plan-time budget.
    """
    max_attempts_help = (
        "plan_pose retry budget per problem (default 100 — the page's "
        "real-solve budget, for the WHOLE benchmark: its retry loop produces "
        "the 99.73 % success rate, and the ROS runner pins the server's "
        "plan-time max_attempts to this value so native and ROS always share "
        "the same envelope; lower it for a fast smoke pass)"
    )
    parser.add_argument("--max-attempts", type=int, default=max_attempts_default,
                        help=max_attempts_help)
def _add_reset_seed_opt(parser) -> None:
    """``--no-reset-seed``: A/B whether the solver RNG is rewound per solve.

    Declared separately from ``_add_solver_opts`` because the ROS-only
    ``ros`` subcommand has no native leg, yet still needs the flag: it pins
    the server's plan-time ``reset_seed_per_plan`` off for the A/B, which is
    the whole point of the knob when only the server reseeds.
    """
    parser.add_argument("--no-reset-seed", action="store_true",
                        help="DIAGNOSTIC: skip mg.reset_seed() before each "
                             "solve (native leg mimics the ROS server's "
                             "drifting RNG; see README 'timing attribution')")
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
    from ..synthetic import DEFAULT_ROBOT_CONFIG

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
def _add_motion_lock_opt(parser) -> None:
    """Gripper-lock parity knob (motion legs only: they are the only ones that
    solve trajectories through the server's locked-finger robot model)."""
    parser.add_argument(
        "--no-mpinets-locks", action="store_true",
        help="DIAGNOSTIC: solve every motion scene with the robot config's "
             "gripper lock (0.04) instead of switching the server to the "
             "per-dataset lock upstream's reference benchmark uses "
             "(mpinets=0.025) — A/B for the success-rate delta",
    )
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

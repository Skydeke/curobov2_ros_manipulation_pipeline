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

def _dump(results: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"Wrote {len(results)} results to {path}", flush=True)
def _fmt_dur(seconds: float) -> str:
    """Human duration: 45s, 12m34s, 2h05m."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    mins = int(seconds // 60)
    if mins < 60:
        return f"{mins}m{int(seconds % 60):02d}s"
    return f"{mins // 60}h{(mins % 60):02d}m"
def _print_wall_time(wall: Dict[str, float], total: float) -> None:
    """Print real (wall-clock) time per leg plus the run total."""
    print("\nWALL TIME (real time per leg):", flush=True)
    for label, seconds in wall.items():
        print(f"  {label}: {_fmt_dur(seconds)}", flush=True)
    print(f"  total: {_fmt_dur(total)}", flush=True)
def _sidecar(output: Optional[str], label: str) -> Optional[str]:
    """``report.json`` + ``core`` -> ``report.core.json`` ('' output -> None)."""
    if not output:
        return None
    root, ext = os.path.splitext(output)
    return f"{root}.{label}{ext or '.json'}"
def _print_run_summary(results, label: str, key: str = "success") -> None:
    n_ok = sum(1 for r in results if r.get(key, False))
    print(f"{label}: {len(results)} problems, {n_ok} succeeded", flush=True)
def _seed_knobs(args) -> Dict[str, bool]:
    """Core-leg RNG diagnostic knobs from CLI args.

    ``--unseeded`` implies ``--no-reset-seed``: the server reseeds neither the
    global generators nor the per-solve solver RNG, so both flags together
    reproduce its drifting-RNG condition.

    ``--unseeded`` is read with ``getattr`` because it only exists where the
    fixed global seeds do, i.e. on the subcommands that run a native leg: the
    ROS-only ``ros`` subcommand skips the *server's* per-plan reseed
    (``--no-reset-seed``, which it does declare) but has no global seeding of
    its own to skip.
    """
    unseeded = getattr(args, "unseeded", False)
    reset_seed_per_problem = not (args.no_reset_seed or unseeded)
    seed_globals = not unseeded
    return {
        "seed_globals": seed_globals,
        "reset_seed_per_problem": reset_seed_per_problem,
    }
def _resolve_tol(value: Optional[float], default: float) -> float:
    """Resolve capability-dependent tolerance defaults (None = capability default)."""
    return default if value is None else float(value)

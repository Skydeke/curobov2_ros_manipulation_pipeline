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

from .cli import build_parser

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

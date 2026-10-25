# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module entry point: ``python3 -m curobov2_ros_extra.benchmark.run``.

The CLI lives in :mod:`.app`; this shim is what makes the package directly
executable (docker/compose_benchmark.yaml and the README both invoke it that
way), mirroring the old single-module ``benchmark/run.py`` behaviour.
"""

import sys

from .app import main

if __name__ == "__main__":
    sys.exit(main())

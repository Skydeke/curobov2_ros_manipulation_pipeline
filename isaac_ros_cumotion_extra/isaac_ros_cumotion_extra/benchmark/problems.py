# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load the robometrics benchmark problems.

Reuses the exact datasets ``curobo_core/curobo/benchmark/motion_plan_benchmark.py``
benchmarks against (``demo``, ``motion_benchmaker``, ``mpinets``), so the two
legs here solve the same problems as the upstream native benchmark.

``robometrics`` ships with the ``curobo[benchmark]`` pip extra (present in the
docker image). The import is lazy and guarded so this module stays importable
in pure-Python environments where robometrics is not installed.

Each problem dict carries:
    start                 -> 7-D pose-free joint configuration (cspace order)
    goal_pose             -> {'position_xyz': [x, y, z],
                              'quaternion_wxyz': [w, x, y, z]}
    obstacles             -> SceneCfg-style {bucket: {name: params}, ...}
    collision_buffer_ik   -> < 0 marks problems the upstream benchmark skips
"""

# Standard Library
from typing import Any, Dict, List, Optional

# Datasets usable via `load_problems` (keys of robometrics' raw loaders).
DATASET_NAMES = ("demo", "motion_benchmaker", "mpinets")

_LOADERS: Dict[str, Any] = {}


def _get_loaders() -> Dict[str, Any]:
    """Import robometrics loaders once (guarded for non-benchmark envs)."""
    global _LOADERS
    if _LOADERS:
        return _LOADERS
    try:
        from robometrics.datasets import (
            demo_raw,
            motion_benchmaker_raw,
            mpinets_raw,
        )
    except ImportError as exc:  # pragma: no cover - box-only dependency
        raise ImportError(
            "robometrics is not installed. It is part of the curobo[benchmark] "
            "pip extra; run this inside the docker image."
        ) from exc
    _LOADERS = {
        "demo": demo_raw,
        "motion_benchmaker": motion_benchmaker_raw,
        "mpinets": mpinets_raw,
    }
    return _LOADERS


def load_problems(dataset: str = "demo") -> Dict[str, List[Dict[str, Any]]]:
    """Load ``{scene_key: [problem, ...]}`` for a benchmark dataset."""
    if dataset not in DATASET_NAMES:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Choose from: {list(DATASET_NAMES)}"
        )
    return _get_loaders()[dataset]()


def filter_scenes(
    problems: Dict[str, List[Dict[str, Any]]],
    scene: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Restrict a ``{scene_key: [problem, ...]}`` dict to one scene.

    ``None`` returns the dict unchanged. An unknown scene key raises
    ``ValueError`` listing the available scenes so a typo fails fast instead
    of silently running the whole dataset on both legs.
    """
    if not scene:
        return problems
    if scene not in problems:
        raise ValueError(
            f"Scene {scene!r} not found in the loaded problems. "
            f"Available scenes: {', '.join(sorted(problems))}"
        )
    return {scene: problems[scene]}
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
from typing import Any, Dict, List, Optional, Tuple

# Datasets usable via `load_problems` (keys of robometrics' raw loaders).
# "full" is the combined motion_benchmaker + mpinets dataset that the
# reference page's tables aggregate (2600 problems), replayed the way the
# upstream script does — as two separate file_paths, each with its own
# mpinets classification.
DATASET_NAMES = ("demo", "motion_benchmaker", "mpinets", "full")

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


def _combined_loader() -> Dict[str, List[Dict[str, Any]]]:
    """Merge motion_benchmaker + mpinets into one ``{scene: [problems]}`` dict.

    Upstream iterates ``file_paths = [motion_benchmaker_raw, mpinets_raw]``
    (the two halves of the page's 2600-problem tables) in one run, so the
    combined dataset is just the union, keyed by scene. Scene keys between the
    two robometrics datasets are disjoint — a collision would mean the merge
    silently hid problems, so it fails fast instead. The merged dict (and the
    mpinets provenance) is cached: the box's 2600-problem pass otherwise
    re-loads both datasets from robometrics on every access.
    """
    global _FULL_CACHE, _MPINETS_SCENE_KEYS_CACHE
    if _FULL_CACHE is not None:
        return _FULL_CACHE
    loaders = _get_loaders()
    loaded: Dict[str, List[Dict[str, Any]]] = {}
    mpinets_seen: List[str] = []
    for name in ("motion_benchmaker", "mpinets"):
        for scene_key, scene_problems in loaders[name]().items():
            if scene_key in loaded:
                raise ValueError(
                    f"Scene {scene_key!r} appears in both motion_benchmaker and "
                    "mpinets; cannot merge the 'full' dataset."
                )
            loaded[scene_key] = scene_problems
            if name == "mpinets":
                mpinets_seen.append(scene_key)
    _FULL_CACHE = loaded
    if _MPINETS_SCENE_KEYS_CACHE is None:
        _MPINETS_SCENE_KEYS_CACHE = frozenset(mpinets_seen)
    return loaded


_FULL_CACHE: Optional[Dict[str, List[Dict[str, Any]]]] = None
_MPINETS_SCENE_KEYS_CACHE: Optional[frozenset] = None


def mpinets_scene_keys() -> frozenset:
    """Scene keys belonging to the mpinets dataset (for the combined "full"
    dataset, where upstream's per-file ``mpinets_data`` flag must classify
    each scene individually — motion_benchmaker problems must not gain the
    mpinets finger-lock robot config). Cached; loading the full dataset
    populates it as a side effect."""
    global _MPINETS_SCENE_KEYS_CACHE
    if _MPINETS_SCENE_KEYS_CACHE is None:
        _MPINETS_SCENE_KEYS_CACHE = frozenset(
            _get_loaders()["mpinets"]().keys()
        )
    return _MPINETS_SCENE_KEYS_CACHE


def load_problems(dataset: str = "demo") -> Dict[str, List[Dict[str, Any]]]:
    """Load ``{scene_key: [problem, ...]}`` for a benchmark dataset."""
    if dataset not in DATASET_NAMES:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Choose from: {list(DATASET_NAMES)}"
        )
    if dataset == "full":
        return _combined_loader()
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


_CONVERTED_BUCKETS = ("sphere", "capsule", "cylinder")


def collision_cache_sizes(
    problems: Dict[str, List[Dict[str, Any]]],
) -> Tuple[int, int]:
    """Solver collision-cache sizes (cuboid slots, mesh slots) a dataset needs.

    curobo's Warp collision kernels launch one thread per (robot sphere, padded
    obstacle slot) per obstacle *type*, so an over-padded cache — the server's
    former deployment default ``{cuboid: 100, mesh: 100, voxel: ...}`` (the
    launch defaults are now 32/4 via ``collision_cache_cuboid`` /
    ``collision_cache_mesh``) — makes every solver iteration run
    a much larger kernel grid than the native leg's ``{obb: n_cubes}`` cache —
    the residual ~7x single-attempt solve gap after the
    ``obstacle_collision_mode:=cuboid`` fix (see the extra README "timing
    attribution" section). The ROS benchmark leg sizes the server's cache to
    these values (and disables the empty no-camera voxel layer) before the
    timed run so both legs collide through native-equivalent kernel grids.

    The semantics mirror the native leg's ``check_problems``
    (``curobo/benchmark/motion_plan_benchmark.py``): per problem it builds
    ``SceneCfg.create(obstacles).get_obb_world()`` and reads the resulting OBB
    cache count, so cuboids *plus* converted sphere/cylinder/capsule all
    occupy the cuboid bucket. The server can run either
    ``obstacle_collision_mode``, routing those prims to the cuboid bucket
    (cuboid, default) or the mesh bucket (mesh), so the cuboid budget is
    ``cuboid + converted`` and the mesh budget is ``mesh + converted`` — exact
    for the default mode and safe in mesh mode.

    Returns:
        ``(cuboid_slots, mesh_slots)`` — the maximum per-scene count the
        solver must hold per type across the dataset (cuboid_slots >= 1).
    """
    cuboid_slots = 0
    mesh_slots = 0
    for scene_problems in problems.values():
        scene_cuboid = scene_mesh = scene_converted = 0
        for problem in scene_problems:
            obstacles = problem.get("obstacles") or {}
            scene_cuboid = max(scene_cuboid, len(obstacles.get("cuboid") or {}))
            scene_mesh = max(scene_mesh, len(obstacles.get("mesh") or {}))
            scene_converted = max(
                scene_converted,
                sum(
                    len(obstacles.get(bucket) or {})
                    for bucket in _CONVERTED_BUCKETS
                ),
            )
        cuboid_slots = max(cuboid_slots, scene_cuboid + scene_converted)
        mesh_slots = max(mesh_slots, scene_mesh + scene_converted)
    return max(cuboid_slots, 1), mesh_slots
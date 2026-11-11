# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The RViz collision-sphere model must be the exact planner model.

Needs a GPU; skips otherwise.
"""

import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import curobo_python_runtime  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FRANKA_CONFIG = os.path.join(REPO, "curobov2_ros", "config", "franka.curobo.yml")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU")


def _bridge():
    handle = curobo_python_runtime.init({})
    curobo_python_runtime.robot_model_build(
        handle, json.dumps({"robot_config_file": FRANKA_CONFIG}))
    return handle


def test_collision_spheres_report_exact_pair_diagnostics():
    """Recompute every configured pair and verify the per-sphere worst value."""
    handle = _bridge()
    q = np.zeros(7, dtype=np.float32)
    result = curobo_python_runtime.collision_spheres_with_attached(handle, q)

    spheres = np.asarray(result["spheres"], dtype=np.float32)
    padding = np.asarray(result["self_padding"], dtype=np.float32).reshape(-1)
    self_distance = np.asarray(result["self_distance"], dtype=np.float32).reshape(-1)
    world_distance = np.asarray(result["world_distance"], dtype=np.float32).reshape(-1)
    assert spheres.shape == (len(self_distance), 4)
    assert padding.shape == self_distance.shape
    assert world_distance.shape == self_distance.shape
    assert len(result["link_names"]) == len(self_distance)
    assert len(result["attached"]) == len(self_distance)

    kin = curobo_python_runtime._REGISTRY[handle]["kin"]
    sc = kin.config.self_collision_config
    pairs = sc.collision_pairs.detach().cpu().numpy().astype(np.int64)
    expected = np.full(len(self_distance), -np.inf, dtype=np.float32)
    radius = spheres[:, 3]
    for a, b in pairs:
        if radius[a] < 0.0 or radius[b] < 0.0:
            continue
        ra = radius[a] + padding[a]
        rb = radius[b] + padding[b]
        centre = np.linalg.norm(spheres[a, :3] - spheres[b, :3])
        penetration = ra + rb - centre
        expected[a] = max(expected[a], penetration)
        expected[b] = max(expected[b], penetration)

    finite = np.isfinite(expected)
    np.testing.assert_allclose(
        self_distance[finite], expected[finite], atol=1e-5)
    assert np.all(np.isneginf(self_distance[~finite]))
    curobo_python_runtime.shutdown(handle)

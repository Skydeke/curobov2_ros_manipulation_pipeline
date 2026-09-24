# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the robometrics problem loading (pure-Python registry part)."""

import pytest

from isaac_ros_cumotion_extra.benchmark.problems import (
    DATASET_NAMES,
    filter_scenes,
    load_problems,
)


class TestDatasetRegistry:
    def test_known_datasets(self):
        assert set(DATASET_NAMES) == {'demo', 'motion_benchmaker', 'mpinets'}

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError):
            load_problems('not_a_dataset')


class TestFilterScenes:
    def test_none_returns_unchanged(self):
        problems = {'a': [1], 'b': [2]}
        assert filter_scenes(problems, None) is problems

    def test_restricts_to_one_scene(self):
        problems = {'a': [1], 'b': [2]}
        assert filter_scenes(problems, 'b') == {'b': [2]}

    def test_unknown_scene_raises_with_listing(self):
        problems = {'a': [1], 'b': [2]}
        with pytest.raises(ValueError, match="Available scenes: a, b"):
            filter_scenes(problems, 'nope')


class TestProblemShape:
    """Structure of each problem dict (skipped when robometrics is absent)."""

    @pytest.fixture(autouse=True)
    def _robometrics(self):
        pytest.importorskip('robometrics')
        yield

    def test_loads_scene_dict(self):
        problems = load_problems('demo')
        assert isinstance(problems, dict)
        assert len(problems) > 0

    def test_problem_fields(self):
        problems = load_problems('demo')
        for scene_key, scene_problems in problems.items():
            assert isinstance(scene_problems, list)
            problem = scene_problems[0]
            assert 'start' in problem
            assert 'goal_pose' in problem
            assert 'obstacles' in problem
            assert isinstance(problem['start'], list)
            gp = problem['goal_pose']
            assert 'position_xyz' in gp
            assert 'quaternion_wxyz' in gp
            assert len(gp['position_xyz']) == 3
            assert len(gp['quaternion_wxyz']) == 4
            assert isinstance(problem['obstacles'], dict)
            break
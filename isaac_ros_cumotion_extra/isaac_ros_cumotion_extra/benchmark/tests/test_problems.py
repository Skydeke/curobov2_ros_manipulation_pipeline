# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the robometrics problem loading (pure-Python registry part)."""

import pytest

from isaac_ros_cumotion_extra.benchmark.problems import (
    DATASET_NAMES,
    collision_cache_sizes,
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


class TestCollisionCacheSizes:
    """Solver collision-cache sizing (pure function, no robometrics needed)."""

    def _problem(self, obstacles):
        return [{"start": [], "goal_pose": {}, "obstacles": obstacles}]

    def test_empty_dataset(self):
        assert collision_cache_sizes({}) == (1, 0)
        assert collision_cache_sizes({"a": []}) == (1, 0)

    def test_cuboids_only(self):
        problems = {
            "s1": self._problem({"cuboid": {"c1": {}, "c2": {}}}),
            "s2": self._problem({"cuboid": {"c1": {}}}),
        }
        assert collision_cache_sizes(problems) == (2, 0)

    def test_converted_prims_count_as_cuboids(self):
        # cuboid mode (default) routes sphere/cylinder/capsule to the cuboid
        # bucket via get_cuboid(); mesh mode routes them to the mesh bucket,
        # so the sizing must include them in BOTH buckets.
        problems = {
            "s1": self._problem(
                {
                    "cuboid": {"c1": {}},
                    "sphere": {"s1": {}},
                    "capsule": {"cap1": {}},
                    "cylinder": {"cyl1": {}},
                }
            )
        }
        assert collision_cache_sizes(problems) == (4, 3)

    def test_max_across_scenes_and_problems(self):
        problems = {
            "s1": self._problem({"cuboid": {"c1": {}, "c2": {}}})
            + self._problem({"cuboid": {"c1": {}}}),
            "s2": self._problem({"cuboid": {"c1": {}, "c2": {}, "c3": {}}}),
        }
        assert collision_cache_sizes(problems) == (3, 0)

    def test_meshes_stay_in_mesh_bucket(self):
        problems = {
            "s1": self._problem(
                {"cuboid": {"c1": {}}, "mesh": {"m1": {}, "m2": {}}}
            )
        }
        assert collision_cache_sizes(problems) == (1, 2)

    def test_meshes_and_converted(self):
        problems = {
            "s1": self._problem(
                {
                    "mesh": {"m1": {}},
                    "cylinder": {"cyl1": {}},
                    "capsule": {"cap1": {}},
                }
            )
        }
        assert collision_cache_sizes(problems) == (2, 3)


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
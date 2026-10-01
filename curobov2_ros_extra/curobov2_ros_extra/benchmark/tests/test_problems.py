# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the robometrics problem loading (pure-Python registry part)."""

import pytest

from curobov2_ros_extra.benchmark.problems import (
    DATASET_NAMES,
    collision_cache_sizes,
    filter_scenes,
    load_problems,
)


class TestDatasetRegistry:
    def test_known_datasets(self):
        assert set(DATASET_NAMES) == {
            'demo', 'motion_benchmaker', 'mpinets', 'full',
        }

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


class TestFullMergeRegistry:
    """The 'full' merge (motion_benchmaker + mpinets) — pure Python, no
    robometrics needed.

    Regression: ``_combined_loader`` assigned ``_MPINETS_SCENE_KEYS_CACHE``
    without a ``global`` declaration, so the module-level read raised
    ``UnboundLocalError`` on the FIRST call. That path was only covered by the
    robometrics-gated tests (skipped locally, they ran first on the box) —
    this class pins it in the pure-Python suite.
    """

    @pytest.fixture
    def _fake(self, monkeypatch):
        from curobov2_ros_extra.benchmark import problems as problems_mod

        datasets = {
            'motion_benchmaker': {
                'bench_a': [
                    {
                        'start': [],
                        'goal_pose': {
                            'position_xyz': [0.0, 0.0, 0.5],
                            'quaternion_wxyz': [1.0, 0.0, 0.0, 0.0],
                        },
                    }
                ],
            },
            'mpinets': {
                'mp_a': [
                    {
                        'start': [],
                        'goal_pose': {
                            'position_xyz': [0.2, 0.2, 0.5],
                            'quaternion_wxyz': [1.0, 0.0, 0.0, 0.0],
                        },
                    }
                ],
            },
        }
        counts = {'n_calls': 0}

        def fake_loaders():
            counts['n_calls'] += 1
            return {
                'motion_benchmaker': lambda: datasets['motion_benchmaker'],
                'mpinets': lambda: datasets['mpinets'],
            }

        monkeypatch.setattr(problems_mod, '_FULL_CACHE', None)
        monkeypatch.setattr(problems_mod, '_MPINETS_SCENE_KEYS_CACHE', None)
        monkeypatch.setattr(problems_mod, '_get_loaders', fake_loaders)
        return problems_mod, datasets, counts

    def test_first_call_populates_both_caches(self, _fake):
        # exactly the path that crashed with UnboundLocalError on the box
        mod, _, _ = _fake
        full = load_problems('full')
        assert set(full) == {'bench_a', 'mp_a'}
        # the merge's side-effect cache is populated (was the local-scope read)
        assert mod.mpinets_scene_keys() == frozenset({'mp_a'})

    def test_second_call_served_from_cache(self, _fake):
        mod, _, counts = _fake
        first = load_problems('full')
        second = load_problems('full')
        assert second is first
        assert counts['n_calls'] == 1  # one loader pass serves both calls

    def test_shared_scene_key_fails_fast(self, _fake):
        mod, datasets, _ = _fake
        datasets['mpinets'] = {'bench_a': []}
        with pytest.raises(ValueError, match='appears in both'):
            load_problems('full')

    def test_preloaded_mpinets_cache_reused_by_merge(self, _fake):
        mod, _, _ = _fake
        # mpinets_scene_keys() loads and caches independently first...
        assert mod.mpinets_scene_keys() == frozenset({'mp_a'})
        # ...and the merge then reuses it (no reload, no classification drift)
        full = load_problems('full')
        assert set(full) == {'bench_a', 'mp_a'}
        assert mod.mpinets_scene_keys() == frozenset({'mp_a'})


class TestMpinetsClassification:
    """Gripper-lock classification — pure Python, no robometrics needed.

    The two lock values are collision *geometry* (how wide the gripper's
    spheres are), so a scene classified as mpinets at 0.04 instead of 0.025 can
    fail a plan the native leg solves. Both legs read this answer, so it is
    pinned here rather than re-derived per leg.
    """

    @pytest.fixture
    def _fake(self, monkeypatch):
        from curobov2_ros_extra.benchmark import problems as problems_mod

        datasets = {
            'mpinets': {'dresser_task_oriented': [], 'tabletop_task_oriented': []},
            'motion_benchmaker': {'bookshelf_tall_panda': []},
        }
        monkeypatch.setattr(
            problems_mod,
            '_get_loaders',
            lambda: {name: (lambda d=datasets[name]: d) for name in datasets},
        )
        return problems_mod

    def test_full_dataset_classifies_per_scene(self, _fake):
        from curobov2_ros_extra.benchmark.problems import (
            DEFAULT_LOCK,
            MPINETS_LOCK,
            is_mpinets_scene,
            mpinets_lock_for_scene,
        )

        mpinets_scenes = frozenset({'dresser_task_oriented', 'tabletop_task_oriented'})
        assert is_mpinets_scene('dresser_task_oriented', mpinets_scenes, 'full')
        assert not is_mpinets_scene('bookshelf_tall_panda', mpinets_scenes, 'full')
        assert (
            mpinets_lock_for_scene('dresser_task_oriented', mpinets_scenes, 'full')
            == MPINETS_LOCK
        )
        assert (
            mpinets_lock_for_scene('bookshelf_tall_panda', mpinets_scenes, 'full')
            == DEFAULT_LOCK
        )

    def test_single_dataset_probes_one_file(self, _fake):
        """A single-dataset run is one file_path upstream, so one probe of an
        mpinets-only scene decides for every scene (no scene set to thread)."""
        from curobov2_ros_extra.benchmark.problems import (
            DEFAULT_LOCK,
            MPINETS_LOCK,
            is_mpinets_scene,
            mpinets_lock_for_scene,
        )

        # mpinets dataset: the probe scene is present -> every scene is mpinets.
        assert is_mpinets_scene('tabletop_task_oriented', None, 'mpinets')
        assert (
            mpinets_lock_for_scene('tabletop_task_oriented', None, 'mpinets')
            == MPINETS_LOCK
        )
        # benchmaker: the probe scene is absent -> every scene keeps the default.
        assert not is_mpinets_scene('bookshelf_tall_panda', None, 'motion_benchmaker')
        assert (
            mpinets_lock_for_scene('bookshelf_tall_panda', None, 'motion_benchmaker')
            == DEFAULT_LOCK
        )

    def test_the_two_values_are_distinct_and_nonzero(self, _fake):
        """Guard the constants themselves: swapping or aliasing them is the
        bug this parity work exists to prevent."""
        from curobov2_ros_extra.benchmark.problems import (
            DEFAULT_LOCK,
            MPINETS_LOCK,
        )

        assert MPINETS_LOCK == 0.025
        assert DEFAULT_LOCK == 0.04
        assert MPINETS_LOCK != DEFAULT_LOCK

    def test_mpinets_gripper_is_narrower(self, _fake):
        """The direction of the difference is load-bearing: mpinets' gripper is
        the tighter one, so a wider lock can only ever lose reach."""
        from curobov2_ros_extra.benchmark.problems import (
            DEFAULT_LOCK,
            MPINETS_LOCK,
        )

        assert MPINETS_LOCK < DEFAULT_LOCK


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

    def test_loads_full_combines_benchmaker_and_mpinets(self):
        """'full' = motion_benchmaker + mpinets (the reference page's 2600
        problems); scene keys are disjoint and per-scene mpinets provenance
        stays recoverable so the combined run classifies each problem exactly
        like the reference script's per-file_path loop."""
        from curobov2_ros_extra.benchmark.problems import (
            mpinets_scene_keys,
        )

        full = load_problems('full')
        mpinets_scenes = mpinets_scene_keys()
        bench_only = load_problems('motion_benchmaker')
        # every mpinets scene survives the merge, and no benchmaker scene
        # collides with an mpinets scene (a collision would hide problems).
        assert mpinets_scenes <= set(full)
        assert bench_only.keys().isdisjoint(mpinets_scenes)
        # a known mpinets scene must classify as mpinets in the combined run.
        assert 'dresser_task_oriented' in mpinets_scenes
        assert any(s in mpinets_scenes for s in full)

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
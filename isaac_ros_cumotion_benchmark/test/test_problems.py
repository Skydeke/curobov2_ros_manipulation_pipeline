import pytest

from isaac_ros_cumotion_benchmark.problems import DATASETS, load_problems


class TestDatasetsRegistered:
    def test_has_expected_keys(self):
        assert set(DATASETS.keys()) == {'demo', 'motion_benchmaker', 'mpinets'}

    def test_all_loaders_are_callable(self):
        for name, loader in DATASETS.items():
            assert callable(loader), f'{name} loader is not callable'


class TestLoadProblems:
    def test_demo_returns_dict_of_lists(self):
        problems = load_problems('demo')
        assert isinstance(problems, dict)
        for key, group in problems.items():
            assert isinstance(key, str)
            assert isinstance(group, list), f'{key} value is not a list'
            for p in group:
                assert isinstance(p, dict)

    def test_every_problem_has_required_keys(self):
        problems = load_problems('demo')
        required = {'start', 'goal_pose', 'obstacles', 'collision_buffer_ik'}
        for scene_key, group in problems.items():
            for i, p in enumerate(group):
                missing = required - set(p.keys())
                assert not missing, (
                    f'{scene_key}[{i}] missing keys: {missing}'
                )

    def test_start_is_list_of_floats(self):
        problems = load_problems('demo')
        for scene_key, group in problems.items():
            for i, p in enumerate(group):
                assert isinstance(p['start'], (list, tuple))
                assert all(isinstance(v, (int, float)) for v in p['start'])

    def test_goal_pose_has_position_and_quaternion(self):
        problems = load_problems('demo')
        for scene_key, group in problems.items():
            for i, p in enumerate(group):
                g = p['goal_pose']
                assert 'position_xyz' in g
                assert 'quaternion_wxyz' in g
                assert len(g['position_xyz']) == 3
                assert len(g['quaternion_wxyz']) == 4

    def test_obstacles_is_dict(self):
        problems = load_problems('demo')
        for scene_key, group in problems.items():
            for i, p in enumerate(group):
                assert isinstance(p['obstacles'], dict)

    def test_collision_buffer_ik_is_numeric(self):
        problems = load_problems('demo')
        for scene_key, group in problems.items():
            for i, p in enumerate(group):
                assert isinstance(p['collision_buffer_ik'], (int, float))

    def test_collision_buffer_ik_negative_skips_problem(self):
        problems = load_problems('demo')
        n_total = 0
        n_skippable = 0
        for group in problems.values():
            for p in group:
                n_total += 1
                if p['collision_buffer_ik'] < 0.0:
                    n_skippable += 1
        assert n_skippable >= 0

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match='Unknown dataset'):
            load_problems('nonexistent')

    def test_motion_benchmaker_returns_dict(self):
        problems = load_problems('motion_benchmaker')
        assert isinstance(problems, dict)
        assert len(problems) > 0

    def test_mpinets_returns_dict(self):
        problems = load_problems('mpinets')
        assert isinstance(problems, dict)
        assert len(problems) > 0

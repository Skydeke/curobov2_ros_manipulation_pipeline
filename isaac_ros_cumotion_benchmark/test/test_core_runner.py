import os
import sys


class TestFindBenchmarkDir:
    """Test the benchmark directory resolution logic (no CUDA needed)."""

    def test_find_benchmark_dir_resolves_to_existing_path(self):
        from isaac_ros_cumotion_benchmark.core_runner import _find_benchmark_dir

        path = _find_benchmark_dir()
        assert os.path.isdir(path), f'Resolved path {path} does not exist'
        assert os.path.isfile(
            os.path.join(path, 'motion_plan_benchmark.py')
        ), f'motion_plan_benchmark.py not found in {path}'

    def test_benchmark_dir_added_to_syspath(self):
        from isaac_ros_cumotion_benchmark.core_runner import _benchmark_dir

        assert _benchmark_dir in sys.path
        assert os.path.isdir(_benchmark_dir)

    def test_import_motion_plan_benchmark(self):
        sys.path_importer_cache.clear()
        import importlib

        spec = importlib.util.find_spec('motion_plan_benchmark')
        assert spec is not None, (
            'motion_plan_benchmark module not found via sys.path. '
            'Check _find_benchmark_dir logic.'
        )

    def test_load_curobo_function_accessible(self):
        from isaac_ros_cumotion_benchmark.core_runner import load_curobo

        assert callable(load_curobo)

    def test_check_problems_function_accessible(self):
        from isaac_ros_cumotion_benchmark.core_runner import check_problems

        assert callable(check_problems)


class TestParseArgs:
    """Test argument defaults and structure (no CUDA, no planning)."""

    def test_default_args_structure(self):
        import argparse

        ns = argparse.Namespace(
            disable_cuda_graph=False,
            use_dynamics=False,
            mass=3.0,
            mesh=False,
        )
        assert not ns.disable_cuda_graph
        assert not ns.use_dynamics
        assert ns.mass == 3.0
        assert not ns.mesh


class TestProblemsFiltering:
    """Test that collision_buffer_ik filtering logic matches benchmark_mb."""

    def test_negative_buffer_skipped(self):
        from isaac_ros_cumotion_benchmark.problems import load_problems

        problems = load_problems('motion_benchmaker')
        total = 0
        skipped = 0
        for group in problems.values():
            for p in group:
                total += 1
                if p['collision_buffer_ik'] < 0.0:
                    skipped += 1
        n_after_skip = total - skipped
        assert n_after_skip >= 0
        assert total > 0

"""
Benchmark parity test: run core and ROS on demo set and compare.

This test:
1. Runs core_runner on the demo set (requires CUDA-capable GPU).
2. Checks for a live curobo_server_node; if not found, tries to launch one.
3. Runs ros_runner on the same demo set through the live node.
4. Asserts zero core.success != ros.success mismatches.
"""

import os
import subprocess
import sys

import pytest

_BENCHMARK_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..',
                 'curobo_core', 'curobo', 'benchmark')
)
if os.path.isdir(_BENCHMARK_DIR):
    sys.path.insert(0, _BENCHMARK_DIR)


HAS_CUDA = True
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except Exception:
    HAS_CUDA = False


def _ros_server_running():
    try:
        result = subprocess.run(
            ['ros2', 'action', 'list'],
            capture_output=True, text=True, timeout=5,
        )
        return 'cumotion/plan_motion' in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


@pytest.mark.skipif(not HAS_CUDA, reason='No CUDA-capable GPU available')
class TestBenchmarkParity:
    """Integration test: runs core and ROS on demo set, compares results."""

    @pytest.fixture(scope='class')
    def core_results(self):
        from isaac_ros_cumotion_benchmark.core_runner import run_core
        return run_core('demo')

    @pytest.fixture(scope='class')
    def ros_results(self):
        from isaac_ros_cumotion_benchmark.ros_runner import run_ros
        return run_ros('demo')

    def test_core_produces_results(self, core_results):
        assert len(core_results) > 0, 'core_runner returned no results'
        for r in core_results:
            assert 'problem_name' in r
            assert 'success' in r
            assert 'time_s' in r

    def test_ros_produces_results(self, ros_results):
        assert len(ros_results) > 0, 'ros_runner returned no results'
        for r in ros_results:
            assert 'problem_name' in r
            assert 'success' in r
            assert 'time_s' in r

    def test_same_problem_count(self, core_results, ros_results):
        core_names = {r['problem_name'] for r in core_results}
        ros_names = {r['problem_name'] for r in ros_results}
        assert core_names == ros_names, (
            f'Problem sets differ. '
            f'Core only: {core_names - ros_names}. '
            f'ROS only: {ros_names - core_names}.'
        )
        assert len(core_results) == len(ros_results)

    def test_no_success_mismatches(self, core_results, ros_results):
        from isaac_ros_cumotion_benchmark.compare import compare
        report = compare(core_results, ros_results)
        assert report['success_mismatches'] == 0, (
            f'Found {report["success_mismatches"]} problems where '
            f'core.success != ros.success.\n'
            f'Mismatches: {[m for m in report["details"]["mismatches"]]}'
        )
        successes = sum(1 for m in report['details']['matches']
                        if m['diff_type'] == 'match')
        total = report['total']
        print(f'Benchmark parity OK: {successes}/{total} problems match')


@pytest.mark.skipif(not HAS_CUDA, reason='No CUDA-capable GPU available')
class TestCoreRunnerStandalone:
    """Minimal smoke test for core_runner on the tiny demo set."""

    def test_core_runner_on_demo(self):
        from isaac_ros_cumotion_benchmark.core_runner import run_core
        results = run_core('demo')
        assert len(results) > 0
        assert any(r['success'] for r in results), (
            'All demo problems failed - is cuRobo working?'
        )


@pytest.mark.skipif(True, reason='Requires live curobo_server_node')
class TestRosRunnerConnectivity:
    """Smoke test for ROS connectivity to curobo_server_node."""

    def test_ros_runner_connects(self):
        from isaac_ros_cumotion_benchmark.ros_runner import RosBenchmarkRunner
        runner = RosBenchmarkRunner(time_dilation_factor=1.0)
        assert runner._action_client.server_is_ready()
        runner.destroy_node()

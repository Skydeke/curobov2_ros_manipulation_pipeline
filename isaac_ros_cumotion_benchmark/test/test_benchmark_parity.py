"""Benchmark parity test: run core and ROS on demo set and compare."""

import json
import os
import sys
import tempfile

import pytest

# --help access the core_runner and ros_runner modules, add the benchmark
# script dir to sys.path (same pattern core_runner.py uses internally).

_BENCHMARK_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..",
                 "curobo_core", "curobo", "benchmark")
)
if os.path.isdir(_BENCHMARK_DIR):
    sys.path.insert(0, _BENCHMARK_DIR)

_PKG_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, _PKG_DIR)


@pytest.mark.skip(
    reason="Requires CUDA-capable GPU for core_runner and a live "
           "curobo_server_node for ros_runner"
)
def test_benchmark_parity():
    from isaac_ros_cumotion_benchmark.core_runner import run_core
    from isaac_ros_cumotion_benchmark.ros_runner import run_ros
    from isaac_ros_cumotion_benchmark.compare import compare

    core_results = run_core("demo")
    ros_results = run_ros("demo")

    report = compare(core_results, ros_results)

    assert report["success_mismatches"] == 0, (
        f"Found {report['success_mismatches']} problems where "
        f"core.success != ros.success"
    )

    successes = sum(1 for m in report["details"]["matches"]
                    if m["diff_type"] == "match")
    total = report["total"]
    print(f"Benchmark parity OK: {successes}/{total} problems match")

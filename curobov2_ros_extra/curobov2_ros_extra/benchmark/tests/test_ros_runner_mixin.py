# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ROS-runner metric helpers (mixin composition, no server required).

Regression guard for the benchmark split: the world mixin's static helpers
must resolve ``_finite_or_none`` through their own class. They used to call
``RosBenchmarkRunner._finite_or_none``, which is not imported in the mixin
module and only failed at run time (NameError on the first successful plan).
"""

import pytest

pytest.importorskip('rclpy')

from types import SimpleNamespace

from curobov2_ros_extra.benchmark.ros_runner_world_mixin import (
    _RosRunnerWorldMixin,
)


def _result(**stats):
    return SimpleNamespace(stats=SimpleNamespace(**stats))


class TestResultMetricHelpers:
    def test_motion_time(self):
        assert _RosRunnerWorldMixin._result_motion_time(
            _result(motion_time_s=1.5)
        ) == 1.5

    def test_solver_time(self):
        assert _RosRunnerWorldMixin._result_solver_time(
            _result(solver_time_s=0.25)
        ) == 0.25

    def test_position_error(self):
        assert _RosRunnerWorldMixin._result_position_error(
            _result(position_error_mm=0.125)
        ) == 0.125

    def test_missing_and_non_finite_values_are_none(self):
        assert _RosRunnerWorldMixin._result_motion_time(_result()) is None
        assert _RosRunnerWorldMixin._result_solver_time(
            SimpleNamespace()
        ) is None
        assert _RosRunnerWorldMixin._result_position_error(
            _result(position_error_mm=float('nan'))
        ) is None
        assert _RosRunnerWorldMixin._result_motion_time(
            _result(motion_time_s=float('inf'))
        ) is None

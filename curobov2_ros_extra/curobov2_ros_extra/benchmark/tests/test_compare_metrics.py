# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the native-vs-ROS parity comparison logic (pure Python)."""

import math

import pytest

from curobov2_ros_extra.benchmark.compare import (
    _grid_table,
    _stat_str,
    compare,
    curobo_style_rows,
    position_error_mm,
    print_reference_comparison,
    print_report,
    print_webpage_summary,
    reference_stats,
    save_report,
    trajectory_jerk,
    trajectory_metrics,
    winner_position_error_mm,
    winner_solve_time,
)

class TestTrajectoryMetrics:
    def test_short_trajectory(self):
        n, path, motion = trajectory_metrics([[0.0, 0.0], [1.0, 0.0]], 0.5)
        assert n == 2
        assert math.isclose(path, 1.0)
        assert math.isclose(motion, 0.5)

    def test_path_length_sums_euclidean_distances(self):
        wp = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
        n, path, motion = trajectory_metrics(wp, 0.025)
        assert n == 3
        assert math.isclose(path, 2.0)  # 1.0 + 1.0
        assert math.isclose(motion, 0.025 * 2)

    def test_single_waypoint(self):
        n, path, motion = trajectory_metrics([[1.0, 2.0]], 0.025)
        assert n == 1
        assert path == 0.0
        assert motion == 0.0

    def test_accepts_lists_of_lists(self):
        n, path, _ = trajectory_metrics([[0, 0], [1, 0]], 0.1)
        assert n == 2
        assert math.isclose(path, 1.0)
class TestTrajectoryJerk:
    """Client-side max|jerk| estimate for the ROS leg (3rd finite difference)."""

    def test_cubic_position_has_constant_jerk(self):
        # q_k = k^3 -> third difference = 6 (dt = 1)
        wp = [[0.0], [1.0], [8.0], [27.0]]
        assert math.isclose(trajectory_jerk(wp, 1.0), 6.0)

    def test_max_abs_across_joints_and_samples(self):
        # joint0: 27 - 3*8 + 3*1 = 6 ; joint1: 30 - 3*(-10) + 3*10 = 90
        wp = [[0.0, 0.0], [1.0, 10.0], [8.0, -10.0], [27.0, 30.0]]
        assert math.isclose(trajectory_jerk(wp, 1.0), 90.0)

    def test_constant_velocity_zero_jerk(self):
        wp = [[0.0], [1.0], [2.0], [3.0], [4.0]]
        assert math.isclose(trajectory_jerk(wp, 1.0), 0.0)

    def test_too_short_returns_none(self):
        assert trajectory_jerk([[0.0], [1.0], [2.0]], 0.025) is None
        assert trajectory_jerk([[0.0], [1.0]], 0.025) is None
        assert trajectory_jerk([], 0.025) is None
class TestPositionErrorMm:
    """Goal-vs-achieved tool position Euclidean distance in mm (pure helper).

    The ROS report row is fed by ``winner_position_error_mm`` (the solver's
    own residual); ``position_error_mm`` remains available for ad-hoc FK
    checks and is tested here for the mm conversion on its own.
    """

    def test_distance_in_mm(self):
        # 10 cm along x -> 100 mm
        assert math.isclose(
            position_error_mm((0.0, 0.0, 0.0), (0.1, 0.0, 0.0)), 100.0
        )

    def test_identity_pose_zero_error(self):
        assert position_error_mm((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)) == 0.0

    def test_3d_distance(self):
        # 3-4-5 triangle: 0.03/0.04 m legs -> 0.05 m hypotenuse -> 50 mm
        assert math.isclose(
            position_error_mm((0.0, 0.0, 0.0), (0.03, 0.04, 0.0)), 50.0
        )

    def test_matches_native_field_semantics(self):
        # Mirrors the native leg's result.position_error * 1000: a 2 mm error
        # is stored as 2.0 in position_error_mm for both legs.
        assert math.isclose(
            position_error_mm((0.0, 0.0, 0.0), (0.0, 0.0, 0.002)), 2.0
        )

    def test_nonpositive_dt_returns_none(self):
        wp = [[0.0], [1.0], [8.0], [27.0]]
        assert trajectory_jerk(wp, 0.0) is None
        assert trajectory_jerk(wp, -1.0) is None

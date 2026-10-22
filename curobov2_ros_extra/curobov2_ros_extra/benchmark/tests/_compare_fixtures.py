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

def _result(name, success=True, time_s=0.1, n_waypoints=10,
            capability='planning', **extra):
    result = {
        'problem_name': name,
        'scene_key': 'test_scene',
        'capability': capability,
        'success': success,
        'time_s': time_s,
    }
    if n_waypoints is not None:
        result['n_waypoints'] = n_waypoints
    result.update(extra)
    return result
def _successful(name, path_length=1.0, motion_time_s=1.0, n_waypoints=10,
                time_s=0.1, **extra):
    return _result(
        name, success=True, time_s=time_s, n_waypoints=n_waypoints,
        path_length=path_length, motion_time_s=motion_time_s, **extra)

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

from ._compare_fixtures import _result, _successful

class TestCompare:
    def test_empty_inputs(self):
        report = compare([], [])
        assert report['total'] == 0
        assert report['matches'] == 0
        assert report['mismatches'] == 0
        assert report['verdict'] == 'PARITY-OK'

    def test_identical_results_all_match(self):
        core = [_successful('a'), _successful('b')]
        ros = [_successful('a'), _successful('b')]
        report = compare(core, ros)
        assert report['total'] == 2
        assert report['matches'] == 2
        assert report['mismatches'] == 0
        assert report['success_mismatches'] == 0
        assert report['verdict'] == 'PARITY-OK'

    def test_success_mismatch_detected(self):
        core = [_successful('a')]
        ros = [_result('a', success=False)]
        report = compare(core, ros)
        assert report['success_mismatches'] == 1
        assert report['mismatches'] == 1
        assert report['details']['mismatches'][0]['diff_type'] == 'success_mismatch'
        assert report['verdict'] == 'PARITY-DELTA'

    def test_both_fail_is_agreement(self):
        core = [_result('a', success=False)]
        ros = [_result('a', success=False)]
        report = compare(core, ros)
        assert report['matches'] == 1
        assert report['mismatches'] == 0

    def test_core_only_problem(self):
        core = [_successful('a'), _successful('b')]
        ros = [_successful('a')]
        report = compare(core, ros)
        assert report['core_only'] == ['b']
        # 'b' is a coverage gap, not a parity mismatch on a common problem.
        assert report['mismatches'] == 0
        assert report['verdict'] == 'PARITY-DELTA'

    def test_ros_only_problem(self):
        core = [_successful('a')]
        ros = [_successful('a'), _successful('b')]
        report = compare(core, ros)
        assert report['ros_only'] == ['b']
        assert report['mismatches'] == 0
        assert report['verdict'] == 'PARITY-DELTA'

    def test_path_length_mismatch_beyond_tolerance(self):
        core = [_successful('a', path_length=1.0)]
        ros = [_successful('a', path_length=1.3)]  # 30% delta
        report = compare(core, ros, path_tolerance=0.02)
        assert report['mismatches'] == 1
        assert report['details']['mismatches'][0]['diff_type'] == 'path_length'

    def test_path_length_within_tolerance(self):
        core = [_successful('a', path_length=1.0)]
        ros = [_successful('a', path_length=1.005)]
        report = compare(core, ros, path_tolerance=0.02)
        assert report['mismatches'] == 0

    def test_waypoint_mismatch(self):
        core = [_successful('a', n_waypoints=100)]
        ros = [_successful('a', n_waypoints=98)]
        report = compare(core, ros)
        assert report['mismatches'] == 1
        assert report['details']['mismatches'][0]['diff_type'] == 'waypoints'

    def test_wall_clock_time_is_informational_not_a_mismatch(self):
        core = [_successful('a', time_s=0.5), _successful('b', time_s=0.4)]
        ros = [_successful('a', time_s=2.0), _successful('b', time_s=1.8)]
        report = compare(core, ros)
        assert report['matches'] == 2
        assert report['mismatches'] == 0
        assert report['summary']['time_overhead_pct'] is not None
        assert report['summary']['time_overhead_pct'] > 100.0

    def test_summary_reports_solver_solve_times(self):
        """The report carries curobo's accumulated retry-loop solver times on
        both legs. On the box the ROS leg's solve_time tracks its wall (~2 s):
        the per-request churn/retry work is charged inside the solver timers,
        not left as RTT around the call."""
        core = [
            _successful('a', time_s=0.065, solve_time_s=0.055),
            _successful('b', time_s=0.066, solve_time_s=0.056),
        ]
        ros = [
            _successful('a', time_s=2.0, solve_time_s=1.96),
            _successful('b', time_s=1.98, solve_time_s=1.98),
        ]
        summary = compare(core, ros)['summary']
        assert math.isclose(summary['avg_solve_time_core'], 0.0555)
        assert math.isclose(summary['avg_solve_time_ros'], 1.97)
        # The solver's own timers claim ~100% of the ROS wall: the ~2 s is
        # solver-loop cost in the server environment, not serialization/RTT.
        assert summary['solve_tracks_wall_pct'] is not None
        assert summary['solve_tracks_wall_pct'] > 90.0
        assert summary['avg_solve_time_ros'] < summary['avg_time_ros']

    def test_report_dictionaries_are_json_serializable(self, tmp_path):
        core = [_successful('a'), _successful('b')]
        ros = [_successful('a'), _successful('b', path_length=0.9)]
        report = compare(core, ros)
        out = tmp_path / 'report.json'
        save_report(report, str(out))
        import json
        loaded = json.loads(out.read_text())
        assert loaded['total'] == 2
class TestWinnerSolveTime:
    """Solver-reported solve time of the winning considered row (ROS leg).

    The ROS server fills ``stats.considered`` only when the request sets
    ``log_considered_trajectories`` (a reporting-only flag, accepted on the
    classic planner); the winner's row carries the same ``result.solve_time``
    the native leg records.
    """

    def test_returns_winner_row_solve_time(self):
        rows = [
            {'problem': 0, 'segment': 0, 'goalset_candidate': 0, 'seed': 0,
             'solve_time': 0.12},
            {'problem': 0, 'segment': 0, 'goalset_candidate': 0, 'seed': 1,
             'solve_time': 0.08},
        ]
        assert winner_solve_time(rows, [0], [1]) == 0.08

    def test_accepts_attribute_rows(self):
        class Row:
            problem = 0
            segment = 0
            goalset_candidate = 0
            seed = 3
            solve_time = 0.055
        assert winner_solve_time([Row()], [0], [3]) == 0.055

    def test_no_match_returns_none(self):
        rows = [
            {'problem': 0, 'segment': 0, 'goalset_candidate': 0, 'seed': 0,
             'solve_time': 0.1},
        ]
        assert winner_solve_time(rows, [0], [7]) is None

    def test_ignores_other_problems_and_segments(self):
        rows = [
            {'problem': 1, 'segment': 0, 'goalset_candidate': 0, 'seed': 0,
             'solve_time': 9.9},
            {'problem': 0, 'segment': 1, 'goalset_candidate': 0, 'seed': 0,
             'solve_time': 9.9},
            {'problem': 0, 'segment': 0, 'goalset_candidate': 0, 'seed': 0,
             'solve_time': 0.07},
        ]
        assert winner_solve_time(rows, [0], [0]) == 0.07

    def test_empty_inputs_are_none(self):
        assert winner_solve_time([], [0], [0]) is None
        assert winner_solve_time(None, [], []) is None
        assert winner_solve_time([{'solve_time': 0.1}], [], []) is None

    def test_unparseable_solve_time_returns_none(self):
        rows = [
            {'problem': 0, 'segment': 0, 'goalset_candidate': 0, 'seed': 0,
             'solve_time': 'not-a-number'},
        ]
        assert winner_solve_time(rows, [0], [0]) is None
class TestWinnerPositionErrorMm:
    """Solver-reported position error of the winning considered row (ROS leg).

    The server fills ``stats.considered`` with one row per seed; the winner's
    row carries the solver's per-seed ``position_error`` (m) as
    ``max_waypoint_error`` — the same convergence metric the native leg records
    as ``result.position_error`` — so ×1000 reproduces the native mm value.
    """

    def test_returns_winner_row_error_in_mm(self):
        rows = [
            {'problem': 0, 'segment': 0, 'goalset_candidate': 0, 'seed': 0,
             'max_waypoint_error': 0.0005},
            {'problem': 0, 'segment': 0, 'goalset_candidate': 0, 'seed': 1,
             'max_waypoint_error': 0.0012},
        ]
        assert winner_position_error_mm(rows, [0], [1]) == 1.2

    def test_accepts_attribute_rows(self):
        class Row:
            problem = 0
            segment = 0
            goalset_candidate = 0
            seed = 3
            max_waypoint_error = 0.002
        assert winner_position_error_mm([Row()], [0], [3]) == 2.0

    def test_zero_error_stays_zero(self):
        rows = [
            {'problem': 0, 'segment': 0, 'goalset_candidate': 0, 'seed': 0,
             'max_waypoint_error': 0.0},
        ]
        assert winner_position_error_mm(rows, [0], [0]) == 0.0

    def test_no_match_returns_none(self):
        rows = [
            {'problem': 0, 'segment': 0, 'goalset_candidate': 0, 'seed': 0,
             'max_waypoint_error': 0.001},
        ]
        assert winner_position_error_mm(rows, [0], [7]) is None

    def test_ignores_other_problems_and_segments(self):
        rows = [
            {'problem': 1, 'segment': 0, 'goalset_candidate': 0, 'seed': 0,
             'max_waypoint_error': 9.9},
            {'problem': 0, 'segment': 1, 'goalset_candidate': 0, 'seed': 0,
             'max_waypoint_error': 9.9},
            {'problem': 0, 'segment': 0, 'goalset_candidate': 0, 'seed': 0,
             'max_waypoint_error': 0.002},
        ]
        assert winner_position_error_mm(rows, [0], [0]) == 2.0

    def test_empty_inputs_are_none(self):
        assert winner_position_error_mm([], [0], [0]) is None
        assert winner_position_error_mm(None, [], []) is None
        assert winner_position_error_mm([{'max_waypoint_error': 0.1}], [], []) is None

    def test_missing_or_unparseable_error_returns_none(self):
        rows = [
            {'problem': 0, 'segment': 0, 'goalset_candidate': 0, 'seed': 0,
             'max_waypoint_error': 'not-a-number'},
        ]
        assert winner_position_error_mm(rows, [0], [0]) is None
        assert winner_position_error_mm(
            [{'problem': 0, 'segment': 0, 'goalset_candidate': 0, 'seed': 0}],
            [0], [0]
        ) is None
class TestReferenceStats:
    """Aggregation for the `reference` subcommand's page-comparison grids."""

    def test_success_pct_over_all_problems(self):
        results = [
            _successful('a', solve_time_s=0.07, position_error_mm=0.5, jerk=3.2,
                        path_length_curobo=1.5, motion_time_curobo=2.0,
                        energy_j=89.27, torque_nm=71.03),
            _successful('b', solve_time_s=0.11, position_error_mm=0.7, jerk=4.0,
                        path_length_curobo=2.5, motion_time_curobo=3.0,
                        energy_j=99.27, torque_nm=81.03),
            _result('c', success=False),
        ]
        stats = reference_stats(results)
        assert stats['Success %'][0] == pytest.approx(66.66666666666666)
        assert stats['Plan Time (s)'][0] == pytest.approx(0.1)  # time_s default
        assert stats['Solve Time (s)'][0] == pytest.approx(0.09)
        assert stats['Position Error (mm)'][0] == pytest.approx(0.6)
        assert stats['Jerk'][0] == pytest.approx(3.6)
        # path/motion prefer the curobo control-point fields
        assert stats['Path Length (rad.)'][0] == pytest.approx(2.0)
        assert stats['Motion Time(s)'][0] == pytest.approx(2.5)
        assert stats['Energy (J)'] == (pytest.approx(94.27), pytest.approx(94.27))
        assert stats['Torque (N·m)'] == (pytest.approx(76.03), pytest.approx(76.03))

    def test_errors_are_success_rate_only(self):
        """Zero successful solves -> Success % only (no statistic rows)."""
        stats = reference_stats([_result('a', success=False)])
        assert stats['Success %'] == (0.0, None)
        assert len(stats) == 1

    def test_filters_inf_timings_like_stat_str(self):
        results = [_successful('a', time_s=float('inf'), position_error_mm=0.5)]
        stats = reference_stats(results)
        assert 'Plan Time (s)' not in stats  # all timings inf -> no row
        assert stats['Position Error (mm)'][0] == pytest.approx(0.5)

    def test_falls_back_to_dense_path_motion_without_curobo_fields(self):
        results = [_successful('a', path_length=1.0, motion_time_s=1.0)]
        stats = reference_stats(results)
        assert stats['Path Length (rad.)'][0] == pytest.approx(1.0)
        assert stats['Motion Time(s)'][0] == pytest.approx(1.0)
        assert 'Position Error (mm)' not in stats
        assert 'Energy (J)' not in stats

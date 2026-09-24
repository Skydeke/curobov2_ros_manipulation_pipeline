# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the native-vs-ROS parity comparison logic (pure Python)."""

import math

import pytest

from isaac_ros_cumotion_extra.benchmark.compare import (
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


class TestPrintReport:
    def test_print_report_runs(self, capsys):
        core = [_successful('a'), _successful('b')]
        ros = [_successful('a'), _successful('b', path_length=0.9)]
        report = compare(core, ros)
        print_report(report)
        captured = capsys.readouterr()
        assert 'VERDICT' in captured.out

    def test_print_report_show_all(self, capsys):
        core = [_successful('a')]
        ros = [_successful('a')]
        print_report(compare(core, ros), show_all=True)
        assert 'per-problem deltas' in capsys.readouterr().out

    def test_print_report_prints_curobo_tables(self, capsys):
        core = [_successful('a'), _successful('b')]
        ros = [_successful('a'), _successful('b', path_length=0.9)]
        print_report(
            compare(core, ros),
            core_results=core,
            ros_results=ros,
        )
        out = capsys.readouterr().out
        assert '== native (curobo_core) ==' in out
        assert '== ros (unified_planner) ==' in out
        assert '+====' in out  # tabulate grid header separator
        assert '| Success %' in out

    def test_print_report_solve_time_info_line(self, capsys):
        """The report prints both legs' solver-reported solve times and makes
        the box behaviour explicit: ros solve tracks the ros wall (the
        per-request cost sits inside the solver timers, not in RTT)."""
        core = [_successful('a', time_s=0.065, solve_time_s=0.055)]
        ros = [_successful('a', time_s=2.0, solve_time_s=1.98)]
        print_report(compare(core, ros), core_results=core, ros_results=ros)
        out = capsys.readouterr().out
        assert 'ros wall avg 2s' in out
        assert 'solve time (info): core avg 0.055s, ros avg 1.98s' in out
        assert 'of the ros wall' in out  # tracks-wall attribution
        assert 'solver-loop cost, not RTT' in out

    def test_print_report_delta_table_shows_solve_columns(self, capsys):
        """The per-problem delta table carries solveC/solveR columns so the
        solver-reported times sit next to the wall clocks per problem (the
        box's solveR==timeR ~2 s vs solveC==timeC ~0.06 s is visible)."""
        core = [_successful('a', time_s=0.065, solve_time_s=0.055, path_length=1.0)]
        ros = [_successful('a', time_s=2.0, solve_time_s=1.98, path_length=1.05)]
        print_report(compare(core, ros), core_results=core, ros_results=ros, show_all=True)
        out = capsys.readouterr().out
        assert '--- per-problem deltas ---' in out
        assert 'solveC' in out and 'solveR' in out
        assert '0.055' in out
        assert '1.98' in out

    def test_print_report_omits_solve_time_line_without_data(self, capsys):
        core = [_successful('a', time_s=0.065)]
        ros = [_successful('a', time_s=2.0)]
        print_report(compare(core, ros), core_results=core, ros_results=ros)
        out = capsys.readouterr().out
        assert 'solve time (info)' not in out

    def test_zero_success_run_does_not_crash(self, capsys):
        """Regression: an all-failed run used to crash printing ``path delta``
        (``NoneType * int`` at ``avg_path_delta * 100``), because the relative
        deltas are only defined over problems where both legs succeeded."""
        core = [_result(f'p{i}', success=False) for i in range(3)]
        ros = [_result(f'p{i}', success=False) for i in range(3)]
        report = compare(core, ros)
        assert report['matches'] == 3  # both fail identically -> agreement
        print_report(report, core_results=core, ros_results=ros)
        out = capsys.readouterr().out
        assert 'path delta: avg -% max -%' in out
        assert 'motion delta: avg -% max -%' in out
        assert 'note: both legs solved 0 problems' in out
        assert 'VERDICT: PARITY-OK' in out

    def test_zero_success_hint_lists_scenes(self, capsys):
        core = [_result('p1', success=False, scene_key='bookshelf_small_panda')]
        ros = [_result('p1', success=False, scene_key='bookshelf_small_panda')]
        print_report(compare(core, ros), core_results=core, ros_results=ros)
        out = capsys.readouterr().out
        assert 'scenes: bookshelf_small_panda' in out
        assert '--scene' in out  # points at the exploration flag


class TestCuroboStyleRows:
    def test_success_rate_and_stats(self):
        results = [
            _successful('a', time_s=0.1, path_length=1.0, motion_time_s=1.0),
            _successful('b', time_s=0.2, path_length=2.0, motion_time_s=2.0),
            _result('c', success=False),
        ]
        rows = curobo_style_rows(results)
        assert rows[0] == ['Success %', '66.67']
        assert rows[1][0] == 'Plan Time (s)'
        assert rows[1][1].startswith('mean: 0.150 \u00b1 0.050')
        assert [r[0] for r in rows] == [
            'Success %', 'Plan Time (s)', 'Path Length (rad.)', 'Motion Time(s)',
        ]

    def test_solve_time_row_only_when_present(self):
        results = [_successful('a', solve_time_s=0.07)]
        labels = [r[0] for r in curobo_style_rows(results)]
        assert labels == [
            'Success %', 'Plan Time (s)', 'Solve Time (s)',
            'Path Length (rad.)', 'Motion Time(s)',
        ]

    def test_empty_results(self):
        rows = curobo_style_rows([])
        assert rows == [
            ['Success %', '0.00'],
            ['Plan Time (s)', '-'],
            ['Path Length (rad.)', '-'],
            ['Motion Time(s)', '-'],
        ]

    def test_reference_rows_use_curobo_fields_when_present(self):
        """Native-leg entries carry the upstream CuroboMetrics fields; the
        table must prefer them (and show Position Error / Jerk rows) exactly
        like the reference benchmark table."""
        results = [_successful(
            'a', solve_time_s=0.07,
            path_length=1.0, motion_time_s=1.0,  # dense fallbacks (parity)
            path_length_curobo=1.5, motion_time_curobo=2.0,
            position_error_mm=0.5, orientation_error_deg=0.1, jerk=3.2,
        )]
        rows = curobo_style_rows(results)
        assert [r[0] for r in rows] == [
            'Success %', 'Plan Time (s)', 'Solve Time (s)',
            'Position Error (mm)', 'Path Length (rad.)', 'Motion Time(s)',
            'Jerk',
        ]
        by_label = dict(rows)
        assert by_label['Path Length (rad.)'].startswith('mean: 1.500')
        assert by_label['Motion Time(s)'].startswith('mean: 2.000')
        assert by_label['Position Error (mm)'].startswith('mean: 0.500')
        assert by_label['Jerk'].startswith('mean: 3.200')

    def test_falls_back_to_dense_metrics_without_curobo_fields(self):
        """ROS-leg entries have no control-point fields -> dense values."""
        results = [_successful('a', path_length=1.0, motion_time_s=1.0)]
        rows = curobo_style_rows(results)
        by_label = dict(rows)
        assert [r[0] for r in rows] == [
            'Success %', 'Plan Time (s)', 'Path Length (rad.)', 'Motion Time(s)',
        ]
        assert by_label['Path Length (rad.)'].startswith('mean: 1.000')
        assert by_label['Motion Time(s)'].startswith('mean: 1.000')
        assert 'Position Error (mm)' not in by_label
        assert 'Jerk' not in by_label

    def test_ros_leg_jerk_row_when_client_side_estimate_present(self):
        """The ROS leg computes max|jerk| client-side (no control points), so its
        table shows a Jerk row too — same layout as the native table."""
        results = [_successful('a', path_length=1.0, motion_time_s=1.0, jerk=12.5)]
        rows = curobo_style_rows(results)
        by_label = dict(rows)
        assert [r[0] for r in rows] == [
            'Success %', 'Plan Time (s)', 'Path Length (rad.)', 'Motion Time(s)',
            'Jerk',
        ]
        assert by_label['Jerk'].startswith('mean: 12.500')

    def test_energy_torque_rows_only_when_data_present(self):
        """Energy (J) / Torque (N·m) rows (reference page layout, after Jerk)
        are emitted only when any successful entry carries the fields — the ROS
        leg without a dynamics model leaves them None and must not gain rows."""
        results = [_successful('a', energy_j=89.27, torque_nm=71.03)]
        rows = curobo_style_rows(results)
        assert [r[0] for r in rows] == [
            'Success %', 'Plan Time (s)', 'Path Length (rad.)', 'Motion Time(s)',
            'Energy (J)', 'Torque (N·m)',
        ]
        by_label = dict(rows)
        assert by_label['Energy (J)'].startswith('mean: 89.270')
        assert by_label['Torque (N·m)'].startswith('mean: 71.030')

    def test_no_energy_torque_rows_without_fields(self):
        results = [_successful('a', path_length=1.0, motion_time_s=1.0)]
        by_label = dict(curobo_style_rows(results))
        assert 'Energy (J)' not in by_label
        assert 'Torque (N·m)' not in by_label


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


class TestStatStr:
    def test_matches_curobo_layout(self):
        out = _stat_str([1.0, 2.0, 3.0, 4.0])
        assert out == (
            'mean: 2.500 \u00b1 1.118  median: 2.500  75%: 3.250  98%: 3.940'
        )

    def test_single_value(self):
        out = _stat_str([3.0])
        assert out == (
            'mean: 3.000 \u00b1 0.000  median: 3.000  75%: 3.000  98%: 3.000'
        )

    def test_filters_none_and_inf(self):
        out = _stat_str([None, 1.0, float('inf')])
        assert out.startswith('mean: 1.000 \u00b1 0.000')

    def test_empty_returns_dash(self):
        assert _stat_str([]) == '-'
        assert _stat_str([None]) == '-'


class TestGridTable:
    def test_grid_layout(self):
        out = _grid_table([['A', '1'], ['BB', '22']], ['M', 'V'])
        assert out == (
            '+----+----+\n'
            '| M  | V  |\n'
            '+====+====+\n'
            '| A  | 1  |\n'
            '+----+----+\n'
            '| BB | 22 |\n'
            '+----+----+'
        )

    def test_empty_rows_still_prints_header(self):
        out = _grid_table([], ['Metric', 'Value'])
        assert out == (
            '+--------+-------+\n'
            '| Metric | Value |\n'
            '+========+=======+'
        )

    def test_long_value_cell_expands_column(self):
        out = _grid_table([['Plan Time (s)', 'mean: 0.038']], ['Metric', 'Value'])
        assert '+---------------+-------------+' in out
        assert '| Plan Time (s) | mean: 0.038 |' in out


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


class TestPrintReferenceComparison:
    def test_prints_both_tables_and_published_grid(self, capsys):
        def one(p_base, e, t):
            return [_successful(
                'a', solve_time_s=0.07, position_error_mm=0.5, jerk=3.2,
                path_length_curobo=1.5, motion_time_curobo=2.0,
                energy_j=e, torque_nm=t,
            ), _successful('b', solve_time_s=0.11, position_error_mm=0.7,
                           jerk=4.0, path_length_curobo=2.5,
                           motion_time_curobo=3.0, energy_j=e + p_base,
                           torque_nm=t + p_base)]

        print_reference_comparison(
            one(10.0, 89.27, 71.03), one(-8.0, 81.41, 62.27), mass=3.0)

        out = capsys.readouterr().out
        # both curobo-style tables are printed again
        assert '== without torque limits (native) ==' in out
        assert '== with torque limits (3 kg) (native) ==' in out
        # then the published-vs-ours comparison grids for both tables
        assert '== published page (RTX 6000 Ada) vs this box — without torque limits ==' in out
        assert '== published page (RTX 6000 Ada) vs this box — with torque limits (3 kg) ==' in out
        for marker in ('Docs mean', 'Ours mean', 'Docs median', 'Ours median'):
            assert marker in out
        # published values appear verbatim in the comparison grid
        assert '0.038' in out and '0.041' in out
        assert '89.270' in out and '78.959' in out
        assert '62.270' in out and '65.345' in out
        # regression guard: the delta note is computed from the real
        # (mean, median) tuples, never from their first characters. With this
        # fixture the worst mean delta is Jerk: |227.365 - 3.6| = 223.765.
        assert 'worst mean delta vs the page: 223.765' in out
        assert 'max |mean delta|' not in out

    def test_empty_results_still_print_published_grid(self, capsys):
        print_reference_comparison([], [])
        out = capsys.readouterr().out
        assert '== without torque limits (native) ==' in out
        # all ours cells are dashes but the grid still lists the page values
        assert '| Success %' in out
        assert '99.73' in out
        assert 'max |mean delta|' not in out  # no data, no delta note

    def test_mass_shown_in_torque_table_title(self, capsys):
        print_reference_comparison([_successful('a')], [_successful('b')], mass=2.0)
        out = capsys.readouterr().out
        assert 'with torque limits (2 kg) (native)' in out


class TestPrintWebpageSummary:
    """The compose-end summary: every suite, page order, native + ROS legs."""

    def _motion(self, **extra):
        base = dict(
            problem_name='a', scene_key='s', capability='planning', success=True,
            time_s=0.04, solve_time_s=0.031, path_length=1.5, motion_time_s=2.0,
            path_length_curobo=3.126, motion_time_curobo=1.25,
            position_error_mm=0.041, orientation_error_deg=0.1, jerk=227.365,
            energy_j=89.27, torque_nm=71.028,
        )
        base.update(extra)
        return base

    def _ik(self, **extra):
        base = dict(
            problem_name='g0', scene_key='s', capability='ik', variant='cfree',
            success=True, time_ms=1.2, position_error_mm=0.3,
            orientation_error_deg=0.05, n_waypoints=0,
        )
        base.update(extra)
        return base

    def _cost(self, **extra):
        base = dict(
            problem_name='c0', scene_key='s', capability='cost', valid=True,
            time_ms=2.0, n_configs=64,
        )
        base.update(extra)
        return base

    def test_suites_print_in_webpage_order_with_both_legs(self, capsys):
        motion, ik, cost = [self._motion()], [self._ik()], [self._cost()]
        print_webpage_summary(
            motion_plain_native=motion, motion_plain_ros=motion,
            motion_torque_native=motion, motion_torque_ros=motion,
            ik_native=ik, ik_ros=ik, cost_native=cost, cost_ros=cost,
        )
        out = capsys.readouterr().out
        i1 = out.index('[1/3] MOTION GENERATION')
        i2 = out.index('[2/3] INVERSE KINEMATICS')
        i3 = out.index('[3/3] KINEMATICS & COLLISION')
        i4 = out.index('[APPENDIX]')
        assert i1 < i2 < i3 < i4
        # both legs of every suite are present, with distinguishable titles
        assert '== without torque limits — native curobo ==' in out
        assert '== without torque limits — ROS (server) ==' in out
        assert '== with torque limits (3 kg) — native curobo ==' in out
        assert '== with torque limits (3 kg) — ROS (server) ==' in out
        assert '== inverse kinematics — native curobo ==' in out
        assert '== inverse kinematics — ROS (/ik_batch) ==' in out
        assert '== kinematics & collision — native curobo ==' in out
        assert '== kinematics & collision — ROS (/fk_batch) ==' in out
        assert '== published page (RTX 6000 Ada) vs this box — without torque limits ==' in out
        assert 'Receipt notes:' in out
        # the receipt states both legs share the page's 100-attempt budget
        # (the runner pins the server's plan-time max_attempts to --max-attempts)
        assert '100 for BOTH legs' in out
        assert 'max_attempts to --max-attempts' in out

    def test_missing_ros_legs_are_flagged_not_crashed(self, capsys):
        print_webpage_summary(
            motion_plain_native=[self._motion()],
            motion_torque_native=[self._motion()],
            ik_native=[self._ik()],
            cost_native=[self._cost()],
        )
        out = capsys.readouterr().out
        assert 'without torque limits — ROS (server): (not run' in out
        assert 'with torque limits (3 kg) — ROS (server): (not run' in out
        assert 'inverse kinematics — ROS (/ik_batch): (not run' in out
        assert 'kinematics & collision — ROS (/fk_batch): (not run' in out
        # the native motion tables exist, so the appendix grids (and their
        # delta note) still print
        assert '== published page (RTX 6000 Ada) vs this box' in out
        assert 'worst mean delta' in out

    def test_motion_section_only_when_native_motion_missing(self, capsys):
        """The appendix needs the two native motion tables to compare."""
        print_webpage_summary(ik_native=[self._ik()])
        out = capsys.readouterr().out
        assert '[APPENDIX]' not in out
        assert '[2/3] INVERSE KINEMATICS' in out
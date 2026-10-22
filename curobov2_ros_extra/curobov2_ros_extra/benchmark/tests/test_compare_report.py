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
        assert '== native (curobo) ==' in out
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

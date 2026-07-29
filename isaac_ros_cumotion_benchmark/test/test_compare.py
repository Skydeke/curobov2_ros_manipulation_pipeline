import json

from isaac_ros_cumotion_benchmark.compare import compare, print_report, save_report


def _result(name, success=True, time_s=0.1, n_waypoints=10):
    return {
        'problem_name': name,
        'scene_key': 'test_scene',
        'success': success,
        'time_s': time_s,
        'n_waypoints': n_waypoints,
    }


class TestCompare:
    def test_empty_inputs(self):
        report = compare([], [])
        assert report['total'] == 0
        assert report['matches'] == 0
        assert report['mismatches'] == 0

    def test_identical_results_all_match(self):
        core = [_result('a'), _result('b')]
        ros = [_result('a'), _result('b')]
        report = compare(core, ros)
        assert report['total'] == 2
        assert report['matches'] == 2
        assert report['mismatches'] == 0
        assert report['success_mismatches'] == 0

    def test_success_mismatch_detected(self):
        core = [_result('a', success=True)]
        ros = [_result('a', success=False)]
        report = compare(core, ros)
        assert report['success_mismatches'] == 1
        assert report['mismatches'] == 1
        assert report['details']['mismatches'][0]['diff_type'] == 'success_mismatch'

    def test_core_only_problem(self):
        core = [_result('a'), _result('b')]
        ros = [_result('a')]
        report = compare(core, ros)
        assert report['core_only'] == 1
        assert report['mismatches'] == 1

    def test_ros_only_problem(self):
        core = [_result('a')]
        ros = [_result('a'), _result('b')]
        report = compare(core, ros)
        assert report['ros_only'] == 1
        assert report['mismatches'] == 1

    def test_time_mismatch_beyond_tolerance(self):
        core = [_result('a', time_s=1.0)]
        ros = [_result('a', time_s=2.0)]
        report = compare(core, ros, tolerance=0.5)
        assert report['mismatches'] == 1
        assert report['details']['mismatches'][0]['diff_type'] == 'time_mismatch'

    def test_time_within_tolerance_is_match(self):
        core = [_result('a', time_s=1.0)]
        ros = [_result('a', time_s=1.000_001)]
        report = compare(core, ros, tolerance=0.01)
        assert report['matches'] == 1
        assert report['mismatches'] == 0

    def test_zero_tolerance(self):
        core = [_result('a', time_s=1.0)]
        ros = [_result('a', time_s=1.000_001)]
        report = compare(core, ros, tolerance=0.0)
        assert report['mismatches'] == 1

    def test_multiple_scenes_mixed(self):
        core = [
            _result('scene1_1', success=True, time_s=0.5),
            _result('scene1_2', success=False),
        ]
        ros = [
            _result('scene1_1', success=True, time_s=0.5),
            _result('scene1_3', success=True),
        ]
        report = compare(core, ros)
        assert report['total'] == 3
        assert report['matches'] == 1
        assert report['core_only'] == 1
        assert report['ros_only'] == 1

    def test_report_structure(self):
        core = [_result('a')]
        ros = [_result('a')]
        report = compare(core, ros)
        assert 'total' in report
        assert 'matches' in report
        assert 'mismatches' in report
        assert 'core_only' in report
        assert 'ros_only' in report
        assert 'success_mismatches' in report
        assert 'details' in report
        assert 'matches' in report['details']
        assert 'mismatches' in report['details']


class TestPrintReport:
    def test_print_no_mismatches(self, capsys):
        report = compare(
            [_result('a', time_s=0.1)],
            [_result('a', time_s=0.1)],
        )
        print_report(report)
        captured = capsys.readouterr().out
        assert 'Matches:' in captured
        assert 'Mismatches:' in captured

    def test_print_mismatches_shows_details(self, capsys):
        report = compare(
            [_result('a', success=True)],
            [_result('a', success=False)],
        )
        print_report(report)
        captured = capsys.readouterr().out
        assert 'MISMATCHES' in captured

    def test_show_all_includes_matches(self, capsys):
        report = compare(
            [_result('a', time_s=0.1)],
            [_result('a', time_s=0.1)],
        )
        print_report(report, show_all=True)
        captured = capsys.readouterr().out
        assert 'ALL MATCHES' in captured

    def test_show_all_false_hides_matches(self, capsys):
        report = compare(
            [_result('a', time_s=0.1)],
            [_result('a', time_s=0.1)],
        )
        print_report(report, show_all=False)
        captured = capsys.readouterr().out
        assert 'ALL MATCHES' not in captured


class TestSaveReport:
    def test_save_and_load_roundtrip(self, tmp_path):
        report = compare(
            [_result('a', success=True, time_s=0.1)],
            [_result('a', success=True, time_s=0.1)],
        )
        path = tmp_path / 'report.json'
        save_report(report, str(path))
        with open(path) as f:
            loaded = json.load(f)
        assert loaded['total'] == 1
        assert loaded['matches'] == 1

    def test_save_with_mismatches(self, tmp_path):
        report = compare(
            [_result('a', success=True)],
            [_result('a', success=False)],
        )
        path = tmp_path / 'mismatch_report.json'
        save_report(report, str(path))
        with open(path) as f:
            loaded = json.load(f)
        assert loaded['success_mismatches'] == 1
        assert len(loaded['details']['mismatches']) == 1

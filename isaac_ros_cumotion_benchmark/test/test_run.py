import argparse

import pytest


def _parse(argv):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)

    p_core = subparsers.add_parser('core')
    p_core.add_argument('--dataset', default='demo',
                        choices=['demo', 'motion_benchmaker', 'mpinets'])
    p_core.add_argument('--output', '-o')
    p_core.set_defaults(func=lambda args: None)

    p_ros = subparsers.add_parser('ros')
    p_ros.add_argument('--dataset', default='demo',
                       choices=['demo', 'motion_benchmaker', 'mpinets'])
    p_ros.add_argument('--time_dilation_factor', type=float, default=1.0)
    p_ros.add_argument('--output', '-o')
    p_ros.set_defaults(func=lambda args: None)

    p_cmp = subparsers.add_parser('compare')
    p_cmp.add_argument('core_results')
    p_cmp.add_argument('ros_results')
    p_cmp.add_argument('--output', '-o')
    p_cmp.add_argument('--show-all', action='store_true')
    p_cmp.set_defaults(func=lambda args: None)

    p_all = subparsers.add_parser('all')
    p_all.add_argument('--dataset', default='demo',
                       choices=['demo', 'motion_benchmaker', 'mpinets'])
    p_all.add_argument('--time_dilation_factor', type=float, default=1.0)
    p_all.add_argument('--output', '-o')
    p_all.add_argument('--save-all', action='store_true')
    p_all.add_argument('--show-all', action='store_true')
    p_all.set_defaults(func=lambda args: None)

    return parser.parse_args(argv)


class TestSubcommands:
    def test_core_defaults(self):
        args = _parse(['core'])
        assert args.command == 'core'
        assert args.dataset == 'demo'
        assert args.output is None

    def test_core_explicit(self):
        args = _parse(['core', '--dataset', 'motion_benchmaker', '-o', 'out.json'])
        assert args.dataset == 'motion_benchmaker'
        assert args.output == 'out.json'

    def test_ros_defaults(self):
        args = _parse(['ros'])
        assert args.command == 'ros'
        assert args.dataset == 'demo'
        assert args.time_dilation_factor == 1.0

    def test_ros_explicit_time_dilation(self):
        args = _parse(['ros', '--time_dilation_factor', '0.5'])
        assert args.time_dilation_factor == 0.5

    def test_compare_requires_two_positional(self):
        args = _parse(['compare', 'core.json', 'ros.json'])
        assert args.command == 'compare'
        assert args.core_results == 'core.json'
        assert args.ros_results == 'ros.json'
        assert not args.show_all

    def test_compare_show_all(self):
        args = _parse(['compare', 'c.json', 'r.json', '--show-all'])
        assert args.show_all

    def test_compare_output(self):
        args = _parse(['compare', 'c.json', 'r.json', '-o', 'report.json'])
        assert args.output == 'report.json'

    def test_all_defaults(self):
        args = _parse(['all'])
        assert args.command == 'all'
        assert args.dataset == 'demo'
        assert args.time_dilation_factor == 1.0
        assert not args.save_all
        assert not args.show_all

    def test_all_with_flags(self):
        args = _parse([
            'all', '--dataset', 'mpinets', '--save-all',
            '--show-all', '-o', 'r.json',
        ])
        assert args.dataset == 'mpinets'
        assert args.save_all
        assert args.show_all
        assert args.output == 'r.json'

    def test_invalid_dataset_rejected(self):
        with pytest.raises(SystemExit):
            _parse(['core', '--dataset', 'invalid'])

    def test_no_command_errors(self):
        with pytest.raises(SystemExit):
            _parse([])


class TestDispatch:
    def test_cmd_core_dispatches(self):
        from isaac_ros_cumotion_benchmark.run import cmd_core

        args = argparse.Namespace(dataset='demo', output=None)
        cmd_core(args)

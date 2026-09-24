# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the benchmark CLI parser + sidecar naming (pure Python)."""

import pytest

from isaac_ros_cumotion_extra.benchmark.run import (
    _seed_knobs,
    _sidecar,
    build_parser,
)


class TestSubcommands:
    def test_core_defaults(self):
        args = build_parser().parse_args(['core'])
        assert args.dataset == 'demo'
        assert args.num_ik_seeds == 32
        assert args.num_trajopt_seeds == 4
        assert args.max_attempts == 1
        assert args.mesh is False
        assert not args.no_cuda_graph
        assert not args.no_reset_seed
        assert not args.unseeded
        assert args.warmup_iters == 3

    def test_seed_knobs_default_to_reference_behaviour(self):
        args = build_parser().parse_args(['core'])
        knobs = _seed_knobs(args)
        assert knobs == {'seed_globals': True, 'reset_seed_per_problem': True}

    def test_no_reset_seed_knob(self):
        args = build_parser().parse_args(['core', '--no-reset-seed'])
        assert _seed_knobs(args) == {
            'seed_globals': True, 'reset_seed_per_problem': False,
        }

    def test_unseeded_implies_no_reset_seed(self):
        args = build_parser().parse_args(['core', '--unseeded'])
        assert _seed_knobs(args) == {
            'seed_globals': False, 'reset_seed_per_problem': False,
        }
        # both flags together are idempotent
        both = build_parser().parse_args(['core', '--unseeded', '--no-reset-seed'])
        assert _seed_knobs(both) == _seed_knobs(args)

    def test_ros_defaults(self):
        args = build_parser().parse_args(['ros'])
        assert args.dataset == 'demo'
        assert args.service_timeout == 30.0
        assert args.call_timeout == 120.0

    def test_compare_requires_two_positional(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(['compare'])
        args = build_parser().parse_args(['compare', 'core.json', 'ros.json', '-o', 'report.json'])
        assert args.core_results == 'core.json'
        assert args.ros_results == 'ros.json'
        assert args.path_tolerance == 0.02
        assert args.motion_tolerance == 0.02
        assert not args.show_all
        assert args.output == 'report.json'

    def test_all_defaults(self):
        args = build_parser().parse_args(['all', '-o', '/tmp/rep.json'])
        assert args.dataset == 'demo'
        assert args.output == '/tmp/rep.json'
        assert args.show_all is False
        assert args.num_ik_seeds == 32
        assert args.num_trajopt_seeds == 4
        assert args.max_attempts == 1
        assert args.mesh is False

    def test_overrides(self):
        args = build_parser().parse_args([
            'core', '--dataset', 'mpinets', '--num-ik-seeds', '16',
            '--num-trajopt-seeds', '2', '--max-attempts', '50',
            '--mesh', '--no-cuda-graph',
        ])
        assert args.dataset == 'mpinets'
        assert args.num_ik_seeds == 16
        assert args.num_trajopt_seeds == 2
        assert args.max_attempts == 50
        assert args.mesh is True
        assert args.no_cuda_graph is True

    def test_show_all_and_tolerances(self):
        args = build_parser().parse_args([
            'compare', 'a.json', 'b.json', '--show-all',
            '--path-tolerance', '0.05', '--motion-tolerance', '0.1',
        ])
        assert args.show_all is True
        assert args.path_tolerance == 0.05
        assert args.motion_tolerance == 0.1

    def test_scene_filter_option(self):
        args = build_parser().parse_args([
            'all', '--dataset', 'mpinets', '--scene', 'dresser_neutral_start',
        ])
        assert args.scene == 'dresser_neutral_start'
        assert build_parser().parse_args(['core']).scene is None
        assert build_parser().parse_args(['ros']).scene is None


class TestSidecar:
    def test_inserts_label_before_extension(self):
        assert _sidecar('/tmp/report.json', 'core') == '/tmp/report.core.json'

    def test_no_output_means_none(self):
        assert _sidecar(None, 'core') is None
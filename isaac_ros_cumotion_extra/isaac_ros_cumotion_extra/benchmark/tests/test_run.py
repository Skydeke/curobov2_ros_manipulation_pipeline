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
        assert args.max_attempts == 100
        assert args.mesh is False
        assert not args.no_cuda_graph
        assert not args.no_reset_seed
        assert not args.unseeded
        assert args.warmup_iters == 3
        assert args.use_dynamics is False
        assert args.mass == 3.0

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
        assert args.use_dynamics is False
        assert args.mass == 3.0

    def test_torque_limited_variant_flags(self):
        args = build_parser().parse_args([
            'core', '--use-dynamics', '--mass', '3.0',
        ])
        assert args.use_dynamics is True
        assert args.mass == 3.0
        ros = build_parser().parse_args(['ros', '--use-dynamics', '--mass', '2.0'])
        assert ros.use_dynamics is True
        assert ros.mass == 2.0

    def test_full_dataset_accepted(self):
        args = build_parser().parse_args(['core', '--dataset', 'full'])
        assert args.dataset == 'full'

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
        assert args.max_attempts == 100
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

    def test_reference_defaults(self):
        args = build_parser().parse_args(['reference'])
        # the page-reproduction envelope: full 2600-problem dataset and the
        # upstream 100-attempt real-solve budget (its retry loop produces the
        # page's 99.73 % success) — unlike the parity subcommands' 1/`demo`.
        assert args.dataset == 'full'
        assert args.max_attempts == 100
        assert args.num_ik_seeds == 32
        assert args.num_trajopt_seeds == 4
        assert args.mass == 3.0
        assert args.warmup_iters == 3
        assert args.output is None
        assert args.scene is None
        assert args.mesh is False
        assert not args.no_cuda_graph

    def test_reference_runs_both_modes_so_no_use_dynamics_flag(self):
        # --use-dynamics is implicit (reference runs plain AND torque passes);
        # passing it is a parser error, and --mass is still honoured.
        with pytest.raises(SystemExit):
            build_parser().parse_args(['reference', '--use-dynamics'])
        args = build_parser().parse_args(['reference', '--mass', '2.0',
                                          '--max-attempts', '1',
                                          '--dataset', 'demo'])
        assert args.mass == 2.0
        assert args.max_attempts == 1
        assert args.dataset == 'demo'

    def test_webpage_defaults(self):
        args = build_parser().parse_args(['webpage'])
        # the compose reproduction envelope: full dataset, 100-attempt budget
        assert args.dataset == 'full'
        assert args.max_attempts == 100
        assert args.mass == 3.0
        assert args.warmup_iters == 3
        # native-only by default; the compose passes --run-ros
        assert args.run_ros is False
        assert args.server_torque is False
        # ik/cost knobs present with their shared defaults
        assert args.batch == 100
        assert args.num_batches == 5
        assert args.num_seeds == 32
        assert args.seed == 2
        assert args.service_timeout == 30.0
        assert args.call_timeout == 120.0
        assert args.output is None
        assert args.scene is None

    def test_webpage_flags(self):
        args = build_parser().parse_args([
            'webpage', '--run-ros', '--server-torque',
            '--dataset', 'demo', '--max-attempts', '1', '--mass', '2.0',
            '--num-batches', '1', '-o', '/tmp/wp.json',
        ])
        assert args.run_ros is True
        assert args.server_torque is True
        assert args.dataset == 'demo'
        assert args.max_attempts == 1
        assert args.mass == 2.0
        assert args.num_batches == 1
        assert args.output == '/tmp/wp.json'
        # --use-dynamics is implicit (webpage runs BOTH motion modes)
        with pytest.raises(SystemExit):
            build_parser().parse_args(['webpage', '--use-dynamics'])


class TestWebpageLegs:
    """`webpage --run-ros` never skips a leg: both motion ROS rows always run,
    each leg ensuring the server's torque mode itself and pinning the server's
    plan-time max_attempts to the run's own --max-attempts budget."""

    def _stub_legs(self, monkeypatch, run_ros_calls, core_calls):
        from isaac_ros_cumotion_extra.benchmark import (
            compare,
            core_cost,
            core_ik,
            core_runner,
            ros_cost,
            ros_ik,
            ros_runner,
        )

        def fake_run_core(**kw):
            core_calls.append(('run_core', kw.get('use_dynamics')))
            return []

        monkeypatch.setattr(core_runner, 'run_core', fake_run_core)
        monkeypatch.setattr(core_ik, 'run_ik_core',
                            lambda **kw: core_calls.append(('run_ik_core', None)) or [])
        monkeypatch.setattr(ros_ik, 'run_ik_ros', lambda **kw: [])
        monkeypatch.setattr(core_cost, 'run_cost_core',
                            lambda **kw: core_calls.append(('run_cost_core', None)) or [])
        monkeypatch.setattr(ros_cost, 'run_cost_ros', lambda **kw: [])
        monkeypatch.setattr(compare, 'print_webpage_summary',
                            lambda **kw: None)

        def fake_run_ros(**kw):
            run_ros_calls.append((
                kw.get('use_dynamics'),
                kw.get('server_dynamics'),
                kw.get('server_payload_mass'),
                kw.get('server_max_attempts'),
                kw.get('apply_mpinets_locks'),
                kw.get('server_reset_seed'),
            ))
            return []

        monkeypatch.setattr(ros_runner, 'run_ros', fake_run_ros)

    def test_run_ros_runs_both_motion_ros_legs(self, monkeypatch, capsys):
        pytest.importorskip('rclpy')  # cmd_webpage imports the ROS leg modules
        from isaac_ros_cumotion_extra.benchmark.run import build_parser, cmd_webpage

        calls = []
        core_calls = []
        self._stub_legs(monkeypatch, calls, core_calls)
        # no --max-attempts flag: the whole-benchmark default 100 must flow
        # through to the server pin on BOTH motion ROS legs
        args = build_parser().parse_args([
            'webpage', '--run-ros', '--dataset', 'demo',
        ])
        assert args.max_attempts == 100
        assert cmd_webpage(args) == 0

        # plain (False) then torque (True) - both native and ROS, in page order,
        # each ROS leg pinning the server's max_attempts to this run's budget
        # and switching the gripper lock per scene by default
        assert calls == [
            # without torque: plain, no payload
            (False, False, 0.0, 100, True, True),
            # with torque limits: torque + payload
            (True, True, 3.0, 100, True, True),
        ]
        # native legs always run: both motion modes + IK + cost
        assert core_calls == [
            ('run_core', False),
            ('run_core', True),
            ('run_ik_core', None),
            ('run_cost_core', None),
        ]
        out = capsys.readouterr().out
        assert 'SKIPPED' not in out
        assert '(not run' not in out

    def test_webpage_without_run_ros_skips_nothing_and_runs_native_legs(
        self, monkeypatch, capsys,
    ):
        pytest.importorskip('rclpy')  # cmd_webpage imports the ROS leg modules
        from isaac_ros_cumotion_extra.benchmark.run import build_parser, cmd_webpage

        calls = []
        core_calls = []
        self._stub_legs(monkeypatch, calls, core_calls)
        args = build_parser().parse_args([
            'webpage', '--dataset', 'demo',
        ])
        assert cmd_webpage(args) == 0
        # native-only: all native legs ran, zero ROS legs, zero SKIPPED messages
        assert calls == []
        assert [c[0] for c in core_calls] == [
            'run_core', 'run_core', 'run_ik_core', 'run_cost_core',
        ]
        out = capsys.readouterr().out
        assert 'SKIPPED' not in out


class TestMpinetsLockFlag:
    """`--no-mpinets-locks` must exist on exactly the subcommands that run a
    motion ROS leg, and on those alone.

    Regression: the flag was first declared in the shared `_add_ros_service_opts`
    helper, which only the ik/cost parsers use — so `ros`/`all`/`webpage`, the
    three subcommands that read `args.no_mpinets_locks` and pass it to
    `run_ros`, would have died with AttributeError at run time while the flag
    sat uselessly on subcommands that never solve a trajectory.
    """

    MOTION_SUBCOMMANDS = ('ros', 'all', 'webpage')
    NON_MOTION_SUBCOMMANDS = ('ik', 'cost', 'ik-ros', 'cost-ros', 'core')

    def test_present_on_every_motion_subcommand(self):
        parser = build_parser()
        for cmd in self.MOTION_SUBCOMMANDS:
            args = parser.parse_args([cmd])
            assert args.no_mpinets_locks is False, cmd
            assert parser.parse_args([cmd, '--no-mpinets-locks']
                                     ).no_mpinets_locks is True, cmd

    def test_absent_from_subcommands_without_a_motion_ros_leg(self):
        parser = build_parser()
        for cmd in self.NON_MOTION_SUBCOMMANDS:
            args = parser.parse_args([cmd])
            assert not hasattr(args, 'no_mpinets_locks'), cmd

    def test_cmd_ros_forwards_the_flag(self, monkeypatch):
        pytest.importorskip('rclpy')  # cmd_ros imports the ROS leg module
        from isaac_ros_cumotion_extra.benchmark import ros_runner
        from isaac_ros_cumotion_extra.benchmark.run import build_parser, cmd_ros

        seen = {}
        monkeypatch.setattr(
            ros_runner, 'run_ros',
            lambda **kw: seen.update(kw) or [],
        )
        cmd_ros(build_parser().parse_args(
            ['ros', '--dataset', 'demo', '--no-mpinets-locks'],
        ))
        assert seen['apply_mpinets_locks'] is False

        cmd_ros(build_parser().parse_args(['ros', '--dataset', 'demo']))
        assert seen['apply_mpinets_locks'] is True


class TestServerResetSeedPin:
    """`--no-reset-seed` must reach the *server*, not just the native leg.

    The node gates its per-plan RNG rewind behind `reset_seed_per_plan`, which
    defaults to False and is enabled only by the benchmark launch file. The
    runner therefore pins the parameter per leg (mirroring
    `set_server_max_attempts`) so an A/B holds regardless of how the server
    was launched - otherwise `--no-reset-seed` would only ever A/B the
    in-process leg and the ROS leg would keep reseeding silently.
    """

    def _webpage_leg_seeds(self, monkeypatch, extra_args):
        pytest.importorskip('rclpy')  # cmd_webpage imports the ROS leg modules
        from isaac_ros_cumotion_extra.benchmark.run import build_parser, cmd_webpage

        holder = TestWebpageLegs()
        calls, core_calls = [], []
        holder._stub_legs(monkeypatch, calls, core_calls)
        args = build_parser().parse_args(
            ['webpage', '--run-ros', '--dataset', 'demo'] + extra_args,
        )
        assert cmd_webpage(args) == 0
        # both motion legs pinned, in page order
        return [c[5] for c in calls]

    def test_webpage_motion_legs_pin_the_server_reset_seed(
        self, monkeypatch, capsys,
    ):
        assert self._webpage_leg_seeds(monkeypatch, []) == [True, True]
        capsys.readouterr()

    def test_webpage_no_reset_seed_reaches_both_motion_legs(
        self, monkeypatch, capsys,
    ):
        assert self._webpage_leg_seeds(
            monkeypatch, ['--no-reset-seed'],
        ) == [False, False]
        capsys.readouterr()

    def test_webpage_unseeded_implies_the_server_reset_seed_is_off(
        self, monkeypatch, capsys,
    ):
        assert self._webpage_leg_seeds(monkeypatch, ['--unseeded']) == [False, False]
        capsys.readouterr()

    def test_cmd_ros_forwards_the_flag(self, monkeypatch):
        pytest.importorskip('rclpy')  # cmd_ros imports the ROS leg module
        from isaac_ros_cumotion_extra.benchmark import ros_runner
        from isaac_ros_cumotion_extra.benchmark.run import build_parser, cmd_ros

        seen = {}
        monkeypatch.setattr(
            ros_runner, 'run_ros',
            lambda **kw: seen.update(kw) or [],
        )
        cmd_ros(build_parser().parse_args(
            ['ros', '--dataset', 'demo', '--no-reset-seed'],
        ))
        assert seen['server_reset_seed'] is False

        cmd_ros(build_parser().parse_args(['ros', '--dataset', 'demo']))
        assert seen['server_reset_seed'] is True

    def test_present_on_every_subcommand_that_reads_it(self):
        # Regression: `cmd_ros` reads `args.no_reset_seed` to pin the server,
        # but `p_ros` never declared the flag (it builds its options by hand
        # instead of calling `_add_solver_opts`), so every `ros` invocation
        # would have died in argparse before the A/B could be requested.
        parser = build_parser()
        for cmd in ('core', 'ros', 'all', 'reference', 'webpage'):
            assert parser.parse_args([cmd]).no_reset_seed is False, cmd
            assert parser.parse_args(
                [cmd, '--no-reset-seed'],
            ).no_reset_seed is True, cmd

    def test_unseeded_stays_off_subcommands_without_a_native_leg(self):
        # `--unseeded` also skips the native leg's fixed global seeds, so it
        # is only meaningful where a native leg runs; `_seed_knobs` reads it
        # defensively so `ros --no-reset-seed` does not need it.
        parser = build_parser()
        for cmd in ('core', 'all', 'reference', 'webpage'):
            assert parser.parse_args([cmd, '--unseeded']).unseeded is True, cmd
        assert not hasattr(parser.parse_args(['ros']), 'unseeded')


class TestRosSubcommandServerPins:
    """`cmd_ros` reads parser attributes, so `p_ros` must declare them.

    `p_ros` builds its options by hand instead of calling `_add_solver_opts`
    (it needs ROS-leg-specific `--use-dynamics`/`--mass` help text), which is
    exactly how these drifts happen: `cmd_ros` grew a `server_max_attempts`
    pin reading `args.max_attempts` that `p_ros` never declared, so every
    `curobo_benchmark ros` died with AttributeError. The cmd-level tests
    catch it too, but they are `importorskip('rclpy')`-gated and therefore
    silent in a non-ROS env, so assert the surface directly.

    (`--no-mpinets-locks` is covered ungated by TestMpinetsLockFlag.)
    """

    def test_ros_declares_the_pins_cmd_ros_reads(self):
        args = build_parser().parse_args(['ros'])
        assert args.max_attempts == 100
        assert args.no_reset_seed is False

    def test_max_attempts_reaches_cmd_ros(self, monkeypatch):
        pytest.importorskip('rclpy')
        from isaac_ros_cumotion_extra.benchmark import ros_runner
        from isaac_ros_cumotion_extra.benchmark.run import build_parser, cmd_ros

        seen = {}
        monkeypatch.setattr(
            ros_runner, 'run_ros',
            lambda **kw: seen.update(kw) or [],
        )
        cmd_ros(build_parser().parse_args(
            ['ros', '--dataset', 'demo', '--max-attempts', '7'],
        ))
        assert seen['server_max_attempts'] == 7


class TestSidecar:
    def test_inserts_label_before_extension(self):
        assert _sidecar('/tmp/report.json', 'core') == '/tmp/report.core.json'

    def test_no_output_means_none(self):
        assert _sidecar(None, 'core') is None
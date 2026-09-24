# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the IK and kinematics/collision benchmark parity (pure Python)."""

import math

import pytest

from isaac_ros_cumotion_extra.benchmark.compare import (
    compare_cost,
    compare_ik,
    cost_style_rows,
    ik_style_rows,
    orientation_error_deg,
)
from isaac_ros_cumotion_extra.benchmark.run import build_parser
from isaac_ros_cumotion_extra.benchmark import synthetic


def _ik_goal(name, success=True, pos_mm=0.1, ori_deg=0.05, time_ms=1.0, **extra):
    return {
        "problem_name": name,
        "scene_key": "ik_cfree",
        "capability": "ik",
        "variant": "cfree",
        "batch": 1,
        "index": 1,
        "n_goals": 1,
        "success": success,
        "time_ms": time_ms,
        "position_error_mm": pos_mm if success else None,
        "orientation_error_deg": ori_deg if success else None,
        **extra,
    }


def _cost_row(name, valid=True, pos=None, quat=None, time_ms=1.0, **extra):
    return {
        "problem_name": name,
        "scene_key": "cost",
        "capability": "cost",
        "batch": 1,
        "index": 1,
        "n_configs": 1,
        "valid": valid,
        "position_xyz": pos if pos is not None else [0.3, 0.0, 0.5],
        "quaternion_wxyz": quat if quat is not None else [1.0, 0.0, 0.0, 0.0],
        "time_ms": time_ms,
        **extra,
    }


class TestOrientationErrorDeg:
    def test_identity_is_zero(self):
        assert orientation_error_deg([1, 0, 0, 0], [1, 0, 0, 0]) == 0.0

    def test_quarter_turn_about_z(self):
        q_goal = [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
        assert math.isclose(orientation_error_deg([1, 0, 0, 0], q_goal), 90.0)

    def test_symmetric(self):
        a = [0.0, 1.0, 0.0, 0.0]
        b = [0.7071, 0.0, 0.7071, 0.0]
        assert math.isclose(
            orientation_error_deg(a, b), orientation_error_deg(b, a)
        )

    def test_half_turn(self):
        assert math.isclose(
            orientation_error_deg([1, 0, 0, 0], [0, 0, 1, 0]), 180.0
        )


class TestCompareIk:
    def test_matching_rows_are_ok(self):
        core = [_ik_goal("ik_cfree_b01_g001"), _ik_goal("ik_cfree_b01_g002")]
        ros = [_ik_goal("ik_cfree_b01_g001"), _ik_goal("ik_cfree_b01_g002")]
        report = compare_ik(core, ros)
        assert report["verdict"] == "PARITY-OK"
        assert report["matches"] == 2
        assert report["mismatches"] == 0

    def test_success_mismatch(self):
        core = [_ik_goal("ik_cfree_b01_g001", success=True)]
        ros = [_ik_goal("ik_cfree_b01_g001", success=False)]
        report = compare_ik(core, ros)
        assert report["verdict"] == "PARITY-DELTA"
        assert report["success_mismatches"] == 1
        assert report["details"]["mismatches"][0]["diff_type"] == "success_mismatch"

    def test_position_error_over_tolerance(self):
        core = [_ik_goal("ik_cfree_b01_g001", pos_mm=1.0)]
        ros = [_ik_goal("ik_cfree_b01_g001", pos_mm=0.1)]
        report = compare_ik(core, ros, position_tolerance_mm=0.5)
        assert report["verdict"] == "PARITY-DELTA"
        assert report["details"]["mismatches"][0]["diff_type"] == "position_error"
        assert report["summary"]["avg_pos_delta_mm"] == pytest.approx(0.9)

    def test_orientation_error_under_tolerance_is_ok(self):
        core = [_ik_goal("ik_cfree_b01_g001", ori_deg=0.2)]
        ros = [_ik_goal("ik_cfree_b01_g001", ori_deg=0.1)]
        assert compare_ik(core, ros)["verdict"] == "PARITY-OK"

    def test_orientation_error_over_tolerance(self):
        core = [_ik_goal("ik_cfree_b01_g001", ori_deg=2.0)]
        ros = [_ik_goal("ik_cfree_b01_g001", ori_deg=0.1)]
        report = compare_ik(core, ros)
        assert report["verdict"] == "PARITY-DELTA"
        assert report["details"]["mismatches"][0]["diff_type"] == "orientation_error"

    def test_both_failed_is_agreement(self):
        core = [_ik_goal("ik_cfree_b01_g001", success=False)]
        ros = [_ik_goal("ik_cfree_b01_g001", success=False)]
        report = compare_ik(core, ros)
        assert report["verdict"] == "PARITY-OK"
        assert report["matches"] == 1

    def test_plain_variant_rows_are_excluded(self):
        core = [
            _ik_goal("ik_cfree_b01_g001"),
            {
                "problem_name": "ik_plain_b01_g001",
                "scene_key": "ik_plain",
                "capability": "ik",
                "variant": "plain",
                "success": True,
                "time_ms": 1.0,
                "position_error_mm": 0.05,
                "orientation_error_deg": 0.01,
            },
        ]
        ros = [_ik_goal("ik_cfree_b01_g001")]
        report = compare_ik(core, ros)
        assert report["verdict"] == "PARITY-OK"  # plain is native-reference only
        assert report["core_only"] == []
        assert report["total"] == 1

    def test_core_only_rows_break_parity(self):
        core = [_ik_goal("ik_cfree_b01_g001"), _ik_goal("ik_cfree_b01_g002")]
        ros = [_ik_goal("ik_cfree_b01_g001")]
        report = compare_ik(core, ros)
        assert report["verdict"] == "PARITY-DELTA"
        assert report["core_only"] == ["ik_cfree_b01_g002"]

    def test_tolerances_are_configurable(self):
        core = [_ik_goal("ik_cfree_b01_g001", ori_deg=5.0)]
        ros = [_ik_goal("ik_cfree_b01_g001", ori_deg=0.0)]
        tight = compare_ik(core, ros, orientation_tolerance_deg=1.0)
        loose = compare_ik(core, ros, orientation_tolerance_deg=10.0)
        assert tight["verdict"] == "PARITY-DELTA"
        assert loose["verdict"] == "PARITY-OK"


class TestCompareCost:
    def test_matching_rows_are_ok(self):
        core = [_cost_row("cost_b01_g001"), _cost_row("cost_b01_g002")]
        ros = [_cost_row("cost_b01_g001"), _cost_row("cost_b01_g002")]
        report = compare_cost(core, ros)
        assert report["verdict"] == "PARITY-OK"
        assert report["matches"] == 2

    def test_validity_mismatch(self):
        core = [_cost_row("cost_b01_g001", valid=True)]
        ros = [_cost_row("cost_b01_g001", valid=False)]
        report = compare_cost(core, ros)
        assert report["verdict"] == "PARITY-DELTA"
        assert report["success_mismatches"] == 1
        assert report["details"]["mismatches"][0]["diff_type"] == "validity"

    def test_pose_delta_over_tolerance(self):
        core = [_cost_row("cost_b01_g001", pos=[0.3, 0.0, 0.5])]
        ros = [_cost_row("cost_b01_g001", pos=[0.3, 0.0, 0.505])]  # 5 mm
        report = compare_cost(core, ros, position_tolerance_mm=1.0)
        assert report["verdict"] == "PARITY-DELTA"
        assert report["details"]["mismatches"][0]["diff_type"] == "pose"
        assert report["summary"]["avg_pos_delta_mm"] == pytest.approx(5.0)

    def test_pose_delta_under_tolerance_ok(self):
        core = [_cost_row("cost_b01_g001")]
        ros = [_cost_row("cost_b01_g001", pos=[0.3001, 0.0, 0.5])]  # 0.1 mm
        assert compare_cost(core, ros, position_tolerance_mm=1.0)["verdict"] == "PARITY-OK"

    def test_orientation_pose_mismatch(self):
        core = [_cost_row("cost_b01_g001", quat=[1.0, 0.0, 0.0, 0.0])]
        ros = [_cost_row("cost_b01_g001", quat=[0.7071, 0.0, 0.0, 0.7071])]
        report = compare_cost(core, ros, orientation_tolerance_deg=1.0)
        assert report["verdict"] == "PARITY-DELTA"
        assert report["details"]["mismatches"][0]["diff_type"] == "pose_orientation"


class TestStyleRows:
    def test_ik_style_rows_success_pct_and_errors(self):
        rows = ik_style_rows(
            [
                _ik_goal("ik_cfree_b01_g001", pos_mm=0.2, ori_deg=0.05),
                _ik_goal("ik_cfree_b01_g002", pos_mm=0.8, ori_deg=0.1),
                _ik_goal("ik_cfree_b01_g003", success=False),
            ]
        )
        by_metric = dict(rows)
        assert by_metric["Success %"] == "66.67"
        assert by_metric["IK Time (ms)"] != "-"
        assert by_metric["Position Error (mm)"].startswith("mean:")
        assert by_metric["Orientation Error (deg)"].startswith("mean:")

    def test_cost_style_rows(self):
        rows = cost_style_rows(
            [
                _cost_row("cost_b01_g001", valid=True, time_ms=4.0),
                _cost_row("cost_b01_g002", valid=False, time_ms=4.0),
            ]
        )
        by_metric = dict(rows)
        assert by_metric["Valid %"] == "50.00"
        assert "mean: 4.000" in by_metric["FK Time (ms)"]
        assert "mean: 4.000" in by_metric["FK Time / Sample (ms)"]


class TestCli:
    def test_ik_subcommand_defaults(self):
        args = build_parser().parse_args(["ik", "-o", "/tmp/ik.json"])
        assert args.batch == 100
        assert args.num_batches == 5
        assert args.seed == 2
        assert args.num_seeds == 32
        assert args.variants == "both"
        assert args.show_all is False
        assert args.output == "/tmp/ik.json"

    def test_ik_core_and_ros_defaults(self):
        core = build_parser().parse_args(["ik-core"])
        assert core.num_seeds == 32
        assert core.variants == "both"
        ros = build_parser().parse_args(["ik-ros"])
        assert ros.service_timeout == 30.0
        assert ros.call_timeout == 120.0
        assert ros.no_size_cache is False
        # ik-ros has no native-only knobs
        assert not hasattr(ros, "variants")

    def test_cost_subcommands_defaults(self):
        cost = build_parser().parse_args(["cost"])
        assert cost.batch == 100
        assert cost.num_batches == 5
        assert cost.seed == 2
        assert not hasattr(cost, "num_seeds")  # FK/validate has no seed restarts
        assert build_parser().parse_args(["cost-core"]).batch == 100
        ros = build_parser().parse_args(["cost-ros"])
        assert ros.call_timeout == 120.0

    def test_compare_capability(self):
        args = build_parser().parse_args([
            "compare", "a.json", "b.json", "--capability", "ik",
            "--position-tolerance-mm", "0.25", "--orientation-tolerance-deg", "1.0",
        ])
        assert args.capability == "ik"
        assert args.position_tolerance_mm == 0.25
        assert args.orientation_tolerance_deg == 1.0
        assert build_parser().parse_args([
            "compare", "a.json", "b.json",
        ]).capability == "planning"

    def test_pose_compare_opts_on_ik(self):
        args = build_parser().parse_args([
            "ik", "--position-tolerance-mm", "2.0",
            "--orientation-tolerance-deg", "5.0", "--show-all",
        ])
        assert args.position_tolerance_mm == 2.0
        assert args.orientation_tolerance_deg == 5.0
        assert args.show_all is True

    def test_robot_config_defaults_to_envelope(self):
        args = build_parser().parse_args(["ik-core"])
        assert args.robot_config == synthetic.DEFAULT_ROBOT_CONFIG


class TestSyntheticModule:
    def test_pure_python_importable(self):
        # Module-level import must not require curobo/ROS.
        assert synthetic.WORLD_IK["cuboid"]["table"]["dims"] == [4.0, 4.0, 0.2]
        assert set(synthetic.WORLD_COST["cuboid"]) == {"table", "cube6"}
        assert synthetic.IK_WORLD_CUBOIDS == 1
        assert synthetic.COST_WORLD_CUBOIDS == 2
        assert synthetic.DEFAULT_ROBOT_CONFIG.endswith("franka.curobo.reference.yml")

    def test_box_only_guards_raise_clearly(self):
        with pytest.raises(ImportError, match="need cuRobo"):
            synthetic.load_ik_goals(batch=2, n_batches=1)
        with pytest.raises(ImportError, match="need cuRobo"):
            synthetic.load_cost_configs(batch=2, n_batches=1)


class TestCanonicalDevice:
    """Warp's wp.device_from_torch needs an explicit CUDA index (see helpers)."""

    def test_bare_cuda_maps_to_index_zero(self):
        assert synthetic._normalize_device_string("cuda") == "cuda:0"

    def test_case_insensitive(self):
        assert synthetic._normalize_device_string("CUDA") == "cuda:0"

    def test_indexed_cuda_unchanged(self):
        assert synthetic._normalize_device_string("cuda:1") == "cuda:1"

    def test_cuda_surrounded_by_whitespace(self):
        assert synthetic._normalize_device_string("  cuda  ") == "cuda:0"

    def test_cpu_unchanged(self):
        assert synthetic._normalize_device_string("cpu") == "cpu"

    def test_non_string_passthrough(self):
        assert synthetic._normalize_device_string(None) is None


class TestOrderRosParitySetup:
    """clear -> size -> add -> warmup ordering against the shared server.

    Ordering is load-bearing: cuRobo raises if the registered scene ever holds
    more cuboids than the active cache capacity, and each leg runs against a
    server whose cache a *previous* leg may have shrunk (the IK leg sizes to
    cuboid=1; with the old clear->add->size order the cost leg's 2-cuboid
    world crashed the box mid-add with "Cannot add cuboid, cache is full").
    The helper is pure Python, so the invariant is unit-testable without ROS.
    """

    @staticmethod
    def _recording_node(calls):
        class _Logger:
            def warn(self, *_args, **_kwargs):
                calls.append(("warn", {}))

        class _Node:
            def clear_world(self, **kwargs):
                calls.append(("clear", kwargs))

            def size_collision_cache(self, **kwargs):
                calls.append(("size", kwargs))

            def add_world(self, **kwargs):
                calls.append(("add", kwargs))

            def get_logger(self):
                return _Logger()

        return _Node()

    @staticmethod
    def _run(calls, **kwargs):
        synthetic.order_ros_parity_setup(
            TestOrderRosParitySetup._recording_node(calls),
            warmup=lambda **kw: calls.append(("warmup", kw)),
            **kwargs,
        )

    def test_order_is_clear_size_add_warmup(self):
        calls = []
        self._run(calls)
        assert [step for step, _ in calls] == ["clear", "size", "add", "warmup"]

    def test_no_size_cache_skips_sizing_but_keeps_order(self):
        calls = []
        self._run(calls, size_collision_cache=False)
        assert [step for step, _ in calls] == ["clear", "warn", "add", "warmup"]

    def test_timeout_reaches_every_service_call_and_warmup(self):
        calls = []
        self._run(calls, timeout=42.0)
        assert len(calls) == 4
        for step, kwargs in calls:
            assert kwargs == {"timeout": 42.0}, step
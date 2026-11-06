# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ordered Cartesian waypoint request shaping.

These pin the pure validation/layout step of ``plan_pose``'s waypoint dispatch:
which tool frame owns which rows of the flat target array, the waypoint-major
multi-tool convention, and the tolerance/count validation. No torch, no cuRobo.
"""

import pytest

from curobo_python_runtime.plan_result import _waypoint_indices_count
from curobo_python_runtime.planning import _ordered_waypoint_layout


def _pose(x):
    return [x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def test_no_waypoint_fields_is_the_legacy_path():
    assert _ordered_waypoint_layout(["tool0"], {"goalset": [_pose(0.0)]}) is None
    assert _ordered_waypoint_layout(
        ["tool0"], {"waypoints": [], "tool_waypoints": []}) is None


def test_single_tool_uses_the_first_configured_frame():
    layout = _ordered_waypoint_layout(
        ["tool0", "left", "right"], {"waypoints": [_pose(0.1), _pose(0.2)]})

    assert layout["num_waypoints"] == 2
    assert layout["frame_rows"] == {"tool0": [0, 1]}
    # Passive frames are padded from the start FK by the caller, not here.
    assert "left" not in layout["frame_rows"]


def test_single_tool_can_name_its_frame_and_declares_the_count():
    layout = _ordered_waypoint_layout(
        ["tool0", "right"],
        {"waypoints": [_pose(0.1)], "tool_frames": ["right"], "waypoint_count": 1})

    assert layout["frame_rows"] == {"right": [0]}


def test_single_tool_rejects_a_declared_count_mismatch():
    with pytest.raises(ValueError, match="waypoint_count"):
        _ordered_waypoint_layout(
            ["tool0"], {"waypoints": [_pose(0.1)], "waypoint_count": 2})


def test_single_tool_rejects_multiple_named_frames():
    with pytest.raises(ValueError, match="single-tool"):
        _ordered_waypoint_layout(
            ["tool0", "right"],
            {"waypoints": [_pose(0.1)], "tool_frames": ["tool0", "right"]})


def test_single_tool_rejects_unknown_and_empty_frame():
    with pytest.raises(ValueError, match="unknown tool frame"):
        _ordered_waypoint_layout(
            ["tool0"], {"waypoints": [_pose(0.1)], "tool_frames": ["nope"]})
    with pytest.raises(ValueError, match="configures none"):
        _ordered_waypoint_layout([], {"waypoints": [_pose(0.1)]})


def test_waypoints_never_mix_with_a_joint_target():
    with pytest.raises(ValueError, match="joint target"):
        _ordered_waypoint_layout(
            ["tool0"],
            {"waypoints": [_pose(0.1)], "joint_target": [0.0, 0.0, 0.0]})


def test_waypoints_and_tool_waypoints_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _ordered_waypoint_layout(
            ["tool0", "right"],
            {"waypoints": [_pose(0.1)], "tool_waypoints": [_pose(0.1)]})


def test_multi_tool_is_waypoint_major():
    # wp0 tool0, wp0 tool1, wp1 tool0, wp1 tool1.
    layout = _ordered_waypoint_layout(
        ["tool0", "left", "right"],
        {
            "tool_frames": ["left", "right"],
            "tool_waypoints": [_pose(0.1), _pose(0.2), _pose(0.3), _pose(0.4)],
        })

    assert layout["num_waypoints"] == 2
    assert layout["frame_rows"] == {"left": [0, 2], "right": [1, 3]}


def test_multi_tool_validates_count_and_frame_arity():
    with pytest.raises(ValueError, match="tool_frames"):
        _ordered_waypoint_layout(
            ["tool0"], {"tool_waypoints": [_pose(0.1)]})

    with pytest.raises(ValueError, match="not a multiple"):
        _ordered_waypoint_layout(
            ["tool0", "left", "right"],
            {"tool_frames": ["left", "right"], "tool_waypoints": [_pose(0.1)]})

    with pytest.raises(ValueError, match="waypoint_count=2"):
        _ordered_waypoint_layout(
            ["tool0", "left", "right"],
            {
                "tool_frames": ["left", "right"],
                "tool_waypoints": [_pose(0.1), _pose(0.2), _pose(0.3)],
                "waypoint_count": 2,
            })

    with pytest.raises(ValueError, match="duplicates"):
        _ordered_waypoint_layout(
            ["tool0", "left"],
            {
                "tool_frames": ["left", "left"],
                "tool_waypoints": [_pose(0.1), _pose(0.2)],
            })


def test_waypoint_array_must_be_k_by_seven():
    with pytest.raises(ValueError, match="K x 7"):
        _ordered_waypoint_layout(["tool0"], {"waypoints": [[0.0, 0.0, 0.0]]})


def test_waypoint_solve_reports_the_target_count():
    class Result:
        waypoint_indices = [3, 7, 11]

    assert _waypoint_indices_count(Result()) == 3
    assert _waypoint_indices_count(object()) == 0

    class Reshapeable:
        class Indices:
            def reshape(self, _shape):
                return [1, 2, 3, 4]

        waypoint_indices = Indices()

    assert _waypoint_indices_count(Reshapeable()) == 4


def test_at_most_256_ordered_waypoints():
    with pytest.raises(ValueError, match="at most 256"):
        _ordered_waypoint_layout(
            ["tool0"], {"waypoints": [_pose(0.1)] * 257})


def test_tolerances_default_to_the_message_defaults():
    layout = _ordered_waypoint_layout(["tool0"], {"waypoints": [_pose(0.1)]})

    assert layout["path_tolerance"] == pytest.approx((0.05, 0.261799388))
    assert layout["final_tolerance"] == pytest.approx((0.005, 0.05))
    assert layout["track_orientation"] is True
    assert layout["outer_iterations"] == 4


def test_tolerances_and_switches_are_forwarded():
    layout = _ordered_waypoint_layout(
        ["tool0"],
        {
            "waypoints": [_pose(0.1)],
            "path_position_tolerance": 0.04,
            "path_orientation_tolerance": 0.2,
            "final_position_tolerance": 0.004,
            "final_orientation_tolerance": 0.03,
            "track_orientation": False,
            "outer_iterations": 7,
        })

    assert layout["path_tolerance"] == pytest.approx((0.04, 0.2))
    assert layout["final_tolerance"] == pytest.approx((0.004, 0.03))
    assert layout["track_orientation"] is False
    assert layout["outer_iterations"] == 7


@pytest.mark.parametrize(
    "field,value",
    [
        ("path_position_tolerance", 0.0),
        ("path_position_tolerance", -0.1),
        ("path_orientation_tolerance", 0.0),
        ("path_orientation_tolerance", 3.2),
        ("final_position_tolerance", 0.0),
        ("final_orientation_tolerance", -0.2),
        ("outer_iterations", 0),
    ],
)
def test_invalid_tolerances_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        _ordered_waypoint_layout(
            ["tool0"], {"waypoints": [_pose(0.1)], field: value})

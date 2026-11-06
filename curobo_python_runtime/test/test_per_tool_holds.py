# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-tool Cartesian holds: wire order -> ToolPoseCriteria order.

``Goalset.tool_trajectory_constraints`` carries 6 ints per named tool in wire
order ``[theta_x, theta_y, theta_z, x, y, z]``; cuRobo's ToolPoseCriteria wants
``[x, y, z, roll, pitch, yaw]``. These tests pin the conversion and the
selection of only the tools that actually hold an axis, without importing torch.
"""

import pytest

from curobo_python_runtime.planning import _per_tool_hold_axes


def test_empty_vector_is_the_legacy_fallback():
    assert _per_tool_hold_axes(None, 2) is None
    assert _per_tool_hold_axes([], 2) is None


def test_each_group_is_converted_to_criteria_order():
    # Wire group: [theta_x, theta_y, theta_z, x, y, z].
    flat = [1, 0, 0, 1, 1, 0] + [0, 1, 0, 0, 0, 1]

    axes = _per_tool_hold_axes(flat, 2)

    # [x, y, z, roll, pitch, yaw]
    assert axes == {
        0: [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        1: [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    }


def test_groups_with_no_hold_are_not_applied():
    flat = [0, 0, 0, 0, 0, 0] + [0, 0, 0, 1, 0, 0]

    axes = _per_tool_hold_axes(flat, 2)

    assert axes == {1: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]}


def test_all_zero_groups_select_nothing_but_are_not_legacy():
    # Present-but-all-zero means "no per-tool holds"; the caller must not fall
    # back to the single trajectory_constraints vector.
    assert _per_tool_hold_axes([0] * 12, 2) == {}


def test_wrong_length_is_a_request_error():
    with pytest.raises(ValueError, match="tool_trajectory_constraints"):
        _per_tool_hold_axes([1, 0, 0, 1, 1, 0], 2)

"""Joint-order normalization tests (ROS-free).

``/joint_states`` order is publisher-defined; the curobo_server resolves
``start_pose`` position lists VERBATIM in cspace order
(``kinematics.cspace.joint_names``). The kortex sim emits the finger joint
FIRST, so readings must be reordered by name before they reach the wire.
These tests pin the canonical-order derivation and the pure reorder helper.
"""

from __future__ import annotations

import pytest

from curobo_task_constructor.core.robot_config import (
    canonical_joint_order,
    reorder_joint_vectors,
)

ARM7 = [f"joint_{i}" for i in range(1, 8)]
KORTEX_ORDER = ARM7 + ["finger_joint"]  # the cspace order


@pytest.fixture
def descriptor_path(tmp_path):
    """kortex.yaml-style descriptor: named configs only (no kinematics)."""
    p = tmp_path / "kortex.yaml"
    lines = ["named_joint_configs:", "  home:"]
    lines += [f"    joint_{i}: 0.0" for i in range(1, 8)]
    lines += ["  gripper_open:", "    finger_joint: 0.0",
              "  gripper_close:", "    finger_joint: 0.7"]
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_canonical_order_falls_back_to_named_configs(descriptor_path):
    # kortex.yaml has no kinematics section; the first-appearance union over
    # named configs reproduces the cspace order (arm ascending, finger last).
    assert canonical_joint_order(descriptor_path) == KORTEX_ORDER


def test_canonical_order_prefers_cspace_joint_names(tmp_path):
    p = tmp_path / "cspace.yml"
    p.write_text("kinematics:\n  cspace:\n    joint_names:\n"
                 "    - finger_joint\n    - joint_1\n")
    assert canonical_joint_order(str(p)) == ["finger_joint", "joint_1"]


def test_canonical_order_none_without_info(tmp_path):
    p = tmp_path / "empty.yml"
    p.write_text("name: kortex\n")
    assert canonical_joint_order(str(p)) is None


def test_reorder_finger_first_into_cspace_order():
    # The log's exact reading: finger FIRST, then joint_1..7.
    names = ["finger_joint"] + ARM7
    positions = [0.695, 0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57]
    out_names, out_pos, out_vel, out_eff = reorder_joint_vectors(
        KORTEX_ORDER, names, positions)
    assert out_names == KORTEX_ORDER
    assert out_pos == [0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57, 0.695]
    assert out_vel == []
    assert out_eff == []


def test_reorder_carries_velocity_and_effort():
    names = ["finger_joint"] + ARM7
    positions = [0.0] * 8
    velocity = [1.0] * 8
    effort = [2.0] * 8
    out_names, _, out_vel, out_eff = reorder_joint_vectors(
        KORTEX_ORDER, names, positions, velocity=velocity, effort=effort)
    assert out_names == KORTEX_ORDER
    assert out_vel == [1.0] * 8
    assert out_eff == [2.0] * 8


def test_reorder_already_canonical_is_identity():
    positions = [0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57, 0.695]
    out_names, out_pos, _, _ = reorder_joint_vectors(
        KORTEX_ORDER, KORTEX_ORDER, positions)
    assert (out_names, out_pos) == (KORTEX_ORDER, positions)


def test_reorder_missing_joint_stays_canonical_prefix():
    # A 7-DOF arm-only reading (no finger) stays a contiguous canonical
    # prefix — the server pads the trailing DOF from its current pose.
    names = ARM7
    positions = [0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57]
    out_names, out_pos, _, _ = reorder_joint_vectors(
        KORTEX_ORDER, names, positions)
    assert out_names == ARM7
    assert out_pos == positions


def test_reorder_drops_unknown_joints():
    names = ["finger_joint", "joint_1", "bogus_joint", "joint_2"]
    positions = [0.1, 0.2, 0.3, 0.4]
    out_names, out_pos, _, _ = reorder_joint_vectors(
        KORTEX_ORDER, names, positions)
    assert out_names == ["joint_1", "joint_2", "finger_joint"]
    assert out_pos == [0.2, 0.4, 0.1]
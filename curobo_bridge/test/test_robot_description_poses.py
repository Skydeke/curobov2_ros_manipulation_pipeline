# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The descriptor's named poses and grippers cross the bridge flattened.

GetJointInfo reports both maps so the RViz Joints tab can offer named poses
without parsing MoveIt's SRDF, and can still open/close a gripper whose jaws are
locked out of the cspace. A nested dict cannot cross pybind without a JSON
dependency, so ``robot_description`` flattens each map into parallel lists; this
locks down that layout.
"""

import curobo_bridge.robot_introspection as introspection

DESCRIPTOR = """
name: test_robot
base_link: base_link
curobo_config: ""

named_joint_configs:
  home:
    joint_1: 0.1
    joint_2: -0.2
  look:
    joint_1: 1.0

gripper_controllers:
  left:
    type: GripperCommand
    name: left_gripper_controller
    action_ns: gripper_cmd
    joints: [finger_joint]
    command_joint: finger_joint
    open: 0.7
    close: 0.0
    max_effort: 5.0
"""


def test_named_configs_and_grippers_are_flattened(tmp_path, monkeypatch):
    path = tmp_path / "robot.yaml"
    path.write_text(DESCRIPTOR)
    # Avoid the ament index: the test passes an absolute path anyway.
    monkeypatch.setattr(
        introspection, "_resolve_descriptor", lambda _arg: str(path))

    desc = introspection.robot_description(str(path), "")

    assert desc["named_config_names"] == ["home", "look"]
    assert desc["named_config_sizes"] == [2, 1]
    assert desc["named_config_joint_names"] == ["joint_1", "joint_2", "joint_1"]
    assert desc["named_config_positions"] == [0.1, -0.2, 1.0]

    assert desc["gripper_names"] == ["left"]
    assert desc["gripper_types"] == ["GripperCommand"]
    assert desc["gripper_controllers"] == ["left_gripper_controller"]
    assert desc["gripper_action_ns"] == ["gripper_cmd"]
    assert desc["gripper_joint_lists"] == ["finger_joint"]
    assert desc["gripper_command_joints"] == ["finger_joint"]
    assert desc["gripper_open_positions"] == [0.7]
    assert desc["gripper_close_positions"] == [0.0]
    assert desc["gripper_max_efforts"] == [5.0]
    assert desc["gripper_time_from_start"] == [2.0]

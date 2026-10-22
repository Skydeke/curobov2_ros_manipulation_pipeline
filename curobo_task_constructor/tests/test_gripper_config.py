"""Gripper-controller registry parsing (robot descriptor, ROS-free).

The general schema is a ``gripper_controllers:`` map keyed by end-effector name;
a single ``gripper_controller:`` block is the legacy one-entry form.
"""

import os
import tempfile

from curobo_task_constructor.core.robot_config import get_gripper_controllers


def _descriptor(text: str) -> str:
    handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    handle.write(text)
    handle.close()
    return handle.name


def test_named_gripper_controllers_map():
    path = _descriptor(
        """
gripper_controllers:
  left:
    type: GripperCommand
    name: left_gripper
    action_ns: gripper_cmd
    joints: [l_finger]
    command_joint: l_finger
    open: 0.0
    close: 0.04
  right:
    type: ParallelGripperCommand
    name: right_gripper
    joints: [r_finger_1, r_finger_2]
"""
    )
    try:
        g = get_gripper_controllers(path)
        assert set(g) == {"left", "right"}
        assert g["left"]["name"] == "left_gripper"
        assert g["left"]["type"] == "GripperCommand"
        assert g["left"]["command_joint"] == "l_finger"
        assert g["left"]["close"] == 0.04
        assert g["right"]["type"] == "ParallelGripperCommand"
        assert g["right"]["joints"] == ["r_finger_1", "r_finger_2"]
        assert g["right"]["command_joint"] == "r_finger_1"
    finally:
        os.unlink(path)


def test_legacy_single_gripper_block_is_the_default_entry():
    path = _descriptor(
        """
gripper_controller:
  name: kate_gripper
  action_ns: gripper_cmd
  joint: finger_1
  open: 0.0
  close: 1.3
  allow_stalling: true
"""
    )
    try:
        g = get_gripper_controllers(path)
        assert set(g) == {"default"}
        assert g["default"]["name"] == "kate_gripper"
        assert g["default"]["joints"] == ["finger_1"]
        assert g["default"]["allow_stalling"] is True
    finally:
        os.unlink(path)


def test_no_gripper_section_is_empty():
    path = _descriptor("name: robot\n")
    try:
        assert get_gripper_controllers(path) == {}
    finally:
        os.unlink(path)

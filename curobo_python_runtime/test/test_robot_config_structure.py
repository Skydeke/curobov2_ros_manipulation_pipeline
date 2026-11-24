# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Robot-YAML structure resolution.

cuRobo's loader accepts ``robot_cfg.kinematics.*``, a root-level ``kinematics``
block (``robot.curobo.yaml``) and a flat document with the kinematics keys at the
root.
"""

import curobo_python_runtime

#: A lock map with a pinned and two unpinned (null) locks.
LOCK_JOINTS = {
    "finger_joint1": 0.025,
    "cam_pan_joint": None,
    "cam_tilt_joint": None,
}
CSPACE = {"joint_names": ["joint_1"]}


def _block(data):
    return curobo_python_runtime._robot_kinematics_block(data)


def test_resolves_a_root_level_kinematics_block():
    """A descriptor keeps ``kinematics:`` at the document root."""
    block = _block(
        {"kinematics": {"lock_joints": LOCK_JOINTS, "cspace": CSPACE}}
    )
    assert block["lock_joints"] == LOCK_JOINTS
    assert block["cspace"] == CSPACE


def test_resolves_robot_cfg_kinematics():
    block = _block({"robot_cfg": {"kinematics": {"lock_joints": LOCK_JOINTS}}})
    assert block["lock_joints"] == LOCK_JOINTS


def test_resolves_a_flat_document():
    block = _block({"lock_joints": LOCK_JOINTS})
    assert block["lock_joints"] == LOCK_JOINTS


def test_splits_unpinned_locks_from_pinned_ones():
    """A null lock value marks an unpinned lock: its value comes from the state."""
    locks = _block({"kinematics": {"lock_joints": LOCK_JOINTS}})["lock_joints"]
    pinned = {k: float(v) for k, v in locks.items() if v is not None}
    unpinned = [k for k, v in locks.items() if v is None]
    assert pinned == {"finger_joint1": 0.025}
    assert unpinned == ["cam_pan_joint", "cam_tilt_joint"]

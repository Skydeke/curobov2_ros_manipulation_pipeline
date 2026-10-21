# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Robot-YAML structure resolution.

cuRobo's loader accepts ``robot_cfg.kinematics.*``, a root-level ``kinematics``
block (``kate.curobo.yaml``) and a flat document with the kinematics keys at the
root. Reading ``lock_joints`` from the wrong level once left the unpinned lock
joints (``lock_joints: {joint: null}``) unreported, so nothing ever drove them
from the joint state and the collision model kept their baked values.
"""

import curobo_bridge

#: A lock map in the shape kate.curobo.yaml uses: pinned + unpinned (null) locks.
LOCK_JOINTS = {
    "j2n6s200_joint_finger_tip_1": 0.025,
    "ptu_pan": None,
    "ptu_tilt": None,
}
CSPACE = {"joint_names": ["j2n6s200_joint_1"]}


def _block(data):
    return curobo_bridge._robot_kinematics_block(data)


def test_resolves_a_root_level_kinematics_block():
    """kate.curobo.yaml keeps ``kinematics:`` at the document root."""
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
    assert pinned == {"j2n6s200_joint_finger_tip_1": 0.025}
    assert unpinned == ["ptu_pan", "ptu_tilt"]

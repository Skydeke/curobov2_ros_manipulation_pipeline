# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""curobo_bridge - the ONLY place cuRobo/torch are called (design section 6). ROS-free module with plain functions. C++ holds opaque integer handles into the
registry below; no torch tensor, no CUDA pointer and no cuRobo object crosses the
boundary."""

from __future__ import annotations

from typing import Dict, Optional

def joint_limits(handle: int):
    """cspace names + position limits + locked joints (PORT: get_joint_info)."""
    import numpy as np

    entry = _REGISTRY[handle]
    kin = entry["kin"]
    limits = kin.get_joint_limits()
    positions = limits.position.detach().cpu().numpy().astype(np.float32)  # [2, dof]
    locked = entry.get("lock_joints", {})
    return {
        "positions": positions,
        "joint_names": list(limits.joint_names),
        "locked_joint_names": list(locked.keys()),
        "locked_positions": [float(v) for v in locked.values()],
        # lock_joints written {joint: null}: out of the cspace but not pinned.
        "unpinned_lock_joint_names": list(entry.get("unpinned_lock_joints", [])),
    }
def fk(handle: int, q):
    """Tool poses for [N, dof] joint positions (PORT: fk_services._compute_poses)."""
    import numpy as np
    import torch
    from curobo.types import JointState

    entry = _REGISTRY[handle]
    kin = entry["kin"]
    device = torch.device("cuda", torch.cuda.current_device())

    q_t = torch.tensor(np.asarray(q, dtype=np.float32), dtype=torch.float32, device=device)
    js = JointState.from_position(q_t, joint_names=kin.joint_names)
    kin_state = kin.compute_kinematics(js)

    # [B, H=1, L, 3/4]; first tool frame. Quaternion is wxyz (cuRobo order).
    positions = kin_state.tool_poses.position[:, 0, 0, :].detach().cpu().numpy()
    quaternions = kin_state.tool_poses.quaternion[:, 0, 0, :].detach().cpu().numpy()

    out = np.zeros((positions.shape[0], 1, 7), dtype=np.float32)
    out[:, 0, 0:3] = positions
    out[:, 0, 3:7] = quaternions
    return out

from .runtime import _REGISTRY

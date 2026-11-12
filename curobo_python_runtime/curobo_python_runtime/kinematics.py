# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""curobo_python_runtime - the ONLY place cuRobo/torch are called (design section 6). ROS-free module with plain functions. C++ holds opaque integer handles into the
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
    names = list(limits.joint_names)
    # A continuous joint is periodic. The descriptor override and the URDF joint
    # type are authoritative; the +/-2*pi limits are only its coordinate window.
    continuous_names = set(entry.get("continuous_joint_names", []))
    continuous_names.update(entry.get("urdf_continuous_joint_names", []))
    return {
        "positions": positions,
        "joint_names": names,
        "continuous": [1 if name in continuous_names else 0 for name in names],
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


def _visualisation_kinematics(entry):
    """All-link Kinematics used only for visualisation.

    The planning model tracks just the configured tool frames, so it cannot
    answer "where is every link". This second instance is built lazily from the
    same resolved config with every link as a tool frame.
    """
    kin = entry.get("viz_kin")
    if kin is None:
        from curobo.kinematics import Kinematics, KinematicsCfg

        kin_cfg = entry["kin_cfg"]
        link_names = list(kin_cfg.kinematics_config.link_name_to_idx_map.keys())
        viz_cfg = KinematicsCfg.from_robot_yaml_file(
            entry["resolved_config"], tool_frames=link_names)
        kin = Kinematics(viz_cfg)
        entry["viz_kin"] = kin
        entry["viz_link_names"] = list(kin.tool_frames)
    return kin


def link_poses(handle: int, q, link_names):
    """Poses of `link_names` for [N, dof] joint positions.

    Returns a dict with `poses` [N, L, 7] (x, y, z, qw, qx, qy, qz) and the
    link names actually returned (unknown names are dropped).
    """
    import numpy as np
    import torch
    from curobo.types import JointState

    entry = _REGISTRY[handle]
    kin = _visualisation_kinematics(entry)
    known = set(kin.tool_frames)
    used = [name for name in link_names if name in known]

    q_t = torch.tensor(np.asarray(q, dtype=np.float32), dtype=torch.float32,
                       device=torch.device("cuda", torch.cuda.current_device()))
    state = kin.compute_kinematics(JointState.from_position(q_t, joint_names=kin.joint_names))
    index = {name: i for i, name in enumerate(kin.tool_frames)}
    out = np.zeros((q_t.shape[0], len(used), 7), dtype=np.float32)
    for li, name in enumerate(used):
        i = index[name]
        out[:, li, 0:3] = state.tool_poses.position[:, 0, i, :].detach().cpu().numpy()
        out[:, li, 3:7] = state.tool_poses.quaternion[:, 0, i, :].detach().cpu().numpy()
    return {"poses": out, "link_names": used}

from .runtime import _REGISTRY

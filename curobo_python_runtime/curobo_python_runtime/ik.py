# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""curobo_python_runtime - the ONLY place cuRobo/torch are called (design section 6). ROS-free module with plain functions. C++ holds opaque integer handles into the
registry below; no torch tensor, no CUDA pointer and no cuRobo object crosses the
boundary."""

from __future__ import annotations

from typing import Dict, Optional

def _build_ik_solver(entry, batch_size, num_seeds, self_collision_check,
                     use_lm_seed, use_cuda_graph):
    """Create the standalone IK solver (PORT: ConfigWrapperIK.build).

    ``num_seeds`` runs that many parallel optimisation trajectories per pose, so
    GPU memory grows with ``batch_size x num_seeds``; the flags bake into the
    solver buffers / CUDA graph at build time.
    """
    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    from curobo.scene import Scene

    batch_size = max(1, int(batch_size))
    num_seeds = max(1, int(num_seeds))
    # Build against an EMPTY scene, never the live one: after a cache shrink the
    # live world may not fit the new cache, and constructing against it raises
    # "Cannot load N cuboids" (observed live, killing a benchmark rebuild). The
    # real world is pushed right after via world_push (PORT: the build tail's
    # update_world(primitives_only_scene())).
    scene = Scene() if self_collision_check else None
    cfg = InverseKinematicsCfg.create(
        robot=entry["robot_cfg"],
        scene_model=scene,
        num_seeds=num_seeds,
        position_tolerance=0.005,
        orientation_tolerance=0.05,
        self_collision_check=bool(self_collision_check),
        collision_cache=entry.get("collision_cache") if self_collision_check else None,
        use_cuda_graph=bool(use_cuda_graph),
        max_batch_size=batch_size,
    )
    # use_lm_seed is a dataclass default (True), not a create() kwarg.
    cfg.use_lm_seed = bool(use_lm_seed)
    solver = InverseKinematics(cfg)
    entry["ik"] = solver
    entry["ik_batch_size"] = batch_size
    entry["ik_num_seeds"] = num_seeds
    entry["ik_self_collision_check"] = bool(self_collision_check)
    entry["ik_use_lm_seed"] = bool(use_lm_seed)
    entry["ik_use_cuda_graph"] = bool(use_cuda_graph)
    return solver
def _ik_solver(entry):
    """The registered IK solver, built on demand if ik_config never ran.

    IKServices always warms up (ik_config) before the first solve, so this
    fallback only covers a direct bridge call; it reuses the cached build
    tunables when present, else the node-param defaults (32 seeds, collisions
    on, CUDA graphs on).
    """
    solver = entry.get("ik")
    if solver is None:
        solver = _build_ik_solver(
            entry,
            batch_size=max(1, int(entry.get("ik_batch_size") or 1)),
            num_seeds=max(1, int(entry.get("ik_num_seeds") or 32)),
            self_collision_check=bool(entry.get("ik_self_collision_check", True)),
            use_lm_seed=bool(entry.get("ik_use_lm_seed", True)),
            use_cuda_graph=bool(entry.get("ik_use_cuda_graph", True)),
        )
    return solver
def ik_config(handle: int, batch_size: int, num_seeds: int,
              self_collision_check: bool = True, use_lm_seed: bool = True,
              use_cuda_graph: bool = True) -> None:
    """(Re)build the IK solver (PORT: ConfigWrapperIK.build)."""
    entry = _REGISTRY[handle]
    _build_ik_solver(
        entry, batch_size, num_seeds, self_collision_check, use_lm_seed,
        use_cuda_graph)
    # PORT: the build tail re-pushes the live world when collision is on, so a
    # rebuild (batch-size or param change) does not leave the solver sceneless.
    if self_collision_check:
        world_push(handle)
def ik(handle: int, poses, tool_frame: str = ""):
    """Solve [N,7] tool poses (PORT: ik_services._solve / _solve_locked)."""
    import numpy as np
    import torch
    from curobo.types import Pose, GoalToolPose
    from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria

    entry = _REGISTRY[handle]
    solver = _ik_solver(entry)
    device = torch.device("cuda", torch.cuda.current_device())
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"ik expects [N,7] poses, got {poses.shape}")

    pose = Pose(
        position=torch.tensor(poses[:, 0:3], dtype=torch.float32, device=device),
        quaternion=torch.tensor(poses[:, 3:7], dtype=torch.float32, device=device),
    )
    tool_frame = tool_frame or solver.kinematics.tool_frames[0]
    if tool_frame not in solver.kinematics.tool_frames:
        raise ValueError(
            f"unknown tool frame {tool_frame!r}; configured: "
            f"{list(solver.kinematics.tool_frames)}")

    # cuRobo requires a goal for every configured tool frame. The frames this
    # request does not target get a dummy pose at the model's default FK; their
    # pose cost is disabled for the solve, so they do not constrain the target
    # arm. Without this a bimanual model rejects a single-frame IK goal with
    # "Ordered link names [...] not a subset of [...]".
    pose_dict = {tool_frame: pose}
    passive_frames = [
        frame for frame in solver.kinematics.tool_frames if frame != tool_frame
    ]
    if passive_frames:
        default_state = solver.kinematics.default_joint_state.unsqueeze(0)
        fk = solver.kinematics.compute_kinematics(default_state)
        for frame in passive_frames:
            here = fk.tool_poses.get_link_pose(frame)
            pose_dict[frame] = Pose(
                position=here.position.repeat(len(poses), 1).contiguous(),
                quaternion=here.quaternion.repeat(len(poses), 1).contiguous())
        solver.update_tool_pose_criteria(
            {frame: ToolPoseCriteria.disabled() for frame in passive_frames})

    goal = GoalToolPose.from_poses(
        pose_dict, ordered_tool_frames=list(solver.kinematics.tool_frames))
    try:
        result = solver.solve_pose(goal_tool_poses=goal)
    finally:
        if passive_frames:
            solver.update_tool_pose_criteria(
                {frame: ToolPoseCriteria() for frame in passive_frames})

    solution = result.solution.detach().cpu().numpy()
    success = result.success.detach().cpu().numpy()
    return {
        "success": bool(np.any(success)),
        "positions": solution.astype(np.float32),
        "valid": success.reshape(-1).astype(int).tolist(),
        "joint_names": list(solver.kinematics.joint_names),
    }


def ik_multi(handle: int, poses, tool_frames):
    """Solve one pose per named tool frame in a SINGLE IK solve.

    This is the multi-tool counterpart of :func:`ik`: every listed frame is an
    active goal at the same time, so the returned joint state puts all of them
    at their requested poses (e.g. both arms of a bimanual robot). Frames not
    listed are unconstrained, exactly like ``ik``'s passive frames.
    """
    import numpy as np
    import torch
    from curobo.types import Pose, GoalToolPose
    from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria

    entry = _REGISTRY[handle]
    solver = _ik_solver(entry)
    device = torch.device("cuda", torch.cuda.current_device())
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"ik_multi expects [N,7] poses, got {poses.shape}")
    tool_frames = [str(frame) for frame in (tool_frames or [])]
    if len(tool_frames) != poses.shape[0]:
        raise ValueError(
            f"ik_multi tool_frames/poses length mismatch: {len(tool_frames)} "
            f"frame(s), {poses.shape[0]} pose(s)")
    if len(set(tool_frames)) != len(tool_frames):
        raise ValueError(f"ik_multi tool_frames contains duplicates: {tool_frames}")
    unknown = [f for f in tool_frames if f not in solver.kinematics.tool_frames]
    if unknown:
        raise ValueError(
            f"unknown tool frame(s) {unknown}; configured: "
            f"{list(solver.kinematics.tool_frames)}")

    pose_dict = {}
    for i, frame in enumerate(tool_frames):
        pose_dict[frame] = Pose(
            position=torch.tensor(
                poses[i:i + 1, 0:3], dtype=torch.float32, device=device),
            quaternion=torch.tensor(
                poses[i:i + 1, 3:7], dtype=torch.float32, device=device))

    passive_frames = [
        frame for frame in solver.kinematics.tool_frames if frame not in pose_dict
    ]
    if passive_frames:
        default_state = solver.kinematics.default_joint_state.unsqueeze(0)
        fk = solver.kinematics.compute_kinematics(default_state)
        for frame in passive_frames:
            here = fk.tool_poses.get_link_pose(frame)
            pose_dict[frame] = Pose(
                position=here.position.repeat(len(tool_frames), 1).contiguous(),
                quaternion=here.quaternion.repeat(len(tool_frames), 1).contiguous())
        solver.update_tool_pose_criteria(
            {frame: ToolPoseCriteria.disabled() for frame in passive_frames})

    goal = GoalToolPose.from_poses(
        pose_dict, ordered_tool_frames=list(solver.kinematics.tool_frames))
    try:
        result = solver.solve_pose(goal_tool_poses=goal)
    finally:
        if passive_frames:
            solver.update_tool_pose_criteria(
                {frame: ToolPoseCriteria() for frame in passive_frames})

    solution = result.solution.detach().cpu().numpy()
    success = result.success.detach().cpu().numpy()
    return {
        "success": bool(np.any(success)),
        "positions": solution.astype(np.float32),
        "valid": success.reshape(-1).astype(int).tolist(),
        "joint_names": list(solver.kinematics.joint_names),
    }

from .runtime import _REGISTRY
from .world_push import world_push

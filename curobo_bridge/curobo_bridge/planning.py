# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""curobo_bridge - the ONLY place cuRobo/torch are called (design section 6). ROS-free module with plain functions. C++ holds opaque integer handles into the
registry below; no torch tensor, no CUDA pointer and no cuRobo object crosses the
boundary."""

from __future__ import annotations

from typing import Dict, Optional

def collision_cache_bucket_counts(handle: int) -> dict:
    """PORT: obstacle_manager._solver_bucket_counts - (cuboid, mesh) slots the
    current scene occupies under the default 'cuboid' collision mode: cuboids
    stay cuboids, sphere/capsule/cylinder ride the cuboid bucket, meshes stay
    meshes (attached, disabled obstacles still occupy slots).
    """
    entry = _REGISTRY[handle]
    scene = entry.get("scene")
    if scene is None:
        return {"cuboid": 0, "mesh": 0}
    n_cuboid = len(getattr(scene, "cuboid", None) or [])
    n_mesh = len(getattr(scene, "mesh", None) or [])
    converted = sum(
        len(getattr(scene, bucket, None) or [])
        for bucket in ("sphere", "capsule", "cylinder")
    )
    return {"cuboid": int(n_cuboid + converted), "mesh": int(n_mesh)}
def render_plan_image(position, velocity, acceleration, names, dt):
    """PORT: plan_plot.render_plan_plot - multi-panel joint-trajectory RGB image.

    position/velocity/acceleration: [T, D] float32 arrays (velocity/acceleration
    may be empty). Returns an HxWx3 uint8 array, or None when matplotlib/PIL are
    unavailable (the caller then skips publishing, as the Python does).
    """
    import numpy as np

    try:
        import io

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image as PILImage
    except Exception:
        return None

    pos = np.asarray(position, dtype=float)
    if pos.ndim == 1:
        pos = pos.reshape(1, -1)
    n, dof = pos.shape

    vel = np.asarray(velocity, dtype=float) if velocity is not None and len(velocity) else None
    if vel is not None:
        if vel.ndim == 1:
            vel = vel.reshape(1, -1)
        vel = vel[:n, :dof]
    else:
        vel = np.zeros((n, dof))
    d = max(float(dt), 1e-6)
    if not np.any(vel):
        vel = np.diff(pos, axis=0, prepend=pos[:1]) / d
    acc = np.diff(vel, axis=0, prepend=vel[:1]) / d
    jrk = np.diff(acc, axis=0, prepend=acc[:1]) / d

    names = list(names or [f"J{i}" for i in range(dof)])
    plot_data = [(pos, "Position (rad)"), (vel, "Velocity (rad/s)"),
                 (acc, "Accel (rad/s^2)"), (jrk, "Jerk (rad/s^3)")]
    n_plots = len(plot_data)
    fig, axes = plt.subplots(n_plots, 1, figsize=(6, 2 * n_plots), dpi=100, sharex=True)
    if n_plots == 1:
        axes = [axes]
    t = np.arange(n) * d
    for ax, (data, ylabel) in zip(axes, plot_data):
        for j in range(dof):
            label = names[j] if j < len(names) else f"J{j}"
            if len(label) > 10:
                label = label[:8] + ".."
            ax.plot(t, data[:, j], linewidth=1.2, label=label)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    axes[0].legend(loc="upper right", fontsize=7, ncol=2)
    axes[-1].set_xlabel("Time (s)", fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return np.array(PILImage.open(buf).convert("RGB"))
def torch_synchronize() -> None:
    """PORT: unified_planner_node.torch_sync_enabled - CPU/GPU ordering bridge.

    Off by default (torch_sync param): a per-frame synchronize() blocks the
    worker on the GPU and starves the viz timers.
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
def reset_seed(handle: int) -> None:
    """PORT: MotionPlanner.reset_seed (reset_seed_per_plan)."""
    entry = _REGISTRY[handle]
    planner = entry.get("motion_planner")
    if planner is not None:
        planner.reset_seed()
def _clamp_to_limits(entry, positions):
    """Snap a joint vector into the cspace position limits.

    Gazebo (and some hardware drivers) report an at-rest joint a hair OUTSIDE
    its [lower, upper] bound - e.g. -3e-9 when the limit is exactly 0.0, which
    is common for robots whose zero pose sits on a joint stop (OpenArm's
    joint4 and lift). cuRobo's cspace constraint then rejects the *start*
    state outright ("Start or End state in collision") and no planner will
    move, even though the state is physically fine. Clamp, don't tolerate: the
    limit check is an exact `lo - q <= 0` test.
    """
    import numpy as np
    try:
        limits = entry["kin"].get_joint_limits().position.detach().cpu().numpy()
    except Exception:
        return positions
    limits = limits.astype(np.float32)
    if limits.shape != (2, positions.shape[-1]):
        return positions
    lo, hi = limits[0], limits[1]
    eps = np.minimum(1e-6, 0.25 * np.maximum(hi - lo, 1e-9))
    return np.clip(positions, lo + eps, hi - eps).astype(positions.dtype)


def plan_cspace(handle: int, req) -> dict:
    """Joint-space plan (PORT: joint_space_planner._plan_trajectory)."""
    import numpy as np
    import torch
    from curobo.types import JointState

    entry = _REGISTRY[handle]
    planner = entry["motion_planner"]
    device = torch.device("cuda", torch.cuda.current_device())

    start = torch.tensor(
        [list(_clamp_to_limits(entry, np.asarray(req["start"], dtype=np.float32)))],
        dtype=torch.float32, device=device)
    goal = torch.tensor(
        [list(_clamp_to_limits(entry, np.asarray(req["joint_target"], dtype=np.float32)))],
        dtype=torch.float32, device=device)
    names = planner.kinematics.joint_names
    start_state = JointState.from_position(start, joint_names=names)
    goal_state = JointState.from_position(goal, joint_names=names)
    result = planner.plan_cspace(
        goal_state, start_state,
        max_attempts=int(req.get("max_attempts", 1)),
        enable_graph_attempt=int(req.get("enable_graph_attempt", 1)))
    return _plan_outcome(planner, result)
def _passive_joint_mask(planner, active_index, passive_indices):
    """Joints that belong to a passive frame and not to the active one."""
    import torch

    try:
        kin = planner.kinematics
        cfg = kin.config.kinematics_config
        matrix = cfg.joint_affects_endeffector
        dof = len(kin.joint_names)
        n_frames = len(planner.tool_frames)
        if matrix is None or matrix.numel() != dof * n_frames:
            return None
        affects = matrix.view(dof, n_frames).bool()
        passive_any = affects[:, list(passive_indices)].any(dim=1)
        active_any = affects[:, active_index]
        mask = passive_any & torch.logical_not(active_any)
        if not bool(mask.any()):
            return None
        return mask.to(dtype=torch.float32, device=matrix.device)
    except Exception:
        return None


def _apply_passive_hold(planner, mask, weight=1.0e4):
    """Enable the target-cspace cost on the passive arm's joints only."""
    saved = []
    seen = set()
    for rollout in planner.trajopt_solver.get_all_rollout_instances():
        for cost in rollout.get_cost_component_by_name('target_cspace_dist'):
            if id(cost) in seen:
                continue
            seen.add(id(cost))
            cfg = cost.config
            saved.append((
                cost,
                cfg.weight.clone(),
                cost._weight.clone(),
                cfg.terminal_dof_weight.clone(),
                cfg.non_terminal_dof_weight.clone(),
                getattr(cfg, 'only_terminal_cost', None),
            ))
            cfg.weight.fill_(float(weight))
            cost._weight.fill_(float(weight))
            cfg.terminal_dof_weight.copy_(mask)
            cfg.non_terminal_dof_weight.copy_(mask)
            if hasattr(cfg, 'only_terminal_cost'):
                cfg.only_terminal_cost = False
    return saved


def _restore_passive_hold(saved):
    if not saved:
        return
    for (cost, weight, live_weight, terminal, non_terminal, only_terminal) in saved:
        cfg = cost.config
        cfg.weight.copy_(weight)
        cost._weight.copy_(live_weight)
        cfg.terminal_dof_weight.copy_(terminal)
        cfg.non_terminal_dof_weight.copy_(non_terminal)
        if only_terminal is not None and hasattr(cfg, 'only_terminal_cost'):
            cfg.only_terminal_cost = bool(only_terminal)


def _per_tool_hold_axes(tool_trajectory_constraints, num_tools):
    """Convert flat per-tool holds into ToolPoseCriteria axis weights.

    `tool_trajectory_constraints` is ``Goalset.tool_trajectory_constraints``:
    6 ints per tool, groups in wire order
    ``[theta_x, theta_y, theta_z, x, y, z]``. Returns
    ``{tool_index: [x, y, z, roll, pitch, yaw]}`` for every tool whose group
    holds at least one axis, or ``None`` when the vector is absent/empty (the
    legacy single-tool behaviour). A non-empty vector of the wrong length is a
    request error, not a silent fallback.
    """
    flat = [int(c) for c in (tool_trajectory_constraints or [])]
    if not flat:
        return None
    if len(flat) != 6 * num_tools:
        raise ValueError(
            f"tool_trajectory_constraints has {len(flat)} value(s) for "
            f"{num_tools} tool frame(s) (expected {6 * num_tools})")
    axes_by_tool = {}
    for i in range(num_tools):
        tx, ty, tz, x, y, z = flat[6 * i:6 * i + 6]
        group = [
            1.0 if x == 1 else 0.0, 1.0 if y == 1 else 0.0,
            1.0 if z == 1 else 0.0, 1.0 if tx == 1 else 0.0,
            1.0 if ty == 1 else 0.0, 1.0 if tz == 1 else 0.0,
        ]
        if any(group):
            axes_by_tool[i] = group
    return axes_by_tool


def plan_pose(handle: int, req) -> dict:
    """Pose plan (PORT: single_planner._plan_trajectory / classic_planner)."""
    import numpy as np
    import torch
    from curobo.types import JointState, Pose, GoalToolPose

    entry = _REGISTRY[handle]
    planner = entry["motion_planner"]
    device = torch.device("cuda", torch.cuda.current_device())

    start = torch.tensor(
        [list(_clamp_to_limits(entry, np.asarray(req["start"], dtype=np.float32)))],
        dtype=torch.float32, device=device)
    start_state = JointState.from_position(start, joint_names=planner.kinematics.joint_names)

    tool_frames_arg = list(req.get("tool_frames") or [])
    tool_poses_arg = req.get("tool_poses")
    goalset_arg = req.get("goalset")
    if tool_frames_arg:
        if tool_poses_arg is None or len(tool_poses_arg) == 0:
            raise ValueError("plan_pose got tool_frames without tool_poses")
        if len(set(tool_frames_arg)) != len(tool_frames_arg):
            raise ValueError(f"tool_frames contains duplicates: {tool_frames_arg}")
        unknown = [f for f in tool_frames_arg if f not in planner.tool_frames]
        if unknown:
            raise ValueError(
                f"unknown tool frame(s) {unknown}; configured: "
                f"{list(planner.tool_frames)}")
        goalset = np.asarray(tool_poses_arg, dtype=np.float32).reshape(-1, 7)
        if goalset.shape[0] != len(tool_frames_arg):
            raise ValueError(
                f"tool_frames/tool_poses length mismatch: {len(tool_frames_arg)} "
                f"frame(s), {goalset.shape[0]} pose(s)")
        num_goalset = 1
    else:
        if goalset_arg is None or len(goalset_arg) == 0:
            raise ValueError(
                "plan_pose requires a 'goalset' of [x,y,z,qw,qx,qy,qz] poses; got no "
                "poses (a joint-space goal must go through plan_cspace / the "
                "joint-space planner).")
        goalset = np.asarray(goalset_arg, dtype=np.float32).reshape(-1, 7)
        num_goalset = len(goalset)

    def make_pose(rows):
        return Pose(
            position=torch.tensor(rows[:, 0:3], dtype=torch.float32, device=device),
            quaternion=torch.tensor(rows[:, 3:7], dtype=torch.float32, device=device))

    from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria

    if tool_frames_arg:
        pose_dict = {
            frame: make_pose(goalset[i:i + 1]) for i, frame in enumerate(tool_frames_arg)
        }
    else:
        pose_dict = {planner.tool_frames[0]: make_pose(goalset)}
    constraint_frame = next(iter(pose_dict))

    # cuRobo requires a goal for every configured tool link. Frames the caller
    # did not target are held at their start joints (cspace cost + seed clamp);
    # they stay planable but do not move. Their tool pose has no cost, so the
    # shared lift can carry the fixed arm.
    passive_frames = [f for f in planner.tool_frames if f not in pose_dict]
    passive_indices = [planner.tool_frames.index(f) for f in passive_frames]
    active_index = planner.tool_frames.index(constraint_frame)
    hold_mask = None
    if passive_frames:
        hold_mask = _passive_joint_mask(planner, active_index, passive_indices)
    passive_criteria = {}
    if passive_frames:
        fk = planner.kinematics.compute_kinematics(start_state)
        for frame in passive_frames:
            here = fk.tool_poses.get_link_pose(frame)
            pose_dict[frame] = Pose(
                position=here.position.repeat(num_goalset, 1).contiguous(),
                quaternion=here.quaternion.repeat(num_goalset, 1).contiguous())
            passive_criteria[frame] = (
                ToolPoseCriteria.disabled() if hold_mask is not None
                else ToolPoseCriteria(
                    terminal_pose_axes_weight_factor=[1.0] * 6,
                    non_terminal_pose_axes_weight_factor=[1.0] * 6))

    if tool_frames_arg or passive_frames:
        goal = GoalToolPose.from_poses(
            pose_dict, ordered_tool_frames=list(planner.tool_frames),
            num_goalset=num_goalset)
    else:
        goal = GoalToolPose.from_poses(pose_dict, num_goalset=num_goalset)
    if passive_criteria:
        planner.update_tool_pose_criteria(passive_criteria)

    # Per-tool holds for a multi-tool request: flat 6 * N, parallel to
    # tool_frames. Absent/empty falls back to the single-vector path below;
    # when present it owns every tool's criteria (never both for a frame).
    per_tool_axes = (
        _per_tool_hold_axes(
            req.get("tool_trajectory_constraints"), len(tool_frames_arg))
        if tool_frames_arg else None)
    per_tool_criteria = None
    if per_tool_axes is not None:
        per_tool_criteria = {
            tool_frames_arg[i]: ToolPoseCriteria(
                non_terminal_pose_axes_weight_factor=axes)
            for i, axes in per_tool_axes.items()
        }
        if per_tool_criteria:
            planner.update_tool_pose_criteria(per_tool_criteria)

    # Goalset.trajectory_constraints: [theta_x, theta_y, theta_z, x, y, z]
    # -> ToolPoseCriteria [x, y, z, roll, pitch, yaw]. Legacy single-tool path:
    # applies to the first named tool only when no per-tool vector was sent.
    constraints = [int(c) for c in (req.get("trajectory_constraints") or [])]
    applied = (per_tool_axes is None and len(constraints) == 6
               and any(c == 1 for c in constraints))
    if applied:
        tx, ty, tz, cx, cy, cz = (1.0 if c == 1 else 0.0 for c in constraints)
        planner.update_tool_pose_criteria({
            constraint_frame: ToolPoseCriteria(
                non_terminal_pose_axes_weight_factor=[cx, cy, cz, tx, ty, tz])})
    hold_saved = None
    if hold_mask is not None:
        hold_saved = _apply_passive_hold(planner, hold_mask)
    try:
        result = planner.plan_pose(
            goal, start_state,
            max_attempts=int(req.get("max_attempts", 1)),
            enable_graph_attempt=int(req.get("enable_graph_attempt", 1)),
            goal_state=start_state if hold_mask is not None else None,
            seed_hold_mask=hold_mask.bool() if hold_mask is not None else None)
    finally:
        _restore_passive_hold(hold_saved)
        if per_tool_criteria:
            planner.update_tool_pose_criteria(
                {frame: ToolPoseCriteria() for frame in per_tool_criteria})
        elif applied:
            planner.update_tool_pose_criteria(
                {constraint_frame: ToolPoseCriteria()})
        if passive_criteria:
            planner.update_tool_pose_criteria(
                {frame: ToolPoseCriteria() for frame in passive_criteria})
    return _plan_outcome(planner, result)

from .runtime import _REGISTRY
from .plan_result import _plan_outcome

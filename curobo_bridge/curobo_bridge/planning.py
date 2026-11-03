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
def _active_tool_frame(planner):
    """Pick the configured active tool frame.

    A bimanual cuRobo config declares one tool frame per arm, but a request
    carries ONE pose goal. ``GoalToolPose`` still needs an entry for every
    configured frame, so the other frames are filled with a dummy pose.

    The active frame is the FIRST configured tool frame. A caller that wants
    the other arm launches with that arm's frame first (the OpenArm launch
    rewrites ``tool_frames`` for ``arm:=right``). Do NOT infer it from the goal
    pose: a grasp near the robot midline can then flip between arms from cycle
    to cycle, while the gripper/perception side is still configured for one arm.

    ``joint_affects_endeffector`` mobility is used only to detect a reduced
    planner (one arm locked out): then the mobile frame is active.

    Returns ``(frames, active_index, counts)``; ``counts`` is empty when the
    model does not expose the matrix.
    """
    frames = list(planner.tool_frames)
    if not frames:
        return frames, 0, []
    try:
        kin = planner.kinematics
        cfg = kin.config.kinematics_config
        matrix = cfg.joint_affects_endeffector
        dof = len(kin.joint_names)
        if matrix is not None and matrix.numel() == dof * len(frames):
            counts = [int(c) for c in matrix.view(dof, len(frames)).sum(dim=0).tolist()]
            if len(set(counts)) > 1:
                return frames, max(range(len(frames)), key=lambda i: (counts[i], -i)), counts
            return frames, 0, counts
    except Exception:
        pass
    return frames, 0, []


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

    goalset_arg = req.get("goalset")
    if goalset_arg is None or len(goalset_arg) == 0:
        raise ValueError(
            "plan_pose requires a 'goalset' of [x,y,z,qw,qx,qy,qz] poses; got no "
            "poses (a joint-space goal must go through plan_cspace / the "
            "joint-space planner).")
    goalset = np.asarray(goalset_arg, dtype=np.float32)
    pose = Pose(
        position=torch.tensor(goalset[:, 0:3], dtype=torch.float32, device=device),
        quaternion=torch.tensor(goalset[:, 3:7], dtype=torch.float32, device=device))
    frames, active_index, _mobility = _active_tool_frame(planner)
    tool_frame = frames[active_index] if frames else planner.tool_frames[0]
    from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria

    # cuRobo's IK requires a goal entry for EVERY configured tool link
    # (`GoalToolPose.reorder_links` raises "Ordered link names [...] not a
    # subset of [...]" otherwise), but only the active frame is targeted. The
    # other frames get a dummy (current FK) goal.
    #
    # The passive arm stays in the cspace (both arms must remain planable),
    # so it is held at its start joints by a target-cspace cost + a seed clamp
    # (see below); its tool pose therefore must have NO pose cost: the arm is
    # fixed relative to the base, and the shared lift is free to carry it.
    # Without a joint hold mask (unusual tree), fall back to the old pose pin.
    #
    # Single-arm robots (TIAGo, KATE) keep the exact old goal, one frame and
    # all.
    passive_frames = [f for i, f in enumerate(frames) if i != active_index]
    passive_indices = [i for i in range(len(frames)) if i != active_index]
    passive_criteria = {}
    hold_mask = None
    hold_saved = []
    if passive_frames:
        hold_mask = _passive_joint_mask(planner, active_index, passive_indices)
        fk = planner.kinematics.compute_kinematics(start_state)
        pose_dict = {tool_frame: pose}
        for frame in passive_frames:
            here = fk.tool_poses.get_link_pose(frame)
            pose_dict[frame] = Pose(
                position=here.position.repeat(len(goalset), 1).contiguous(),
                quaternion=here.quaternion.repeat(len(goalset), 1).contiguous())
            # If the passive joints can be held, its tool pose must be free:
            # the arm is fixed relative to the base, so the shared lift carries
            # it, and a world-frame pose cost would forbid that lift motion.
            # Without a hold mask (unusual tree), keep the old pose pin.
            passive_criteria[frame] = (
                ToolPoseCriteria.disabled() if hold_mask is not None
                else ToolPoseCriteria(
                    terminal_pose_axes_weight_factor=[1.0] * 6,
                    non_terminal_pose_axes_weight_factor=[1.0] * 6))
        goal = GoalToolPose.from_poses(
            pose_dict, ordered_tool_frames=frames, num_goalset=len(goalset))
        planner.update_tool_pose_criteria(passive_criteria)
    else:
        goal = GoalToolPose.from_poses({tool_frame: pose}, num_goalset=len(goalset))

    # PORT: single_planner._apply_pose_constraints - hold the requested Cartesian
    # axes along the whole path (Goalset.trajectory_constraints order
    # [theta_x, theta_y, theta_z, x, y, z] -> ToolPoseCriteria order
    # [x, y, z, roll, pitch, yaw]). This is what makes a CartesianPath solve
    # trace the straight line; without it the solve curves and the achieved
    # fraction check rejects it. The MotionPlanner is SHARED, so the default
    # (unconstrained) criteria are restored right after the solve.
    constraints = [int(c) for c in (req.get("trajectory_constraints") or [])]
    applied = len(constraints) == 6 and any(c == 1 for c in constraints)
    if applied:
        tx, ty, tz, cx, cy, cz = (1.0 if c == 1 else 0.0 for c in constraints)
        planner.update_tool_pose_criteria({
            tool_frame: ToolPoseCriteria(
                non_terminal_pose_axes_weight_factor=[cx, cy, cz, tx, ty, tz])})
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
        if applied:
            planner.update_tool_pose_criteria({tool_frame: ToolPoseCriteria()})
        if passive_frames:
            planner.update_tool_pose_criteria(
                {frame: ToolPoseCriteria() for frame in passive_frames})
    return _plan_outcome(planner, result)

from .runtime import _REGISTRY
from .plan_result import _plan_outcome

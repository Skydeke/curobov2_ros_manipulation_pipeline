# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""curobo_python_runtime - the ONLY place cuRobo/torch are called (design section 6). ROS-free module with plain functions. C++ holds opaque integer handles into the
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


def _waypoint_pose_array(value, name):
    """Validate a waypoint field and return K rows of 7 floats.

    Deliberately torch/numpy-free so a malformed request is rejected before any
    GPU import (the tensors are built in _run_ordered_waypoints).
    """
    if value is None:
        return []
    try:
        rows = list(value)
    except TypeError as exc:
        raise ValueError(
            f"{name} must be a list of [x,y,z,qw,qx,qy,qz] poses") from exc
    if not rows:
        return []
    # Accept a single flat 7-vector as one pose, like the old reshape(-1, 7).
    if len(rows) == 7 and not hasattr(rows[0], "__len__"):
        rows = [rows]
    poses = []
    for row in rows:
        try:
            values = [float(v) for v in row]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must be a list of [x,y,z,qw,qx,qy,qz] poses") from exc
        if len(values) != 7:
            raise ValueError(
                f"{name} must be a K x 7 array of [x,y,z,qw,qx,qy,qz] poses, "
                f"got {len(values)} value(s) in a row")
        poses.append(values)
    return poses


def _ordered_waypoint_options(req):
    """Validate the waypoint tolerances / switches shared by both layouts.

    cuRobo's validate_pose_tolerance requires a finite positive position
    tolerance (metres) and an orientation tolerance in ``(0, pi]``. Reject a bad
    request here with a field name instead of letting the solver raise.
    """
    import math

    def tolerance(name, default):
        try:
            value = float(req.get(name, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a number") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return value

    path_position = tolerance("path_position_tolerance", 0.05)
    path_orientation = tolerance("path_orientation_tolerance", 0.261799388)
    final_position = tolerance("final_position_tolerance", 0.005)
    final_orientation = tolerance("final_orientation_tolerance", 0.05)
    for name, value in (
            ("path_orientation_tolerance", path_orientation),
            ("final_orientation_tolerance", final_orientation)):
        if value > math.pi:
            raise ValueError(f"{name} must be in (0, pi] radians")

    try:
        outer_iterations = int(req.get("outer_iterations", 4))
    except (TypeError, ValueError) as exc:
        raise ValueError("outer_iterations must be an integer >= 1") from exc
    if outer_iterations < 1:
        raise ValueError("outer_iterations must be an integer >= 1")

    return {
        "path_tolerance": (path_position, path_orientation),
        "final_tolerance": (final_position, final_orientation),
        "track_orientation": bool(req.get("track_orientation", True)),
        "outer_iterations": outer_iterations,
    }


def _ordered_waypoint_layout(configured_frames, req):
    """Validate an ordered waypoint request.

    Returns ``None`` when the request has no waypoint path, otherwise a dict:

      source           flat K x 7 (single tool) or K*N x 7 (waypoint-major) rows
      num_waypoints    K
      frame_rows       {tool frame: row indices into `source`}, targeted frames only
      path_tolerance   (metres, radians)
      final_tolerance  (metres, radians)
      track_orientation / outer_iterations

    Passive configured frames are deliberately absent from `frame_rows`; the
    caller pads them by repeating the start FK, exactly like the legacy
    multi-tool path holds a non-targeted arm.
    """
    waypoints = req.get("waypoints")
    tool_waypoints = req.get("tool_waypoints")
    has_waypoints = waypoints is not None and len(waypoints) > 0
    has_tool_waypoints = tool_waypoints is not None and len(tool_waypoints) > 0
    if not has_waypoints and not has_tool_waypoints:
        return None
    if req.get("joint_target"):
        raise ValueError(
            "a waypoint path cannot mix with a joint target "
            "(target_joint_positions)")
    if has_waypoints and has_tool_waypoints:
        raise ValueError(
            "waypoints and tool_waypoints are mutually exclusive; send one "
            "single-tool path or one multi-tool waypoint-major path")

    frames = [str(frame) for frame in (req.get("tool_frames") or [])]
    configured = [str(frame) for frame in (configured_frames or [])]
    try:
        declared = int(req.get("waypoint_count", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("waypoint_count must be a non-negative integer") from exc
    if declared < 0:
        raise ValueError("waypoint_count must be a non-negative integer")

    if has_waypoints:
        if len(frames) > 1:
            raise ValueError(
                "waypoints is a single-tool path but tool_frames names "
                f"{len(frames)} frames; use tool_waypoints for multiple tools")
        if frames:
            frame = frames[0]
            if frame not in configured:
                raise ValueError(
                    f"unknown tool frame {frame!r}; configured: {configured}")
        elif configured:
            frame = configured[0]
        else:
            raise ValueError(
                "waypoints needs a tool frame but the robot configures none")
        source = _waypoint_pose_array(waypoints, "waypoints")
        count = len(source)
        if count < 1:
            raise ValueError("waypoints must contain at least one pose")
        if declared and declared != count:
            raise ValueError(
                f"waypoint_count={declared} but waypoints holds {count} pose(s)")
        active = {frame: list(range(count))}
    else:
        if not frames:
            raise ValueError(
                "tool_waypoints requires tool_frames (one name per tracked tool)")
        if len(set(frames)) != len(frames):
            raise ValueError(f"tool_frames contains duplicates: {frames}")
        unknown = [frame for frame in frames if frame not in configured]
        if unknown:
            raise ValueError(
                f"unknown tool frame(s) {unknown}; configured: {configured}")
        source = _waypoint_pose_array(tool_waypoints, "tool_waypoints")
        num_tools = len(frames)
        if declared:
            expected = declared * num_tools
            if len(source) != expected:
                raise ValueError(
                    f"tool_waypoints holds {len(source)} pose(s) for "
                    f"waypoint_count={declared} and {num_tools} tool frame(s) "
                    f"(expected {expected})")
            count = declared
        else:
            if len(source) % num_tools != 0:
                raise ValueError(
                    f"tool_waypoints holds {len(source)} pose(s), which is "
                    f"not a multiple of {num_tools} tool frame(s)")
            count = len(source) // num_tools
            if count < 1:
                raise ValueError(
                    "tool_waypoints must contain at least one pose per tool")
        # Waypoint-major: wp0 tool0, wp0 tool1, ..., wp1 tool0, ...
        active = {
            frame: [wp * num_tools + i for wp in range(count)]
            for i, frame in enumerate(frames)
        }

    # cuRobo's interpolate_pose_waypoints accepts 1..256 ordered targets.
    if count > 256:
        raise ValueError(
            f"an ordered waypoint path supports at most 256 waypoints, got {count}")

    layout = {
        "source": source,
        "num_waypoints": count,
        "frame_rows": active,
        "configured_frames": configured,
    }
    layout.update(_ordered_waypoint_options(req))
    return layout


def _run_ordered_waypoints(planner, start_state, layout, req):
    """Build GoalToolPose [1,K,L,1,3/4] and call MotionPlanner.plan_pose_waypoints.

    Every configured tool frame must have a target in the goal (cuRobo compares
    the frame lists exactly). Frames the caller did not target are padded with
    the start FK repeated K times, so they ride along unchanged.
    """
    import numpy as np
    import torch
    from curobo.types import Pose, GoalToolPose

    device = start_state.position.device
    source = np.asarray(layout["source"], dtype=np.float32).reshape(-1, 7)
    frame_rows = layout["frame_rows"]
    count = int(layout["num_waypoints"])

    def make_pose(rows):
        rows = np.asarray(rows, dtype=np.float32).reshape(-1, 7)
        return Pose(
            position=torch.tensor(rows[:, 0:3], dtype=torch.float32, device=device),
            quaternion=torch.tensor(rows[:, 3:7], dtype=torch.float32, device=device))

    passive = [frame for frame in planner.tool_frames if frame not in frame_rows]
    fk = planner.kinematics.compute_kinematics(start_state) if passive else None

    positions = []
    quaternions = []
    for frame in planner.tool_frames:
        if frame in frame_rows:
            pose = make_pose(source[frame_rows[frame]])
        else:
            here = fk.tool_poses.get_link_pose(frame)
            pose = Pose(
                position=here.position.repeat(count, 1).contiguous(),
                quaternion=here.quaternion.repeat(count, 1).contiguous())
        positions.append(pose.position)
        quaternions.append(pose.quaternion)

    goal = GoalToolPose(
        tool_frames=list(planner.tool_frames),
        # [1, K, L, 1, 3/4]: one ordered path, one goalset.
        position=torch.stack(positions, dim=1).unsqueeze(0).unsqueeze(3).contiguous(),
        quaternion=torch.stack(quaternions, dim=1).unsqueeze(0).unsqueeze(3).contiguous(),
    )

    # A passive arm (configured frame not in the path) is held exactly like the
    # legacy multi-tool pose path: seed clamp + target-cspace cost on the joints
    # that drive it and not the active arm.
    active_indices = [
        planner.tool_frames.index(frame) for frame in frame_rows
        if frame in planner.tool_frames]
    passive_indices = [planner.tool_frames.index(frame) for frame in passive]
    hold_mask = None
    if len(active_indices) == 1 and passive_indices:
        hold_mask = _passive_joint_mask(planner, active_indices[0], passive_indices)
    hold_saved = _apply_passive_hold(planner, hold_mask) if hold_mask is not None else None
    try:
        result = planner.plan_pose_waypoints(
            goal, start_state,
            path_tolerance=layout["path_tolerance"],
            final_tolerance=layout["final_tolerance"],
            max_attempts=max(1, int(req.get("max_attempts", 1))),
            track_orientation=layout["track_orientation"],
            outer_iterations=layout["outer_iterations"],
            goal_state=start_state if hold_mask is not None else None,
            seed_hold_mask=hold_mask.bool() if hold_mask is not None else None)
    finally:
        _restore_passive_hold(hold_saved)
    return _plan_outcome(planner, result)


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

    # Ordered Cartesian waypoint path (MULTIPOINT / Cartesian Waypoint): EVERY
    # target is visited in order along one trajectory, unlike the candidate
    # `goalset` below where the solver picks one. Legacy plan_pose is untouched
    # when no waypoint field is present.
    waypoint_layout = _ordered_waypoint_layout(planner.tool_frames, req)
    if waypoint_layout is not None:
        return _run_ordered_waypoints(planner, start_state, waypoint_layout, req)

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

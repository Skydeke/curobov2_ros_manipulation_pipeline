# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""curobo_bridge - the ONLY place cuRobo/torch are called (design section 6). ROS-free module with plain functions. C++ holds opaque integer handles into the
registry below; no torch tensor, no CUDA pointer and no cuRobo object crosses the
boundary."""

from __future__ import annotations

from typing import Dict, Optional

def mpc_solve_goal_state(handle: int, goal_raw, seed_q):
    """PORT: mpc_planner.MPCController._solve_goal_state (multi-seed IK anchor)."""
    import numpy as np
    import torch
    from curobo.types import GoalToolPose, JointState, Pose

    entry = _REGISTRY[handle]
    solver = _mpc(entry)
    device = torch.device("cuda", torch.cuda.current_device())
    try:
        ik = _mpc_ensure_ik_solver(entry)
        raw = list(goal_raw)
        pose = Pose(
            position=torch.tensor([raw[0:3]], dtype=torch.float32, device=device),
            quaternion=torch.tensor([raw[3:7]], dtype=torch.float32, device=device),
        )
        goal = GoalToolPose.from_poses({ik.kinematics.tool_frames[0]: pose})
        seed_state = None
        seed = np.asarray(seed_q, dtype=np.float32).reshape(1, -1)
        if seed.size:
            seed_state = JointState.from_position(
                torch.tensor(seed[:, :solver.action_dim], dtype=torch.float32, device=device),
                joint_names=solver.joint_names,
            )
        result = ik.solve_pose(goal_tool_poses=goal, current_state=seed_state)
        if not bool(result.success.reshape(-1)[0].item()):
            return {"success": False, "solution": None}
        sol = result.solution.reshape(-1, solver.action_dim)[0:1].clone()
        return {"success": True, "solution": sol.detach().cpu().numpy().astype(np.float32)}
    except Exception:
        return {"success": False, "solution": None}
def _mpc_apply_goal(entry, solver, goal, raw, seed_q):
    """PORT: mpc_planner.MPCController._apply_goal."""
    if solver.update_goal_tool_poses(goal, run_ik=False):
        return True
    solved = mpc_solve_goal_state(entry, raw, seed_q if seed_q is not None else [])
    if solved["success"]:
        import numpy as np
        import torch
        from curobo.types import JointState

        device = torch.device("cuda", torch.cuda.current_device())
        solver.update_goal_tool_poses(goal, run_ik=False)
        sol = torch.tensor(
            np.asarray(solved["solution"], dtype=np.float32), dtype=torch.float32,
            device=device)
        solver.update_goal_state(
            JointState.from_position(sol, joint_names=solver.joint_names))
        solver.enable_joint_position_tracking()
        return True
    solver.update_goal_tool_poses(goal, run_ik=False)
    return False
def mpc_update_goal_pose(handle: int, goal_raw, run_ik: bool) -> bool:
    """PORT: mpc_planner.MPCController._apply_goal -> update_goal_tool_poses."""
    entry = _REGISTRY[handle]
    solver = _mpc(entry)
    raw = list(goal_raw)
    goal = _mpc_goal_tool_pose(solver, raw)
    entry["mpc_goal"] = raw
    applied = bool(solver.update_goal_tool_poses(goal, run_ik=bool(run_ik)))
    entry["mpc_goal_applied"] = applied
    return applied
def mpc_update_goal_state(handle: int, q) -> None:
    """PORT: mpc_planner.MPCController._apply_goal joint branch."""
    import numpy as np
    import torch
    from curobo.types import JointState

    entry = _REGISTRY[handle]
    solver = _mpc(entry)
    device = torch.device("cuda", torch.cuda.current_device())
    sol = torch.tensor(
        np.asarray(q, dtype=np.float32).reshape(1, -1), dtype=torch.float32, device=device)
    solver.update_goal_state(JointState.from_position(sol, joint_names=solver.joint_names))
    solver.enable_joint_position_tracking()
def _mpc_state_from_dict(state):
    import numpy as np
    import torch
    from curobo.types import JointState

    device = torch.device("cuda", torch.cuda.current_device())
    pos = torch.tensor(
        np.asarray(state["position"], dtype=np.float32).reshape(1, -1),
        dtype=torch.float32, device=device)
    js = JointState.from_position(pos, joint_names=None)
    for key in ("velocity", "acceleration"):
        if state.get(key) is not None:
            val = np.asarray(state[key], dtype=np.float32)
            if val.size:
                setattr(js, key, torch.tensor(
                    val.reshape(1, -1), dtype=torch.float32, device=device))
    return js
def _mpc_js_to_dict(js):
    """{position, velocity, acceleration} as float32 [n, dof] lists (or None)."""
    import numpy as np

    def _arr(t):
        if t is None:
            return None
        arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
        while arr.ndim > 2:
            arr = arr[0]
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr.astype(np.float32)

    if js is None:
        return None
    return {
        "position": _arr(js.position),
        "velocity": _arr(getattr(js, "velocity", None)),
        "acceleration": _arr(getattr(js, "acceleration", None)),
    }
def mpc_cost_breakdown(handle: int) -> dict:
    """PORT: mpc_planner.MPCController._cost_breakdown.

    Per-term COST and CONSTRAINT values (horizon-summed) for the last solve,
    read from the metrics rollout's already-populated _current_metrics (never
    rebatched - see the crash-safety note in the Python source).
    """
    entry = _REGISTRY[handle]
    solver = _mpc(entry)
    try:
        m = solver.trajectory_execution_manager.get_current_metrics()
        if m is None:
            return {}
        cc = m.costs_and_constraints
        out = {}
        for name, val in zip(cc.costs.names, cc.costs.values):
            if name == "tool_pose":
                out["cost_tool_pose_pos"] = float(val[0, :, 0].sum().item())
                out["cost_tool_pose_orient"] = float(val[0, :, 1].sum().item())
            else:
                out[f"cost_{name}"] = float(val.sum().item())
        for name, val in zip(cc.constraints.names, cc.constraints.values):
            if name == "cspace":
                out["con_cspace_bound"] = float(val.sum().item())
            else:
                out[f"con_{name}"] = float(val.sum().item())
        return out
    except Exception:
        return {}
def mpc_step(handle: int, state, paced: bool):
    """PORT: mpc_planner.MPCController.step / step_paced."""
    import torch
    from curobo.types import JointState

    entry = _REGISTRY[handle]
    solver = _mpc(entry)
    current_state = _mpc_state_from_dict(state)
    # joint names were required for the solver to label the state.
    current_state = JointState.from_position(
        current_state.position, joint_names=solver.joint_names)

    if paced:
        result = solver.optimize_action_sequence(current_state)
        action = getattr(result, "action_sequence", None)
    else:
        result = solver.optimize_next_action(current_state)
        action = getattr(result, "next_action", None)

    ee_position = None
    try:
        kin = solver.compute_kinematics(current_state)
        ee_position = [
            float(v) for v in kin.tool_poses.position.reshape(-1, 3)[0].cpu().tolist()]
    except Exception:
        ee_position = None

    if action is None or getattr(action, "position", None) is None:
        action_dict = None
    else:
        action_dict = _mpc_js_to_dict(action.clone())

    horizon_position = None
    try:
        state_seq = getattr(result, "robot_state_sequence", None)
        js = getattr(state_seq, "joint_state", None) if state_seq is not None else None
        if js is not None and getattr(js, "position", None) is not None:
            pos = js.position
            while pos.ndim > 2:
                pos = pos[0]
            if pos.ndim == 1:
                pos = pos.unsqueeze(0)
            names = list(getattr(js, "joint_names", None) or [])
            try:
                from curobo.types import JointState
                probe = JointState.from_position(pos, joint_names=names)
                active = solver.kinematics.get_active_js(probe)
                pos = active.position
            except Exception:
                pass
            horizon_position = pos.detach().cpu().numpy().astype("float32")
    except Exception:
        horizon_position = None

    entry["mpc_last_result"] = result
    # PORT: mpc_planner._publish_costs reads result.position_error /
    # rotation_error for the deprecated pose_* fields of MpcCosts.
    def _first(tensor):
        try:
            flat = tensor.reshape(-1)
            return float(flat[0].item()) if flat.numel() else None
        except Exception:
            return None

    return {
        "action": action_dict,
        "ee_position": ee_position,
        "horizon_position": horizon_position,
        "pose_pos_err_m": _first(getattr(result, "position_error", None)),
        "pose_rot_err_rad": _first(getattr(result, "rotation_error", None)),
    }
def mpc_metrics(handle: int) -> str:
    """PORT: mpc_planner.MPCController._cost_breakdown (as a JSON dict)."""
    import json

    entry = _REGISTRY[handle]
    solver = _mpc(entry)
    try:
        m = solver.trajectory_execution_manager.get_current_metrics()
        if m is None:
            return "{}"
        cc = m.costs_and_constraints
        out = {}
        for name, val in zip(cc.costs.names, cc.costs.values):
            if name == "tool_pose":
                out["cost_tool_pose_pos"] = float(val[0, :, 0].sum().item())
                out["cost_tool_pose_orient"] = float(val[0, :, 1].sum().item())
            else:
                out[f"cost_{name}"] = float(val.sum().item())
        for name, val in zip(cc.constraints.names, cc.constraints.values):
            if name == "cspace":
                out["con_cspace_bound"] = float(val.sum().item())
            else:
                out[f"con_{name}"] = float(val.sum().item())
        return json.dumps(out)
    except Exception:
        return "{}"

from .runtime import _REGISTRY
from .ik import ik
from .mpc import _mpc, _mpc_ensure_ik_solver, _mpc_goal_tool_pose

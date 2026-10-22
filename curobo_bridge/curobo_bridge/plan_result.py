# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""curobo_bridge - the ONLY place cuRobo/torch are called (design section 6). ROS-free module with plain functions. C++ holds opaque integer handles into the
registry below; no torch tensor, no CUDA pointer and no cuRobo object crosses the
boundary."""

from __future__ import annotations

from typing import Dict, Optional

def _as_2d_list(tensor):
    import numpy as np

    if tensor is None:
        return None
    arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr.astype("float32")
def _flat_list(value):
    """PORT: single_planner.SinglePlanner._flat_values (best-effort flatten)."""
    if value is None:
        return []
    try:
        if hasattr(value, "reshape"):
            value = value.reshape(-1)
        if hasattr(value, "tolist"):
            return list(value.tolist())
        return list(value)
    except Exception:
        return []
def _select_goal_index(result) -> int:
    """PORT: single_planner._select_goal_index - the winner candidate index of a
    plan_pose goalset solve (0 when goalset_index is None, -1 on failure)."""
    if result is None:
        return -1
    succ = getattr(result, "success", False)
    if hasattr(succ, "item") and not bool(succ.item()):
        return -1
    if not succ:
        return -1
    idx = getattr(result, "goalset_index", None)
    if idx is None:
        return 0
    try:
        flat = idx.view(-1) if hasattr(idx, "view") else idx
        return int(flat[0].item())
    except (TypeError, ValueError, IndexError):
        return -1
def _select_seed_index(result) -> int:
    """PORT: single_planner._select_seed_index - the winner restart (seed) index
    (seed_rank row 0 argmin, else seed_cost argmin, else 0; -1 on failure)."""
    if result is None:
        return -1
    succ = getattr(result, "success", None)
    if succ is None:
        return -1
    try:
        ok = bool(succ.item()) if hasattr(succ, "item") else bool(succ)
        if not ok:
            return -1
    except Exception:
        return -1
    rank = getattr(result, "seed_rank", None)
    if rank is not None:
        try:
            row0 = rank[0]
            winner = row0.argmin().item() if hasattr(row0, "argmin") else row0[0]
            return int(winner)
        except Exception:
            pass
    cost = getattr(result, "seed_cost", None)
    if cost is not None:
        try:
            row0 = cost[0]
            winner = row0.argmin().item() if hasattr(row0, "argmin") else row0
            return int(winner)
        except Exception:
            pass
    return 0
def considered_rows(result, segment_i=0, fallback_candidate=None):
    """PORT: single_planner.SinglePlanner._segment_considered_rows.

    One row per seed of a solved segment. The whole-task-only fields
    (waypoint_cost/path_length/clearance) stay 0, as in the Python.
    """
    if result is None:
        return []
    success = _flat_list(getattr(result, "success", None))
    costs = _flat_list(getattr(result, "seed_cost", None))
    perr = _flat_list(getattr(result, "position_error", None))
    gidx = getattr(result, "goalset_index", None)
    gidx_flat = _flat_list(gidx)
    n_links = 1
    if gidx is not None and getattr(gidx, "ndim", 0) >= 3:
        try:
            n_links = int(gidx.shape[-1])
        except Exception:
            n_links = 1
    # PORT: single_planner._segment_considered_rows(row, 0,
    # self._selected_goal_indexes[0]) - the fallback candidate for a failed seed
    # is the WINNER candidate index, not 0.
    if fallback_candidate is None:
        fallback_candidate = _select_goal_index(result)
    n_seeds = len(success) if success else 1
    if n_seeds <= 0:
        n_seeds = 1
    solve_time = getattr(result, "solve_time", 0.0)
    rows = []
    for s in range(n_seeds):
        ok = bool(success[s]) if s < len(success) else False
        cost = float(costs[s]) if costs and s < len(costs) else 0.0
        err = float(perr[s]) if perr and s < len(perr) else 0.0
        cand = 0
        if gidx_flat and s * n_links < len(gidx_flat):
            try:
                cand = int(gidx_flat[s * n_links])
            except (TypeError, ValueError):
                cand = 0
        elif not ok:
            cand = max(0, int(fallback_candidate))
        rows.append({
            "problem": 0, "segment": segment_i, "goalset_candidate": cand, "seed": s,
            "success": ok, "cost": cost, "waypoint_cost": 0.0, "path_length": 0.0,
            "clearance": 0.0, "max_waypoint_error": err, "solve_time": float(solve_time),
        })
    return rows
def _position_error_mm(result):
    """Winner candidate's solver-reported position error in mm (NaN if absent)."""
    import math

    try:
        perr = getattr(result, "position_error", None)
        if perr is None:
            return math.nan
        flat = perr.reshape(-1) if hasattr(perr, "reshape") else None
        if flat is None:
            return math.nan
        return float(flat[0].item() if hasattr(flat[0], "item") else flat[0]) * 1000.0
    except Exception:
        return math.nan
def _knot_motion_time(planner, result):
    """PORT: single_planner.SinglePlanner._knot_motion_time.

    Native measures the KNOT solution (``js_solution.dt * (H_knots - offset)``,
    ``offset = 2 * interpolation_steps`` for bspline, else 1) - not the
    interpolated waypoints. NaN when unavailable.
    """
    import math

    try:
        from curobo._src.types.control_space import ControlSpace

        js = getattr(result, "js_solution", None)
        if js is None:
            return math.nan
        h = int(js.position.shape[-2])
        dt = getattr(js, "dt", None)
        if dt is None:
            return math.nan
        if hasattr(dt, "detach"):
            dt = dt.detach()
        if hasattr(dt, "flatten"):
            flat = dt.flatten()
            dt = flat[0] if len(flat) else None
        if hasattr(dt, "item"):
            dt = dt.item()
        dt = float(dt)
        interp_steps = None
        try:
            interp_steps = getattr(planner.trajopt_solver, "interpolation_steps", None)
        except Exception:
            interp_steps = None
        if getattr(js, "control_space", None) in ControlSpace.bspline_types():
            if interp_steps is None or int(interp_steps) <= 0:
                return math.nan
            offset = 2 * int(interp_steps)
        else:
            offset = 1
        return max(0.0, dt * (h - offset))
    except Exception:
        return math.nan
def _plan_outcome(planner, result):
    """PORT: single_planner result shaping + unified_planner._build_planning_stats."""
    import math

    names = list(planner.kinematics.joint_names)
    base = {
        "success": False, "feasible": False, "status": "no solution",
        "positions": None, "velocity": None, "acceleration": None,
        "joint_names": names, "goalset_index": -1, "seed_index": -1,
        "total_time": math.nan,
        # PORT: the solve was attempted with >= 1 seed, so the candidate tally
        # and the (unsolved) segment row are present even on failure.
        "problems": 1, "candidates_generated": 1, "candidates_solved": 0,
        "candidates_pruned": 0, "waypoints_planned": 0,
        "solver_time_s": math.nan, "motion_time_s": math.nan,
        "position_error_mm": math.nan, "waypoint_status": [0],
    }
    if result is None:
        return base

    trajectory = result.get_interpolated_plan()
    # PORT: unified_planner._result_trajectory uses traj.joint_names verbatim
    # (the plan carries the full joint set, fingers included).
    traj_names = getattr(trajectory, "joint_names", None)
    if traj_names:
        names = list(traj_names)

    success = result.success
    try:
        success = bool(success.reshape(-1)[0].item()) if hasattr(success, "reshape") \
            else bool(success)
    except Exception:
        success = False

    positions = _as_2d_list(trajectory.position)
    velocity = _as_2d_list(getattr(trajectory, "velocity", None))
    acceleration = _as_2d_list(getattr(trajectory, "acceleration", None))

    # PORT: single_planner._tally_candidates
    try:
        generated = max(1, int(result.num_seeds))
    except Exception:
        generated = 1
    solved = 1 if success else 0

    # PORT: unified_planner._solver_time_s
    solver_time = math.nan
    try:
        solver_time = float(result.solve_time)
    except Exception:
        try:
            solver_time = float(result.solve_time.detach().flatten()[0].item())
        except Exception:
            try:
                solver_time = float(result.total_time)
            except Exception:
                solver_time = math.nan

    return {
        "success": success,
        "feasible": bool(getattr(result, "feasible", success)),
        "status": str(getattr(result, "status", None) or ""),
        "positions": positions,
        "velocity": velocity,
        "acceleration": acceleration,
        "joint_names": names,
        # PORT: single_planner._selected_goal_indexes / _selected_seed_index
        # (the WINNER candidate / seed, not a constant 0).
        "goalset_index": max(0, _select_goal_index(result)),
        "seed_index": max(0, _select_seed_index(result)),
        "total_time": solver_time,
        "problems": 1,
        "candidates_generated": generated,
        "candidates_solved": solved,
        "candidates_pruned": 0,
        "waypoints_planned": 1,
        "solver_time_s": solver_time,
        "motion_time_s": _knot_motion_time(planner, result),
        # PORT: single_planner._winner_position_error_mm - the winner candidate's
        # solver-reported position error. The Python reads it from the considered
        # rows' max_waypoint_error (== result.position_error per seed); the
        # native leg's own metric is ``result.position_error * 1000`` and that is
        # what the service always exposes here.
        "position_error_mm": _position_error_mm(result),
        "waypoint_status": [1] if success else [],
        "considered": considered_rows(result),
    }

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""curobo_bridge - the ONLY place cuRobo/torch are called (design section 6). ROS-free module with plain functions. C++ holds opaque integer handles into the
registry below; no torch tensor, no CUDA pointer and no cuRobo object crosses the
boundary."""

from __future__ import annotations

from typing import Dict, Optional

def retarget_build(handle: int, cfg_json: str) -> None:
    """Build the MotionRetargeter (PORT: retarget_controller.build_solver)."""
    import json

    from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
    from curobo.motion_retargeter import MotionRetargeter, MotionRetargeterCfg
    from curobo.scene import Scene

    entry = _REGISTRY[handle]
    cfg = json.loads(cfg_json)
    tool = cfg.get("tool_frame") or entry["kin"].tool_frames[0]
    w_pos = float(cfg.get("position_weight", 1.0))
    w_rot = float(cfg.get("orientation_weight", 1.0))
    solver = MotionRetargeter(MotionRetargeterCfg.create(
        robot=entry["robot_cfg"],
        tool_pose_criteria={
            tool: ToolPoseCriteria.track_position_and_orientation(
                xyz=[w_pos, w_pos, w_pos], rpy=[w_rot, w_rot, w_rot],
            )
        },
        scene_model=Scene(),
        use_mpc=bool(cfg.get("use_mpc", False)),
        num_seeds_global=64,
        self_collision_check=True,
    ))
    entry["retargeter"] = solver
    entry["retarget_tool"] = tool
def _retargeter(entry):
    solver = entry.get("retargeter")
    if solver is None:
        raise ValueError("Retargeter not built; call retarget_build first")
    return solver
def retarget_reset(handle: int) -> None:
    """PORT: retarget_controller.setup -> solver.reset()."""
    _retargeter(_REGISTRY[handle]).reset()
def retarget_step(handle: int, goal_raw):
    """PORT: retarget_controller.step -> solve_frame + FK position error input."""
    import numpy as np

    from curobo.types import GoalToolPose, Pose

    entry = _REGISTRY[handle]
    solver = _retargeter(entry)
    raw = list(goal_raw)
    pose = Pose.from_list(raw)
    tool = entry.get("retarget_tool") or solver.kinematics.tool_frames[0]
    goal = GoalToolPose.from_poses(
        {tool: pose},
        ordered_tool_frames=list(solver.kinematics.tool_frames),
        num_goalset=1,
    )
    result = solver.solve_frame(goal)
    js = result.joint_state
    ee_position = None
    try:
        kin = solver.kinematics.compute_kinematics(js)
        ee_position = [
            float(v) for v in kin.tool_poses.position.reshape(-1, 3)[0].cpu().tolist()]
    except Exception:
        ee_position = None
    return {"action": _mpc_js_to_dict(js), "ee_position": ee_position}
def retarget_load_scene(handle: int, scene_json: str) -> None:
    """PORT: retarget_controller.update_world (IK solvers' update_world)."""
    import json

    from curobo.scene import Scene

    entry = _REGISTRY[handle]
    solver = _retargeter(entry)
    specs = json.loads(scene_json) if scene_json else []
    scene = Scene()
    for spec in specs:
        primitive = _to_scene(spec)
        attr = {0: "cuboid", 1: "sphere", 2: "capsule", 3: "cylinder", 4: "mesh"}[
            int(spec["type"])]
        bucket = getattr(scene, attr)
        if bucket is None:
            setattr(scene, attr, [primitive])
        else:
            bucket.append(primitive)
    global_ik = getattr(solver, "_global_ik_solver", None)
    if global_ik is not None:
        global_ik.update_world(scene)
    local_ik = getattr(solver, "_local_ik_solver", None)
    if local_ik is not None:
        local_ik.update_world(scene)

from .runtime import _REGISTRY
from .scene import _to_scene
from .mpc_step import _mpc_js_to_dict

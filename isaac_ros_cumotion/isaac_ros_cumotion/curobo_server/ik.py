"""IK handler for ComputeIK service.

Uses the shared ``MotionPlanner.ik_solver`` (built at startup) to solve IK
for one or more Cartesian goal poses in a single GPU batch.

Goal layout (see ``ComputeIK.srv``):
  ``num_goalset == 0`` → legacy: ``use_goalset`` governs the shape
    ``use_goalset=False`` (default): batch=N, goalset=1
    ``use_goalset=True``:             batch=1, goalset=N
  ``num_goalset  > 0`` → explicit: ``batch = len(goal_poses) / num_goalset``
    ``seed_states``: 1 (shared) or ``batch_size`` entries
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

import torch

from curobo.types import GoalToolPose, JointState, Pose

from isaac_ros_cumotion_interfaces.srv import ComputeIK
from sensor_msgs.msg import JointState as RosJointState

from .context import CuroboContext
from .conversions import cu_joint_state_to_ros
from .world import sync_world


def _ros_joint_states_to_cu(
    ros_states: List[RosJointState],
    dof: int,
    device: torch.device,
) -> Optional[JointState]:
    if not ros_states or len(ros_states) == 0:
        return None
    pos_list = []
    for js in ros_states:
        p = list(js.position)
        if len(p) < dof:
            p.extend([0.0] * (dof - len(p)))
        pos_list.append(p[:dof])
    pos_tensor = torch.tensor(pos_list, dtype=torch.float32, device=device)
    return JointState.from_position(pos_tensor)


def _extract_per_goal_results(
    ik_result, chunk_batch: int, num_goalset: int,
) -> Tuple[List[bool], List[RosJointState], List[float], List[float]]:
    """Extract per-goal success/solutions from an IK solver result.

    Batch mode (num_goalset == 1): each problem has one goal →
        success[b, 0] is the per-goal answer.

    Goalset mode (num_goalset > 1): each problem has N alternatives.
        Per-goal reachability is derived from ``goalset_index`` across
        all returned seeds.
    """
    successes: List[bool] = [False] * (chunk_batch * num_goalset)
    solutions: List[RosJointState] = [RosJointState()] * (chunk_batch * num_goalset)
    pos_errs: List[float] = [0.0] * (chunk_batch * num_goalset)
    rot_errs: List[float] = [0.0] * (chunk_batch * num_goalset)

    if ik_result.success is None:
        return successes, solutions, pos_errs, rot_errs

    n_ret = ik_result.success.shape[1]  # return_seeds

    if num_goalset == 1:
        for b in range(chunk_batch):
            idx = b
            ok = bool(ik_result.success[b].any().item())
            successes[idx] = ok
            if ok and ik_result.js_solution is not None:
                solutions[idx] = cu_joint_state_to_ros(ik_result.js_solution[b])
            if ik_result.position_error is not None:
                pos_errs[idx] = float(ik_result.position_error[b].max().item())
            if ik_result.rotation_error is not None:
                rot_errs[idx] = float(ik_result.rotation_error[b].max().item())
    else:
        for b in range(chunk_batch):
            for s in range(n_ret):
                if not bool(ik_result.success[b, s].item()):
                    continue
                gs_idx = -1
                if ik_result.goalset_index is not None:
                    gs_idx = int(ik_result.goalset_index[b, s, 0].item())
                if gs_idx < 0 or gs_idx >= num_goalset:
                    continue
                goal_flat = b * num_goalset + gs_idx
                if not successes[goal_flat]:
                    successes[goal_flat] = True
                    if ik_result.js_solution is not None:
                        solutions[goal_flat] = cu_joint_state_to_ros(ik_result.js_solution[s])
                    if ik_result.position_error is not None:
                        pos_errs[goal_flat] = float(ik_result.position_error[b, s].item())
                    if ik_result.rotation_error is not None:
                        rot_errs[goal_flat] = float(ik_result.rotation_error[b, s].item())

    return successes, solutions, pos_errs, rot_errs


def handle_compute_ik(context: CuroboContext, request, response, lock):
    t0 = time.perf_counter()
    sync_world(context)
    try:
        num_goals = len(request.goal_poses)
        if num_goals == 0:
            response.success = []
            response.solutions = []
            response.position_error = []
            response.rotation_error = []
            response.solve_time_s = 0.0
            return response

        tool_frame = request.tool_frame if request.tool_frame else ""
        num_gs = request.num_goalset

        # --- determine layout -------------------------------------------------
        if num_gs == 0:
            if request.use_goalset:
                batch_size, num_gs = 1, num_goals
            else:
                batch_size, num_gs = num_goals, 1
        else:
            if num_goals % num_gs != 0:
                context.logger.warning(
                    f"ComputeIK: len(goal_poses)={num_goals} not divisible by "
                    f"num_goalset={num_gs}; truncating to "
                    f"{(num_goals // num_gs) * num_gs}."
                )
                num_goals = (num_goals // num_gs) * num_gs
            batch_size = num_goals // num_gs

        # --- flatten pose list ------------------------------------------------
        pose_list = [
            [p.position.x, p.position.y, p.position.z,
             p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z]
            for p in request.goal_poses[:num_goals]
        ]

        with lock:
            ik_solver = context.motion_planner.ik_solver
            max_batch = ik_solver.config.max_batch_size
            max_gs = ik_solver.config.max_goalset
            dof = ik_solver.kinematics.dof
            frame = tool_frame or context.motion_planner.kinematics.tool_frames[0]

            # --- resolve seed states -----------------------------------------
            num_states = len(request.seed_states)
            shared_seed: Optional[JointState] = None
            seed_tensor: Optional[torch.Tensor] = None
            if num_states == 1:
                shared_seed = _ros_joint_states_to_cu(request.seed_states, dof, context.device)
            elif num_states == batch_size:
                full = _ros_joint_states_to_cu(request.seed_states, dof, context.device)
                if full is not None:
                    seed_tensor = full.position
            elif num_states > 0:
                context.logger.warning(
                    f"ComputeIK: seed_states count ({num_states}) != 1 and != "
                    f"batch_size ({batch_size}); proceeding with solver-default seeding."
                )

            # --- validate limits ----------------------------------------------
            if num_gs > max_gs:
                context.logger.warning(
                    f"ComputeIK: num_goalset={num_gs} exceeds max_goalset={max_gs}; "
                    f"capping at {max_gs}."
                )
                num_gs = max_gs
            if batch_size > max_batch:
                context.logger.warning(
                    f"ComputeIK: effective batch_size={batch_size} exceeds "
                    f"max_batch_size={max_batch}; will split into "
                    f"{math.ceil(batch_size / max_batch)} chunks."
                )

            # --- solve in chunks ----------------------------------------------
            all_success: List[bool] = []
            all_solutions: List[RosJointState] = []
            all_pos_err: List[float] = []
            all_rot_err: List[float] = []
            solve_time = 0.0

            for chunk_start in range(0, batch_size, max_batch):
                chunk_end = min(chunk_start + max_batch, batch_size)
                chunk_batch = chunk_end - chunk_start

                # Build goal tensors: [chunk_batch, num_gs, 7]
                chunk_poses = []
                for b in range(chunk_start, chunk_end):
                    base = b * num_gs
                    chunk_poses.extend(pose_list[base:base + num_gs])

                chunk_positions = torch.tensor(
                    [[p[0], p[1], p[2]] for p in chunk_poses],
                    dtype=torch.float32, device=context.device,
                ).view(chunk_batch, num_gs, 3)
                chunk_quaternions = torch.tensor(
                    [[p[3], p[4], p[5], p[6]] for p in chunk_poses],
                    dtype=torch.float32, device=context.device,
                ).view(chunk_batch, num_gs, 4)

                goal_pose = Pose(position=chunk_positions, quaternion=chunk_quaternions)
                goal_tool_pose = GoalToolPose.from_poses(
                    {frame: goal_pose}, num_goalset=num_gs,
                )

                cur_state: Optional[JointState] = None
                if shared_seed is not None and chunk_batch == 1:
                    cur_state = shared_seed
                elif seed_tensor is not None:
                    cur_state = JointState.from_position(
                        seed_tensor[chunk_start:chunk_end]
                    )

                ik_result = ik_solver.solve_pose(
                    goal_tool_pose,
                    current_state=cur_state,
                    return_seeds=ik_solver.config.num_seeds if num_gs > 1 else 1,
                )

                if hasattr(ik_result, "solve_time"):
                    solve_time += ik_result.solve_time

                chunk_success, chunk_solutions, chunk_pos, chunk_rot = (
                    _extract_per_goal_results(ik_result, chunk_batch, num_gs)
                )
                all_success.extend(chunk_success)
                all_solutions.extend(chunk_solutions)
                all_pos_err.extend(chunk_pos)
                all_rot_err.extend(chunk_rot)

            response.success = all_success
            response.solutions = all_solutions
            response.position_error = all_pos_err
            response.rotation_error = all_rot_err
            response.solve_time_s = solve_time

        return response

    except Exception as e:
        context.logger.error(f"ComputeIK failed: {e}")
        response.success = [False] * len(request.goal_poses) if hasattr(request, "goal_poses") else []
        response.solutions = []
        response.position_error = []
        response.rotation_error = []
        response.solve_time_s = 0.0
        return response

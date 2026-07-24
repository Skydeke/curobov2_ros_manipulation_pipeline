"""IK handler for ComputeIK service.

Uses the shared ``MotionPlanner.ik_solver`` (built at startup) to solve IK
for one or more Cartesian goal poses in a single GPU batch.
"""

from __future__ import annotations

import time
from typing import Optional

import torch

from curobo.types import GoalToolPose, Pose

from isaac_ros_cumotion_interfaces.srv import ComputeIK
from sensor_msgs.msg import JointState as RosJointState

from .context import CuroboContext
from .conversions import cu_joint_state_to_ros


def handle_compute_ik(context: CuroboContext, request, response, lock):
    t0 = time.perf_counter()
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

        # Convert ROS poses to cuRobo Pose tensor
        pose_list = [
            [p.position.x, p.position.y, p.position.z,
             p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z]
            for p in request.goal_poses
        ]

        with lock:
            ik_solver = context.motion_planner.ik_solver
            max_batch = ik_solver.config.max_batch_size
            frame = tool_frame or context.motion_planner.kinematics.tool_frames[0]

            # Build base seed tensor [1, 1, dof] if seed state was provided
            base_seed: Optional[torch.Tensor] = None
            if len(request.seed_state.position) > 0:
                dof = ik_solver.kinematics.dof
                seed_positions = list(request.seed_state.position)
                if len(seed_positions) < dof:
                    seed_positions.extend([0.0] * (dof - len(seed_positions)))
                base_seed = torch.tensor(
                    seed_positions,
                    dtype=torch.float32,
                    device=context.device,
                ).view(1, 1, -1)

            # Allocate result accumulators
            all_success = []
            all_solutions = []
            all_pos_err = []
            all_rot_err = []
            solve_time = 0.0

            for chunk_start in range(0, num_goals, max_batch):
                chunk_end = min(chunk_start + max_batch, num_goals)
                chunk = pose_list[chunk_start:chunk_end]
                chunk_size = len(chunk)

                goal_positions = torch.tensor(
                    [[p[0], p[1], p[2]] for p in chunk],
                    dtype=torch.float32,
                    device=context.device,
                )
                goal_quaternions = torch.tensor(
                    [[p[3], p[4], p[5], p[6]] for p in chunk],
                    dtype=torch.float32,
                    device=context.device,
                )

                goal_pose = Pose(position=goal_positions, quaternion=goal_quaternions)
                goal_tool_pose = GoalToolPose.from_poses({frame: goal_pose}, num_goalset=1)

                ik_result = ik_solver.solve_pose(
                    goal_tool_pose,
                )

                if hasattr(ik_result, "solve_time"):
                    solve_time += ik_result.solve_time

                if ik_result.success is not None:
                    best_success = ik_result.success.any(dim=-1)
                    for i in range(chunk_size):
                        ok = bool(best_success[i].item())
                        all_success.append(ok)
                        if ok and ik_result.js_solution is not None:
                            all_solutions.append(cu_joint_state_to_ros(ik_result.js_solution[i]))
                        else:
                            all_solutions.append(RosJointState())
                    if ik_result.position_error is not None:
                        all_pos_err.extend(float(ik_result.position_error[i].item()) for i in range(chunk_size))
                    else:
                        all_pos_err.extend([0.0] * chunk_size)
                    if ik_result.rotation_error is not None:
                        all_rot_err.extend(float(ik_result.rotation_error[i].item()) for i in range(chunk_size))
                    else:
                        all_rot_err.extend([0.0] * chunk_size)
                else:
                    all_success.extend([False] * chunk_size)
                    all_solutions.extend([RosJointState()] * chunk_size)
                    all_pos_err.extend([0.0] * chunk_size)
                    all_rot_err.extend([0.0] * chunk_size)

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

"""Trajectory optimization handler for OptimizeTrajectory action.

Wraps TrajOptSolver.solve_pose() / solve_cspace() directly — no graph-search
fallback, exposes raw optimizer parameters.
"""

from __future__ import annotations

from isaac_ros_cumotion_interfaces.action import OptimizeTrajectory
from curobo.types import GoalToolPose, JointState as CuJointState, Pose

from .context import CuroboContext
from .conversions import cu_solution_to_joint_trajectory


def handle_optimize_trajectory(
    context: CuroboContext,
    goal_handle,
    js_buffer,
    lock,
    motion_planner,
):
    goal: OptimizeTrajectory.Goal = goal_handle.request
    result = OptimizeTrajectory.Result()
    feedback = OptimizeTrajectory.Feedback()

    try:
        # --- resolve start state ---
        has_pose_goals = len(goal.goal_poses) > 0
        has_joint_goal = len(goal.goal_joint_state.position) > 0

        if not has_pose_goals and not has_joint_goal:
            result.success = False
            result.message = "Exactly one of goal_poses or goal_joint_state must be non-empty"
            goal_handle.abort(result)
            return result

        if len(goal.start_state.position) > 0:
            start_state = motion_planner.kinematics.get_active_js(
                CuJointState.from_position(
                    position=motion_planner.device_cfg.to_device(
                        goal.start_state.position
                    ).unsqueeze(0),
                    joint_names=list(goal.start_state.name),
                )
            )
        elif js_buffer is not None:
            js = js_buffer
            start_state = CuJointState.from_position(
                position=motion_planner.device_cfg.to_device(
                    js["position"]
                ).unsqueeze(0),
                joint_names=list(js["joint_names"]),
            )
            start_state = motion_planner.kinematics.get_active_js(start_state)
        else:
            result.success = False
            result.message = "No start state available (no joint_states topic data)"
            goal_handle.abort(result)
            return result

        # --- build common trajopt kwargs ---
        num_seeds = goal.num_seeds if goal.num_seeds > 0 else None
        finetune_attempts = goal.finetune_attempts if goal.finetune_attempts > 0 else 1
        dt = None
        if goal.trajectory_dt > 0:
            dt = motion_planner.device_cfg.to_device(
                [[goal.trajectory_dt]]
            )

        if has_joint_goal:
            # --- joint-space goal (solve_cspace) ---
            feedback.phase = "seeding"
            goal_handle.publish_feedback(feedback)

            goal_state = motion_planner.kinematics.get_active_js(
                CuJointState.from_position(
                    position=motion_planner.device_cfg.to_device(
                        list(goal.goal_joint_state.position)
                    ).unsqueeze(0),
                    joint_names=list(goal.goal_joint_state.name),
                )
            )

            with lock:
                feedback.phase = "optimizing"
                goal_handle.publish_feedback(feedback)
                trajopt_result = motion_planner.trajopt_solver.solve_cspace(
                    goal_state,
                    start_state,
                    return_seeds=1,
                    num_seeds=num_seeds,
                    dt=dt,
                    finetune_attempts=finetune_attempts,
                )

            result.matched_goal_index = -1

        else:
            # --- Cartesian pose goal(s) ---
            num_goals = len(goal.goal_poses)
            pose_list = [
                [p.position.x, p.position.y, p.position.z,
                 p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z]
                for p in goal.goal_poses
            ]
            bp = Pose.from_batch_list(pose_list)
            goal_pose = Pose(
                position=bp.position.contiguous().view(1, -1, 3),
                quaternion=bp.quaternion.contiguous().view(1, -1, 4),
            )

            tool_frame = goal.tool_frame if goal.tool_frame else motion_planner.tool_frames[0]
            goal_tool_poses = GoalToolPose.from_poses(
                {tool_frame: goal_pose},
                ordered_tool_frames=[tool_frame],
                num_goalset=num_goals,
            )

            with lock:
                feedback.phase = "seeding"
                goal_handle.publish_feedback(feedback)
                feedback.phase = "optimizing"
                goal_handle.publish_feedback(feedback)
                trajopt_result = motion_planner.trajopt_solver.solve_pose(
                    goal_tool_poses,
                    start_state,
                    return_seeds=1,
                    num_seeds=num_seeds,
                    dt=dt,
                    finetune_attempts=finetune_attempts,
                )

            if (trajopt_result is not None
                    and trajopt_result.success is not None
                    and trajopt_result.success.any().item()):
                if (hasattr(trajopt_result, "goalset_index")
                        and trajopt_result.goalset_index is not None):
                    result.matched_goal_index = int(trajopt_result.goalset_index.item())
                else:
                    result.matched_goal_index = 0
            else:
                result.matched_goal_index = -1

        feedback.phase = "finetuning"
        goal_handle.publish_feedback(feedback)

        feedback.phase = "done"
        goal_handle.publish_feedback(feedback)

        # --- process result ---
        if (trajopt_result is not None
                and trajopt_result.success is not None
                and trajopt_result.success.any().item()):
            result.success = True
            result.message = "Trajectory optimization succeeded"
            result.trajectory = cu_solution_to_joint_trajectory(
                trajopt_result.js_solution,
                trajopt_result.js_solution.dt.item(),
            )
            result.position_error = float(
                trajopt_result.position_error.max().item()
                if trajopt_result.position_error is not None else 0.0
            )
            result.rotation_error = float(
                trajopt_result.rotation_error.max().item()
                if trajopt_result.rotation_error is not None else 0.0
            )
            result.solve_time_s = float(
                trajopt_result.solve_time if hasattr(trajopt_result, "solve_time") else 0.0
            )
            result.feasible = bool(
                trajopt_result.feasible.any().item()
                if trajopt_result.feasible is not None else True
            )
        else:
            result.success = False
            result.message = (
                str(trajopt_result.status)
                if trajopt_result is not None and hasattr(trajopt_result, "status")
                and trajopt_result.status
                else "Trajectory optimization failed"
            )
            result.position_error = 0.0
            result.rotation_error = 0.0
            result.solve_time_s = 0.0
            result.feasible = False

        if result.success:
            goal_handle.succeed(result)
        else:
            goal_handle.abort(result)
        return result

    except Exception as e:
        context.logger.error(f"OptimizeTrajectory failed: {e}")
        result.success = False
        result.message = str(e)
        result.position_error = 0.0
        result.rotation_error = 0.0
        result.solve_time_s = 0.0
        result.feasible = False
        goal_handle.abort(result)
        return result

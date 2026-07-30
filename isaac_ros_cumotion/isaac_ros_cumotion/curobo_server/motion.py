"""Motion planning handlers for PlanMotion and PlanGrasp actions."""

from __future__ import annotations

from copy import deepcopy

from curobo.types import GoalToolPose, JointState as CuJointState, Pose

import numpy as np

from isaac_ros_cumotion_interfaces.action import PlanGrasp, PlanMotion
from moveit_msgs.msg import MoveItErrorCodes

from .context import CuroboContext
from .conversions import cu_solution_to_joint_trajectory
from .world import sync_world


def _get_start_state(context, goal, js_buffer):
    """Resolve the start state from the action goal or the joint-state buffer."""
    if len(goal.start_state.position) > 0:
        return context.motion_planner.kinematics.get_active_js(
            CuJointState.from_position(
                position=context.motion_planner.device_cfg.to_device(
                    goal.start_state.position
                ).unsqueeze(0),
                joint_names=list(goal.start_state.name),
            )
        )
    if js_buffer is not None:
        js = js_buffer
        state = CuJointState.from_position(
            position=context.motion_planner.device_cfg.to_device(
                js["position"]
            ).unsqueeze(0),
            joint_names=list(js["joint_names"]),
        )
        if js.get("velocity") is not None:
            state.velocity = context.motion_planner.device_cfg.to_device(
                js["velocity"]
            ).unsqueeze(0)
        return context.motion_planner.kinematics.get_active_js(state)
    return None


def handle_plan_motion(context: CuroboContext, goal_handle, js_buffer, lock, motion_planner):
    goal: PlanMotion.Goal = goal_handle.request
    result = PlanMotion.Result()
    result.error_code = MoveItErrorCodes()
    feedback = PlanMotion.Feedback()

    try:
        sync_world(context)

        time_dilation_factor = goal.time_dilation_factor
        if time_dilation_factor == 0.0:
            time_dilation_factor = 0.1

        # --- resolve start state ---
        start_state = _get_start_state(context, goal, js_buffer)
        if start_state is None:
            result.success = False
            result.message = "No start state available (no joint_states topic data)"
            result.error_code.val = MoveItErrorCodes.INVALID_ROBOT_STATE
            goal_handle.abort(result)
            return result

        # --- determine goal type ---
        has_pose_goals = len(goal.goal_poses) > 0
        has_joint_goal = len(goal.goal_joint_state.position) > 0

        if not has_pose_goals and not has_joint_goal:
            result.success = False
            result.message = "Exactly one of goal_poses or goal_joint_state must be non-empty"
            result.error_code.val = MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS
            goal_handle.abort(result)
            return result

        motion_gen_result = None

        if has_joint_goal:
            # --- joint-space goal ---
            feedback.phase = "seeding"
            goal_handle.publish_feedback(feedback)

            goal_state = context.motion_planner.kinematics.get_active_js(
                CuJointState.from_position(
                    position=context.motion_planner.device_cfg.to_device(
                        list(goal.goal_joint_state.position)
                    ).unsqueeze(0),
                    joint_names=list(goal.goal_joint_state.name),
                )
            )
            with lock:
                feedback.phase = "trajopt"
                goal_handle.publish_feedback(feedback)
                motion_gen_result = context.motion_planner.plan_cspace(
                    goal_state, start_state,
                )
            result.matched_goal_index = -1

        elif has_pose_goals:
            # --- Cartesian pose goal(s) ---
            num_goals = len(goal.goal_poses)
            num_goalset = num_goals if goal.plan_goal_set else 1

            pose_list = [
                [p.position.x, p.position.y, p.position.z,
                 p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z]
                for p in goal.goal_poses
            ]

            tool_frame = goal.tool_frame if goal.tool_frame else context.motion_planner.tool_frames[0]
            max_attempts_val = (
                context.node.get_parameter('max_attempts').get_parameter_value().integer_value
                if context.node is not None and context.node.has_parameter('max_attempts')
                else 100
            )

            motion_gen_result = None
            matched_goal_offset = 0
            max_goalset = (
                int(context.node.get_parameter('max_goalset').value)
                if context.node is not None and context.node.has_parameter('max_goalset')
                else 12
            )
            chunk_size = 1 if num_goalset == 1 else min(num_goalset, max_goalset)

            for chunk_start in range(0, num_goalset, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_goalset)
                chunk_poses = pose_list[chunk_start:chunk_end]
                n_chunk = chunk_end - chunk_start

                if n_chunk == 1:
                    goal_pose = Pose.from_list(chunk_poses[0])
                    goal_pose.position = goal_pose.position.view(1, -1)
                    goal_pose.quaternion = goal_pose.quaternion.view(1, -1)
                else:
                    bp = Pose.from_batch_list(chunk_poses)
                    goal_pose = Pose(
                        position=bp.position.contiguous(),
                        quaternion=bp.quaternion.contiguous(),
                    )

                goal_tool_poses = GoalToolPose.from_poses(
                    {tool_frame: goal_pose},
                    ordered_tool_frames=[tool_frame],
                    num_goalset=n_chunk,
                )

                with lock:
                    feedback.phase = "seeding"
                    goal_handle.publish_feedback(feedback)
                    context.motion_planner.reset_seed()
                    if goal.enable_graph_search:
                        result_chunk = context.motion_planner.plan_pose(
                            goal_tool_poses, start_state,
                            max_attempts=max_attempts_val,
                            enable_graph_attempt=1,
                        )
                    else:
                        result_chunk = context.motion_planner.plan_pose(
                            goal_tool_poses, start_state,
                            max_attempts=max_attempts_val,
                        )

                if (result_chunk is not None
                        and result_chunk.success is not None
                        and result_chunk.success.any().item()):
                    motion_gen_result = result_chunk
                    if hasattr(result_chunk, "goalset_index") and result_chunk.goalset_index is not None:
                        result.matched_goal_index = int(result_chunk.goalset_index.item()) + matched_goal_offset
                    else:
                        result.matched_goal_index = matched_goal_offset
                    break

                matched_goal_offset += n_chunk

            if motion_gen_result is None:
                result.matched_goal_index = -1

        feedback.phase = "done"
        goal_handle.publish_feedback(feedback)

        # --- process result ---
        if (motion_gen_result is not None
                and motion_gen_result.success is not None
                and motion_gen_result.success.any().item()):
            result.success = True
            result.message = "Planning succeeded"
            result.error_code.val = MoveItErrorCodes.SUCCESS
            result.trajectory = cu_solution_to_joint_trajectory(
                motion_gen_result.js_solution,
                motion_gen_result.js_solution.dt.item(),
            )
            result.planning_time_s = float(
                motion_gen_result.total_time if hasattr(motion_gen_result, "total_time") else 0.0
            )
        else:
            result.success = False
            result.message = (
                str(motion_gen_result.status)
                if motion_gen_result is not None and hasattr(motion_gen_result, "status") and motion_gen_result.status
                else "Motion planning failed"
            )
            result.error_code.val = MoveItErrorCodes.PLANNING_FAILED
            result.planning_time_s = 0.0

        if result.success:
            goal_handle.succeed(result)
        else:
            goal_handle.abort(result)
        return result

    except Exception as e:
        context.logger.error(f"PlanMotion failed: {e}")
        result.success = False
        result.message = str(e)
        result.error_code.val = MoveItErrorCodes.PLANNING_FAILED
        goal_handle.abort(result)
        return result


def handle_plan_grasp(context: CuroboContext, goal_handle, js_buffer, lock, motion_planner):
    """Handle PlanGrasp action by calling MotionPlanner.plan_grasp()."""
    goal: PlanGrasp.Goal = goal_handle.request
    result = PlanGrasp.Result()
    feedback = PlanGrasp.Feedback()

    try:
        start_state = _get_start_state(context, goal, js_buffer)
        if start_state is None:
            result.success = False
            result.message = "No start state available"
            result.matched_goal_index = -1
            goal_handle.abort(result)
            return result

        if len(goal.grasp_poses) == 0:
            result.success = False
            result.message = "grasp_poses is empty"
            result.matched_goal_index = -1
            goal_handle.abort(result)
            return result

        # Build goal poses
        pose_list = [
            [p.position.x, p.position.y, p.position.z,
             p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z]
            for p in goal.grasp_poses
        ]
        bp = Pose.from_batch_list(pose_list)
        flat_pose = Pose(
            position=bp.position.contiguous().view(-1, 3),
            quaternion=bp.quaternion.contiguous().view(-1, 4),
        )

        tool_frame = goal.tool_frame if goal.tool_frame else context.motion_planner.tool_frames[0]
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frame: flat_pose},
            ordered_tool_frames=[tool_frame],
            num_goalset=flat_pose.position.shape[0],
        )

        feedback.phase = "approach"
        goal_handle.publish_feedback(feedback)

        with lock:
            grasp_result = context.motion_planner.plan_grasp(
                goal_tool_poses,
                start_state,
                grasp_approach_offset=goal.grasp_approach_offset,
                grasp_lift_offset=goal.grasp_lift_offset,
                grasp_approach_in_tool_frame=False,
                grasp_lift_in_tool_frame=goal.grasp_lift_in_tool_frame,
                plan_approach_to_grasp=goal.plan_approach_to_grasp,
                plan_grasp_to_lift=goal.plan_grasp_to_lift,
            )

        if (grasp_result is not None
                and grasp_result.success is not None
                and grasp_result.success.any().item()):
            result.success = True
            result.message = "Grasp planning succeeded"
            result.planning_time_s = float(grasp_result.planning_time) if hasattr(grasp_result, "planning_time") else 0.0

            # Approach trajectory
            feedback.phase = "approach"
            goal_handle.publish_feedback(feedback)
            if goal.plan_approach_to_grasp and grasp_result.approach_interpolated_trajectory is not None:
                approach_dt = float(grasp_result.approach_trajectory_dt.item()) if grasp_result.approach_trajectory_dt is not None else 0.025
                result.approach_trajectory = cu_solution_to_joint_trajectory(
                    grasp_result.approach_interpolated_trajectory, approach_dt,
                )

            # Grasp trajectory
            feedback.phase = "grasp"
            goal_handle.publish_feedback(feedback)
            if grasp_result.grasp_interpolated_trajectory is not None:
                grasp_dt = float(grasp_result.grasp_trajectory_dt.item()) if grasp_result.grasp_trajectory_dt is not None else 0.025
                result.grasp_trajectory = cu_solution_to_joint_trajectory(
                    grasp_result.grasp_interpolated_trajectory, grasp_dt,
                )

            # Lift trajectory
            if goal.plan_grasp_to_lift and grasp_result.lift_interpolated_trajectory is not None:
                feedback.phase = "lift"
                goal_handle.publish_feedback(feedback)
                lift_dt = float(grasp_result.lift_trajectory_dt.item()) if grasp_result.lift_trajectory_dt is not None else 0.025
                result.lift_trajectory = cu_solution_to_joint_trajectory(
                    grasp_result.lift_interpolated_trajectory, lift_dt,
                )

            result.matched_goal_index = int(grasp_result.goalset_index.item()) if grasp_result.goalset_index is not None else 0
        else:
            result.success = False
            result.message = (
                str(grasp_result.status)
                if grasp_result is not None and hasattr(grasp_result, "status") and grasp_result.status
                else "Grasp planning failed"
            )
            result.matched_goal_index = -1

        feedback.phase = "done"
        goal_handle.publish_feedback(feedback)

        if result.success:
            goal_handle.succeed(result)
        else:
            goal_handle.abort(result)
        return result

    except Exception as e:
        context.logger.error(f"PlanGrasp failed: {e}")
        result.success = False
        result.message = str(e)
        result.matched_goal_index = -1
        goal_handle.abort(result)
        return result

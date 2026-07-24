"""Motion retargeting handler for RetargetMotion action.

Wraps MotionRetargeter.solve_sequence() for offline batch retargeting of a
sequence of tool poses into a joint trajectory.
"""

from __future__ import annotations

from isaac_ros_cumotion_interfaces.action import RetargetMotion
from curobo.motion_retargeter import MotionRetargeter, MotionRetargeterCfg, SequenceGoalToolPose
from curobo.types import (
    JointState as CuJointState,
    ToolPoseCriteria,
)

from .context import CuroboContext
from .conversions import cu_joint_state_to_ros, cu_solution_to_joint_trajectory

import torch


def handle_retarget_motion(context: CuroboContext, goal_handle, lock, motion_planner):
    goal: RetargetMotion.Goal = goal_handle.request
    result = RetargetMotion.Result()
    feedback = RetargetMotion.Feedback()

    try:
        num_frames = goal.num_frames
        tool_frames = list(goal.tool_frames)
        num_links = len(tool_frames)

        if num_frames <= 0 or num_links == 0:
            result.success = False
            result.message = "num_frames > 0 and at least one tool_frame required"
            goal_handle.abort(result)
            return result

        total_poses_expected = num_frames * num_links
        if len(goal.tool_pose_sequence) != total_poses_expected:
            result.success = False
            result.message = (
                f"Expected {total_poses_expected} poses "
                f"({num_frames} frames x {num_links} links), "
                f"got {len(goal.tool_pose_sequence)}"
            )
            goal_handle.abort(result)
            return result

        feedback.phase = "setup"
        goal_handle.publish_feedback(feedback)

        # Resolve start state
        if len(goal.start_state.position) > 0:
            start_state = motion_planner.kinematics.get_active_js(
                CuJointState.from_position(
                    position=motion_planner.device_cfg.to_device(
                        goal.start_state.position
                    ).unsqueeze(0),
                    joint_names=list(goal.start_state.name),
                )
            )
        else:
            start_state = motion_planner.kinematics.get_active_js(
                motion_planner.kinematics.default_joint_state.clone()
            )

        # Build SequenceGoalToolPose from flat pose list
        device = motion_planner.device_cfg.device
        poses_np = [
            [p.position.x, p.position.y, p.position.z,
             p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z]
            for p in goal.tool_pose_sequence
        ]
        poses_tensor = torch.tensor(poses_np, dtype=torch.float32, device=device)
        # Reshape to (num_frames, num_envs=1, num_links, num_goalset=1, 7)
        poses_tensor = poses_tensor.view(num_frames, 1, num_links, 1, 7)

        seq_poses = SequenceGoalToolPose(
            tool_frames=tool_frames,
            position=poses_tensor[..., :3],
            quaternion=poses_tensor[..., 3:7],
        )

        # Build tool_pose_criteria for each tracked frame
        tool_pose_criteria = {}
        for tf in tool_frames:
            tool_pose_criteria[tf] = ToolPoseCriteria.track_position_and_orientation(
                xyz=[1.0, 1.0, 1.0], rpy=[0.5, 0.5, 0.5],
            )

        feedback.phase = "solving"
        goal_handle.publish_feedback(feedback)

        with lock:
            cfg = MotionRetargeterCfg.create(
                robot=motion_planner.robot_config,
                scene_model=motion_planner.scene_model,
                self_collision_check=motion_planner.self_collision_check,
                device_cfg=motion_planner.device_cfg,
                tool_pose_criteria=tool_pose_criteria,
            )
            retargeter = MotionRetargeter(cfg)
            retarget_result = retargeter.solve_sequence(seq_poses)

        feedback.phase = "done"
        goal_handle.publish_feedback(feedback)

        if retarget_result is not None and retarget_result.joint_state is not None:
            result.success = True
            result.message = "Retargeting succeeded"
            js = retarget_result.joint_state
            dt = 0.033  # ~30 fps default
            if hasattr(js, "dt") and js.dt is not None:
                dt = float(js.dt.item()) if js.dt.numel() == 1 else 0.033
            result.trajectory = cu_solution_to_joint_trajectory(js, dt)
            result.solve_time_s = 0.0
        else:
            result.success = False
            result.message = "Retargeting returned no solution"
            result.solve_time_s = 0.0

        if result.success:
            goal_handle.succeed(result)
        else:
            goal_handle.abort(result)
        return result

    except Exception as e:
        context.logger.error(f"RetargetMotion failed: {e}")
        result.success = False
        result.message = str(e)
        result.solve_time_s = 0.0
        goal_handle.abort(result)
        return result

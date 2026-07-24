from __future__ import annotations

import math

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MoveItErrorCodes, RobotTrajectory

from curobo.types import GoalToolPose, JointState as CuJointState, Pose
import torch

from .context import CuroboContext
from .conversions import cu_solution_to_joint_trajectory
from .world import _rebuild_world


def handle_move_group_action(context: CuroboContext, goal_handle, js_buffer, lock):
    goal: MoveGroup.Goal = goal_handle.request
    result = MoveGroup.Result()

    try:
        plan_req = goal.request
        scene_diff = goal.planning_options.planning_scene_diff

        context.logger.info(f"MoveGroup request: {len(plan_req.goal_constraints)} constraint(s), "
                            f"{len(scene_diff.world.collision_objects)} world diff objects, "
                            f"start_state joints={len(plan_req.start_state.joint_state.position)}, "
                            f"vel_scale={plan_req.max_velocity_scaling_factor}, "
                            f"acc_scale={plan_req.max_acceleration_scaling_factor}")

        # ---- update world from planning scene diff ----
        world_objs = scene_diff.world.collision_objects
        if world_objs:
            for obj in world_objs:
                context.world_objects[obj.id] = obj
            context.logger.info(f"Updated world with {len(world_objs)} objects, total={len(context.world_objects)}")
            _rebuild_world(context)

        # ---- resolve start state ----
        start_state = None
        if len(plan_req.start_state.joint_state.position) > 0:
            context.logger.info("Using request's start_state")
            start_state = context.motion_planner.kinematics.get_active_js(
                CuJointState.from_position(
                    position=context.motion_planner.device_cfg.to_device(
                        list(plan_req.start_state.joint_state.position)
                    ).unsqueeze(0),
                    joint_names=list(plan_req.start_state.joint_state.name),
                )
            )

            # Unwrap continuous joints: MoveIt normalizes joint angles to (-π, π],
            # which can flip a joint at ±π (e.g. +3.14 → -3.14).  Unwrap the
            # planner's start_state to match the robot's actual js_buffer values
            # modulo 2π, so the trajectory's first commanded position is
            # numerically close to where the robot really is.
            if js_buffer is not None:
                buf = CuJointState.from_position(
                    position=context.motion_planner.device_cfg.to_device(
                        js_buffer["position"]
                    ).unsqueeze(0),
                    joint_names=list(js_buffer["joint_names"]),
                )
                buf_active = context.motion_planner.kinematics.get_active_js(buf)
                diff = start_state.position - buf_active.position
                mask = diff.abs() > math.pi
                if mask.any().item():
                    unwrap = torch.where(diff > math.pi, -2.0 * math.pi,
                                         torch.where(diff < -math.pi, 2.0 * math.pi, 0.0))
                    start_state.position = start_state.position + unwrap
                    context.logger.info(
                        f"Unwrapped {mask.sum().item()} joint(s) by ±2π to match js_buffer"
                    )

        if start_state is None or plan_req.start_state.is_diff:
            if js_buffer is None:
                context.logger.error("No joint state available")
                result.error_code.val = MoveItErrorCodes.INVALID_ROBOT_STATE
                goal_handle.abort(result)
                return result

            state = CuJointState.from_position(
                position=context.motion_planner.device_cfg.to_device(
                    js_buffer["position"]
                ).unsqueeze(0),
                joint_names=list(js_buffer["joint_names"]),
            )
            if js_buffer.get("velocity") is not None:
                state.velocity = context.motion_planner.device_cfg.to_device(
                    js_buffer["velocity"]
                ).unsqueeze(0)
            current_js = context.motion_planner.kinematics.get_active_js(state)

            if start_state is not None and plan_req.start_state.is_diff:
                start_state.position += current_js.position
                start_state.velocity += current_js.velocity
            else:
                start_state = current_js

        # ---- resolve goal ----
        goal_tool_poses = None
        constraints = plan_req.goal_constraints[0]
        context.logger.info(f"Goal: {len(constraints.joint_constraints)} joint, "
                            f"{len(constraints.position_constraints)} pos, "
                            f"{len(constraints.orientation_constraints)} orient constraints")

        if len(constraints.joint_constraints) > 0:
            goal_config = [c.position for c in constraints.joint_constraints]
            goal_jnames = [c.joint_name for c in constraints.joint_constraints]
            context.logger.info(f"Joint-space goal: names={goal_jnames}, positions={goal_config}")

            goal_state = context.motion_planner.kinematics.get_active_js(
                CuJointState.from_position(
                    position=context.motion_planner.device_cfg.to_device(
                        goal_config
                    ).view(1, -1),
                    joint_names=goal_jnames,
                )
            )

            cartesian_link = plan_req.cartesian_speed_limited_link
            if cartesian_link:
                context.logger.info(f"Cartesian path requested (link={cartesian_link}), using plan_cspace")
                with lock:
                    motion_gen_result = context.motion_planner.plan_cspace(
                        goal_state, start_state,
                    )
            else:
                context.logger.info("No cartesian path flag, computing FK for plan_pose")
                fk_state = context.motion_planner.kinematics.compute_kinematics(goal_state)
                tool_frame = context.motion_planner.tool_frames[0]
                ee_pose = fk_state.tool_poses.get_link_pose(tool_frame)
                goal_tool_poses = GoalToolPose.from_poses(
                    {tool_frame: ee_pose},
                    ordered_tool_frames=[tool_frame],
                    num_goalset=1,
                )
                with lock:
                    motion_gen_result = context.motion_planner.plan_pose(
                        goal_tool_poses, start_state,
                    )

            success = motion_gen_result is not None and motion_gen_result.success is not None and motion_gen_result.success.any().item()
            context.logger.info(f"plan_{'cspace' if cartesian_link else 'pose'} done, success={success}")
            result.error_code.val = (
                MoveItErrorCodes.SUCCESS if success else MoveItErrorCodes.PLANNING_FAILED
            )

        elif len(constraints.position_constraints) > 0 and len(constraints.orientation_constraints) > 0:
            pc = constraints.position_constraints[0]
            oc = constraints.orientation_constraints[0]

            pos = pc.constraint_region.primitive_poses[0].position
            orientation = oc.orientation
            pose_list = [pos.x, pos.y, pos.z, orientation.w, orientation.x, orientation.y, orientation.z]
            context.logger.info(f"Pose goal: position=({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}), "
                                f"orientation=({orientation.w:.3f}, {orientation.x:.3f}, {orientation.y:.3f}, {orientation.z:.3f})")

            goal_pose = Pose.from_list(pose_list)
            goal_pose.position = goal_pose.position.view(1, -1)
            goal_pose.quaternion = goal_pose.quaternion.view(1, -1)

            tool_frame = context.motion_planner.tool_frames[0]
            context.logger.info(f"Using tool_frame={tool_frame}")
            goal_tool_poses = GoalToolPose.from_poses(
                {tool_frame: goal_pose},
                ordered_tool_frames=[tool_frame],
                num_goalset=1,
            )

            with lock:
                context.logger.info("Calling plan_pose...")
                motion_gen_result = context.motion_planner.plan_pose(
                    goal_tool_poses, start_state,
                )
            success = motion_gen_result is not None and motion_gen_result.success is not None and motion_gen_result.success.any().item()
            context.logger.info(f"plan_pose done, success={success}")

            result.error_code.val = (
                MoveItErrorCodes.SUCCESS if success else MoveItErrorCodes.PLANNING_FAILED
            )

        else:
            context.logger.error("Goal constraints not supported")
            result.error_code.val = MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS
            goal_handle.abort(result)
            return result

        # ---- build response ----
        if result.error_code.val == MoveItErrorCodes.SUCCESS:
            result.trajectory_start = plan_req.start_state

            # Apply velocity/acceleration scaling from MoveGroup request
            vel_scale = plan_req.max_velocity_scaling_factor
            acc_scale = plan_req.max_acceleration_scaling_factor
            if vel_scale is None or vel_scale <= 0.0 or vel_scale > 1.0:
                vel_scale = 1.0
            if acc_scale is None or acc_scale <= 0.0 or acc_scale > 1.0:
                acc_scale = 1.0
            time_scaling = min(vel_scale, acc_scale)
            if time_scaling < 1.0:
                context.logger.info(f"Scaling trajectory timing by {time_scaling:.3f} "
                                    f"(vel={vel_scale:.3f}, acc={acc_scale:.3f})")

            robot_traj = RobotTrajectory()
            traj_src = motion_gen_result
            js_for_traj = traj_src.js_solution
            traj_dt = js_for_traj.dt.item() if hasattr(js_for_traj, 'dt') and js_for_traj.dt is not None else 0.0
            num_pts = js_for_traj.position.shape[-2] if hasattr(js_for_traj, 'position') and js_for_traj.position is not None else 0
            traj_duration = (num_pts - 1) * traj_dt
            context.logger.info(f"Trajectory: {num_pts} waypoints, dt={traj_dt:.4f}s, "
                                f"duration={traj_duration:.2f}s, time_scaling={time_scaling}")
            robot_traj.joint_trajectory = cu_solution_to_joint_trajectory(
                js_for_traj,
                traj_dt,
                time_scaling=time_scaling,
            )
            result.planned_trajectory = robot_traj

            result.planning_time = float(
                motion_gen_result.total_time
                if hasattr(motion_gen_result, "total_time")
                else 0.0
            )
            context.logger.info(f"MoveGroup succeeded: planning_time={result.planning_time:.3f}s, "
                                f"{len(robot_traj.joint_trajectory.points)} waypoints")
            goal_handle.succeed(result)
        else:
            err = result.error_code.val
            if motion_gen_result is None:
                result.error_code.val = MoveItErrorCodes.PLANNING_FAILED
            context.logger.error(f"MoveGroup failed: error_code={err}")
            goal_handle.abort(result)

        return result

    except Exception as e:
        context.logger.error(f"MoveGroup action failed: {e}")
        import traceback
        context.logger.error(traceback.format_exc())
        result.error_code.val = MoveItErrorCodes.PLANNING_FAILED
        goal_handle.abort(result)
        return result

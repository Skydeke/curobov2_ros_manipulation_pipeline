# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import List, Tuple

from curobo.types import GoalToolPose, JointState as CuJointState, Pose
from geometry_msgs.msg import Pose as RosPose
from isaac_ros_cumotion.cumotion_planner import CumotionActionServer
from isaac_ros_cumotion_interfaces.action import MotionPlan
from moveit_msgs.msg import MoveItErrorCodes
import numpy as np
import rclpy
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from scipy.spatial.transform import Rotation as R


class CumotionGoalSetPlannerServer(CumotionActionServer):

    def __init__(self):
        super().__init__()
        self._goal_set_planner_server = ActionServer(
            self, MotionPlan, "cumotion/motion_plan", self.motion_plan_execute_callback
        )

    def warmup(self):
        self.get_logger().info("warming up cuMotion, wait until ready")
        self.motion_gen.warmup(enable_graph=True, num_warmup_iterations=10)
        self.get_logger().info("cuMotion is ready for planning queries!")

    def toggle_link_collision(self, collision_link_names: List[str], enable_flag: bool):
        if len(collision_link_names) > 0:
            if enable_flag:
                for k in collision_link_names:
                    self.motion_gen.kinematics.config.kinematics_config.enable_link_spheres(
                        k
                    )
            else:
                for k in collision_link_names:
                    self.motion_gen.kinematics.config.kinematics_config.disable_link_spheres(
                        k
                    )

    def get_goal_poses(self, plan_req: MotionPlan.Goal) -> Pose:
        if plan_req.goal_pose.header.frame_id != self.motion_gen.kinematics.base_link:
            self.get_logger().error(
                "Planning frame: "
                + plan_req.goal_pose.header.frame_id
                + " is not same as motion gen frame: "
                + self.motion_gen.kinematics.base_link
            )
            return False, MoveItErrorCodes.INVALID_LINK_NAME, []
        poses = []
        for k in plan_req.goal_pose.poses:
            poses.append(
                [
                    k.position.x,
                    k.position.y,
                    k.position.z,
                    k.orientation.w,
                    k.orientation.x,
                    k.orientation.y,
                    k.orientation.z,
                ]
            )
        if len(poses) == 0:
            self.get_logger().error("No goal pose found")
            return False, MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS, poses
        goal_pose = Pose.from_batch_list(poses)
        goal_pose = Pose(
            position=goal_pose.position.contiguous().view(1, -1, 3),
            quaternion=goal_pose.quaternion.contiguous().view(1, -1, 4),
        )
        return True, MoveItErrorCodes.SUCCESS, goal_pose

    def _create_transform_matrix(self, pose: RosPose) -> np.ndarray:
        """Create a 4x4 homogeneous transformation matrix from a ROS pose."""
        world_pose_mat = np.eye(4)
        world_pose_mat[:3, :3] = R.from_quat(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ]
        ).as_matrix()
        world_pose_mat[:3, 3] = np.asarray(
            [pose.position.x, pose.position.y, pose.position.z]
        )
        return world_pose_mat

    def motion_plan_execute_callback(self, goal_handle):
        try:
            self.motion_gen.reset_seed()
            with self.lock:
                self.planner_busy = True
            self.get_logger().info("Executing goal...")

            time_dilation_factor = goal_handle.request.time_dilation_factor
            if time_dilation_factor == 0.0:
                time_dilation_factor = 0.1
                self.get_logger().warn("Cannot set time_dilation_factor = 0.0")
            self.get_logger().info(
                "Planning with time_dilation_factor: " + str(time_dilation_factor)
            )

            result = MotionPlan.Result()
            result.success = False
            result.error_code = MoveItErrorCodes()

            objects_to_clear = [], [], [], []
            if goal_handle.request.use_planning_scene:
                self.get_logger().info("Updating planning scene")
                scene = goal_handle.request.world
                world_objects = scene.collision_objects
                if goal_handle.request.enable_aabb_clearing:
                    padding = goal_handle.request.object_esdf_clearing_padding
                    if goal_handle.request.plan_grasp:
                        world_pose_object = self.get_object_pose(
                            goal_handle.request.world_frame,
                            goal_handle.request.object_frame,
                        )
                        objects_to_clear = self.calculate_aabbs_to_clear(
                            world_pose_object=world_pose_object,
                            mesh_resource=goal_handle.request.mesh_resource,
                            object_esdf_clearing_padding=padding,
                        )
                    elif goal_handle.request.plan_pose:
                        goal_pose = goal_handle.request.goal_pose.poses[0]
                        objects_to_clear = self.calculate_aabbs_to_clear(
                            world_pose_object=goal_pose,
                            mesh_resource=goal_handle.request.mesh_resource,
                            object_esdf_clearing_padding=padding,
                            object_shape=goal_handle.request.object_shape,
                            object_scale=goal_handle.request.object_scale,
                        )
                world_update_status = self.update_world_objects(
                    world_objects, objects_to_clear
                )
                if not world_update_status:
                    result.error_code.val = (
                        MoveItErrorCodes.COLLISION_CHECKING_UNAVAILABLE
                    )
                    self.get_logger().error("World update failed.")
                    goal_handle.abort(result)
                    return result

            # Early-return for scene-update-only requests (no planning needed).
            # This check must happen BEFORE reading the joint state so that a
            # scene-only goal sent at startup (before any /joint_states arrive)
            # succeeds reliably.
            plan_req = goal_handle.request
            if (
                not plan_req.plan_grasp
                and not plan_req.plan_cspace
                and not plan_req.plan_pose
            ):
                self.get_logger().info("No planning type set, scene update only")
                result.success = True
                goal_handle.succeed(result)
                return result

            # --- Read start state (only reached when actual planning is needed) ---
            start_state = None
            if plan_req.use_current_state:
                with self.lock:
                    if self._CumotionActionServer__js_buffer is None:
                        self.get_logger().error(
                            "joint_state was not received from "
                            + self._CumotionActionServer__joint_states_topic
                        )
                        result.error_code.val = MoveItErrorCodes.INVALID_ROBOT_STATE
                        goal_handle.abort(result)
                        return result
                    js_position = list(
                        self._CumotionActionServer__js_buffer["position"]
                    )
                    js_velocity = list(
                        self._CumotionActionServer__js_buffer["velocity"]
                    )
                    js_names = list(
                        self._CumotionActionServer__js_buffer["joint_names"]
                    )
                    # js_buffer intentionally kept; overwritten by the next /joint_states callback

                lock_js = self.motion_gen.kinematics.lock_jointstate
                if (
                    lock_js is not None
                    and getattr(lock_js, "joint_names", None) is not None
                ):
                    lock_names = list(lock_js.joint_names)
                    for i, name in enumerate(js_names):
                        if name in lock_names:
                            idx = lock_names.index(name)
                            lock_js.position[idx] = float(js_position[i])

                state = CuJointState.from_position(
                    position=self._CumotionActionServer__tensor_args.to_device(
                        js_position
                    ).unsqueeze(0),
                    joint_names=js_names,
                )
                state.velocity = self._CumotionActionServer__tensor_args.to_device(
                    js_velocity
                ).unsqueeze(0)
                start_state = self.motion_gen.kinematics.get_active_js(state)
            elif len(plan_req.start_state.position) > 0:
                start_state = self.motion_gen.kinematics.get_active_js(
                    CuJointState.from_position(
                        position=self._CumotionActionServer__tensor_args.to_device(
                            plan_req.start_state.position
                        ).unsqueeze(0),
                        joint_names=plan_req.start_state.name,
                    )
                )
            else:
                self.get_logger().error("joint state in start state was empty")
                result.error_code.val = MoveItErrorCodes.INVALID_ROBOT_STATE
                goal_handle.abort(result)
                return result

            motion_gen_result = None
            if plan_req.plan_grasp:
                self.get_logger().info(
                    "Planning to Grasp Object with stop at offset distance"
                )
                success, error_code, poses = self.get_goal_poses(plan_req)
                self.get_logger().info(
                    f"Success, Error Code): {success}, {error_code}!"
                )
                if not success:
                    result.error_code.val = error_code
                    goal_handle.abort(result)
                    return result

                grasp_offset_pose = self.get_cu_pose_from_ros_pose(
                    plan_req.grasp_offset_pose
                )
                retract_offset_pose = self.get_cu_pose_from_ros_pose(
                    plan_req.retract_offset_pose
                )

                grasp_approach_offset = -0.15
                if (
                    grasp_offset_pose is not None
                    and hasattr(grasp_offset_pose, "position")
                    and grasp_offset_pose.position is not None
                ):
                    grasp_approach_offset = (
                        grasp_offset_pose.position.contiguous().view(-1)[2].item()
                    )
                grasp_lift_offset = 0.15
                if (
                    retract_offset_pose is not None
                    and hasattr(retract_offset_pose, "position")
                    and retract_offset_pose.position is not None
                ):
                    grasp_lift_offset = (
                        retract_offset_pose.position.contiguous().view(-1)[2].item()
                    )

                flat_pose = Pose(
                    position=poses.position.view(-1, 3),
                    quaternion=poses.quaternion.view(-1, 4),
                )
                goal_tool_poses = GoalToolPose.from_poses(
                    {self.motion_gen.tool_frames[0]: flat_pose},
                    ordered_tool_frames=[self.motion_gen.tool_frames[0]],
                    num_goalset=poses.position.shape[1],
                )
                with self.lock:
                    motion_gen_result = self.motion_gen.plan_grasp(
                        goal_tool_poses,
                        start_state,
                        grasp_approach_offset=grasp_approach_offset,
                        grasp_lift_offset=grasp_lift_offset,
                        grasp_approach_in_tool_frame=False,
                        grasp_lift_in_tool_frame=False,
                        plan_approach_to_grasp=plan_req.plan_approach_to_grasp,
                        plan_grasp_to_lift=plan_req.plan_grasp_to_retract,
                        disable_collision_links=plan_req.disable_collision_links,
                    )
                if (
                    motion_gen_result is not None
                    and motion_gen_result.success is not None
                    and motion_gen_result.success.any().item()
                ):
                    result.error_code = MoveItErrorCodes(val=MoveItErrorCodes.SUCCESS)

                    if motion_gen_result.approach_interpolated_trajectory is not None:
                        approach_dt = (
                            motion_gen_result.approach_trajectory_dt.item()
                            if motion_gen_result.approach_trajectory_dt is not None
                            else 0.025
                        )
                        traj = self.get_joint_trajectory(
                            motion_gen_result.approach_interpolated_trajectory,
                            approach_dt,
                        )
                        result.planned_trajectory.append(traj)

                    if motion_gen_result.grasp_interpolated_trajectory is not None:
                        grasp_dt = (
                            motion_gen_result.grasp_trajectory_dt.item()
                            if motion_gen_result.grasp_trajectory_dt is not None
                            else 0.025
                        )
                        traj = self.get_joint_trajectory(
                            motion_gen_result.grasp_interpolated_trajectory,
                            grasp_dt,
                        )
                        result.planned_trajectory.append(traj)

                    if (
                        plan_req.plan_grasp_to_retract
                        and motion_gen_result.lift_interpolated_trajectory is not None
                    ):
                        lift_dt = (
                            motion_gen_result.lift_trajectory_dt.item()
                            if motion_gen_result.lift_trajectory_dt is not None
                            else 0.025
                        )
                        traj = self.get_joint_trajectory(
                            motion_gen_result.lift_interpolated_trajectory,
                            lift_dt,
                        )
                        result.planned_trajectory.append(traj)

                    result.planning_time = motion_gen_result.planning_time
                    result.success = True
                    if motion_gen_result.goalset_index is not None:
                        result.goal_index = motion_gen_result.goalset_index.item()
                else:
                    result.success = False
                    result.message = (
                        str(motion_gen_result.status)
                        if motion_gen_result is not None and motion_gen_result.status
                        else "Grasp planning failed"
                    )
                    result.error_code.val = MoveItErrorCodes.PLANNING_FAILED
            elif plan_req.plan_cspace:
                self.get_logger().info("Planning CSpace target")
                if len(plan_req.goal_state.position) <= 0:
                    self.get_logger().error("goal state is empty")
                    result.error_code.val = MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS
                    goal_handle.abort(result)
                    return result
                goal_state = self.motion_gen.kinematics.get_active_js(
                    CuJointState.from_position(
                        position=self._CumotionActionServer__tensor_args.to_device(
                            plan_req.goal_state.position
                        ).unsqueeze(0),
                        joint_names=plan_req.goal_state.name,
                    )
                )
                with self.lock:
                    self.toggle_link_collision(plan_req.disable_collision_links, False)
                    motion_gen_result = self.motion_gen.plan_cspace(
                        goal_state,
                        start_state,
                        max_attempts=self._CumotionActionServer__max_attempts,
                    )
                    self.toggle_link_collision(plan_req.disable_collision_links, True)

            elif plan_req.plan_pose:
                self.get_logger().info("Planning Pose target")
                success, error_code, poses = self.get_goal_poses(plan_req)
                if not success:
                    result.error_code.val = error_code
                    goal_handle.abort(result)
                    return result

                num_goalset = poses.position.shape[1]
                flat_pose = Pose(
                    position=poses.position.view(-1, 3),
                    quaternion=poses.quaternion.view(-1, 4),
                )
                goal_tool_poses = GoalToolPose.from_poses(
                    {self.motion_gen.tool_frames[0]: flat_pose},
                    ordered_tool_frames=[self.motion_gen.tool_frames[0]],
                    num_goalset=num_goalset,
                )
                with self.lock:
                    self.toggle_link_collision(plan_req.disable_collision_links, False)
                    motion_gen_result = self.motion_gen.plan_pose(
                        goal_tool_poses,
                        start_state,
                        max_attempts=self._CumotionActionServer__max_attempts,
                    )
                    self.toggle_link_collision(plan_req.disable_collision_links, True)

            if (
                not plan_req.plan_grasp
                and motion_gen_result is not None
                and motion_gen_result.success is not None
                and motion_gen_result.success.any().item()
            ):
                result.error_code.val = MoveItErrorCodes.SUCCESS
                traj = self.get_joint_trajectory(
                    motion_gen_result.js_solution,
                    motion_gen_result.js_solution.dt.item(),
                )
                result.planning_time = (
                    motion_gen_result.total_time
                    if hasattr(motion_gen_result, "total_time")
                    else 0.0
                )
                result.planned_trajectory.append(traj)
                result.success = True
                if (
                    hasattr(motion_gen_result, "goalset_index")
                    and motion_gen_result.goalset_index is not None
                ):
                    result.goal_index = motion_gen_result.goalset_index.item()
            elif motion_gen_result is None:
                self.get_logger().error("Motion planning failed: result is None")
                result.error_code.val = MoveItErrorCodes.PLANNING_FAILED
            elif not plan_req.plan_grasp:
                self.get_logger().error("Motion planning failed")
                result.error_code.val = MoveItErrorCodes.PLANNING_FAILED

            if not plan_req.plan_grasp:
                self.get_logger().info(
                    "returned planning result (query, success): "
                    + str(self._CumotionActionServer__query_count)
                    + " "
                    + str(
                        motion_gen_result.success.item()
                        if motion_gen_result is not None
                        and motion_gen_result.success is not None
                        else False
                    )
                )

            self._CumotionActionServer__query_count += 1
            if result.success:
                goal_handle.succeed(result)
            else:
                goal_handle.abort(result)
            return result
        finally:
            with self.lock:
                self.planner_busy = False


def main(args=None):
    rclpy.init(args=args)
    cumotion_action_server = CumotionGoalSetPlannerServer()
    executor = MultiThreadedExecutor()
    executor.add_node(cumotion_action_server)
    try:
        executor.spin()
    except KeyboardInterrupt:
        cumotion_action_server.get_logger().info("KeyboardInterrupt, shutting down.\n")
    cumotion_action_server.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Dict, List, Set, Tuple

from curobo.scene import Cuboid, Cylinder, Mesh, Sphere
from curobo.types import GoalToolPose, JointState as CuJointState, Pose
from geometry_msgs.msg import Pose as RosPose
from isaac_ros_cumotion.cumotion_planner import CumotionActionServer
from isaac_ros_cumotion_interfaces.action import MotionPlan
from moveit_msgs.msg import CollisionObject
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.msg import PlanningScene
from shape_msgs.msg import SolidPrimitive
import numpy as np
import rclpy
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from scipy.spatial.transform import Rotation as R
import torch


class CumotionGoalSetPlannerServer(CumotionActionServer):

    def __init__(self):
        super().__init__()

        self._world_objects: Dict[str, CollisionObject] = {}
        self._attached_object_ids: Set[str] = set()
        self._planning_scene_sub = self.create_subscription(
            PlanningScene, "/planning_scene", self._on_planning_scene, 10
        )

        self._goal_set_planner_server = ActionServer(
            self, MotionPlan, "cumotion/motion_plan", self.motion_plan_execute_callback
        )

    def _on_planning_scene(self, msg: PlanningScene):
        world_updated = False

        for obj in msg.world.collision_objects:
            if obj.operation == CollisionObject.ADD:
                self._world_objects[obj.id] = obj
                world_updated = True
            elif obj.operation == CollisionObject.REMOVE:
                self._world_objects.pop(obj.id, None)
                self._attached_object_ids.discard(obj.id)
                world_updated = True

        for aco in msg.robot_state.attached_collision_objects:
            obj = aco.object
            if obj.operation == CollisionObject.ADD:
                self._world_objects.pop(obj.id, None)
                self._attached_object_ids.add(obj.id)
                self._attach_object_to_link(obj, aco.link_name)
                world_updated = True
            elif obj.operation == CollisionObject.REMOVE:
                self._attached_object_ids.discard(obj.id)
                self._detach_object_from_link(aco.link_name)
                world_updated = True

        if world_updated:
            self.update_world_objects(list(self._world_objects.values()))

    def _build_cu_pose(self, ros_pose, frame_id=None):
        return [
            ros_pose.position.x, ros_pose.position.y, ros_pose.position.z,
            ros_pose.orientation.w, ros_pose.orientation.x,
            ros_pose.orientation.y, ros_pose.orientation.z,
        ]

    def _attach_object_to_link(self, obj: CollisionObject, link_name: str):
        spheres_list = []
        for i, prim in enumerate(obj.primitives):
            pose = obj.primitive_poses[i]
            cu_pose = self._build_cu_pose(pose)

            if prim.type == SolidPrimitive.BOX:
                obstacle = Cuboid(
                    name=f"{obj.id}_{i}",
                    pose=cu_pose,
                    dims=[prim.dimensions[0], prim.dimensions[1], prim.dimensions[2]],
                )
            elif prim.type == SolidPrimitive.SPHERE:
                obstacle = Sphere(
                    name=f"{obj.id}_{i}",
                    pose=cu_pose,
                    radius=prim.dimensions[SolidPrimitive.SPHERE_RADIUS],
                )
            elif prim.type == SolidPrimitive.CYLINDER:
                obstacle = Cylinder(
                    name=f"{obj.id}_{i}",
                    pose=cu_pose,
                    height=prim.dimensions[SolidPrimitive.CYLINDER_HEIGHT],
                    radius=prim.dimensions[SolidPrimitive.CYLINDER_RADIUS],
                )
            else:
                continue

            cu_spheres = obstacle.get_bounding_spheres(
                num_spheres=100, surface_radius=0.01
            )
            for s in cu_spheres:
                spheres_list.append([s.pose[0], s.pose[1], s.pose[2], s.radius])

        for i, mesh in enumerate(obj.meshes):
            mesh_pose = obj.mesh_poses[i]
            cu_mesh_pose = self._build_cu_pose(mesh_pose)
            verts = [[v.x, v.y, v.z] for v in mesh.vertices]
            tris = [[v.vertex_indices[0], v.vertex_indices[1], v.vertex_indices[2]]
                    for v in mesh.triangles]
            obstacle = Mesh(
                name=f"{obj.id}_mesh_{i}",
                pose=cu_mesh_pose,
                vertices=verts,
                faces=tris,
            )
            cu_spheres = obstacle.get_bounding_spheres(
                num_spheres=100, surface_radius=0.01
            )
            for s in cu_spheres:
                spheres_list.append([s.pose[0], s.pose[1], s.pose[2], s.radius])

        if not spheres_list:
            self.get_logger().warn(f"No spheres generated for attached object {obj.id}")
            return

        sphere_tensor = torch.tensor(
            spheres_list, device=self.motion_gen.device_cfg.device, dtype=torch.float32
        )
        kin_config = self.motion_gen.kinematics.config.kinematics_config

        if link_name not in kin_config.link_name_to_idx_map:
            link_idx = len(kin_config.link_name_to_idx_map)
            kin_config.link_name_to_idx_map[link_name] = link_idx
        else:
            link_idx = kin_config.link_name_to_idx_map[link_name]

        existing = torch.nonzero(kin_config.link_sphere_idx_map == link_idx).view(-1)
        num_spheres = sphere_tensor.shape[0]

        if existing.shape[0] < num_spheres:
            spheres_to_add = num_spheres - existing.shape[0]
            n_configs = kin_config.link_spheres.shape[0]
            dev = kin_config.link_spheres.device
            dtype = kin_config.link_spheres.dtype
            idx_dtype = kin_config.link_sphere_idx_map.dtype

            new_idx = torch.full((spheres_to_add,), link_idx, device=dev, dtype=idx_dtype)
            kin_config.link_sphere_idx_map = torch.cat(
                [kin_config.link_sphere_idx_map, new_idx]
            )
            new_sph = torch.zeros((n_configs, spheres_to_add, 4), device=dev, dtype=dtype)
            kin_config.link_spheres = torch.cat([kin_config.link_spheres, new_sph], dim=1)

            if kin_config.reference_link_spheres is not None:
                new_ref = torch.zeros(
                    (n_configs, spheres_to_add, 4), device=dev, dtype=dtype
                )
                kin_config.reference_link_spheres = torch.cat(
                    [kin_config.reference_link_spheres, new_ref], dim=1
                )
            kin_config.total_spheres += spheres_to_add

        kin_config.update_link_spheres(
            link_name=link_name, sphere_position_radius=sphere_tensor
        )
        self.motion_gen.kinematics.update_batch_size(1, 1, reset_buffers=True)
        self.get_logger().info(
            f"Attached {sphere_tensor.shape[0]} spheres to link '{link_name}' for object '{obj.id}'"
        )

    def _detach_object_from_link(self, link_name: str):
        kin_config = self.motion_gen.kinematics.config.kinematics_config
        if link_name in kin_config.link_name_to_idx_map:
            kin_config.disable_link_spheres(link_name=link_name)
            self.get_logger().info(f"Detached spheres from link '{link_name}'")
        else:
            self.get_logger().warn(f"Link '{link_name}' not found for detachment")

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

    def _log_collision_scene_state(self):
        world_ids = list(self._world_objects.keys())
        attached_ids = list(self._attached_object_ids)
        js = self._CumotionActionServer__js_buffer
        if js is not None:
            pos = js.get("position", [])
            names = js.get("joint_names", [])
            pos_str = ", ".join(f"{p:.4f}" for p in pos[:7])
            self.get_logger().info(
                f"  joints[{len(names)}]: [{pos_str}]"
            )
        self.get_logger().info(
            f"  world_objects ({len(world_ids)}): {world_ids}"
        )
        self.get_logger().info(
            f"  attached_objects ({len(attached_ids)}): {attached_ids}"
        )
        for oid in world_ids:
            obj = self._world_objects[oid]
            for i, prim in enumerate(obj.primitives):
                pose = obj.primitive_poses[i]
                dims = list(prim.dimensions)
                p = pose.position
                self.get_logger().info(
                    f"    {oid}: prim[{i}] pos=({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) "
                    f"dims={dims}"
                )

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
                scene = goal_handle.request.world
                for obj in scene.collision_objects:
                    if obj.operation == CollisionObject.ADD:
                        self._world_objects[obj.id] = obj
                    elif obj.operation == CollisionObject.REMOVE:
                        self._world_objects.pop(obj.id, None)
                        self._attached_object_ids.discard(obj.id)
                self.update_world_objects(list(self._world_objects.values()))
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
                world_update_status = True
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
                self._log_collision_scene_state()
                result.error_code.val = MoveItErrorCodes.PLANNING_FAILED
            elif not plan_req.plan_grasp:
                self.get_logger().error("Motion planning failed")
                self._log_collision_scene_state()
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

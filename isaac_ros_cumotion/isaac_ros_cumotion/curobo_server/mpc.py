"""Model-predictive control handler.

Manages a stateful MPCSolver instance with a ROS timer that drives the
control loop. Lifecycle: ControlMPC action → timer → StopMPC service.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from curobo.model_predictive_control import (
    ModelPredictiveControl as MPCSolver,
    ModelPredictiveControlCfg,
)
from curobo.types import GoalToolPose, JointState as CuJointState, Pose

from geometry_msgs.msg import Pose as RosPose
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import JointState as RosJointState

from isaac_ros_cumotion_interfaces.action import ControlMPC
from isaac_ros_cumotion_interfaces.srv import StopMPC, UpdateMPCGoal

from .context import CuroboContext
from .conversions import (
    cu_joint_state_to_ros,
    cu_solution_to_joint_trajectory,
    ros_pose_to_cu_pose,
)


class MPCIntegration:
    """Drives one MPCSolver instance with a ROS 2 timer and action/service servers.

    The solver is built lazily on the first ``ControlMPC`` call and is
    persistent across start/stop cycles so warm-start state is preserved.
    """

    def __init__(self, node, context: CuroboContext, lock: threading.Lock,
                 cb_group: MutuallyExclusiveCallbackGroup):
        self._node = node
        self._context = context
        self._lock = lock
        self._cb_group = cb_group

        self._mpc: Optional[MPCSolver] = None
        self._mpc_timer = None
        self._mpc_running = False
        self._mpc_num_steps = 0
        self._mpc_start_time = 0.0
        self._mpc_current_state: Optional[CuJointState] = None
        self._active_goal_handle = None
        self._mpc_goal_tool_poses = None
        self._active_joint_names = None
        self._goal_completed = False

        # Publisher for commanded joint state
        self._cmd_pub = node.create_publisher(
            RosJointState, "cumotion/mpc/commanded_joint_state", 10,
        )

        # Service servers
        self._update_goal_srv = node.create_service(
            UpdateMPCGoal, "cumotion/mpc/update_goal",
            self._on_update_goal, callback_group=cb_group,
        )
        self._stop_mpc_srv = node.create_service(
            StopMPC, "cumotion/mpc/stop",
            self._on_stop_mpc, callback_group=cb_group,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def handle_control_mpc(self, context, goal_handle, lock, motion_planner):
        goal: ControlMPC.Goal = goal_handle.request
        result = ControlMPC.Result()

        try:
            # Build MPCSolver lazily (shares device/config from context)
            if self._mpc is None:
                self._build_mpc(motion_planner)
                self._node.get_logger().info("MPCSolver initialised")

            # Resolve start state
            if len(goal.start_state.position) > 0:
                start_state_raw = CuJointState.from_position(
                    position=motion_planner.device_cfg.to_device(
                        goal.start_state.position
                    ).unsqueeze(0),
                    joint_names=list(goal.start_state.name),
                )
                start_state = motion_planner.kinematics.get_active_js(start_state_raw)
            elif self._node._js_buffer is not None:
                js = self._node._js_buffer
                start_state = self._filter_joint_state(js)
                if start_state is None:
                    result.success = False
                    result.message = "Failed to filter joint state"
                    goal_handle.abort(result)
                    return result
            else:
                result.success = False
                result.message = "No start state available"
                goal_handle.abort(result)
                return result

            # Store the active joint names for later filtering
            self._active_joint_names = motion_planner.kinematics.joint_names

            # Resolve tool frame
            tool_frame = goal.tool_frame if goal.tool_frame else motion_planner.tool_frames[0]

            # Build goal poses
            if len(goal.goal_poses) == 0:
                result.success = False
                result.message = "goal_poses is empty"
                goal_handle.abort(result)
                return result

            pose_list = [
                [p.position.x, p.position.y, p.position.z,
                 p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z]
                for p in goal.goal_poses
            ]
            bp = Pose.from_batch_list(pose_list)
            flat_pose = Pose(
                position=bp.position.contiguous().view(-1, 3),
                quaternion=bp.quaternion.contiguous().view(-1, 4),
            )
            goal_tool_poses = GoalToolPose.from_poses(
                {tool_frame: flat_pose},
                ordered_tool_frames=[tool_frame],
                num_goalset=flat_pose.position.shape[0],
            )

            # Publish initial feedback
            feedback = ControlMPC.Feedback()
            feedback.phase = "setup"
            goal_handle.publish_feedback(feedback)

            # Setup MPC (this is synchronous)
            with lock:
                self._mpc.setup(start_state)
                self._mpc.update_goal_tool_poses(goal_tool_poses)

            # Store the goal handle and state BEFORE starting the timer
            self._active_goal_handle = goal_handle
            self._goal_completed = False
            self._mpc_num_steps = 0
            self._mpc_start_time = time.monotonic()
            self._mpc_current_state = start_state.clone()
            self._mpc_goal_tool_poses = goal_tool_poses

            # Start the timer
            dt = goal.optimization_dt if goal.optimization_dt > 0 else float(
                self._mpc.config.optimization_dt
            )
            timer_period_s = dt
            self._mpc_timer = self._node.create_timer(
                timer_period_s, self._mpc_timer_cb, callback_group=self._cb_group,
            )
            self._mpc_running = True

            feedback.phase = "running"
            goal_handle.publish_feedback(feedback)

            # Return a successful result - the action will be marked as complete
            # but the timer will continue running in the background.
            # The StopMPC service will stop the timer and complete the action.
            result.success = True
            result.message = "MPC started successfully. Use StopMPC service to stop."
            result.num_steps = 0
            result.total_control_time_s = 0.0
            result.final_state = RosJointState()
            
            # Mark the goal as completed so the timer doesn't try to complete it again
            self._goal_completed = True
            return result

        except Exception as e:
            context.logger.error(f"ControlMPC setup failed: {e}")
            result.success = False
            result.message = str(e)
            try:
                goal_handle.abort(result)
            except Exception:
                pass
            return result

    # ------------------------------------------------------------------
    # Helper to filter joint state to active joints
    # ------------------------------------------------------------------

    def _filter_joint_state(self, js_dict: dict) -> Optional[CuJointState]:
        """Filter a joint state dictionary to only active joints."""
        if js_dict is None:
            return None
        
        motion_planner = self._context.motion_planner
        active_names = motion_planner.kinematics.joint_names
        
        try:
            # Find indices of active joints in the message
            indices = [js_dict["joint_names"].index(n) for n in active_names]
            positions = [js_dict["position"][i] for i in indices]
            
            return CuJointState.from_position(
                position=motion_planner.device_cfg.to_device(
                    positions
                ).unsqueeze(0),
                joint_names=list(active_names),
            )
        except ValueError as e:
            self._context.logger.warn(f"Failed to filter joint state: {e}")
            return None

    # ------------------------------------------------------------------
    # Timer callback — drives the MPC loop
    # ------------------------------------------------------------------

    def _mpc_timer_cb(self):
        """Called at the MPC rate. Calls optimise_next_action and publishes."""
        if not self._mpc_running:
            return

        # If the goal has been completed (the action result was sent), we still
        # run the MPC but we don't try to publish feedback or complete the action.
        if self._goal_completed:
            # Run MPC without feedback
            try:
                self._run_mpc_step(publish_feedback=False)
            except Exception as e:
                self._context.logger.error(f"MPC timer callback failed: {e}")
                self._mpc_running = False
                if self._mpc_timer is not None:
                    self._mpc_timer.cancel()
                    self._mpc_timer = None
            return

        # Check if the goal handle is valid
        if self._active_goal_handle is None:
            self._context.logger.warn("MPC timer called but no active goal handle - stopping")
            self._stop_mpc_internal(success=False, message="No active goal handle")
            return

        try:
            self._run_mpc_step(publish_feedback=True)
        except Exception as e:
            self._context.logger.error(f"MPC timer callback failed: {e}")
            self._stop_mpc_internal(success=False, message=str(e))

    def _run_mpc_step(self, publish_feedback: bool = True):
        """Run a single MPC step and optionally publish feedback."""
        feedback = ControlMPC.Feedback()
        feedback.phase = "running"

        # Get current state from buffer and filter to active joints
        current_state = None
        if self._node._js_buffer is not None:
            current_state = self._filter_joint_state(self._node._js_buffer)
        
        if current_state is None:
            # Fall back to last known state
            current_state = self._mpc_current_state
            if current_state is None:
                self._context.logger.warn("No current state available for MPC")
                return

        with self._lock:
            # Check if the current state has changed significantly
            if self._mpc_current_state is not None and current_state is not None:
                if (current_state.position.shape[1] == 
                    self._mpc_current_state.position.shape[1]):
                    state_diff = (current_state.position - self._mpc_current_state.position).abs().max().item()
                    if state_diff > 0.01:
                        self._context.logger.info(
                            f"State changed significantly ({state_diff:.4f}), re-initializing..."
                        )
                        self._mpc.setup(current_state)
                        if self._mpc_goal_tool_poses is not None:
                            self._mpc.update_goal_tool_poses(self._mpc_goal_tool_poses)
                else:
                    self._context.logger.warn(
                        f"Joint count mismatch: current={current_state.position.shape[1]}, "
                        f"stored={self._mpc_current_state.position.shape[1]}, re-initializing..."
                    )
                    self._mpc.setup(current_state)
                    if self._mpc_goal_tool_poses is not None:
                        self._mpc.update_goal_tool_poses(self._mpc_goal_tool_poses)
            
            mpc_result = self._mpc.optimize_next_action(current_state)

        self._mpc_num_steps += 1
        self._mpc_current_state = current_state.clone() if current_state is not None else None

        if mpc_result is not None and mpc_result.next_action is not None:
            # Publish commanded joint state
            ros_cmd = cu_joint_state_to_ros(
                mpc_result.next_action, 
                stamp=self._node.get_clock().now().to_msg()
            )
            self._cmd_pub.publish(ros_cmd)

            # Publish action feedback if requested
            if publish_feedback:
                feedback.current_action = ros_cmd
                feedback.solve_time_s = float(mpc_result.solve_time) if hasattr(mpc_result, "solve_time") else 0.0
                if hasattr(mpc_result, "position_error") and mpc_result.position_error is not None:
                    feedback.tracking_position_error = float(mpc_result.position_error.max().item())
                if hasattr(mpc_result, "rotation_error") and mpc_result.rotation_error is not None:
                    feedback.tracking_rotation_error = float(mpc_result.rotation_error.max().item())
            else:
                feedback.solve_time_s = 0.0
                feedback.tracking_position_error = -1.0
                feedback.tracking_rotation_error = -1.0
        else:
            feedback.solve_time_s = 0.0
            feedback.tracking_position_error = -1.0
            feedback.tracking_rotation_error = -1.0

        # Publish feedback only if requested and the goal is still active
        if (publish_feedback and self._active_goal_handle is not None 
            and self._mpc_running and not self._goal_completed):
            try:
                if self._active_goal_handle.is_active:
                    self._active_goal_handle.publish_feedback(feedback)
            except Exception as e:
                self._context.logger.warn(f"Failed to publish feedback: {e}")
                # If we can't publish feedback, the goal is probably dead
                self._stop_mpc_internal(success=False, message="Goal handle lost")

    # ------------------------------------------------------------------
    # Service handlers
    # ------------------------------------------------------------------

    def _on_update_goal(self, request, response):
        if not self._mpc_running:
            response.success = False
            response.message = "MPC is not running"
            return response

        try:
            pose_list = [
                [p.position.x, p.position.y, p.position.z,
                 p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z]
                for p in request.goal_poses
            ]
            if not pose_list:
                response.success = False
                response.message = "goal_poses is empty"
                return response

            bp = Pose.from_batch_list(pose_list)
            flat_pose = Pose(
                position=bp.position.contiguous().view(-1, 3),
                quaternion=bp.quaternion.contiguous().view(-1, 4),
            )
            tool_frame = request.tool_frame if request.tool_frame else self._context.motion_planner.tool_frames[0]
            goal_tool_poses = GoalToolPose.from_poses(
                {tool_frame: flat_pose},
                ordered_tool_frames=[tool_frame],
                num_goalset=flat_pose.position.shape[0],
            )

            with self._lock:
                self._mpc.update_goal_tool_poses(goal_tool_poses)
                self._mpc_goal_tool_poses = goal_tool_poses

            response.success = True
            response.message = "Goal updated"
        except Exception as e:
            self._context.logger.error(f"UpdateMPCGoal failed: {e}")
            response.success = False
            response.message = str(e)

        return response

    def _on_stop_mpc(self, request, response):
        if not self._mpc_running:
            response.success = False
            response.message = "MPC is not running"
            return response

        self._stop_mpc_internal(success=True, message="Stopped by service", service_response=response)
        return response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stop_mpc_internal(self, success: bool, message: str,
                           service_response=None):
        """Stop the MPC timer and complete the action."""
        self._mpc_running = False
        
        if self._mpc_timer is not None:
            self._mpc_timer.cancel()
            self._mpc_timer = None

        total_time = time.monotonic() - self._mpc_start_time if self._mpc_start_time > 0 else 0.0

        # Build the result
        result = ControlMPC.Result()
        result.success = success
        result.message = message
        result.num_steps = self._mpc_num_steps
        result.total_control_time_s = float(total_time)
        if self._mpc_current_state is not None:
            ros_state = cu_joint_state_to_ros(self._mpc_current_state)
            result.final_state = ros_state

        # Only try to complete the goal if it hasn't been completed yet
        if not self._goal_completed and self._active_goal_handle is not None:
            try:
                if hasattr(self._active_goal_handle, 'is_active') and self._active_goal_handle.is_active:
                    if success:
                        self._active_goal_handle.succeed(result)
                    else:
                        self._active_goal_handle.abort(result)
                else:
                    self._context.logger.info(f"Goal already completed/canceled, not sending result")
            except Exception as e:
                self._context.logger.warn(f"Failed to complete goal: {e}")
            finally:
                self._active_goal_handle = None
                self._goal_completed = True

        # Also handle the service response if this was called from a service
        if service_response is not None:
            service_response.success = success
            service_response.message = message
            service_response.num_steps = self._mpc_num_steps
            service_response.total_control_time_s = float(total_time)
            if self._mpc_current_state is not None:
                ros_state = cu_joint_state_to_ros(self._mpc_current_state)
                service_response.final_state = ros_state

    def _build_mpc(self, motion_planner):
        """Construct MPCSolver sharing device/config from the motion planner."""
        ik_cfg = motion_planner.config.ik_solver_config
        scene_collision_cfg = motion_planner.config.scene_collision_cfg
        scene_model = (
            scene_collision_cfg.scene_model if scene_collision_cfg is not None else None
        )

        cfg = ModelPredictiveControlCfg.create(
            robot=ik_cfg.robot_config,
            scene_model=scene_model,
            self_collision_check=ik_cfg.self_collision_check,
            device_cfg=motion_planner.device_cfg,
            use_cuda_graph=ik_cfg.core_cfg.use_cuda_graph,
        )
        self._mpc = MPCSolver(cfg)
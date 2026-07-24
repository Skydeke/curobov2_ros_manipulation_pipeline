"""Model-predictive control handler.

Manages a stateful MPCSolver instance with a ROS timer that drives the
control loop. Lifecycle: ControlMPC action → timer → StopMPC service.
"""

from __future__ import annotations

import threading
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

import time


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
        self._mpc_goal_handle = None
        self._mpc_timer = None
        self._mpc_running = False
        self._mpc_num_steps = 0
        self._mpc_start_time = 0.0
        self._mpc_current_state: Optional[CuJointState] = None
        self._active_goal_handle = None

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
        feedback = ControlMPC.Feedback()

        try:
            # Build MPCSolver lazily (shares device/config from context)
            if self._mpc is None:
                self._build_mpc(motion_planner)
                self._node.get_logger().info("MPCSolver initialised")

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
            elif self._node._js_buffer is not None:
                js = self._node._js_buffer
                start_state = CuJointState.from_position(
                    position=motion_planner.device_cfg.to_device(
                        js["position"]
                    ).unsqueeze(0),
                    joint_names=list(js["joint_names"]),
                )
                start_state = motion_planner.kinematics.get_active_js(start_state)
            else:
                result.success = False
                result.message = "No start state available"
                goal_handle.abort(result)
                return result

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

            feedback.phase = "setup"
            goal_handle.publish_feedback(feedback)

            with lock:
                self._mpc.setup(start_state, tool_frames=[tool_frame])
                self._mpc.update_goal_tool_poses(goal_tool_poses)

            self._active_goal_handle = goal_handle
            self._mpc_num_steps = 0
            self._mpc_start_time = time.monotonic()
            self._mpc_current_state = start_state.clone()

            # Start timer at the MPC rate
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

            # The action remains active until StopMPC is called.
            # We return a deferred result (the action API will not send the
            # result until goal_handle.succeed()/abort() is called).
            # The timer callback will complete the action when stopped.
            return result  # Placeholder — see _finish_mpc for the real result

        except Exception as e:
            context.logger.error(f"ControlMPC setup failed: {e}")
            result.success = False
            result.message = str(e)
            goal_handle.abort(result)
            return result

    # ------------------------------------------------------------------
    # Timer callback — drives the MPC loop
    # ------------------------------------------------------------------

    def _mpc_timer_cb(self):
        """Called at the MPC rate. Calls optimise_next_action and publishes."""
        if not self._mpc_running or self._active_goal_handle is None:
            return

        try:
            feedback = ControlMPC.Feedback()
            feedback.phase = "running"

            # Get current state from buffer
            if self._node._js_buffer is not None:
                js = self._node._js_buffer
                current_state = CuJointState.from_position(
                    position=self._context.motion_planner.device_cfg.to_device(
                        js["position"]
                    ).unsqueeze(0),
                    joint_names=list(js["joint_names"]),
                )
            else:
                current_state = self._mpc_current_state

            with self._lock:
                mpc_result = self._mpc.optimize_next_action(current_state)

            self._mpc_num_steps += 1
            self._mpc_current_state = current_state.clone() if current_state is not None else None

            if mpc_result is not None and mpc_result.next_action is not None:
                # Publish commanded joint state
                ros_cmd = cu_joint_state_to_ros(mpc_result.next_action, stamp=self._node.now().to_msg())
                self._cmd_pub.publish(ros_cmd)

                # Publish action feedback
                feedback.current_action = ros_cmd
                feedback.solve_time_s = float(mpc_result.solve_time) if hasattr(mpc_result, "solve_time") else 0.0
                if mpc_result.position_error is not None:
                    feedback.tracking_position_error = float(mpc_result.position_error.max().item())
                if mpc_result.rotation_error is not None:
                    feedback.tracking_rotation_error = float(mpc_result.rotation_error.max().item())
            else:
                feedback.solve_time_s = 0.0
                feedback.tracking_position_error = -1.0
                feedback.tracking_rotation_error = -1.0

            if self._active_goal_handle is not None:
                self._active_goal_handle.publish_feedback(feedback)

        except Exception as e:
            self._context.logger.error(f"MPC timer callback failed: {e}")
            self._stop_mpc_internal(success=False, message=str(e))

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

        if service_response is not None:
            service_response.success = success
            service_response.message = message
            service_response.num_steps = self._mpc_num_steps
            service_response.total_control_time_s = float(total_time)
            if self._mpc_current_state is not None:
                ros_state = cu_joint_state_to_ros(self._mpc_current_state)
                service_response.final_state = ros_state

        if self._active_goal_handle is not None:
            result = ControlMPC.Result()
            result.success = success
            result.message = message
            result.num_steps = self._mpc_num_steps
            result.total_control_time_s = float(total_time)
            if self._mpc_current_state is not None:
                ros_state = cu_joint_state_to_ros(self._mpc_current_state)
                result.final_state = ros_state

            if success:
                self._active_goal_handle.succeed(result)
            else:
                self._active_goal_handle.abort(result)
            self._active_goal_handle = None

    def _build_mpc(self, motion_planner):
        """Construct MPCSolver sharing device/config from the motion planner."""
        cfg = ModelPredictiveControlCfg.create(
            robot=motion_planner.robot_config,
            scene_model=motion_planner.scene_model,
            self_collision_check=motion_planner.self_collision_check,
            device_cfg=motion_planner.device_cfg,
            use_cuda_graph=motion_planner.config.use_cuda_graph,
            max_goalset=motion_planner.config.max_goalset,
            max_batch_size=motion_planner.config.max_batch_size,
        )
        self._mpc = MPCSolver(cfg)
        self._mpc.warmup()

"""FK handler for ComputeFK service.

Uses ``context.motion_planner.kinematics.compute_kinematics()`` to compute
forward kinematics for one or more joint configurations on the GPU.
"""

from __future__ import annotations

import time

import torch

from isaac_ros_cumotion_interfaces.srv import ComputeFK

from .context import CuroboContext


def handle_compute_fk(context: CuroboContext, request, response, lock):
    try:
        num_configs = len(request.joint_states)
        if num_configs == 0:
            response.success = True
            response.message = "No joint states provided"
            response.tool_poses = []
            response.resolved_frame_names = []
            response.num_configs = 0
            response.num_frames = 0
            response.solve_time_s = 0.0
            return response

        t_start = time.perf_counter()
        kin = context.motion_planner.kinematics

        # Determine which tool frames to query
        if len(request.tool_frames) > 0:
            frame_names = list(request.tool_frames)
        else:
            frame_names = list(kin.tool_frames)

        with lock:
            # Stack all joint configurations
            positions = torch.stack([
                torch.tensor(js.position, dtype=torch.float32, device=context.device)
                for js in request.joint_states
            ])

            from curobo.types import JointState as CuJointState
            joint_state = CuJointState.from_position(
                position=positions,
                joint_names=list(request.joint_states[0].name),
            )

            kin_state = kin.compute_kinematics(joint_state)

        # Extract tool poses: [B, H, L, 3/4] with B=num_configs, H=1, L=num_frames
        tp = kin_state.tool_poses
        num_frames = len(frame_names)

        ros_poses = []
        for i in range(num_configs):
            for frame_name in frame_names:
                if frame_name in tp.tool_frames:
                    li = tp.tool_frames.index(frame_name)
                    pos = tp.position[i, 0, li]
                    quat = tp.quaternion[i, 0, li]
                else:
                    pos = torch.zeros(3, device=context.device)
                    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=context.device)

                from geometry_msgs.msg import Pose as RosPose
                p = RosPose()
                p.position.x = float(pos[0].item())
                p.position.y = float(pos[1].item())
                p.position.z = float(pos[2].item())
                p.orientation.w = float(quat[0].item())
                p.orientation.x = float(quat[1].item())
                p.orientation.y = float(quat[2].item())
                p.orientation.z = float(quat[3].item())
                ros_poses.append(p)

        dt = time.perf_counter() - t_start

        response.success = True
        response.message = f"FK solved for {num_configs} configs x {num_frames} frames"
        response.tool_poses = ros_poses
        response.resolved_frame_names = frame_names
        response.num_configs = num_configs
        response.num_frames = num_frames
        response.solve_time_s = float(dt)

        return response

    except Exception as e:
        context.logger.error(f"ComputeFK failed: {e}")
        response.success = False
        response.message = str(e)
        response.tool_poses = []
        response.resolved_frame_names = []
        response.num_configs = 0
        response.num_frames = 0
        response.solve_time_s = 0.0
        return response

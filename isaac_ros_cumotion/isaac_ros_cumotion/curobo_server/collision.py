"""Collision-checking handler for CheckCollision service.

Uses ``MotionPlanner.scene_collision_checker`` (the shared
``RobotCollisionChecker`` built inside ``MotionPlanner``) to evaluate
world- and self-collision distances for one or more joint configurations.
"""

from __future__ import annotations

import torch

from curobo.types import JointState as CuJointState

from isaac_ros_cumotion_interfaces.srv import CheckCollision

from .context import CuroboContext


def handle_check_collision(context: CuroboContext, request, response):
    try:
        num_states = len(request.joint_states)
        response.in_collision = [False] * num_states
        response.world_collision_distance = [0.0] * num_states
        response.self_collision_distance = [0.0] * num_states

        if num_states == 0:
            return response

        collision_checker = context.motion_planner.scene_collision_checker
        kinematics = context.motion_planner.kinematics

        for i, js in enumerate(request.joint_states):
            if len(js.position) == 0:
                response.in_collision[i] = True
                continue

            cu_js = CuJointState.from_position(
                position=context.motion_planner.device_cfg.to_device(
                    list(js.position)
                ).unsqueeze(0),
                joint_names=list(js.name),
            )
            active = kinematics.get_active_js(cu_js)

            # Collision checking
            cd = collision_checker.compute_collision_distance(active)

            response.in_collision[i] = bool(cd.in_collision.item()) if cd.in_collision is not None else False

            if cd.world_collision_distance is not None:
                response.world_collision_distance[i] = float(cd.world_collision_distance.item())
            if cd.self_collision_distance is not None:
                response.self_collision_distance[i] = float(cd.self_collision_distance.item())

        return response

    except Exception as e:
        context.logger.error(f"CheckCollision failed: {e}")
        num_states = len(request.joint_states) if hasattr(request, "joint_states") else 0
        response.in_collision = [True] * num_states
        response.world_collision_distance = [0.0] * num_states
        response.self_collision_distance = [0.0] * num_states
        return response

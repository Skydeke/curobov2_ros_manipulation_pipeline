"""Collision-checking handler for CheckCollision service.

Uses ``MotionPlanner.scene_collision_checker.get_sphere_distance`` (the
real cuRobo API) to evaluate world-collision distances for one or more
joint configurations.  Self-collision is not directly exposed through
``SceneCollision`` so it is reported as 0.0 (use the planning-based path
if you need self-collision awareness).
"""

from __future__ import annotations

import torch

from curobo.types import JointState as CuJointState
from curobo._src.geom.collision.buffer_collision import CollisionBuffer

from isaac_ros_cumotion_interfaces.srv import CheckCollision

from .context import CuroboContext
from .world import sync_world


def handle_check_collision(context: CuroboContext, request, response):
    sync_world(context)
    try:
        num_states = len(request.joint_states)
        response.in_collision = [False] * num_states
        response.world_collision_distance = [0.0] * num_states
        response.self_collision_distance = [0.0] * num_states

        if num_states == 0:
            return response

        collision_checker = context.motion_planner.scene_collision_checker
        kinematics = context.motion_planner.kinematics
        device_cfg = context.motion_planner.device_cfg

        for i, js in enumerate(request.joint_states):
            if len(js.position) == 0:
                response.in_collision[i] = True
                continue

            cu_js = CuJointState.from_position(
                position=device_cfg.to_device(
                    list(js.position)
                ).unsqueeze(0),
                joint_names=list(js.name),
            )
            active = kinematics.get_active_js(cu_js)

            kin_state = kinematics.compute_kinematics(active)
            b, h, num_spheres = kin_state.robot_spheres.shape[0], kin_state.robot_spheres.shape[1], kin_state.robot_spheres.shape[2]

            collision_buffer = CollisionBuffer.from_shape(
                (b, h, num_spheres, 4), device_cfg
            )
            weight = device_cfg.to_device([1.0])
            activation_distance = device_cfg.to_device([0.0])

            dist = collision_checker.get_sphere_collision(
                kin_state, collision_buffer, weight, activation_distance
            )

            response.in_collision[i] = bool((dist > 0).any().item())
            response.world_collision_distance[i] = float(dist.max().item())

        return response

    except Exception as e:
        context.logger.error(f"CheckCollision failed: {e}")
        num_states = len(request.joint_states) if hasattr(request, "joint_states") else 0
        response.in_collision = [True] * num_states
        response.world_collision_distance = [0.0] * num_states
        response.self_collision_distance = [0.0] * num_states
        return response

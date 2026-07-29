"""World update handlers for UpdateWorld and PublishStaticPlanningScene services."""

from __future__ import annotations

import os
from typing import Dict, List

from curobo.scene import Cuboid, Cylinder, Mesh, Scene, Sphere
from moveit_msgs.msg import CollisionObject as MoveItCollisionObject
from moveit_msgs.msg import MoveItErrorCodes

from isaac_ros_cumotion_interfaces.srv import UpdateWorld
from isaac_ros_cumotion.curobo_server.moveit_scene_file_parser import MoveItSceneFileReader

from .context import CuroboContext
from .conversions import collision_object_to_scene_objects


def _rebuild_world(context: CuroboContext) -> bool:
    """Rebuild the full world model from ``context.world_objects`` and push to cuRobo."""
    cuboid_list: List[Cuboid] = []
    sphere_list: List[Sphere] = []
    cylinder_list: List[Cylinder] = []
    mesh_list: List[Mesh] = []

    for obj in context.world_objects.values():
        cu_objs, ok = collision_object_to_scene_objects(obj)
        for cu_obj in cu_objs:
            if isinstance(cu_obj, Cuboid):
                cuboid_list.append(cu_obj)
            elif isinstance(cu_obj, Cylinder):
                cylinder_list.append(cu_obj)
            elif isinstance(cu_obj, Sphere):
                sphere_list.append(cu_obj)
            elif isinstance(cu_obj, Mesh):
                mesh_list.append(cu_obj)

    world_model = Scene(
        cuboid=cuboid_list,
        cylinder=cylinder_list,
        sphere=sphere_list,
        mesh=mesh_list,
    )
    context.motion_planner.update_world(world_model)
    return True


def handle_update_world(context: CuroboContext, request: UpdateWorld.Request,
                        response: UpdateWorld.Response):
    operation = request.operation

    try:
        if operation == UpdateWorld.Request.CLEAR_ALL:
            context.world_objects.clear()
            _rebuild_world(context)
            response.success = True
            response.message = "World cleared"
            response.num_obstacles_in_world = 0
            return response

        elif operation == UpdateWorld.Request.REMOVE:
            for oid in request.remove_ids:
                context.world_objects.pop(oid, None)
            _rebuild_world(context)
            response.success = True
            response.message = f"Removed {len(request.remove_ids)} object(s)"
            response.num_obstacles_in_world = len(context.world_objects)
            return response

        elif operation in (UpdateWorld.Request.ADD, UpdateWorld.Request.REPLACE):
            for obj in request.objects:
                if operation == UpdateWorld.Request.ADD:
                    if obj.id in context.world_objects:
                        response.success = False
                        response.message = f"Object '{obj.id}' already exists; use REPLACE to overwrite"
                        response.num_obstacles_in_world = len(context.world_objects)
                        return response
                context.world_objects[obj.id] = obj
            _rebuild_world(context)
            response.success = True
            response.message = f"{'Replaced' if operation == UpdateWorld.Request.REPLACE else 'Added'} {len(request.objects)} object(s)"
            response.num_obstacles_in_world = len(context.world_objects)
            return response

        else:
            response.success = False
            response.message = f"Unknown operation: {operation}"
            response.num_obstacles_in_world = len(context.world_objects)
            return response

    except Exception as e:
        context.logger.error(f"UpdateWorld failed: {e}")
        response.success = False
        response.message = str(e)
        response.num_obstacles_in_world = len(context.world_objects)
        return response


def handle_publish_static_scene(context: CuroboContext, request, response):
    scene_file = request.scene_file_path
    if not scene_file and context.node is not None:
        scene_file = context.node.get_parameter(
            "moveit_collision_objects_scene_file"
        ).get_parameter_value().string_value
    if not scene_file:
        response.success = False
        response.message = "No static planning scene file path provided"
        response.status = 1
        return response
    if not os.path.exists(scene_file):
        response.success = False
        response.message = f"Scene file not found: {scene_file}"
        response.status = 2
        return response
    try:
        reader = MoveItSceneFileReader()
        scene_msg = reader.parse_scene_file(scene_file)
        for obj in scene_msg.world.collision_objects:
            context.world_objects[obj.id] = obj
        _rebuild_world(context)
        response.planning_scene = scene_msg
        response.success = True
        response.message = "Planning scene published successfully."
        response.status = 0
    except Exception as e:
        context.logger.error(f"Failed to publish planning scene: {e}")
        response.success = False
        response.message = f"Failed to publish planning scene: {e}"
        response.status = 3
    return response

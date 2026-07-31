"""Attach/detach handler for AttachObject action.

Operates on ``MotionPlanner.kinematics`` directly — does NOT build a second
``Kinematics`` instance. Supports both the sphere-list fast path and the
primitive+auto-fit path.
"""

from __future__ import annotations

from typing import List

from curobo.scene import Cuboid, Cylinder, Mesh, Sphere
from geometry_msgs.msg import Point, Pose
from moveit_msgs.msg import CollisionObject as MoveItCollisionObject

import torch

from isaac_ros_cumotion_interfaces.action import AttachObject
from shape_msgs.msg import SolidPrimitive

from .context import CuroboContext


def _attach_spheres_to_link(
    context: CuroboContext,
    object_id: str,
    parent_link: str,
    spheres: List[List[float]],
):
    """Attach pre-computed collision spheres to a link via kinematics config."""
    sphere_tensor = torch.tensor(
        spheres,
        device=context.device,
        dtype=torch.float32,
    )
    kin_config = context.motion_planner.kinematics.config.kinematics_config

    if parent_link not in kin_config.link_name_to_idx_map:
        link_idx = len(kin_config.link_name_to_idx_map)
        kin_config.link_name_to_idx_map[parent_link] = link_idx
    else:
        link_idx = kin_config.link_name_to_idx_map[parent_link]

    existing = torch.nonzero(kin_config.link_sphere_idx_map == link_idx).view(-1)
    num_spheres = sphere_tensor.shape[0]

    if existing.shape[0] < num_spheres:
        spheres_to_add = num_spheres - existing.shape[0]
        n_configs = kin_config.link_spheres.shape[0]
        dev = kin_config.link_spheres.device
        dtype = kin_config.link_spheres.dtype
        idx_dtype = kin_config.link_sphere_idx_map.dtype

        new_idx = torch.full((spheres_to_add,), link_idx, device=dev, dtype=idx_dtype)
        kin_config.link_sphere_idx_map = torch.cat([kin_config.link_sphere_idx_map, new_idx])
        new_sph = torch.zeros((n_configs, spheres_to_add, 4), device=dev, dtype=dtype)
        kin_config.link_spheres = torch.cat([kin_config.link_spheres, new_sph], dim=1)

        if kin_config.reference_link_spheres is not None:
            new_ref = torch.zeros((n_configs, spheres_to_add, 4), device=dev, dtype=dtype)
            kin_config.reference_link_spheres = torch.cat([kin_config.reference_link_spheres, new_ref], dim=1)
        kin_config.total_spheres += spheres_to_add

    kin_config.update_link_spheres(
        link_name=parent_link,
        sphere_position_radius=sphere_tensor,
    )
    context.motion_planner.kinematics.update_batch_size(1, 1, reset_buffers=True)


def _detach_from_link(context: CuroboContext, parent_link: str):
    """Disable all spheres on a given link."""
    kin_config = context.motion_planner.kinematics.config.kinematics_config
    if parent_link in kin_config.link_name_to_idx_map:
        kin_config.disable_link_spheres(link_name=parent_link)


def _fit_spheres_from_primitive(
    context: CuroboContext,
    primitive: SolidPrimitive,
    object_id: str,
) -> List[List[float]]:
    """Fit bounding spheres to a SolidPrimitive."""
    prim_type = primitive.type
    dims = list(primitive.dimensions)
    pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # identity — caller applies pose offset

    if prim_type == SolidPrimitive.BOX:
        obstacle = Cuboid(
            name=f"{object_id}_prim",
            pose=pose,
            dims=[dims[SolidPrimitive.BOX_X], dims[SolidPrimitive.BOX_Y], dims[SolidPrimitive.BOX_Z]],
        )
    elif prim_type == SolidPrimitive.SPHERE:
        obstacle = Sphere(
            name=f"{object_id}_prim",
            pose=pose,
            radius=dims[SolidPrimitive.SPHERE_RADIUS],
        )
    elif prim_type == SolidPrimitive.CYLINDER:
        obstacle = Cylinder(
            name=f"{object_id}_prim",
            pose=pose,
            height=dims[SolidPrimitive.CYLINDER_HEIGHT],
            radius=dims[SolidPrimitive.CYLINDER_RADIUS],
        )
    else:
        return []

    num_spheres = 100
    cu_spheres = obstacle.get_bounding_spheres(
        num_spheres=num_spheres,
        surface_radius=0.01,
    )
    return [[s.pose[0], s.pose[1], s.pose[2], s.radius] for s in cu_spheres]


def _transform_spheres(
    spheres: List[List[float]],
    pose: Pose,
) -> List[List[float]]:
    """Apply a geometry_msgs/Pose offset to [x, y, z, r] sphere lists.

    Rotation is applied to the centre before the translation; radii are
    unaffected (uniform-scale assumption, matching cuRobo's Sphere type).
    """
    q = [
        pose.orientation.w,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
    ]
    norm = (q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2) ** 0.5
    if norm > 0.0:
        q = [v / norm for v in q]
    w, x, y, z = q
    tx, ty, tz = pose.position.x, pose.position.y, pose.position.z

    out = []
    for s in spheres:
        px, py, pz, r = s
        # Rotate (row-vector convention: v' = R^T @ v with this R).
        rx = (1 - 2 * (y * y + z * z)) * px + (2 * (x * y - z * w)) * py + (2 * (x * z + y * w)) * pz
        ry = (2 * (x * y + z * w)) * px + (1 - 2 * (x * x + z * z)) * py + (2 * (y * z - x * w)) * pz
        rz = (2 * (x * z - y * w)) * px + (2 * (y * z + x * w)) * py + (1 - 2 * (x * x + y * y)) * pz
        out.append([rx + tx, ry + ty, rz + tz, r])
    return out


def attach_collision_object(
    context: CuroboContext,
    object_id: str,
    parent_link: str,
    collision_object: MoveItCollisionObject,
    lock=None,
) -> int:
    """Attach a MoveIt CollisionObject (primitives + poses) to a link as spheres.

    Used by the planning-scene attach path (AttachedCollisionObject). Returns the
    number of spheres attached, or 0 if no supported geometry was found.
    """
    spheres: List[List[float]] = []
    for i, prim in enumerate(collision_object.primitives):
        fitted = _fit_spheres_from_primitive(context, prim, f"{object_id}_prim_{i}")
        if not fitted:
            continue
        if i < len(collision_object.primitive_poses):
            fitted = _transform_spheres(fitted, collision_object.primitive_poses[i])
        spheres.extend(fitted)

    if not spheres:
        return 0

    if lock is not None:
        with lock:
            _attach_spheres_to_link(context, object_id, parent_link, spheres)
    else:
        _attach_spheres_to_link(context, object_id, parent_link, spheres)
    return len(spheres)


def handle_attach_object(context: CuroboContext, goal_handle, js_buffer, lock):
    goal: AttachObject.Goal = goal_handle.request
    result = AttachObject.Result()
    feedback = AttachObject.Feedback()

    try:
        if not goal.attach:
            # --- Detach ---
            _detach_from_link(context, goal.parent_link)
            context.attached_objects.pop(goal.object_id, None)
            result.success = True
            result.message = f"Object '{goal.object_id}' detached"
            result.num_spheres_attached = 0
            feedback.status = "detached"
            goal_handle.publish_feedback(feedback)
            goal_handle.succeed(result)
            return result

        # --- Attach ---
        object_id = goal.object_id
        parent_link = goal.parent_link

        # Option A: exact spheres provided
        spheres: List[List[float]] = []
        if len(goal.sphere_centers) > 0 and len(goal.sphere_radii) > 0:
            if len(goal.sphere_centers) != len(goal.sphere_radii):
                result.success = False
                result.message = "sphere_centers and sphere_radii must have the same length"
                result.num_spheres_attached = 0
                goal_handle.abort(result)
                return result

            # Apply pose_in_parent_link offset
            px = goal.pose_in_parent_link.position.x
            py = goal.pose_in_parent_link.position.y
            pz = goal.pose_in_parent_link.position.z
            for i, c in enumerate(goal.sphere_centers):
                spheres.append([c.x + px, c.y + py, c.z + pz, goal.sphere_radii[i]])

        # Option B: primitive shape + auto-fit
        elif goal.primitive.type != 0:
            spheres = _fit_spheres_from_primitive(context, goal.primitive, object_id)

        # Fallback: single sphere from fallback radius
        if not spheres:
            if goal.fallback_sphere_radius > 0.0:
                px = goal.pose_in_parent_link.position.x
                py = goal.pose_in_parent_link.position.y
                pz = goal.pose_in_parent_link.position.z
                spheres.append([px, py, pz, goal.fallback_sphere_radius])
            else:
                result.success = False
                result.message = "No collision geometry provided and fallback_sphere_radius <= 0"
                result.num_spheres_attached = 0
                goal_handle.abort(result)
                return result

        # Auto-detach if already attached
        if object_id in context.attached_objects:
            _detach_from_link(context, parent_link)
            context.attached_objects.pop(object_id, None)

        with lock:
            _attach_spheres_to_link(context, object_id, parent_link, spheres)

        context.attached_objects[object_id] = goal
        result.success = True
        result.message = f"Object '{object_id}' attached with {len(spheres)} spheres"
        result.num_spheres_attached = len(spheres)
        feedback.status = "attached"
        goal_handle.publish_feedback(feedback)
        goal_handle.succeed(result)
        return result

    except Exception as e:
        context.logger.error(f"AttachObject failed: {e}")
        result.success = False
        result.message = str(e)
        result.num_spheres_attached = 0
        goal_handle.abort(result)
        return result

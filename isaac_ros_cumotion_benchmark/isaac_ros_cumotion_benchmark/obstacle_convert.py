from typing import Dict, List, Union

from geometry_msgs.msg import Pose as RosPose
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header


def _pose_list_to_ros(msg: RosPose, pose_list):
    msg.position.x = float(pose_list[0])
    msg.position.y = float(pose_list[1])
    msg.position.z = float(pose_list[2])
    msg.orientation.w = float(pose_list[3])
    msg.orientation.x = float(pose_list[4])
    msg.orientation.y = float(pose_list[5])
    msg.orientation.z = float(pose_list[6])


def _make_identity_pose() -> RosPose:
    p = RosPose()
    p.orientation.w = 1.0
    return p


def _cuboid_dict_to_collision_object(name: str, data: dict, frame_id: str = "world") -> CollisionObject:
    co = CollisionObject()
    co.header = Header(frame_id=frame_id)
    co.id = name
    _pose_list_to_ros(co.pose, data["pose"])

    prim = SolidPrimitive()
    prim.type = SolidPrimitive.BOX
    prim.dimensions = [float(d) for d in data["dims"]]

    co.primitives = [prim]
    co.primitive_poses = [_make_identity_pose()]
    co.operation = CollisionObject.ADD
    return co


def _cylinder_dict_to_collision_object(name: str, data: dict, frame_id: str = "world") -> CollisionObject:
    co = CollisionObject()
    co.header = Header(frame_id=frame_id)
    co.id = name
    _pose_list_to_ros(co.pose, data["pose"])

    prim = SolidPrimitive()
    prim.type = SolidPrimitive.CYLINDER
    prim.dimensions = [float(data["height"]), float(data["radius"])]

    co.primitives = [prim]
    co.primitive_poses = [_make_identity_pose()]
    co.operation = CollisionObject.ADD
    return co


def _sphere_dict_to_collision_object(name: str, data: dict, frame_id: str = "world") -> CollisionObject:
    co = CollisionObject()
    co.header = Header(frame_id=frame_id)
    co.id = name
    _pose_list_to_ros(co.pose, data["pose"])

    prim = SolidPrimitive()
    prim.type = SolidPrimitive.SPHERE
    prim.dimensions = [float(data["radius"])]

    co.primitives = [prim]
    co.primitive_poses = [_make_identity_pose()]
    co.operation = CollisionObject.ADD
    return co


def obstacles_dict_to_collision_objects(
    obstacles: dict, frame_id: str = "world"
) -> List[CollisionObject]:
    result = []

    for name, data in obstacles.get("cuboid", {}).items():
        result.append(_cuboid_dict_to_collision_object(name, data, frame_id))

    for name, data in obstacles.get("cylinder", {}).items():
        result.append(_cylinder_dict_to_collision_object(name, data, frame_id))

    for name, data in obstacles.get("sphere", {}).items():
        result.append(_sphere_dict_to_collision_object(name, data, frame_id))

    return result


def scene_objects_to_collision_objects(
    objects: List, frame_id: str = "world"
) -> List[CollisionObject]:
    from curobo.scene import Cuboid, Cylinder, Sphere

    result = []
    for obj in objects:
        if isinstance(obj, Cuboid):
            d = {"dims": obj.dims, "pose": obj.pose}
            result.append(_cuboid_dict_to_collision_object(obj.name, d, frame_id))
        elif isinstance(obj, Cylinder):
            d = {"height": obj.height, "radius": obj.radius, "pose": obj.pose}
            result.append(_cylinder_dict_to_collision_object(obj.name, d, frame_id))
        elif isinstance(obj, Sphere):
            d = {"radius": obj.radius, "pose": obj.pose}
            result.append(_sphere_dict_to_collision_object(obj.name, d, frame_id))
    return result

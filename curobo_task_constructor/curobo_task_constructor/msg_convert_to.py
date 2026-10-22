"""ROS message converters for task solutions (MTC Solution.toMsg surface). ``executor.solution_to_dict`` produces ROS-free dicts with MTC ``Solution``
field names; this module converts those dicts to/from
``curobo_task_constructor_interfaces`` messages. ROS message classes are
imported lazily inside the converters (the ``viz.py`` pattern), so the core
stays importable without ROS and unit tests inject duck-typed stand-ins."""

from __future__ import annotations

import math
from typing import Any, Optional

try:
    import yaml
except Exception:  # pragma: no cover - exotic envs without yaml
    yaml = None


#: Reserved collision-object id carrying the "clear the whole world" /
#: "release whatever is attached" scene ops across the wire. ``moveit_msgs``'
#: PlanningScene is a NAME-KEYED diff (there is no wildcard REMOVE), so the
#: clear-all ops — which the in-process executor applies as the robot's
#: ``remove_all_objects()`` / ``detach_object(None)`` — need their own marker
#: to survive the message round trip. A generic PlanningScene consumer reads it
#: as a plain REMOVE of an unknown id, i.e. a no-op; ``msg_to_scene_diff``
#: recognizes it and restores the boolean op. Without it a scene-setup task
#: cannot clear the server world, its re-adds are refused as duplicate names,
#: and the pipeline plans against the previous cycle's objects forever.


SCENE_CLEAR_ALL_ID = "__curobo_clear_all__"


# ----------------------------------------------------------------------
# dict -> ROS messages
# ----------------------------------------------------------------------
def _uint32(value) -> int:
    """Clamp to a valid rosidl ``uint32`` field value. Stage-local record ids default to -1 until a stage emits (and the
    scene-only chain synthesizes id/stage_id -1 outright). Python field
    checks are off by default, so a -1 assignment sails through message
    construction and only detonates later in C serialization
    (``OverflowError: can't convert negative value to unsigned int``
    inside ``send_response``/``publish`` — past any try/except around the
    conversion, and fatal to the node for a service response)."""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def solution_info_to_msg(info: dict, msg_cls=None):
    """SolutionInfo dict -> message (MTC fields only)."""
    if msg_cls is None:
        from curobo_task_constructor_interfaces.msg import SolutionInfo
        msg_cls = SolutionInfo
    msg = msg_cls()
    msg.id = _uint32(info.get("id", 0))
    cost = info.get("cost", float("inf"))
    try:
        cost = float(cost)
    except (TypeError, ValueError):
        cost = float("inf")
    msg.cost = cost
    msg.comment = str(info.get("comment", "") or "")
    msg.stage_id = _uint32(info.get("stage_id", 0))
    msg.planner_id = str(info.get("planner_id", "") or "")
    for marker in (info.get("markers", None) or []):
        msg.markers.append(marker)
    return msg


def joint_trajectory_to_msg(names, points, msg_cls=None, point_cls=None,
                            dt: float = 0.05, velocities=None):
    """(names, point-lists) -> RobotTrajectory joint_trajectory.

    Timing is display-only (the curobo server retimes on execution);
    ``dt`` spacing matches the trajectory display default. ``velocities``
    (optional, one list per point) rides along so a replayed trajectory reaches
    the server with the solver's own joint speeds — MTC's RobotTrajectory
    carries them and the open-loop streamer clamps/tapers them.
    """
    if msg_cls is None or point_cls is None:
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        msg_cls = JointTrajectory
        point_cls = JointTrajectoryPoint
    msg = msg_cls()
    msg.joint_names = [str(n) for n in (names or [])]
    vels = list(velocities or [])
    for i, pt in enumerate(points or []):
        point = point_cls()
        point.positions = [float(v) for v in (pt or [])]
        if i < len(vels) and vels[i]:
            point.velocities = [float(v) for v in vels[i]]
        total = float(i) * float(dt)
        sec = int(total)
        point.time_from_start.sec = sec
        point.time_from_start.nanosec = int((total - sec) * 1e9)
        msg.points.append(point)
    return msg


def _collision_object(name, spec, operation, msg_cls=None):
    """ObjectSpec dict -> moveit CollisionObject (cuboid/sphere/cylinder)."""
    if msg_cls is None:
        from moveit_msgs.msg import CollisionObject
        msg_cls = CollisionObject
    from shape_msgs.msg import SolidPrimitive
    msg = msg_cls()
    msg.id = str(name)
    shape = str((spec or {}).get("shape", "") or "cuboid").lower()
    pose = (spec or {}).get("pose") or {}
    dims = list((spec or {}).get("dimensions") or [])
    prim = SolidPrimitive()
    if shape in ("sphere",):
        prim.type = SolidPrimitive.SPHERE
        r = float(dims[0]) if dims else 0.05
        prim.dimensions = [r]
    elif shape in ("cylinder", "capsule"):
        prim.type = SolidPrimitive.CYLINDER
        h = float(dims[1]) if len(dims) > 1 else 0.1
        r = float(dims[0]) if dims else 0.05
        prim.dimensions = [h, r]
    else:
        prim.type = SolidPrimitive.BOX
        while len(dims) < 3:
            dims.append(0.05)
        prim.dimensions = [float(v) for v in dims[:3]]
    msg.primitives.append(prim)
    msg.primitive_poses.append(_pose_to_msg(pose))
    msg.operation = operation
    return msg


def _pose_to_msg(pose, msg_cls=None):
    if msg_cls is None:
        from geometry_msgs.msg import Pose
        msg_cls = Pose
    pose = pose or {}
    msg = msg_cls()
    msg.position.x = float(pose.get("x", 0.0))
    msg.position.y = float(pose.get("y", 0.0))
    msg.position.z = float(pose.get("z", 0.0))
    msg.orientation.x = float(pose.get("qx", 0.0))
    msg.orientation.y = float(pose.get("qy", 0.0))
    msg.orientation.z = float(pose.get("qz", 0.0))
    msg.orientation.w = float(pose.get("qw", 1.0))
    return msg


def scene_diff_to_msg(diff: dict, msg_cls=None):
    """scene_diff dict -> PlanningScene. Two conventions have no MTC counterpart and ride reserved ids:

    - Detach: an attached object entry whose ``operation`` is REMOVE detaches
      that id (the server treats an empty ``link_names`` the same way). - Clear-all: ``remove_all`` / ``detach_all`` become a REMOVE of
      ``SCENE_CLEAR_ALL_ID`` in the world / attached lists (a name-keyed diff
      has no wildcard REMOVE)."""
    if msg_cls is None:
        from moveit_msgs.msg import PlanningScene
        msg_cls = PlanningScene
    from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
    msg = msg_cls()
    msg.is_diff = True
    diff = diff or {}
    if diff.get("remove_all"):
        clear = CollisionObject()
        clear.id = SCENE_CLEAR_ALL_ID
        clear.operation = CollisionObject.REMOVE
        msg.world.collision_objects.append(clear)
    if diff.get("detach_all"):
        entry = AttachedCollisionObject()
        entry.object.id = SCENE_CLEAR_ALL_ID
        entry.object.operation = CollisionObject.REMOVE
        msg.robot_state.attached_collision_objects.append(entry)
    for spec in (diff.get("added", None) or []):
        msg.world.collision_objects.append(
            _collision_object(spec.get("name", ""), spec,
                              CollisionObject.ADD))
    for name in (diff.get("removed", None) or []):
        remove = CollisionObject()
        remove.id = str(name)
        remove.operation = CollisionObject.REMOVE
        msg.world.collision_objects.append(remove)
    attached = diff.get("attached")
    if attached is not None:
        entry = AttachedCollisionObject()
        entry.object.id = str(attached)
        entry.object.operation = CollisionObject.ADD
        msg.robot_state.attached_collision_objects.append(entry)
    detached = diff.get("detached")
    if detached is not None:
        entry = AttachedCollisionObject()
        entry.object.id = str(detached)
        entry.object.operation = CollisionObject.REMOVE
        msg.robot_state.attached_collision_objects.append(entry)
    return msg


def _merge_scene_diff(before: dict, after: dict) -> dict:
    """Net scene effect of a segment (before + after ops, display order)."""
    merged = {"added": [], "removed": [], "attached": None, "detached": None,
              "remove_all": False, "detach_all": False}
    for part in (before or {}, after or {}):
        merged["added"].extend(part.get("added", None) or [])
        merged["removed"].extend(part.get("removed", None) or [])
        # A clear-all in either half applies to the whole segment, so it
        # latches rather than overwriting.
        merged["remove_all"] = merged["remove_all"] or bool(
            part.get("remove_all"))
        merged["detach_all"] = merged["detach_all"] or bool(
            part.get("detach_all"))
        if part.get("attached") is not None:
            merged["attached"] = part["attached"]
            merged["detached"] = None
        if part.get("detached") is not None:
            merged["detached"] = part["detached"]
            merged["attached"] = None
    return merged


def sub_trajectory_to_msg(seg: dict, msg_cls=None):
    """One sub_trajectories dict entry -> SubTrajectory message."""
    if msg_cls is None:
        from curobo_task_constructor_interfaces.msg import SubTrajectory
        msg_cls = SubTrajectory
    from curobo_task_constructor_interfaces.msg import SolutionInfo
    from moveit_msgs.msg import RobotTrajectory
    seg = seg or {}
    msg = msg_cls()
    msg.info = solution_info_to_msg(seg.get("info", {}) or {},
                                    msg_cls=SolutionInfo)
    for name in ((seg.get("execution_info", {}) or {}).get(
            "controller_names", None) or []):
        msg.execution_info.controller_names.append(str(name))
    traj = seg.get("trajectory", {}) or {}
    robot_traj = RobotTrajectory()
    robot_traj.joint_trajectory = joint_trajectory_to_msg(
        traj.get("joint_names", None) or [],
        traj.get("points", None) or [],
        velocities=traj.get("velocities", None) or [])
    msg.trajectory = robot_traj
    diff = seg.get("scene_diff", {}) or {}
    msg.scene_diff = scene_diff_to_msg(
        _merge_scene_diff(diff.get("before"), diff.get("after")))
    msg.replay_goal_yaml = _dump_yaml(seg.get("replay_goal"))
    return msg


def solution_to_msg(d: dict, msg_cls=None):
    """solution_to_dict output -> Solution message (MTC ``Solution.toMsg``)."""
    if msg_cls is None:
        from curobo_task_constructor_interfaces.msg import Solution
        msg_cls = Solution
    from curobo_task_constructor_interfaces.msg import SubSolution
    d = d or {}
    msg = msg_cls()
    msg.task_id = str(d.get("task_id", "") or "")
    from moveit_msgs.msg import PlanningScene
    scene = PlanningScene()
    for spec in (d.get("start_scene", None) or []):
        from moveit_msgs.msg import CollisionObject
        scene.world.collision_objects.append(
            _collision_object(spec.get("name", ""), spec,
                              CollisionObject.ADD))
    msg.start_scene = scene
    for sub in (d.get("sub_solution", None) or []):
        entry = SubSolution()
        from curobo_task_constructor_interfaces.msg import SolutionInfo
        entry.info = solution_info_to_msg(sub.get("info", {}) or {},
                                          msg_cls=SolutionInfo)
        for child_id in (sub.get("sub_solution_id", None) or []):
            # uint32[]: a never-emitted child carries solution_id -1.
            entry.sub_solution_id.append(_uint32(child_id))
        msg.sub_solution.append(entry)
    for seg in (d.get("sub_trajectories", None) or []):
        msg.sub_trajectory.append(sub_trajectory_to_msg(seg))
    return msg


def _dump_yaml(value) -> str:
    if value is None:
        return ""
    if yaml is None:
        return ""
    try:
        return yaml.safe_dump(value, sort_keys=False) or ""
    except Exception:
        return ""


def _load_yaml(text: str):
    if not text or yaml is None:
        return None
    try:
        return yaml.safe_load(text)
    except Exception:
        return None


# ----------------------------------------------------------------------
# ROS messages -> dicts (execute server side)
# ----------------------------------------------------------------------

__all__ = [
    'SCENE_CLEAR_ALL_ID',
    '_uint32',
    'solution_info_to_msg',
    'joint_trajectory_to_msg',
    '_collision_object',
    '_pose_to_msg',
    'scene_diff_to_msg',
    '_merge_scene_diff',
    'sub_trajectory_to_msg',
    'solution_to_msg',
    '_dump_yaml',
    '_load_yaml',
]

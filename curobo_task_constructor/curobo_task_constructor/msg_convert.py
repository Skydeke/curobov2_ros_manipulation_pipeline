"""ROS message converters for task solutions (MTC Solution.toMsg surface).

``executor.solution_to_dict`` produces ROS-free dicts with MTC ``Solution``
field names; this module converts those dicts to/from
``curobo_task_constructor_interfaces`` messages. ROS message classes are
imported lazily inside the converters (the ``viz.py`` pattern), so the core
stays importable without ROS and unit tests inject duck-typed stand-ins.

Wire shapes mirror ``moveit_task_constructor_msgs`` exactly:

- ``Solution``: ``task_id``, ``start_scene`` (PlanningScene), ``sub_solution``
  (SubSolution[]), ``sub_trajectory`` (SubTrajectory[]).
- ``SubTrajectory``: ``info`` (SolutionInfo), ``execution_info``
  (TrajectoryExecutionInfo), ``trajectory`` (RobotTrajectory),
  ``scene_diff`` (PlanningScene), ``replay_goal_yaml`` (the curobo segment
  planning goal; see the .msg comment).
- ``ExecuteTaskSolution``: goal carries the full ``Solution``; result is a
  ``moveit_msgs/MoveItErrorCodes``; feedback reports ``sub_id``/``sub_no``
  per driven sub-trajectory.
"""

from __future__ import annotations

import math
from typing import Any, Optional

try:
    import yaml
except Exception:  # pragma: no cover - exotic envs without yaml
    yaml = None


# ----------------------------------------------------------------------
# dict -> ROS messages
# ----------------------------------------------------------------------
def _uint32(value) -> int:
    """Clamp to a valid rosidl ``uint32`` field value.

    Stage-local record ids default to -1 until a stage emits (and the
    scene-only chain synthesizes id/stage_id -1 outright). Python field
    checks are off by default, so a -1 assignment sails through message
    construction and only detonates later in C serialization
    (``OverflowError: can't convert negative value to unsigned int``
    inside ``send_response``/``publish`` — past any try/except around the
    conversion, and fatal to the node for a service response). MTC
    reserves 0 as "no solution", so clamp there.
    """
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
                            dt: float = 0.05):
    """(names, point-lists) -> RobotTrajectory joint_trajectory.

    Timing is display-only (the curobo server retimes on execution);
    ``dt`` spacing matches the trajectory display default.
    """
    if msg_cls is None or point_cls is None:
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        msg_cls = JointTrajectory
        point_cls = JointTrajectoryPoint
    msg = msg_cls()
    msg.joint_names = [str(n) for n in (names or [])]
    for i, pt in enumerate(points or []):
        point = point_cls()
        point.positions = [float(v) for v in (pt or [])]
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
    """scene_diff dict ({added, removed, attached, detached}) -> PlanningScene.

    Detach is the one convention without an MTC counterpart: an attached
    object entry whose ``operation`` is REMOVE detaches that id (the server
    treats an empty ``link_names`` the same way).
    """
    if msg_cls is None:
        from moveit_msgs.msg import PlanningScene
        msg_cls = PlanningScene
    from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
    msg = msg_cls()
    msg.is_diff = True
    diff = diff or {}
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
    merged = {"added": [], "removed": [], "attached": None, "detached": None}
    for part in (before or {}, after or {}):
        merged["added"].extend(part.get("added", None) or [])
        merged["removed"].extend(part.get("removed", None) or [])
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
        traj.get("points", None) or [])
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
def msg_to_solution_info(msg) -> dict:
    cost = float(getattr(msg, "cost", float("inf")))
    if isinstance(cost, float) and math.isnan(cost):
        cost = float("inf")
    return {
        "id": int(getattr(msg, "id", 0) or 0),
        "cost": cost,
        "comment": str(getattr(msg, "comment", "") or ""),
        "stage_id": int(getattr(msg, "stage_id", 0) or 0),
        "planner_id": str(getattr(msg, "planner_id", "") or ""),
    }


def msg_to_joint_trajectory(msg) -> dict:
    jt = getattr(msg, "joint_trajectory", msg)
    return {
        "joint_names": [str(n) for n in (getattr(jt, "joint_names", []) or [])],
        "points": [[float(v) for v in (getattr(pt, "positions", []) or [])]
                   for pt in (getattr(jt, "points", []) or [])],
    }


def _pose_to_dict(pose: Any) -> Optional[dict]:
    """Pose-like -> flat ``{x,y,z,qx,qy,qz,qw}`` dict (for markers).

    Accepts flat dicts, ``Pose3`` (list position/orientation), ROS
    geometry_msgs poses and duck-typed equivalents; anything else (None,
    partial dicts) yields None. The missing function that kept every
    attempt/failure marker empty: callers referenced it, msg_convert never
    defined it, and the callers' broad excepts swallowed the AttributeError.
    """
    if pose is None:
        return None
    if isinstance(pose, dict):
        if {"x", "y", "z"} <= set(pose):
            out = {k: float(pose.get(k, 0.0)) for k in ("x", "y", "z")}
            for k, default in (("qx", 0.0), ("qy", 0.0),
                               ("qz", 0.0), ("qw", 1.0)):
                try:
                    out[k] = float(pose.get(k, default))
                except (TypeError, ValueError):
                    return None
            return out
        return None

    def component(obj: Any, keys: tuple, defaults: tuple):
        if isinstance(obj, (list, tuple)):
            if len(obj) < len(keys):
                return None
            try:
                return [float(obj[i]) for i in range(len(keys))]
            except (TypeError, ValueError, IndexError):
                return None
        if isinstance(obj, dict):
            try:
                return [float(obj.get(k, d)) for k, d in zip(keys, defaults)]
            except (TypeError, ValueError):
                return None
        try:
            return [float(getattr(obj, k)) for k in keys]
        except (TypeError, ValueError, AttributeError):
            return None

    pos = component(getattr(pose, "position", None), ("x", "y", "z"),
                    (0.0, 0.0, 0.0))
    ori = component(getattr(pose, "orientation", None), ("x", "y", "z", "w"),
                    (0.0, 0.0, 0.0, 1.0))
    if pos is None or ori is None:
        return None
    return {"x": pos[0], "y": pos[1], "z": pos[2],
            "qx": ori[0], "qy": ori[1], "qz": ori[2], "qw": ori[3]}


def msg_to_pose(pose) -> dict:
    pos = getattr(pose, "position", pose)
    ori = getattr(pose, "orientation", None)
    get = lambda o, k, default=0.0: float(
        getattr(o, k, default) if not isinstance(o, dict)
        else o.get(k, default))
    return {
        "x": get(pos, "x"), "y": get(pos, "y"), "z": get(pos, "z"),
        "qx": get(ori, "x") if ori is not None else 0.0,
        "qy": get(ori, "y") if ori is not None else 0.0,
        "qz": get(ori, "z") if ori is not None else 0.0,
        "qw": get(ori, "w", 1.0) if ori is not None else 1.0,
    }


def _byte_to_int(value, default=0) -> int:
    """A ``byte``-typed field value (or int) -> plain int.

    ``moveit_msgs/CollisionObject.operation`` is ``byte``: the generated
    constants are single bytes (``ADD == b'\\x00'``) and after a DDS round
    trip the field reads back as ``bytes``. ``int()`` on that raises
    ``ValueError: invalid literal for int() with base 10`` — which is how
    the execute server rejected every scene-carrying goal while pure-motion
    goals (empty diffs, loop body never runs) sailed through. Indexing the
    single byte yields the numeric op on both sides of the wire.
    """
    if isinstance(value, (bytes, bytearray)):
        return int(value[0]) if len(value) else int(default or 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default or 0)


def msg_to_scene_diff(msg) -> dict:
    """PlanningScene -> scene_diff dict (inverse of scene_diff_to_msg)."""
    from moveit_msgs.msg import CollisionObject
    out = {"added": [], "removed": [], "attached": None, "detached": None}
    world = getattr(msg, "world", None)
    for co in (getattr(world, "collision_objects", []) or []):
        op = _byte_to_int(getattr(co, "operation", None), 0)
        if op == _byte_to_int(CollisionObject.REMOVE):
            out["removed"].append(str(getattr(co, "id", "")))
            continue
        spec = {"name": str(getattr(co, "id", "")), "shape": "cuboid",
                "pose": None, "dimensions": None, "mesh_path": None,
                "vertices": None, "triangles": None}
        poses = list(getattr(co, "primitive_poses", []) or [])
        primitives = list(getattr(co, "primitives", []) or [])
        if primitives:
            from shape_msgs.msg import SolidPrimitive
            prim = primitives[0]
            dims = [float(v) for v in (getattr(prim, "dimensions", []) or [])]
            t = int(getattr(prim, "type", SolidPrimitive.BOX))
            if t == int(SolidPrimitive.SPHERE):
                spec["shape"] = "sphere"
            elif t == int(SolidPrimitive.CYLINDER):
                spec["shape"] = "cylinder"
            spec["dimensions"] = dims
        if poses:
            spec["pose"] = msg_to_pose(poses[0])
        out["added"].append(spec)
    state = getattr(msg, "robot_state", None)
    for entry in (getattr(state, "attached_collision_objects", []) or []):
        obj = getattr(entry, "object", entry)
        name = str(getattr(obj, "id", ""))
        op = _byte_to_int(getattr(obj, "operation", None), 0)
        if op == _byte_to_int(CollisionObject.REMOVE):
            out["detached"] = name
        else:
            out["attached"] = name
    return out


def msg_to_sub_trajectory(msg) -> dict:
    traj = getattr(msg, "trajectory", None)
    return {
        "info": msg_to_solution_info(getattr(msg, "info", None)),
        "execution_info": {
            "controller_names": [
                str(n) for n in (getattr(
                    getattr(msg, "execution_info", None),
                    "controller_names", []) or [])],
        },
        "trajectory": msg_to_joint_trajectory(traj) if traj is not None
        else {"joint_names": [], "points": []},
        "scene_diff": {
            "before": msg_to_scene_diff(getattr(msg, "scene_diff", None)),
            "after": {"added": [], "removed": [], "attached": None,
                      "detached": None},
        },
        "replay_goal": _load_yaml(getattr(msg, "replay_goal_yaml", "")),
    }


def msg_to_solution_dict(msg) -> dict:
    """Solution message -> solution_to_dict-shaped dict (for execute_dict)."""
    return {
        "task_id": str(getattr(msg, "task_id", "") or ""),
        "start_scene": [],
        "sub_solution": [
            {"info": msg_to_solution_info(getattr(sub, "info", None)),
             "sub_solution_id": [int(i) for i in (
                 getattr(sub, "sub_solution_id", []) or [])]}
            for sub in (getattr(msg, "sub_solution", []) or [])
        ],
        "sub_trajectories": [
            msg_to_sub_trajectory(sub)
            for sub in (getattr(msg, "sub_trajectory", []) or [])
        ],
    }


# ----------------------------------------------------------------------
# StageDescription / Property helpers (panel + TaskDescription)
# ----------------------------------------------------------------------
def interface_flags(stage) -> int:
    """Resolved interface as MTC StageDescription.flags bits (0x01 reads
    start, 0x02 reads end, 0x04 writes next start, 0x08 writes prev end)."""
    try:
        start, end = stage.interface_flags()
    except Exception:
        return 0
    flags = 0
    if getattr(start, "read", False):
        flags |= 0x01
    if getattr(end, "read", False):
        flags |= 0x02
    if getattr(end, "write", False):
        flags |= 0x04
    if getattr(start, "write", False):
        flags |= 0x08
    return flags


def sphere_marker(ns: str, marker_id: int, pose: dict, size: float,
                  color, marker_cls=None):
    """One SPHERE marker (MTC stage start/goal frames)."""
    if marker_cls is None:
        from visualization_msgs.msg import Marker
        marker_cls = Marker
    mark = marker_cls()
    mark.header.frame_id = "world"
    mark.ns = str(ns)
    mark.id = int(marker_id)
    mark.type = marker_cls.SPHERE
    mark.action = marker_cls.ADD
    mark.pose = _pose_to_msg(pose)
    mark.scale.x = mark.scale.y = mark.scale.z = float(size)
    mark.color.r, mark.color.g, mark.color.b, mark.color.a = (
        float(color[0]), float(color[1]), float(color[2]), float(color[3]))
    return mark


def sphere_list_marker(ns: str, marker_id: int, positions: list, size: float,
                       color, marker_cls=None):
    """One SPHERE_LIST marker (goal candidates)."""
    if marker_cls is None:
        from visualization_msgs.msg import Marker
        marker_cls = Marker
    from geometry_msgs.msg import Point
    mark = marker_cls()
    mark.header.frame_id = "world"
    mark.ns = str(ns)
    mark.id = int(marker_id)
    mark.type = marker_cls.SPHERE_LIST
    mark.action = marker_cls.ADD
    mark.scale.x = mark.scale.y = mark.scale.z = float(size)
    mark.color.r, mark.color.g, mark.color.b, mark.color.a = (
        float(color[0]), float(color[1]), float(color[2]), float(color[3]))
    for pose in (positions or []):
        pt = Point()
        pt.x, pt.y, pt.z = (float(pose.get("x", 0.0)),
                            float(pose.get("y", 0.0)),
                            float(pose.get("z", 0.0)))
        mark.points.append(pt)
    return mark
def property_to_msg(name: str, value: Any, description: str = "",
                    type_name: str = "", msg_cls=None):
    """One Property message (MTC Property shape)."""
    if msg_cls is None:
        from curobo_task_constructor_interfaces.msg import Property
        msg_cls = Property
    msg = msg_cls()
    msg.name = str(name)
    msg.description = str(description or "")
    msg.type = str(type_name or type(value).__name__)
    msg.value = _dump_yaml(value)
    return msg


def stage_description_to_msg(stage_id: int, parent_id: int, name: str,
                             flags: int, properties: list, msg_cls=None):
    """StageDescription message (MTC shape)."""
    if msg_cls is None:
        from curobo_task_constructor_interfaces.msg import StageDescription
        msg_cls = StageDescription
    msg = msg_cls()
    msg.id = int(stage_id)
    msg.parent_id = int(parent_id)
    msg.name = str(name)
    msg.flags = int(flags)
    for prop in (properties or []):
        msg.properties.append(prop)
    return msg

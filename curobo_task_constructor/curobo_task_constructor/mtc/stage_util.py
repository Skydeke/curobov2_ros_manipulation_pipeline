"""MTC-shaped stage wrappers (``stages`` namespace). Mirrors ``moveit.task_constructor.stages`` constructor/setter shape::

    move = stages.MoveRelative("x +0.2", cartesian)
    move.group = "arm"
    move.setDirection(Vector3Stamped(...))

Method names match MTC (``setGoal``/``setDirection``/``attachObject``/…);
each wrapper converts itself to a ``StageSpec`` node (same ``params_yaml``
schema the registry stages parse), so the wire format never changes and
``Task.plan()`` can drive the existing ``TaskExecutor`` locally. Planners may
be solver objects (``core.CartesianPath``), wire strings (``"classic"`` /
``"joint_space"``) or ``SetPlanner`` ints — the string form is what the pick
pipeline's trees carry on the wire."""

from __future__ import annotations

import math
from typing import Any




def _dump_params(params: dict) -> str:
    try:
        import yaml
        try:
            text = yaml.safe_dump(dict(params), sort_keys=False)
        except Exception:
            # A value that is not YAML-serializable (e.g. a Constraints
            # message via setProperty) must not wipe the whole stage config:
            # fall back per value.
            safe = {}
            for key, value in dict(params).items():
                try:
                    yaml.safe_dump({key: value})
                    safe[key] = value
                except Exception:
                    safe[key] = repr(value)
            text = yaml.safe_dump(safe, sort_keys=False)
        return text or ""
    except Exception:
        return ""


def _vec_norm(v) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in v))


def _planner_name(planner: Any) -> Any:
    """Planner object/int/str -> the wire ``planner`` param (None = default).

    A solver object resolves through ``planner_key()`` (the cuRobo
    ``SetPlanner`` int it selects); a ``PipelinePlanner`` whose pipeline maps
    to no backend key resolves to ``None`` — the server default — rather than
    raising, so a stage can still be built, dumped and inspected. Strings and
    ints pass through verbatim (``planner_key`` in ``_util`` maps them).
    """
    if planner is None:
        return None
    if isinstance(planner, (str, int)):
        return planner
    name = getattr(planner, "wire_name", None)
    if isinstance(name, str):
        return name
    key_fn = getattr(planner, "planner_key", None)
    if callable(key_fn):
        try:
            key = key_fn()
        except Exception:
            return None
        return None if key is None else key
    return None


def _group_link_names(group: Any) -> list:
    """Link names out of a joint-model group (MTC JointModelGroup).

    Tries the common accessors (C++ getJointModelNames/getLinkModelNames,
    python joint_names/link_names, plain iterables); falls back to the
    string form so an unknown object degrades to one pair, never a crash.
    """
    for attr in ("get_joint_model_names", "getJointModelNames",
                 "get_link_model_names", "getLinkModelNames",
                 "joint_names", "link_names", "links"):
        try:
            value = getattr(group, attr, None)
            names = value() if callable(value) else value
            if names:
                return [str(n) for n in names]
        except Exception:
            continue
    try:
        return [str(n) for n in group]
    except Exception:
        return [str(group)]


def _frame_of(header: Any, default: str = "world") -> str:
    try:
        fid = getattr(header, "frame_id", "") or ""
        if isinstance(header, dict):
            fid = header.get("frame_id", "") or ""
        fid = str(fid).lower()
        if fid in ("world", "map", "odom", "base_link", "base"):
            return "world"
        if fid in ("hand", "tool", "tool_link", "ee", "eef", "gripper"):
            return "hand"
        return default
    except Exception:
        return default


def _split_twist_like(direction: Any):
    """Return (header, linear_xyz|None, angular_xyz|None) for ROS msg-likes."""
    if direction is None:
        return None, None, None
    # geometry_msgs Vector3Stamped: .header + .vector
    if hasattr(direction, "vector") and not hasattr(direction, "twist"):
        v = direction.vector
        header = getattr(direction, "header", None)
        return header, [float(v.x), float(v.y), float(v.z)], None
    # geometry_msgs TwistStamped: .header + .twist.{linear,angular}
    if hasattr(direction, "twist"):
        tw = direction.twist
        header = getattr(direction, "header", None)
        lin = [float(tw.linear.x), float(tw.linear.y), float(tw.linear.z)]
        ang = [float(tw.angular.x), float(tw.angular.y), float(tw.angular.z)]
        return header, lin, ang
    # bare Vector3 / Twist
    if hasattr(direction, "x") and hasattr(direction, "y") and hasattr(direction, "z"):
        return None, [float(direction.x), float(direction.y), float(direction.z)], None
    if hasattr(direction, "linear") and hasattr(direction, "angular"):
        lin = direction.linear
        ang = direction.angular
        return None, [float(lin.x), float(lin.y), float(lin.z)], \
            [float(ang.x), float(ang.y), float(ang.z)]
    if isinstance(direction, dict) and {"x", "y", "z"} <= set(direction):
        return None, [float(direction["x"]), float(direction["y"]),
                      float(direction["z"])], None
    return None, None, None


def _pose_dict_of(goal: Any):
    if isinstance(goal, dict) and {"x", "y", "z"} <= set(goal):
        out = {"x": float(goal.get("x", 0.0)), "y": float(goal.get("y", 0.0)),
               "z": float(goal.get("z", 0.0)),
               "qx": float(goal.get("qx", 0.0)), "qy": float(goal.get("qy", 0.0)),
               "qz": float(goal.get("qz", 0.0)), "qw": float(goal.get("qw", 1.0))}
        return out
    pos = getattr(goal, "position", None)
    ori = getattr(goal, "orientation", None)
    if pos is not None and ori is not None:
        return {"x": float(pos.x), "y": float(pos.y), "z": float(pos.z),
                "qx": float(ori.x), "qy": float(ori.y),
                "qz": float(ori.z), "qw": float(ori.w)}
    stamped = getattr(goal, "pose", None)
    if stamped is not None:
        return _pose_dict_of(stamped)
    # NOTE: PointStamped has no orientation to keep at build time, so it is
    # not a valid pose-slot value (MoveTo.setGoal handles the singular form
    # as a {"point"} goal resolved against the live start state instead).
    return None

__all__ = [
    '_dump_params',
    '_vec_norm',
    '_planner_name',
    '_group_link_names',
    '_frame_of',
    '_split_twist_like',
    '_pose_dict_of',
]

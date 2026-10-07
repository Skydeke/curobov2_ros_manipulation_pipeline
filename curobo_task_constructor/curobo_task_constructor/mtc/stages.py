"""MTC-shaped stage wrappers (``stages`` namespace).

Mirrors ``moveit.task_constructor.stages`` constructor/setter shape::

    move = stages.MoveRelative("x +0.2", cartesian)
    move.group = "arm"
    move.setDirection(Vector3Stamped(...))

Each wrapper converts itself to a ``StageSpec`` node (same ``params_yaml``
schema the registry stages parse), so the wire format never changes and
``Task.plan()`` can drive the existing ``TaskExecutor`` locally. Planners may
be solver objects (``core.CartesianPath``), wire strings (``"classic"`` /
``"joint_space"``) or ``SetPlanner`` ints — the string form is what the pick
pipeline's trees carry on the wire.
"""

from __future__ import annotations

import math
from typing import Any


def _dump_params(params: dict) -> str:
    try:
        import yaml
        text = yaml.safe_dump(dict(params), sort_keys=False)
        return text or ""
    except Exception:
        return ""


def _vec_norm(v) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in v))


def _planner_name(planner: Any) -> Any:
    """Planner object/int/str -> the wire ``planner`` param (None = default)."""
    if planner is None:
        return None
    if isinstance(planner, (str, int)):
        return planner
    name = getattr(planner, "wire_name", None)
    if isinstance(name, str):
        return name
    key = getattr(planner, "planner_key", None)
    if callable(key):
        return int(key())
    return None


class _StageBase:
    stage_type = ""
    is_generator = False

    def __init__(self, name: str = "", planner: Any = None):
        self.name = name or self.stage_type
        self.planner = planner
        self.group: str = ""
        self.params: dict = {}

    def _planner_params(self) -> dict:
        out: dict = {}
        wire = _planner_name(self.planner)
        if wire is not None:
            out["planner"] = wire
        extra = getattr(self.planner, "extra_params", None)
        if callable(extra):
            out.update(extra() or {})
        return out

    def _common_params(self) -> dict:
        params = dict(self.params)
        params.update(self._planner_params())
        if self.group and "group" not in params:
            params["group"] = self.group
        return params

    def to_spec(self):
        from curobo_task_constructor.graph.spec import StageSpec
        return StageSpec(stage_type=self.stage_type, name=self.name,
                         container_type="", children=[],
                         params_yaml=_dump_params(self._common_params()))

    def to_dict(self) -> dict:
        return self.to_spec().to_dict()


class _MotionBase(_StageBase):
    """Motion stages carry planning control (MTC ``num_planning_attempts``).

    Ranking is explicit, never a hidden default: every motion stage declares
    its cost term (MTC ``cost::PathLength`` et al) via ``setCost``. The
    builders in the pick pipeline set ``path_length`` on each motion leg;
    generators cost 0 and containers accumulate their children, so neither
    carries a term.
    """

    #: Ranking terms understood by ``stages._util.cost_of``.
    COST_TERMS = ("auto", "path_length", "waypoints", "solver_cost", "inf")

    def __init__(self, name: str = "", planner: Any = None,
                 planning_attempts: int = 3):
        super().__init__(name, planner)
        self.planning_attempts = int(planning_attempts)

    def setCost(self, term: str) -> None:
        """Declare the ranking term explicitly (default ``auto``)."""
        if term not in self.COST_TERMS:
            raise ValueError(f"unknown cost term {term!r}; "
                             f"expected one of {list(self.COST_TERMS)}")
        self.params["cost"] = term

    def _common_params(self) -> dict:
        params = super()._common_params()
        params["planning_attempts"] = int(self.planning_attempts)
        return params


class CurrentState(_StageBase):
    stage_type = "current_state"
    is_generator = True

    def __init__(self, name: str = "current state", planner: Any = None):
        super().__init__(name, planner)

    def require_not_attached(self, objects) -> None:
        """Refuse when any listed object is already attached (MTC
        PredicateFilter "already attached and cannot be picked")."""
        self.params["require_not_attached"] = [str(o) for o in (objects or [])]

    def require_attached(self, objects) -> None:
        self.params["require_attached"] = [str(o) for o in (objects or [])]


class FixedState(_StageBase):
    stage_type = "fixed_state"

    def __init__(self, name: str = "fixed state", planner: Any = None):
        super().__init__(name, planner)
        self.goal_name: str = ""

    def setGoal(self, goal: Any) -> None:
        if isinstance(goal, str):
            self.goal_name = goal
            self.params["goal"] = {"name": goal}
        elif isinstance(goal, dict):
            self.params["goal"] = {"joints": dict(goal)}
        else:
            raise TypeError(f"FixedState.setGoal: unsupported {type(goal)}")


class MoveRelative(_MotionBase):
    stage_type = "move_relative"

    def __init__(self, name: str = "move relative", planner: Any = None,
                 planning_attempts: int = 3):
        super().__init__(name, planner, planning_attempts)

    def setDirection(self, direction: Any) -> None:
        """Accept Vector3Stamped / TwistStamped / dict / axis-dict (MTC parity)."""
        # dict joint offsets: {"joint": delta}
        if isinstance(direction, dict) and not {"x", "y", "z"} <= set(direction):
            self.params["joint_offsets"] = {str(k): float(v)
                                            for k, v in direction.items()}
            return
        header, linear, angular = _split_twist_like(direction)
        if angular is not None and _vec_norm(angular) > 0.0:
            norm = _vec_norm(angular)
            self.params["rotation"] = {
                "axis": [float(v) / norm for v in angular],
                "angle": float(norm),
                "frame": _frame_of(header, default="hand"),
            }
        if linear is not None and _vec_norm(linear) > 0.0:
            norm = _vec_norm(linear)
            self.params["axis"] = {
                "xyz": [float(v) / norm for v in linear],
                "frame": _frame_of(header, default="world"),
            }
            self.params["distance"] = float(norm)
        elif angular is None and linear is None:
            raise TypeError(
                "MoveRelative.setDirection: expected Vector3Stamped, "
                "TwistStamped, or {joint: delta} dict")

    def setDistance(self, distance: float) -> None:
        self.params["distance"] = float(distance)


class MoveTo(_MotionBase):
    stage_type = "move_to"

    def __init__(self, name: str = "move to", planner: Any = None,
                 planning_attempts: int = 3):
        super().__init__(name, planner, planning_attempts)

    def setGoal(self, goal: Any) -> None:
        if isinstance(goal, str):
            self.params["goal"] = {"name": goal}
        elif isinstance(goal, dict):
            # {"joint": value} sparse joints goal
            self.params["goal"] = {"joints": {str(k): float(v)
                                              for k, v in goal.items()}}
        elif isinstance(goal, (list, tuple)):
            self.params["goal"] = {"joints": [float(v) for v in goal]}
        else:
            pose = _pose_dict_of(goal)
            if pose is None:
                raise TypeError(f"MoveTo.setGoal: unsupported {type(goal)}")
            self.params["goal"] = {"pose": pose}

    def setGoals(self, poses) -> None:
        """N candidate poses in ONE goalset; the planner resolves the set and
        reports the winner (the pick's pre-grasp/descend fan-out)."""
        dicts = [_pose_dict_of(p) for p in (poses or [])]
        if any(d is None for d in dicts):
            raise TypeError("MoveTo.setGoals: every goal must be a pose")
        self.params["goal"] = {"poses": dicts}


class Connect(_MotionBase):
    stage_type = "connect"

    def __init__(self, name: str = "connect", planner: Any = None,
                 planning_attempts: int = 3):
        super().__init__(name, planner, planning_attempts)


class CartesianPath(_MotionBase):
    """Straight-line solve with whole-path axis holds (MTC CartesianPath).

    The hold is derived by the stage (pin every axis start and goal agree
    on) unless ``setHold`` pins it explicitly; ``setCheckStraightness(False)``
    is the free-space opt-out (the pick's last-resort strategy).
    """

    stage_type = "cartesian_path"

    def __init__(self, name: str = "cartesian", planner: Any = None,
                 planning_attempts: int = 3):
        super().__init__(name, planner, planning_attempts)
        # MTC parity: a Cartesian stage enforces the line it planned. The raw
        # stage defaults the gate off (a bowed solve is the optimizer's
        # answer); the pick's line-constrained legs keep it on.
        self.params["check_straightness"] = True

    def setGoal(self, goal: Any) -> None:
        pose = _pose_dict_of(goal)
        if pose is None:
            raise TypeError(f"CartesianPath.setGoal: unsupported {type(goal)}")
        self.params.setdefault("goal", {})["pose"] = pose

    def setGoals(self, poses) -> None:
        dicts = [_pose_dict_of(p) for p in (poses or [])]
        if any(d is None for d in dicts):
            raise TypeError("CartesianPath.setGoals: every goal must be a pose")
        self.params.setdefault("goal", {})["poses"] = dicts

    def setRelative(self, x: float = 0.0, y: float = 0.0, z: float = 0.0,
                    frame: str = "world") -> None:
        """Offset from FK(start), orientation frozen (the retreat/lift)."""
        self.params["goal"] = {"relative": {
            "x": float(x), "y": float(y), "z": float(z), "frame": str(frame)}}

    def setHold(self, hold) -> None:
        self.params["hold"] = [int(c) for c in (hold or [])]

    def setCheckStraightness(self, enabled: bool) -> None:
        self.params["check_straightness"] = bool(enabled)


class ModifyScene(_StageBase):
    """Scene mutation (MTC ModifyPlanningScene equivalent).

    Emits exactly the ``modify_scene`` params the stage parses — no planner
    or attempt keys, so ``{"detach": name}`` stays exactly that on the wire.
    """

    stage_type = "modify_scene"

    def __init__(self, name: str = "modify scene"):
        super().__init__(name, None)

    def add(self, spec: dict) -> None:
        self.params["add"] = dict(spec)

    def remove(self, name: str) -> None:
        self.params["remove"] = str(name)

    def remove_all(self) -> None:
        self.params["remove_all"] = True

    def attach(self, name: str) -> None:
        self.params["attach"] = str(name)

    def detach(self, name: str) -> None:
        self.params["detach"] = str(name)

    def detach_all(self) -> None:
        self.params["detach_all"] = True

    def allow_collisions(self, object_name: str, links,
                         enabled: bool) -> None:
        self.params["allow_collisions"] = {
            "object": str(object_name),
            "links": [str(link) for link in (links or [])],
            "enabled": bool(enabled),
        }


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
    return None

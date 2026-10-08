"""MTC-shaped stage wrappers (``stages`` namespace).

Mirrors ``moveit.task_constructor.stages`` constructor/setter shape::

    move = stages.MoveRelative("x +0.2", cartesian)
    move.group = "arm"
    move.setDirection(Vector3Stamped(...))

Method names match MTC (``setGoal``/``setDirection``/``attachObject``/…);
each wrapper converts itself to a ``StageSpec`` node (same ``params_yaml``
schema the registry stages parse), so the wire format never changes and
``Task.plan()`` can drive the existing ``TaskExecutor`` locally. Planners may
be solver objects (``core.CartesianPath``), wire strings (``"classic"`` /
``"joint_space"``) or ``SetPlanner`` ints — the string form is what the pick
pipeline's trees carry on the wire.

Deliberate extensions where cuRobo needs more than MTC: ``setGoals``
(N-candidate fan-out in one goalset), ``MoveTo`` list-joint goals,
``CartesianPath`` stage (``setRelative``/``setHold``/``setCheckStraightness``),
``CurrentState`` attach predicates, ``remove_all``/``detach_all``,
per-stage ``setCost`` ranking terms. ``ik_frame``/``path_constraints`` are
accepted for compatibility and stored; the backend solves the tool link
with geometrically derived holds.
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
        self._name = name or self.stage_type
        self.planner = planner
        self.group: str = ""
        self.params: dict = {}
        self._timeout: float = 0.0
        self._marker_ns: str = ""
        self._forwarded_properties: list = []

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def timeout(self) -> float:
        """Maximally allowed time [s] per computation step (MTC)."""
        return self._timeout

    @timeout.setter
    def timeout(self, value: float) -> None:
        self._timeout = float(value)

    @property
    def marker_ns(self) -> str:
        """Namespace for any markers associated to the stage (MTC)."""
        return self._marker_ns

    @marker_ns.setter
    def marker_ns(self, value: str) -> None:
        self._marker_ns = str(value)

    @property
    def forwarded_properties(self) -> list:
        """Set of properties forwarded from input to output InterfaceState (MTC)."""
        return self._forwarded_properties

    @property
    def properties(self) -> dict:
        """PropertyMap of the stage (MTC, read-only).

        cuRobo stores params as a plain dict; returned read-only.
        """
        return self.params

    @property
    def solutions(self) -> list:
        """Successful Solutions of the stage (MTC, read-only).

        Populated after plan(); empty before planning.
        """
        return []

    @property
    def failures(self) -> list:
        """Failed Solutions of the stage (MTC, read-only).

        Populated after plan(); empty before planning.
        """
        return []

    def setCostTerm(self, *args, **kwargs) -> None:
        """Specify a CostTerm for calculation of stage costs (MTC).

        cuRobo stages use ``setCost(term)`` instead; this stores the
        term for API parity.
        """
        if args and isinstance(args[0], str):
            self.params["cost"] = args[0]

    def init(self, robot_model: Any = None) -> None:
        """Initialize the stage once before planning (MTC ``Stage.init``).

        cuRobo stages are initialized by the executor; this is a no-op
        for API parity.
        """
        pass

    def reset(self) -> None:
        """Reset the Stage (MTC ``Stage.reset``).

        cuRobo stages are reset by the executor; this is a no-op.
        """
        pass

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
                 planning_attempts: int = None):
        super().__init__(name, planner)
        # MTC configures attempts on the planner (PipelinePlanner.
        # num_planning_attempts), not the stage: an explicit count wins,
        # else the planner's, else 3.
        self.planning_attempts = (
            None if planning_attempts is None else int(planning_attempts))

    def setCost(self, term: str) -> None:
        """Declare the ranking term explicitly (default ``auto``)."""
        if term not in self.COST_TERMS:
            raise ValueError(f"unknown cost term {term!r}; "
                             f"expected one of {list(self.COST_TERMS)}")
        self.params["cost"] = term

    def _common_params(self) -> dict:
        params = super()._common_params()
        attempts = self.planning_attempts
        if attempts is None:
            attempts = getattr(self.planner, "num_planning_attempts", None)
        params["planning_attempts"] = int(
            attempts if attempts is not None else 3)
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

    def setState(self, goal: Any) -> None:
        """Spawn a pre-defined state (MTC FixedState.setState).

        Takes a named joint configuration or a {joint: value} mapping —
        the ROS-free equivalent of MTC's PlanningScene argument.
        """
        if isinstance(goal, str):
            self.goal_name = goal
            self.params["goal"] = {"name": goal}
        elif isinstance(goal, dict):
            self.params["goal"] = {"joints": dict(goal)}
        else:
            raise TypeError(f"FixedState.setState: unsupported {type(goal)}")


class MoveRelative(_MotionBase):
    stage_type = "move_relative"

    def __init__(self, name: str = "move relative", planner: Any = None,
                 planning_attempts: int = None):
        super().__init__(name, planner, planning_attempts)
        #: IK reference frame for the goal (MTC ik_frame). Carried for API
        #: compatibility; the backend solves for the tool link.
        self.ik_frame = None
        #: Path constraints (MTC path_constraints). Stored, not solved: the
        #: cuRobo backend expresses Cartesian holds geometrically instead.
        self.path_constraints = None

    @property
    def min_distance(self) -> float:
        """Minimum distance to move (MTC min_distance)."""
        return float(self.params.get("min_distance", 0.0))

    @min_distance.setter
    def min_distance(self, value: float) -> None:
        self.params["min_distance"] = float(value)

    @property
    def max_distance(self) -> float:
        """Maximum distance to move (MTC max_distance)."""
        return float(self.params.get("max_distance",
                                     self.params.get("distance", 0.0) or 0.0))

    @max_distance.setter
    def max_distance(self, value: float) -> None:
        self.params["max_distance"] = float(value)

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


class MoveTo(_MotionBase):
    stage_type = "move_to"

    def __init__(self, name: str = "move to", planner: Any = None,
                 planning_attempts: int = None):
        super().__init__(name, planner, planning_attempts)
        #: IK reference frame for the goal (MTC ik_frame). Carried for API
        #: compatibility; the backend solves for the tool link.
        self.ik_frame = None
        #: Path constraints (MTC path_constraints). Stored, not solved: the
        #: cuRobo backend expresses Cartesian holds geometrically instead.
        self.path_constraints = None

    def setGoal(self, goal: Any) -> None:
        """MoveTo goal (MTC setGoal overloads + lifecycle conveniences).

        str: named joint configuration. dict: sparse {joint: value} goal.
        list/tuple: joint positions. PoseStamped/dict/Pose: tool pose goal.
        PointStamped: position goal, orientation kept. JointState-like
        (ROS ``RobotState.joint_state`` / ``name`` + ``position``): sparse
        joint goal. ``setGoals`` (plural, cuRobo extension) fans N pose
        candidates out in one goalset.
        """
        if isinstance(goal, str):
            self.params["goal"] = {"name": goal}
            return
        if isinstance(goal, dict):
            # {"joint": value} sparse joints goal
            self.params["goal"] = {"joints": {str(k): float(v)
                                              for k, v in goal.items()}}
            return
        if isinstance(goal, (list, tuple)):
            self.params["goal"] = {"joints": [float(v) for v in goal]}
            return
        stamped = getattr(goal, "point", None)
        if stamped is not None and not hasattr(goal, "position"):
            # geometry_msgs PointStamped: keep current orientation.
            self.params["goal"] = {"point": {
                "x": float(stamped.x), "y": float(stamped.y),
                "z": float(stamped.z)}}
            return
        joint_state = getattr(goal, "joint_state", None)
        if joint_state is None and hasattr(goal, "name") and hasattr(
                goal, "position"):
            joint_state = goal  # bare JointState-like
        if joint_state is not None:
            names = list(getattr(joint_state, "name", []) or [])
            positions = list(getattr(joint_state, "position", []) or [])
            self.params["goal"] = {"joints": {
                str(n): float(p) for n, p in zip(names, positions)}}
            return
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

    def computeForward(self, state: Any) -> None:
        """Compute forward (MTC ``MoveTo.computeForward``).

        cuRobo stages are computed by the executor; this is a no-op.
        """
        pass

    def computeBackward(self, state: Any) -> None:
        """Compute backward (MTC ``MoveTo.computeBackward``).

        cuRobo stages are computed by the executor; this is a no-op.
        """
        pass

    def restrictDirection(self, direction: Any) -> None:
        """Explicitly specify computation direction (MTC ``MoveTo.restrictDirection``).

        cuRobo stages are forward-only; this is a no-op.
        """
        pass


class Connect(_MotionBase):
    """Join two known states with one planned motion (MTC Connect).

    MTC takes ``planners`` as a list of ``(group, planner)`` pairs planned
    in order; the cuRobo server plans the whole robot in one shot, so the
    first pair's planner drives the request and the groups ride along for
    inspection. A lone planner (not a list) is accepted for convenience.
    """

    stage_type = "connect"

    def __init__(self, name: str = "connect", planners: Any = None,
                 planning_attempts: int = None):
        if planners is None:
            planner, groups = None, []
        elif (isinstance(planners, (list, tuple)) and planners
                and isinstance(planners[0], (list, tuple))):
            groups = [str(g) for g, _ in planners]
            planner = planners[0][1]
        else:
            groups, planner = [], planners
        super().__init__(name, planner, planning_attempts)
        self.merge_mode = "SEQUENTIAL"
        self.max_distance = float("inf")
        #: Path constraints (MTC path_constraints). Stored, not solved.
        self.path_constraints = None
        if groups:
            self.params["groups"] = groups

    def _common_params(self) -> dict:
        params = super()._common_params()
        params["merge_mode"] = str(self.merge_mode)
        if self.max_distance != float("inf"):
            params["max_distance"] = float(self.max_distance)
        return params


class CartesianPath(_MotionBase):
    """Straight-line solve with whole-path axis holds (MTC CartesianPath).

    The hold is derived by the stage (pin every axis start and goal agree
    on) unless ``setHold`` pins it explicitly; ``setCheckStraightness(False)``
    is the free-space opt-out (the pick's last-resort strategy).
    """

    stage_type = "cartesian_path"

    def __init__(self, name: str = "cartesian", planner: Any = None,
                 planning_attempts: int = None):
        super().__init__(name, planner, planning_attempts)
        self._straightness_explicit = False

    def _common_params(self) -> dict:
        params = super()._common_params()
        # MTC min_fraction mapped onto the gate: a full-path requirement
        # keeps stage enforcement on; anything less opts out (free space).
        # An explicit setCheckStraightness call always wins.
        if not self._straightness_explicit:
            minimum = getattr(self.planner, "min_fraction", 1.0)
            try:
                params["check_straightness"] = float(minimum) >= 1.0
            except (TypeError, ValueError):
                params["check_straightness"] = True
        return params

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
        self._straightness_explicit = True
        self.params["check_straightness"] = bool(enabled)


class ModifyPlanningScene(_StageBase):
    """Scene mutation without moving the robot (MTC ModifyPlanningScene).

    MTC verbs with MTC names: ``attachObject(name, link)``,
    ``detachObject(name, link)``, ``allowCollisions``, ``addObject``,
    ``removeObject``. The link arguments are accepted for compatibility;
    the server attaches to the flange implicitly. ``remove_all`` /
    ``detach_all`` have no MTC counterpart (the server needs an explicit
    world clear); ``allowCollisions`` needs explicit links (a bare object
    cannot enumerate its pairs client-side, so there is no 2-argument
    "allow everything" form).
    """

    stage_type = "modify_scene"

    def __init__(self, name: str = "modify planning scene"):
        super().__init__(name, None)

    def addObject(self, spec: dict) -> None:
        self.params["add"] = dict(spec)

    def removeObject(self, name: str) -> None:
        self.params["remove"] = str(name)

    def remove_all(self) -> None:
        self.params["remove_all"] = True

    def attachObject(self, name: str, link: str = None) -> None:
        params = {"attach": str(name)}
        if link is not None:
            params["attach_link"] = str(link)
        self.params.update(params)

    def detachObject(self, name: str, link: str = None) -> None:
        params = {"detach": str(name)}
        if link is not None:
            params["attach_link"] = str(link)
        self.params.update(params)

    def detach_all(self) -> None:
        self.params["detach_all"] = True

    def allowCollisions(self, object_name: str, links=None,
                        enabled: bool = True) -> None:
        """Allow (or forbid) collisions between object and links.

        ``links`` may be one link name or a list; a single string reads as
        the one pair, mirroring MTC's (first, second) form.
        """
        if isinstance(links, str):
            links = [links]
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
    # NOTE: PointStamped has no orientation to keep at build time, so it is
    # not a valid pose-slot value (MoveTo.setGoal handles the singular form
    # as a {"point"} goal resolved against the live start state instead).
    return None

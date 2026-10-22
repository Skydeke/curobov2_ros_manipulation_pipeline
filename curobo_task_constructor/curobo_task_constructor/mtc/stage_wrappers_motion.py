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



from curobo_task_constructor.mtc.stage_util import *
from curobo_task_constructor.mtc.stage_wrappers_base import *

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

    def setMinMaxDistance(self, min_distance: float,
                          max_distance: float) -> None:
        """Set the accepted distance range (MTC ``MoveRelative``)."""
        self.min_distance = float(min_distance)
        self.max_distance = float(max_distance)

    def setPathConstraints(self, constraints: Any) -> None:
        """Set path constraints (MTC ``MoveRelative``).

        Stored, not solved: the cuRobo backend expresses Cartesian holds
        geometrically instead.
        """
        self.path_constraints = constraints


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
        """MoveTo goal (MTC setGoal overloads + lifecycle conveniences). str: named joint configuration. dict: sparse {joint: value} goal."""
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

    def setPathConstraints(self, constraints: Any) -> None:
        """Set path constraints (MTC ``MoveTo``).

        Stored, not solved: the cuRobo backend expresses Cartesian holds
        geometrically instead.
        """
        self.path_constraints = constraints

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


class GripperCommand(_StageBase):
    """Move one end effector's jaws without planning.

    Carries a named gripper and a target; the executor sends it to that
    gripper's controller action instead of a trajectory. Blank fields fall
    back to the robot descriptor's ``gripper_controllers`` default entry.
    """

    stage_type = "gripper_command"

    def __init__(self, name: str = "gripper command", planner: Any = None):
        super().__init__(name or "gripper command", planner)

    def setGripper(self, gripper: str) -> "GripperCommand":
        self.params["gripper"] = str(gripper)
        return self

    def setJoint(self, joint: str) -> "GripperCommand":
        self.params["joint"] = str(joint)
        return self

    def setPosition(self, position: float) -> "GripperCommand":
        self.params["position"] = float(position)
        return self

    def setEffort(self, effort: float) -> "GripperCommand":
        self.params["effort"] = float(effort)
        return self

    def setMaxEffort(self, max_effort: float) -> "GripperCommand":
        self.params["max_effort"] = float(max_effort)
        return self

    def setController(self, controller: str) -> "GripperCommand":
        self.params["controller"] = str(controller)
        return self

    def setActionNamespace(self, action_ns: str) -> "GripperCommand":
        self.params["action_ns"] = str(action_ns)
        return self


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

    def setPathConstraints(self, constraints: Any) -> None:
        """Set path constraints (MTC ``Connect``).

        Stored, not solved.
        """
        self.path_constraints = constraints


class CartesianPath(_MotionBase):
    """Straight-line Cartesian solve (MTC ``solvers::CartesianPath``). MTC's CartesianPath is a *solver*, not a stage: it interpolates a straight
    EEF path from the start to the goal at ``step_size`` and reports the
    fraction of that path it reached. cuRobo has no Cartesian solve mode, so
    this stage requests the line as whole-path axis holds (the
    ``ToolPoseCriteria`` cost) and the server's CARTESIAN planner solves it;
    there is no client-side gate, and no fractional-path gate server-side
    either (it was removed -)."""

    stage_type = "cartesian_path"

    def __init__(self, name: str = "cartesian", planner: Any = None,
                 planning_attempts: int = None):
        super().__init__(name, planner, planning_attempts)

    # ------------------------------------------------------------------
    # MTC solvers::CartesianPath surface (delegated to the backing solver so
    # the stage and the solver never disagree; params mirror it for the wire)
    # ------------------------------------------------------------------
    def _solver(self) -> Any:
        """The backing ``core.CartesianPath`` when one was supplied."""
        if callable(getattr(self.planner, "setStepSize", None)):
            return self.planner
        return None

    def setStepSize(self, step_size: float) -> None:
        """Set the step size between consecutive waypoints (m)."""
        solver = self._solver()
        if solver is not None:
            solver.setStepSize(step_size)
        self.params["step_size"] = float(step_size)

    def setPrecision(self, precision: Any) -> None:
        """Set the precision of the linear path (``CartesianPrecision``)."""
        solver = self._solver()
        if solver is not None:
            solver.setPrecision(precision)
        translational = getattr(precision, "translational", precision)
        try:
            self.params["precision"] = float(translational)
        except (TypeError, ValueError):
            pass

    def setIKFrame(self, *args) -> None:
        """Set the frame to move linearly (MTC ``CartesianPath::setIKFrame``)."""
        solver = self._solver()
        if solver is not None:
            solver.setIKFrame(*args)
        for arg in args:
            if isinstance(arg, str):
                self.params["ik_frame"] = arg
                return

    def setJumpThreshold(self, *args) -> None:
        """Deprecated no-op (MTC ``CartesianPath::setJumpThreshold``)."""
        solver = self._solver()
        if solver is not None:
            solver.setJumpThreshold(*args)

    def setMaxVelocityScaling(self, *args) -> None:
        """Deprecated alias for ``setMaxVelocityScalingFactor`` (MTC)."""
        solver = self._solver()
        if solver is not None:
            solver.setMaxVelocityScaling(*args)

    def setMaxAccelerationScaling(self, *args) -> None:
        """Deprecated alias for ``setMaxAccelerationScalingFactor`` (MTC)."""
        solver = self._solver()
        if solver is not None:
            solver.setMaxAccelerationScaling(*args)

    def getPlannerId(self) -> str:
        """Name of the backing planner (MTC ``getPlannerId``)."""
        solver = self._solver()
        if solver is not None and callable(getattr(solver, "getPlannerId", None)):
            return solver.getPlannerId()
        return "CartesianPath"

    # ------------------------------------------------------------------
    # cuRobo goal forms
    # ------------------------------------------------------------------
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
        """Pin the held axes explicitly (int8[6]; all-zero = free space)."""
        self.params["hold"] = [int(c) for c in (hold or [])]

    def setCheckStraightness(self, enabled: bool) -> None:
        """Deprecated: the hold already asks for a line.

        Kept for older builders. ``False`` clears the hold (a free-space
        move); ``True`` is a no-op (the derived hold already asks for a line).
        """
        if not enabled:
            self.params["hold"] = [0, 0, 0, 0, 0, 0]

__all__ = [
    'CurrentState',
    'FixedState',
    'MoveRelative',
    'MoveTo',
    'GripperCommand',
    'Connect',
    'CartesianPath',
]

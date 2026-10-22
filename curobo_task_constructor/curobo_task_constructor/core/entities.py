"""Stage — the abstract building block of a task graph. Mirrors MoveIt Task Constructor's taxonomy (paper Sec. III-B / Fig. 2):

- ``GeneratorStage``    — populates BOTH interfaces from nothing. - ``PropagatingStage``  — reads one interface, computes, writes the other;
                          supports forward and backward propagation."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from curobo_task_constructor.core.robot import PlanRequest, PlanResult, RobotInterface
from curobo_task_constructor.core.state import InterfaceState



FORWARD = "forward"
BACKWARD = "backward"


def chain_hook(existing, handler):
    """Compose a new hook onto an existing one, if any.

    ``ContainerStage.init`` (and ``ComputeIK.init``) wrap a child's
    ``on_solution`` handler so chain composition / candidate collection runs
    the moment a child emits. Callers that attach their OWN hooks after
    ``init()`` MUST chain onto that handler instead of replacing it — a
    replacement severs the container's threading and no root solution can
    ever lift.
    """
    if existing is None:
        return handler

    def _composed(*args, **kwargs):
        existing(*args, **kwargs)
        return handler(*args, **kwargs)

    return _composed


class InitStageError(Exception):
    """Raised when a stage/container fails to initialize (MTC's
    InitStageException equivalent). Aggregates child errors."""

    def __init__(self, stage_name: str, message: str):
        self.stage_name = stage_name
        super().__init__(f"[{stage_name}] {message}")


class PropertyInitializerSource:
    """OR-combinable flags defining a source to initialize a property from
    (MTC ``Stage.PropertyInitializerSource``). Used in
    ``PropertyMap.configureInitFrom()``.
    """

    PARENT = 0x01
    INTERFACE = 0x02


@dataclass(frozen=True)
class InterfaceFlag:
    read: bool
    write: bool

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        bits = ("R" if self.read else "-") + ("W" if self.write else "-")
        return bits


class InterfaceType(Enum):
    """MTC interface taxonomy — what a stage does with start/end interfaces.

    Values are (start_flag, end_flag).
    """

    GENERATOR = (InterfaceFlag(False, True), InterfaceFlag(False, True))
    PROPAGATOR_FORWARD = (InterfaceFlag(True, False), InterfaceFlag(False, True))
    PROPAGATOR_BACKWARD = (InterfaceFlag(False, True), InterfaceFlag(True, False))
    CONNECTING = (InterfaceFlag(True, False), InterfaceFlag(True, False))

    @property
    def start(self) -> InterfaceFlag:
        return self.value[0]

    @property
    def end(self) -> InterfaceFlag:
        return self.value[1]


@dataclass
class StageFailure:
    """A failed attempt: (from_state|None, to_state|None, message)."""

    from_state: Optional[InterfaceState]
    to_state: Optional[InterfaceState]
    message: str
    #: Record id, unique with solutions in this stage's id space (MTC
    #: numbers stored solutions and failures together; solved[]/failed[]
    #: in StageStatistics reference these ids).
    failure_id: int = -1
    #: Recording stage (MTC: failures live in the stage that made them, so
    #: no reverse lookup is ever needed). Set by Stage._fail; excluded from
    #: equality so identical failures from sibling stages sharing one input
    #: state (e.g. every strategy's pre_grasp failing the same IK) stay
    #: distinct records instead of all attributing to the first match.
    stage: Any = field(default=None, repr=False, compare=False)


@dataclass
class Solution:
    """A trajectory segment (or zero-length scene mutation) connecting two
    InterfaceStates, plus its cost and the raw service response that
    produced it."""

    start: InterfaceState
    end: InterfaceState
    trajectory: Optional[list] = None  # JointStateLike waypoints; None = mutation
    cost: float = 0.0
    comment: str = ""
    response: Any = None  # raw TrajectoryResult / service response
    stage: Any = field(default=None, repr=False)
    solution_id: int = -1
    # For containers: the child solutions this composed solution chains.
    children: list = field(default_factory=list)
    # Execution surface: the PlanRequest to re-solve/execute, if motion.
    plan_request: Optional[PlanRequest] = None
    # Execution surface: a non-planned gripper command (MoveIt
    # GripperCommand). When set, ``plan_request`` is None and the executor
    # sends the command to the gripper controller instead of a trajectory.
    gripper_command: Optional[dict] = None
    # Scene ops this solution materializes before its motion (computed from
    # the scene delta it introduced).
    scene_ops: list = field(default_factory=list)
    # MTC parity: failure marking and markers.
    _is_failure: bool = field(default=False, repr=False)
    _markers: list = field(default_factory=list, repr=False)

    def key(self):
        return (self.start.key(), self.end.key())

    def is_motion(self) -> bool:
        return self.trajectory is not None or self.plan_request is not None

    def duration(self) -> float:
        if not self.trajectory:
            return 0.0
        return max(0.0, len(self.trajectory) - 1)

    @property
    def isFailure(self) -> bool:
        """True if the trajectory is marked as a failure (MTC)."""
        return self._is_failure

    @property
    def markers(self) -> list:
        """Markers to visualize important aspects of the trajectory (MTC)."""
        return self._markers

    def markAsFailure(self, comment: str = "") -> None:
        """Mark the SubTrajectory as a failure (MTC)."""
        self._is_failure = True
        if comment:
            self.comment = comment

    def toMsg(self, introspection: Any = None) -> dict:
        """Convert to the ROS message Solution (MTC).

        Returns a dict with the solution's fields (cuRobo does not use
        ROS messages internally).
        """
        return {
            "start": self.start.key() if self.start else None,
            "end": self.end.key() if self.end else None,
            "cost": self.cost,
            "comment": self.comment,
            "isFailure": self._is_failure,
            "trajectory": self.trajectory or [],
            "markers": self._markers,
        }

__all__ = [
    'FORWARD',
    'BACKWARD',
    'chain_hook',
    'InitStageError',
    'PropertyInitializerSource',
    'InterfaceFlag',
    'InterfaceType',
    'StageFailure',
    'Solution',
]

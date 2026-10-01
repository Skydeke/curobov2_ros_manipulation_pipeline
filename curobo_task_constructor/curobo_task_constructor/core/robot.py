"""Normalized planning request/result types and the abstract RobotInterface.

The core is ROS-free: stages talk to a ``RobotInterface`` (the cuRobo server
surface) through these lightweight dataclasses. The ROS deployment adapter
(``robot/curobo.py``) converts them to/from isaac_ros_cumotion_interfaces
message types. Unit tests inject a ``MockCuroboServer`` implementing the
same interface — this is what makes the framework testable without a GPU or
a running curobo_server.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from curobo_task_constructor.core.state import ObjectSpec


class ServiceError(RuntimeError):
    """Raised when the curobo server is unreachable or rejects a request."""


@dataclass
class GoalsetSpec:
    """One segment of a goal path (mirror of isaac_ros Goalset.msg)."""

    poses: list = field(default_factory=list)  # Pose-like candidate(s)
    target_joint_positions: list = field(default_factory=list)  # joint-space
    allowed_collisions: list = field(default_factory=list)  # link names
    trajectory_constraints: list = field(default_factory=list)  # axis holds


@dataclass
class PlanRequest:
    """A whole-task planning request (mirror of TrajectoryGoal.msg).

    ``start_pose`` is a JointStateLike — the state the task starts from. Each
    goalset is one segment; cuRobo solves the whole path in one call.
    """

    goalsets: list = field(default_factory=list)
    start_pose: Any = None  # JointStateLike or None (server uses current)
    planner: Any = None  # planner key/int; None -> server default


@dataclass
class PlanResult:
    """Normalized trajectory solve result (mirror of TrajectoryResult.msg)."""

    success: bool
    message: str = ""
    trajectory: list = field(default_factory=list)  # JointStateLike waypoints
    dt: float = 0.0
    cost: float = float("inf")  # path cost used for ranking
    raw: Any = None  # the raw service response (TrajectoryResult in ROS)
    selected_goal_index: list = field(default_factory=list)
    waypoint_status: list = field(default_factory=list)
    stats: Any = None

    @property
    def last_state(self) -> Any:
        return self.trajectory[-1] if self.trajectory else None


class RobotInterface(ABC):
    """The injectable cuRobo-server surface the stages compute against.

    Mission-critical property: every method is synchronous and blocking. The
    executor's compute loop is single-threaded (like MTC's), so the real
    rclpy adapter simply spins until each service round-trip completes.
    """

    #: Message class used to build JointState-likes (None in the ROS-free
    #: core/tests -> a stub is used; the rclpy adapter sets
    #: ``sensor_msgs.msg.JointState``).
    joint_state_cls = None

    #: Message class used to build Pose-likes (None -> an internal stub;
    #: the rclpy adapter sets ``geometry_msgs.msg.Pose``).
    pose_cls = None

    # ----- state -------------------------------------------------------
    @abstractmethod
    def get_current_joint_state(self) -> Any:
        """Current robot joint state (JointStateLike)."""

    @abstractmethod
    def get_object_pose(self, name: str) -> Any:
        """World pose of scene object `name`, or None when unknown."""

    def get_object_spec(self, name: str) -> Any:
        """Full description of scene object `name` as an ``ObjectSpec``, or
        None when unknown.

        Preferred over ``get_object_pose`` wherever the caller intends to
        RE-ADD the object, because a pose alone cannot rebuild it: the shape and
        the size are lost, and an implementation that guesses them describes the
        world wrongly. Default: None, so an interface that only tracks poses
        still works — the caller falls back.
        """
        return None

    def get_named_joint_config(self, name: str) -> Optional[list]:
        """Joint positions for a named config from the robot's own config
        YAML (used by FixedState). Default: no named configs."""
        return None

    def get_attached_objects(self) -> list:
        """Names of scene objects currently attached to the robot (the MTC
        planning-scene attached-object query; Sec 6 reverse-sync surface).
        Default: nothing known to be attached."""
        return []

    # ----- kinematics --------------------------------------------------
    @abstractmethod
    def fk(self, joint_state: Any, link: Optional[str] = None) -> Any:
        """Forward kinematics: pose of ``link`` (default: tool tip)."""

    def fk_batch(self, joint_states: list, link: Optional[str] = None) -> list:
        """FK for several states at once, returning one pose per input.

        The wire carries this natively (``Fk.srv`` takes a
        ``JointState[]``), so the ROS adapter overrides it with a single
        round-trip; the default loops so tests need no batch support. Used by
        the Cartesian stage to measure a whole trajectory's straightness in one
        call instead of one per waypoint.
        """
        return [self.fk(js, link) for js in (joint_states or [])]

    @abstractmethod
    def ik(self, pose: Any, seed: Optional[Any] = None) -> Any:
        """Inverse kinematics: joint state reaching ``pose`` or None."""

    def ik_batch(self, poses: list) -> list:
        """IK for several poses; returns [(joint_state|None, valid_bool)]."""
        return [self.ik(p) for p in poses]  # default: one-by-one

    # ----- planning ----------------------------------------------------
    @abstractmethod
    def set_planner(self, planner: Any) -> None:
        """Select the active cuRobo planner (SetPlanner)."""

    @abstractmethod
    def plan(self, request: PlanRequest) -> PlanResult:
        """Solve one whole-task request (TrajectoryGeneration)."""

    def plan_batch(self, requests: list) -> list:
        """Solve several whole-task requests in ONE batched call
        (TrajectoryGenerationBatch). Default: sequential ``plan``."""
        return [self.plan(r) for r in requests]

    @abstractmethod
    def execute(self, request: PlanRequest) -> PlanResult:
        """Plan and drive the arm (SendTrajectory action)."""

    # ----- scene -------------------------------------------------------
    @abstractmethod
    def add_object(self, spec: ObjectSpec) -> bool:
        """Register an obstacle in the server's collision scene."""

    @abstractmethod
    def remove_object(self, name: str) -> bool:
        """Remove an obstacle from the server's collision scene."""

    def remove_all_objects(self) -> None:  # pragma: no cover - convenience
        pass

    @abstractmethod
    def attach_object(self, name: str) -> bool:
        """Attach scene obstacle `name` to the robot flange."""

    @abstractmethod
    def detach_object(self, name: Optional[str] = None) -> bool:
        """Detach the attached obstacle (server detaches by name or all)."""

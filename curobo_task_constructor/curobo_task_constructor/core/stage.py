"""Stage — the abstract building block of a task graph.

Mirrors MoveIt Task Constructor's taxonomy (paper Sec. III-B / Fig. 2):

- ``GeneratorStage``    — populates BOTH interfaces from nothing.
- ``PropagatingStage``  — reads one interface, computes, writes the other;
                          supports forward and backward propagation.
- ``ConnectingStage``   — reads a PAIR of already-known states (start and
                          end) and solves a trajectory joining them.

A stage declares which interface type it is; containers validate adjacency
at ``init()`` time (a stage that only *writes* an interface must be
followed/preceded by one that *reads* it — the same rule MTC enforces).

New capabilities are added by writing a ``Stage`` subclass and registering
it via ``@register_stage("name")`` — never by editing an enum or the
executor.
"""

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
    # Scene ops this solution materializes before its motion (computed from
    # the scene delta it introduced).
    scene_ops: list = field(default_factory=list)

    def key(self):
        return (self.start.key(), self.end.key())

    def is_motion(self) -> bool:
        return self.trajectory is not None or self.plan_request is not None

    def duration(self) -> float:
        if not self.trajectory:
            return 0.0
        return max(0.0, len(self.trajectory) - 1)


class Stage(ABC):
    """Abstract base class of every task-graph node (containers included)."""

    #: The stage's interface type — subclasses must set this.
    interface: InterfaceType = InterfaceType.PROPAGATOR_FORWARD

    def __init__(self, name: Optional[str] = None, params: Optional[dict] = None):
        self.name = name or self.default_name()
        self.params = dict(params or {})
        self.parent: Optional["Stage"] = None

        # Pull interfaces: states received from the neighbouring stages.
        self.start_pull: list = []  # read side (forward)
        self.end_pull: list = []  # read side (backward)

        # Written interfaces: states this stage produced.
        self._starts: list = []
        self._ends: list = []

        self.solutions: list = []
        self.failures: list = []
        self.children: list = []  # containers only

        self.stage_id = -1
        self.robot: Optional[RobotInterface] = None
        self._computed = False  # generators run once per task
        self._emitted_keys: set = set()
        self.attempt_count = 0
        self.compute_time = 0.0
        # Propagation direction resolved by the containing container
        # (EitherWay stages): None (unresolved) | "forward" | "backward".
        self._flow: Optional[str] = None
        # Optional batch session (Alternatives collect whole-task requests).
        self._batch: Any = None
        # Event hooks (tests / introspection attach here):
        self.on_solution: Optional[Callable[[Solution], None]] = None
        self.on_failure: Optional[Callable[[StageFailure], None]] = None

    # ------------------------------------------------------------------
    # Stage identity / description
    # ------------------------------------------------------------------
    @classmethod
    def stage_type(cls) -> str:
        """Registered registry name (set by @register_stage)."""
        return getattr(cls, "_registry_name", cls.__name__.lower())

    def default_name(self) -> str:
        return self.stage_type()

    def describe(self) -> dict:
        return {"name": self.name, "type": self.stage_type()}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def init(self, base_scene, robot: RobotInterface) -> None:
        """Validate configuration against the task's base scene.

        ``base_scene`` is a SceneDiff describing the task's starting world
        (objects available for attach/grasp etc.).
        """
        self.robot = robot

    @abstractmethod
    def compute(self) -> None:
        """Push new (InterfaceState, Solution) pairs."""

    def can_compute(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Interface plumbing (used by containers)
    # ------------------------------------------------------------------
    def interface_flags(self):
        """(start_flag, end_flag) this stage exposes for adjacency checks.

        Leaf stages report their declared ``InterfaceType``; containers
        report the flags resolved from their boundary children at init.
        """
        return self.interface.start, self.interface.end

    def required_flags(self):
        """The interface this stage needs; identical to ``interface_flags``
        except for AUTO-direction propagators, which report the direction
        the container resolved for them."""
        return self.interface_flags()

    def resolve(self, expected_start, expected_end=None) -> None:
        """Resolve propagation direction from the neighbouring stages.

        Fixed-interface stages are a no-op here; ``PropagatingEitherWay``
        adopts FORWARD or BACKWARD based on what the container expects (MTC
        ``resolveInterface``), and containers recurse into their children.
        """

    def push_start(self, state: InterfaceState) -> None:
        for existing in self.start_pull:
            if existing.key() == state.key():
                return
        self.start_pull.append(state)

    def push_end(self, state: InterfaceState) -> None:
        for existing in self.end_pull:
            if existing.key() == state.key():
                return
        self.end_pull.append(state)

    def written_starts(self) -> list:
        return list(self._starts)

    def written_ends(self) -> list:
        return list(self._ends)

    def num_children(self) -> int:
        return len(self.children)

    # ------------------------------------------------------------------
    # Solution production
    # ------------------------------------------------------------------
    def _emit(self, solution: Solution) -> None:
        key = solution.key()
        if key in self._emitted_keys:
            return
        self._emitted_keys.add(key)
        solution.stage = self
        solution.solution_id = len(self.solutions)
        self.solutions.append(solution)
        self.attempt_count += 1
        if self.on_solution:
            self.on_solution(solution)

    def _fail(self, from_state, to_state, message: str) -> None:
        failure = StageFailure(from_state, to_state, message)
        self.failures.append(failure)
        self.attempt_count += 1
        if self.on_failure:
            self.on_failure(failure)

    def spawn(self, state: InterfaceState, cost: float = 0.0, comment: str = "",
              response: Any = None) -> None:
        """Generator helper: same state at both interfaces, no trajectory."""
        self._starts.append(state)
        self._ends.append(state)
        self._emit(Solution(state, state, trajectory=None, cost=cost,
                            comment=comment, response=response))

    def send_forward(self, from_state, to_state, trajectory=None, cost=0.0,
                     comment: str = "", response: Any = None,
                     plan_request: Optional[PlanRequest] = None,
                     scene_ops: Optional[list] = None) -> None:
        """Forward propagator helper: writes the end interface."""
        self._ends.append(to_state)
        self._emit(Solution(from_state, to_state, trajectory=trajectory,
                            cost=cost, comment=comment, response=response,
                            plan_request=plan_request, scene_ops=scene_ops or []))

    def send_backward(self, from_state, to_state, trajectory=None, cost=0.0,
                      comment: str = "", response: Any = None,
                      plan_request: Optional[PlanRequest] = None) -> None:
        """Backward propagator helper: writes the start interface."""
        self._starts.append(from_state)
        self._emit(Solution(from_state, to_state, trajectory=trajectory,
                            cost=cost, comment=comment, response=response,
                            plan_request=plan_request))

    def lift_solution(self, solution: Solution) -> None:
        """Container helper: forward a child solution as our own."""
        from curobo_task_constructor.core.container import ContainerStage
        if isinstance(self, ContainerStage):
            sol = Solution(
                start=solution.start, end=solution.end,
                trajectory=solution.trajectory, cost=solution.cost,
                comment=solution.comment, response=solution.response,
                children=[solution],
            )
            self._emit(sol)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run_compute(self) -> None:
        """Timed, counted execution of ``compute()`` (the MTC runCompute
        equivalent)."""
        self.attempt_count += 0  # attempts counted on emit/fail
        started = time.perf_counter()
        try:
            self.compute()
        finally:
            self.compute_time += time.perf_counter() - started

    def reset(self) -> None:
        """Reset per-task state so the stage can be re-planned (MTC reset)."""
        self.start_pull = []
        self.end_pull = []
        self._starts = []
        self._ends = []
        self.solutions = []
        self.failures = []
        self._computed = False
        self._emitted_keys = set()
        self.attempt_count = 0
        self.compute_time = 0.0
        self._flow = None
        self._batch = None

    def subtree_stages(self) -> list:
        """This stage + all descendants (depth-first)."""
        out = [self]
        for child in self.children:
            out.extend(child.subtree_stages())
        return out


class GeneratorStage(Stage):
    """Populates both interfaces from nothing (MTC generator)."""

    interface = InterfaceType.GENERATOR

    def can_compute(self) -> bool:
        return not self._computed

    def run_compute(self) -> None:
        # Generators run exactly once per task.
        if self._computed:
            return
        self._computed = True  # set before compute so re-entry is safe
        super().run_compute()


class PropagatingStage(Stage):
    """Reads one interface, computes, writes the opposite one.

    Fixeds-flow propagator (MTC ``PropagatingForward``): reads the start
    interface, writes the end interface. Subclasses implement
    ``compute_forward(state)``.
    """

    interface = InterfaceType.PROPAGATOR_FORWARD

    def can_compute(self) -> bool:
        return bool(self.start_pull)

    def compute(self) -> None:
        work, self.start_pull = self.start_pull, []  # consume the pull
        for state in work:
            self.compute_forward(state)

    def compute_forward(self, state: InterfaceState) -> None:
        raise NotImplementedError(f"{type(self).__name__} must implement compute_forward")


class BackwardPropagatingStage(PropagatingStage):
    """Backward propagation: read end, write start (trajectories reversed
    where needed, same as MTC ``PropagatingBackward``)."""

    interface = InterfaceType.PROPAGATOR_BACKWARD

    def can_compute(self) -> bool:
        return bool(self.end_pull)

    def compute(self) -> None:
        work, self.end_pull = self.end_pull, []  # consume the pull
        for state in work:
            self.compute_backward(state)

    def compute_backward(self, state: InterfaceState) -> None:
        raise NotImplementedError(f"{type(self).__name__} must implement compute_backward")


class PropagatingEitherWay(PropagatingStage):
    """A propagator that may run forward (read start) or backward (read end).

    The containing serial container resolves the direction from its
    neighbours — what MTC's ``PropagatingEitherWay`` does in
    ``resolveInterface``: a stage whose output has to feed the *previous*
    sibling runs backward (e.g. the pick/place approach stages), otherwise it
    runs forward. Subclasses implement ``compute_forward`` and
    ``compute_backward``.

    ``direction`` may pin one side (MTC ``restrictDirection``): "auto"
    (default), "forward", or "backward".
    """

    def __init__(self, name: Optional[str] = None, params: Optional[dict] = None,
                 direction: str = "auto"):
        super().__init__(name, params)
        if direction not in ("auto", "forward", "backward"):
            raise ValueError(f"direction must be auto|forward|backward, got {direction!r}")
        self._configured_direction = direction

    # -- interface resolution ------------------------------------------
    def interface_flags(self):
        if self._flow == BACKWARD:
            return InterfaceType.PROPAGATOR_BACKWARD.start, InterfaceType.PROPAGATOR_BACKWARD.end
        return InterfaceType.PROPAGATOR_FORWARD.start, InterfaceType.PROPAGATOR_FORWARD.end

    def required_flags(self):
        return self.interface_flags()

    def resolve(self, expected_start, expected_end=None) -> None:
        flow = None
        if expected_start is not None:
            if expected_start.read and not expected_start.write:
                flow = FORWARD
            elif expected_start.write and not expected_start.read:
                flow = BACKWARD
            elif not expected_start.read and not expected_start.write:
                flow = self._resolve_from_end(expected_end)
        else:
            flow = self._resolve_from_end(expected_end)
        if flow is None:
            raise InitStageError(
                self.name,
                f"propagator cannot satisfy expected interface start={expected_start} end={expected_end}")
        if self._configured_direction != "auto" and flow != self._configured_direction:
            raise InitStageError(
                self.name,
                f"configured direction {self._configured_direction!r} "
                f"does not match expected one {flow!r}")
        self._flow = flow

    def _resolve_from_end(self, expected_end):
        if expected_end is None:
            # No expectation at all -> default to forward (only reachable in
            # degenerate graphs; MTC would error on UNKNOWN).
            return FORWARD
        if expected_end.write and not expected_end.read:
            return FORWARD
        if expected_end.read and not expected_end.write:
            return BACKWARD
        return None

    # -- execution ------------------------------------------------------
    def can_compute(self) -> bool:
        if self._flow == BACKWARD:
            return bool(self.end_pull)
        return bool(self.start_pull)

    def compute(self) -> None:
        if self._flow == BACKWARD:
            work, self.end_pull = self.end_pull, []
            for state in work:
                self.compute_backward(state)
        else:
            work, self.start_pull = self.start_pull, []
            for state in work:
                self.compute_forward(state)

    def compute_forward(self, state: InterfaceState) -> None:
        raise NotImplementedError(f"{type(self).__name__} must implement compute_forward")

    def compute_backward(self, state: InterfaceState) -> None:
        raise NotImplementedError(f"{type(self).__name__} must implement compute_backward")


class ConnectingStage(Stage):
    """Reads a pair of already-known states and solves a trajectory joining
    them."""

    interface = InterfaceType.CONNECTING

    def can_compute(self) -> bool:
        return bool(self.start_pull) and bool(self.end_pull)


class TrajectoryStage(PropagatingStage):
    """A forward propagator whose work is a single whole-task PlanRequest.

    Splits "build the request" from "solve it" so containers can collect the
    requests of several motion children into ONE ``plan_batch`` call
    (Alternatives batching — a pure optimization, invisible to stage authors
    above the container).
    """

    #: engine/planner key for set_planner before solving (None = leave).
    planner: Any = None

    def build_plan_request(self, start: InterfaceState) -> Optional[PlanRequest]:
        """Build the whole-task request from a start state (may call
        robot.fk/ik for relative goals). Return None when no goal is
        computable."""
        raise NotImplementedError

    def make_end_state(self, start: InterfaceState, result: PlanResult,
                       raw: Any = None) -> Optional[InterfaceState]:
        """Derive the end state (joints + scene) from a successful solve."""
        raise NotImplementedError

    def prepare(self) -> bool:
        """Build pending (start, request) pairs WITHOUT planning.

        Consumes the start pull (delivered states are either built into a
        request or dropped), so ``can_compute`` goes quiet once no new input
        remains.
        """
        work, self.start_pull = self.start_pull, []
        self._pending = []
        for s in work:
            req = self.build_plan_request(s)
            if req is not None:
                self._pending.append((s, req))
        return bool(self._pending)

    def solve_pending(self) -> None:
        if self._batch is not None:
            # The parent Alternatives container solves all pending requests
            # in ONE plan_batch call (Sec. 2 optimization).
            for start, req in self._pending:
                self._batch.defer(self, start, req)
            return
        for start, req in self._pending:
            try:
                result = self.robot.plan(req)
            except Exception as exc:  # ServiceError etc.
                self._fail(start, None, f"plan call failed: {exc}")
                continue
            if not result.success:
                self._fail(start, None, result.message or "plan failed")
                continue
            end = self.make_end_state(start, result, raw=result)
            if end is not None:
                self.send_forward(start, end, trajectory=result.trajectory,
                                  cost=self._cost_of(result), comment="",
                                  response=result.raw, plan_request=req)

    def commit_result(self, start: InterfaceState, req: PlanRequest,
                      result: PlanResult) -> None:
        """Commit the result of a batched solve for (start, req)."""
        if not result.success:
            self._fail(start, None, result.message or "plan failed")
            return
        end = self.make_end_state(start, result, raw=result)
        if end is not None:
            self.send_forward(start, end, trajectory=result.trajectory,
                              cost=self._cost_of(result), comment="",
                              response=result.raw, plan_request=req)

    def compute(self) -> None:
        if self.prepare():
            self.solve_pending()

    def _cost_of(self, result: PlanResult) -> float:
        """Ranking cost of a solve: explicit cost if populated, else the
        (seeded) considered rows or a waypoint-count proxy."""
        if result.cost != float("inf"):
            return float(result.cost)
        stats = getattr(result, "stats", None)
        if stats is not None:
            rows = getattr(stats, "considered", None)
            if rows:
                costs = [getattr(r, "cost", float("inf")) for r in rows]
                costs = [c for c in costs if c != float("inf")]
                if costs:
                    return float(min(costs))
        return float(len(result.trajectory)) if result.trajectory else 0.0
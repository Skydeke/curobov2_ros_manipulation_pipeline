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


from curobo_task_constructor.core.entities import *

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
        #: Record sequence shared by solutions and failures (MTC numbers
        #: stored solutions and failures in one id space per stage).
        self._record_seq = 0
        self.attempt_count = 0
        self.compute_time = 0.0
        self._accounted_time = 0.0
        # Planner calls in the current compute pass not yet "spent" on an
        # emitted record. attempt_count = planner calls + records from
        # computes that made none (generators, mutations, validation
        # failures): a record consumes one pending try instead of counting
        # again, so best-of-3 reads 3, not 4. Reset per pass in run_compute.
        self._pending_tries = 0
        # Propagation direction resolved by the containing container
        # (EitherWay stages): None (unresolved) | "forward" | "backward".
        self._flow: Optional[str] = None
        # Optional batch session (Alternatives collect whole-task requests).
        self._batch: Any = None
        # Event hooks (tests / introspection attach here):
        self.on_solution: Optional[Callable[[Solution], None]] = None
        self.on_failure: Optional[Callable[[StageFailure], None]] = None
        # Fired on every planner attempt (each robot.plan call inside a
        # stage's multi-attempt loop), so introspection can stream
        # attempts/compute-time per try instead of only on emit. See
        # _note_plan_attempt; chain with chain_hook like the other two.
        self.on_progress: Optional[Callable[[Stage], None]] = None
        # Fired for successful tries that are NOT ranked (the seeds a
        # best-of-N loop discards): every attempt leaves a visible row, so
        # attempts == successful + failed rows always. Ranked winners still
        # go through on_solution (store + lift); containers never see this
        # hook, so ranking is untouched. Chain with chain_hook.
        self.on_considered: Optional[Callable[[Solution], None]] = None

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
    def _count_record(self) -> None:
        """Count one emitted record (solution or failure).

        A record spends one pending planner try instead of counting again;
        a record from a compute that made no planner call (generators,
        mutations, validation failures) counts 1. So attempt_count reads as
        planner calls plus try-less records: best-of-3 with one winner is 3,
        three failed tries is 3, a generator run is 1.
        """
        if self._pending_tries > 0:
            self._pending_tries -= 1
        else:
            self.attempt_count += 1

    def _next_record_id(self) -> int:
        """Next id in this stage's shared record space (MTC numbers stored
        solutions and failures together; solved[]/failed[] reference them)."""
        record_id = self._record_seq
        self._record_seq += 1
        return record_id

    def _emit(self, solution: Solution) -> None:
        key = solution.key()
        if key in self._emitted_keys:
            return
        self._emitted_keys.add(key)
        solution.stage = self
        solution.solution_id = self._next_record_id()
        self.solutions.append(solution)
        self._count_record()
        if self.on_solution:
            self.on_solution(solution)

    def _fail(self, from_state, to_state, message: str) -> None:
        failure = StageFailure(from_state, to_state, message,
                               failure_id=self._next_record_id())
        failure.stage = self
        self.failures.append(failure)
        self._count_record()
        if self.on_failure:
            self.on_failure(failure)
        # Tick statistics live too (on_progress drives publishTaskState):
        # without this the ✗ column only moves at pass end while the red
        # row itself already streams via on_failure.
        if self.on_progress is not None:
            self.on_progress(self)

    def _consider(self, solution: Solution) -> None:
        """Store a successful try without lifting it. MTC stores every computed solution; best-of-N ranking still picks
        the cheapest via cost sort, and containers only ever see the
        promoted winner. Fires ``on_considered`` for consumers that want
        per-try visibility."""
        solution.stage = self
        solution.solution_id = self._next_record_id()
        self.solutions.append(solution)
        self._count_record()
        if self.on_considered is not None:
            self.on_considered(solution)

    def _note_plan_attempt(self) -> None:
        """Record one planner call and stream it (multi-attempt loops). Best-of-N stages (MoveTo, TrajectoryStage) call the planner several
        times but emit once, so without this the panel would sit still until
        the Nth try finished. Each call counts and fires ``on_progress``;
        the node publishes a per-stage statistics snapshot on it, so
        attempts and compute time walk 1, 2, 3 live."""
        self.attempt_count += 1
        self._pending_tries += 1
        if self.on_progress is not None:
            self.on_progress(self)

    def _timed_plan(self, req) -> Any:
        """robot.plan timed into compute_time as it happens. MTC accumulates compute time per compute; attempts here span seconds
        each, so each call ticks the stage clock live instead of only at
        pass end. run_compute subtracts the accounted share, so the total
        stays exactly the wall duration (never double-counted)."""
        self.robot.plan_scene(
            getattr(req, "scene", None), getattr(req, "start_pose", None))
        started = time.perf_counter()
        try:
            return self.robot.plan(req)
        finally:
            dt = time.perf_counter() - started
            self.compute_time += dt
            self._accounted_time += dt

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

    def _promote_forward(self, from_state, to_state, solution: Solution) -> None:
        """Lift an already-stored winner downstream without re-storing it.

        Best-of-N legs store every try live (via _consider) so the panel
        walks each attempt the moment it solves; only the cheapest may feed
        downstream. Re-emitting the winner through send_forward would store
        (and count) it a second time, so promotion writes the end interface
        and fires the hook chain (container lift + live republish of the
        same global id, deduplicated by receivers) exactly like an emit
        minus the store.
        """
        self._ends.append(to_state)
        if self.on_solution is not None:
            self.on_solution(solution)

    def _promote_backward(self, from_state, to_state, solution: Solution) -> None:
        """Backward variant of _promote_forward (writes the start interface)."""
        self._starts.append(from_state)
        if self.on_solution is not None:
            self.on_solution(solution)

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
        self._accounted_time = 0.0  # per-pass live-ticked share, see below
        self._pending_tries = 0  # per-pass: records spend only this pass's tries
        started = time.perf_counter()
        try:
            self.compute()
        finally:
            # Tries that timed themselves via _timed_plan already ticked
            # compute_time live; add only the unaccounted remainder so the
            # total stays exactly the wall duration.
            self.compute_time += time.perf_counter() - started - self._accounted_time

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
        self._record_seq = 0
        self.attempt_count = 0
        self.compute_time = 0.0
        self._accounted_time = 0.0
        self._pending_tries = 0
        self._flow = None
        self._batch = None

    def subtree_stages(self) -> list:
        """This stage + all descendants (depth-first)."""
        out = [self]
        for child in self.children:
            out.extend(child.subtree_stages())
        return out

__all__ = [
    'Stage',
]

"""Containers — compose stages into task graphs (MTC paper Sec. III-C / MTC's
``core/src/container.cpp`` semantics).

A container is itself a ``Stage`` (nestable), and its *interface* is derived
from its boundary children:

- ``SerialContainer``  — children run in order; adjacent interface types are
  validated at init with MTC's exact ``connect()`` rules; a container
  solution exists only when *every* child produced one (costs accumulate).
- ``Alternatives``     — same input(s) to all children; every child solution
  becomes a container solution. When all children are whole-task motion
  stages, their requests are collected into ONE ``plan_batch`` call (a pure
  optimization, invisible to stages above the container).
- ``Fallbacks``        — children run in order; only the first child that
  produces a solution counts.
- ``IndependentComponents`` — children act on disjoint joint groups; every
  child solution lifts (intended for bimanual work; lowest priority).

Adjacency rules (from ``SerialContainerPrivate::connect``): an edge between
``prev`` and ``next`` is valid iff

    prev writes its END interface and next reads its START interface   (forward)
  OR
    prev reads its END interface and next writes its START interface   (backward)

This is what lets an ``approach``-style stage (reads end, writes start) sit
between a ``Connect`` and the generator-wrapped ``ComputeIK`` in the
pick&place graph: the approach resolves BACKWARD because the Connect expects
the container to *write* its start.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from curobo_task_constructor.core.stage import (
    BACKWARD,
    FORWARD,
    InterfaceFlag,
    InterfaceType,
    Stage,
    InitStageError,
    Solution,
    TrajectoryStage,
)
from curobo_task_constructor.core.state import InterfaceState
from curobo_task_constructor.core.robot import RobotInterface

__all__ = [
    "ContainerStage",
    "SerialContainer",
    "Alternatives",
    "Fallbacks",
    "IndependentComponents",
    "invert_flags",
    "GENERATE_INTERFACE",
]

#: The root container of a task presents a GENERATE interface (it writes both
#: the start and the end interface — a full task spans start to finish).
GENERATE_INTERFACE = (InterfaceFlag(False, True), InterfaceFlag(False, True))


def invert_flags(start: InterfaceFlag, end: InterfaceFlag):
    """MTC's ``invert()``: swap read<->write on both interfaces.

    Used by the serial resolution cascade — the next child must satisfy
    ``invert(previous.requiredInterface()) & START_IF_MASK``.
    """
    return (InterfaceFlag(end.write, end.read),
            InterfaceFlag(start.write, start.read))


def _same_joints(a, b) -> bool:
    pa = list(getattr(a, "position", None) or [])
    pb = list(getattr(b, "position", None) or [])
    return len(pa) == len(pb) and all(abs(x - y) < 1e-9 for x, y in zip(pa, pb))


def _concat_trajectories(chain) -> list:
    """Concatenate child waypoint lists, dropping duplicated boundary joint
    configurations. Returns None when the chain has no motion at all."""
    out = []
    for s in chain:
        traj = s.trajectory
        if not traj:
            continue  # zero-length scene mutation contributes no waypoints
        if out and _same_joints(out[-1], traj[0]):
            traj = traj[1:]
        out.extend(traj)
    return out if out else None


class _BatchSession:
    """Collects the whole-task requests of several children of an
    ``Alternatives`` container and solves them in ONE plan_batch call."""

    def __init__(self, robot: RobotInterface):
        self.robot = robot
        self._jobs = []  # (stage, start_state, PlanRequest)

    def defer(self, stage, start, req) -> None:
        self._jobs.append((stage, start, req))

    def flush(self) -> None:
        if not self._jobs:
            return
        requests = [req for _, _, req in self._jobs]
        results = self.robot.plan_batch(requests)
        for (stage, start, req), result in zip(self._jobs, results):
            stage.commit_result(start, req, result)


class ContainerStage(Stage, ABC):
    """Base class of every container.

    A container validates its children's interface adjacency at ``init()``
    (reject at build time, exactly like MTC) and — after ``resolve()`` has
    run — advertises the interface derived from its boundary children.
    """

    def __init__(self, name=None, params=None):
        super().__init__(name, params)
        self.children: list = []
        self._resolved = False
        self._start_flag: InterfaceFlag = None
        self._end_flag: InterfaceFlag = None
        self._edges: list = []  # per adjacent pair: FORWARD | BACKWARD
        # External pulls not yet forwarded to boundary children.
        self._fed_start = 0
        self._fed_end = 0
        # Solution lookups for chain composition (identity-keyed).
        self._by_end: dict = {}
        self._by_start: dict = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def stage_type(cls) -> str:
        """The ``container_type`` wire name (``stage_kind``); containers are
        never registered in STAGE_REGISTRY, so fall back to the class name
        only for exotic unlabeled subclasses."""
        kind = getattr(cls, "stage_kind", None)
        return kind if kind else super().stage_type()

    def add(self, child: Stage) -> Stage:
        """Append a child (returns it, for chaining)."""
        child.parent = self
        self.children.append(child)
        return child

    def insert(self, child: Stage, index: int = -1) -> Stage:
        child.parent = self
        self.children.insert(index if index >= 0 else len(self.children), child)
        return child

    def num_children(self) -> int:
        return len(self.children)

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------
    def interface_flags(self):
        if not self._resolved:
            return self.interface.start, self.interface.end  # pre-resolve best guess
        return self._start_flag, self._end_flag

    def required_flags(self):
        return self.interface_flags()

    def resolve(self, expected_start, expected_end=None):
        """Run the child resolution cascade for this container kind."""
        if not self.children:
            raise InitStageError(self.name, "container requires at least one child")
        if expected_start is None and expected_end is None:
            raise InitStageError(self.name, "cannot resolve to an unknown interface")
        self._resolve_children(expected_start, expected_end)
        self._resolved = True

    @abstractmethod
    def _resolve_children(self, expected_start, expected_end) -> None:
        ...

    def _boundary_validate(self, expected_start, expected_end) -> None:
        if expected_start is not None and self._start_flag != expected_start:
            raise InitStageError(
                self.name,
                f"start interface ({self._start_flag}) does not match "
                f"expected ({expected_start})")
        if expected_end is not None and self._end_flag != expected_end:
            raise InitStageError(
                self.name,
                f"end interface ({self._end_flag}) does not match "
                f"expected ({expected_end})")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def init(self, base_scene, robot: RobotInterface) -> None:
        if not self.children:
            raise InitStageError(self.name, "container requires at least one child")
        super().init(base_scene, robot)
        for child in self.children:
            child.init(base_scene, robot)
        # Hook every child's solutions so chain composition / lifting runs
        # the moment a child emits (MTC's onNewSolution callback).
        for child in self.children:
            upstream = child.on_solution

            def _handler(sol, child=child, upstream=upstream):
                if upstream is not None:
                    upstream(sol)
                self._on_child_solution(child, sol)

            child.on_solution = _handler

    def _on_child_solution(self, child: Stage, solution: Solution) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Input feeding / output propagation
    # ------------------------------------------------------------------
    def _has_unfed_external(self) -> bool:
        if self._start_flag is not None and self._start_flag.read and \
                len(self.start_pull) > self._fed_start:
            return True
        if self._end_flag is not None and self._end_flag.read and \
                len(self.end_pull) > self._fed_end:
            return True
        return False

    def can_compute(self) -> bool:
        return self._has_unfed_external() or any(
            child.can_compute() for child in self.children)

    def _push_written(self, target: list, state: InterfaceState) -> None:
        for existing in target:
            if existing.key() == state.key():
                return
        target.append(state)

    def _index_solution(self, sol: Solution) -> None:
        self._by_end.setdefault(id(sol.end), []).append(sol)
        self._by_start.setdefault(id(sol.start), []).append(sol)

    def _lift(self, sol: Solution) -> None:
        """Emit a container-level solution (keyed dedup via Stage._emit)."""
        self._emit(sol)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def describe(self) -> dict:
        out = super().describe()
        out["container_type"] = self.stage_type()
        out["children"] = [child.describe() for child in self.children]
        return out

    def reset(self) -> None:
        for child in self.children:
            child.reset()
        super().reset()
        self._resolved = False
        self._start_flag = self._end_flag = None
        self._edges = []
        self._fed_start = 0
        self._fed_end = 0
        self._by_end = {}
        self._by_start = {}


class SerialContainer(ContainerStage):
    """Children run in order; MTC's flagship container.

    A solution of the container is a complete chain — one solution per child,
    each child's start state identical to its predecessor's end state — with
    accumulated cost. Partial chains are kept internally; they only surface
    once every child contributed.
    """

    #: container_type name used in StageSpec trees / describe()
    stage_kind = "serial"

    def _resolve_children(self, expected_start, expected_end) -> None:
        first, last = self.children[0], self.children[-1]

        # First child must satisfy the container's expected START interface.
        first.resolve(expected_start, None)
        prev_s, prev_e = first.required_flags()

        n = len(self.children)
        self._edges = [None] * (n - 1)
        for i in range(1, n):
            inv_s, inv_e = invert_flags(prev_s, prev_e)
            child = self.children[i]
            child.resolve(inv_s, None)
            self._check_edge(i - 1, i)
            prev_s, prev_e = child.required_flags()

        self._start_flag, self._end_flag = first.required_flags()[0], last.required_flags()[1]
        self._boundary_validate(expected_start, expected_end)

    def _check_edge(self, i: int, j: int) -> None:
        prev = self.children[i]
        nxt = self.children[j]
        prev_s, prev_e = prev.required_flags()
        nxt_s, nxt_e = nxt.required_flags()
        if prev_e.write and nxt_s.read:
            self._edges[i] = FORWARD
        elif prev_e.read and nxt_s.write:
            self._edges[i] = BACKWARD
        else:
            raise InitStageError(
                self.name,
                f"cannot connect end interface of '{prev.name}' "
                f"({prev_s} {prev_e}) to start interface of '{nxt.name}' "
                f"({nxt_s} {nxt_e})")

    # -- compute -------------------------------------------------------
    def compute(self) -> None:
        # First solution wins — stop scheduling (same rule as Fallbacks,
        # container.py). Without this, the serial chain keeps RE-CONNECTING:
        # every new upstream end (e.g. each completed alternative variant
        # chain's lift pose) re-feeds the first child's pull, re-running the
        # whole chain (return -> forbid -> open -> detach) once per variant.
        # That redundant re-plan of an already-solved stage is exactly what
        # wedged the curobo server's CUDA-graph capture right after the pick
        # task finished planning: the reconnect re-plan of the 'open' stage
        # left gpu_lock held forever, so nothing ever executed. The first
        # complete chain is the one best() picks anyway.
        if self.solutions:
            return
        self._feed_external()
        for i, child in enumerate(self.children):
            if not child.can_compute():
                continue
            child.run_compute()
            self._propagate_outputs(child, i)

    def _feed_external(self) -> None:
        if self._start_flag is not None and self._start_flag.read:
            pool = self.start_pull[self._fed_start:]
            if pool:
                self._fed_start = len(self.start_pull)
                for st in pool:
                    self.children[0].push_start(st)
        if self._end_flag is not None and self._end_flag.read:
            pool = self.end_pull[self._fed_end:]
            if pool:
                self._fed_end = len(self.end_pull)
                for st in pool:
                    self.children[-1].push_end(st)

    def _propagate_outputs(self, child: Stage, i: int) -> None:
        n = len(self.children)
        # forward edges: child's END output feeds the next child's start
        if i < n - 1 and self._edges[i] == FORWARD:
            for st in child.written_ends():
                self.children[i + 1].push_start(st)
        elif i == n - 1 and self._end_flag is not None and self._end_flag.write:
            for st in child.written_ends():
                self._push_written(self._ends, st)
        # backward edges: child's START output feeds the previous child's end
        if i > 0 and self._edges[i - 1] == BACKWARD:
            for st in child.written_starts():
                self.children[i - 1].push_end(st)
        elif i == 0 and self._start_flag is not None and self._start_flag.write:
            for st in child.written_starts():
                self._push_written(self._starts, st)

    # -- solution composition ------------------------------------------
    def _on_child_solution(self, child: Stage, sol: Solution) -> None:
        if sol is None:
            return
        self._index_solution(sol)
        n = len(self.children)
        i = self.children.index(child)
        # full chains only: predecessor span [0..i-1] and successor [i+1..n-1]
        incoming = self._traces(sol.start, BACKWARD, 0, i - 1)
        outgoing = self._traces(sol.end, FORWARD, i + 1, n - 1)
        for inc in incoming:
            for out in outgoing:
                self._lift_chain(inc + [sol] + out)

    def _traces(self, state, direction, lo: int, hi: int) -> list:
        """All solution paths covering exactly child indices ``lo..hi`` that
        thread through ``state`` in the given direction.

        BACKWARD walks solutions *ending* at ``state`` (following them from
        the container start), FORWARD walks solutions *starting* at it. Each
        returned path is ordered by ascending child index.
        """
        if lo > hi:
            return [[]]
        pool = (self._by_end.get(id(state), []) if direction == BACKWARD
                else self._by_start.get(id(state), []))
        paths = []
        for sol in pool:
            k = self.children.index(sol.stage)
            if not (lo <= k <= hi):
                continue
            if direction == BACKWARD:
                for rest in self._traces(sol.start, BACKWARD, lo, k - 1):
                    paths.append(rest + [sol])
            else:
                for rest in self._traces(sol.end, FORWARD, k + 1, hi):
                    paths.append([sol] + rest)
        return paths

    def _lift_chain(self, chain: list) -> None:
        if not chain:
            return
        start = chain[0].start
        end = chain[-1].end
        cost = sum(s.cost for s in chain)
        comment = " -> ".join(s.stage.name for s in chain)
        lifted = Solution(
            start=start, end=end,
            trajectory=_concat_trajectories(chain),
            cost=cost, comment=comment,
            children=list(chain),
            scene_ops=[op for s in chain for op in (s.scene_ops or [])],
        )
        self._lift(lifted)


class ParallelContainerBase(ContainerStage):
    """Common behaviour of Alternatives/Fallbacks/IndependentComponents:
    every child connects to the container's own interfaces, and the
    container requires all children to declare the same interface."""

    def _resolve_children(self, expected_start, expected_end) -> None:
        ref = None
        for child in self.children:
            child.resolve(expected_start, expected_end)
            req = child.required_flags()
            if ref is None:
                ref = req
            elif req != ref:
                raise InitStageError(
                    self.name,
                    "all children of a parallel container must declare the "
                    "same interface")
        self._start_flag, self._end_flag = ref if ref else (None, None)
        self._boundary_validate(expected_start, expected_end)

    def _feed_all(self) -> None:
        pool_s = self.start_pull[self._fed_start:]
        pool_e = self.end_pull[self._fed_end:]
        if pool_s:
            self._fed_start = len(self.start_pull)
        if pool_e:
            self._fed_end = len(self.end_pull)
        for child in self.children:
            s, e = child.required_flags()
            if pool_s and s.read:
                for st in pool_s:
                    child.push_start(st)
            if pool_e and e.read:
                for st in pool_e:
                    child.push_end(st)

    def _propagate_outputs(self, child: Stage, i: int) -> None:
        s, e = child.required_flags()
        if e.write:
            for st in child.written_ends():
                self._push_written(self._ends, st)
        if s.write:
            for st in child.written_starts():
                self._push_written(self._starts, st)

    def _lift_child_solution(self, sol: Solution) -> None:
        lifted = Solution(
            start=sol.start, end=sol.end,
            trajectory=sol.trajectory, cost=sol.cost, comment=sol.comment,
            children=[sol], scene_ops=list(sol.scene_ops or []),
            plan_request=sol.plan_request,
        )
        self._lift(lifted)


class Alternatives(ParallelContainerBase):
    """Same input(s) to all children; every child solution becomes a
    solution of the container.

    When every child is a whole-task motion stage, their requests are solved
    in ONE ``plan_batch`` call — an optimization invisible to the stage
    authors above this container.
    """

    stage_kind = "alternatives"

    def compute(self) -> None:
        self._feed_all()
        session = self._open_batch() if self._all_motion_children() else None
        try:
            for i, child in enumerate(self.children):
                if child.can_compute():
                    child.run_compute()
                    self._propagate_outputs(child, i)
        finally:
            if session is not None:
                session.flush()
                for child in self.children:
                    child._batch = None

    def _all_motion_children(self) -> bool:
        return bool(self.children) and all(
            isinstance(c, TrajectoryStage) for c in self.children)

    def _open_batch(self):
        session = _BatchSession(self.robot)
        for child in self.children:
            child._batch = session
        return session

    def _on_child_solution(self, child: Stage, sol: Solution) -> None:
        if sol is not None:
            self._lift_child_solution(sol)


class Fallbacks(ParallelContainerBase):
    """Children run in order; only the first child that produces a solution
    counts (MTC's ``Fallbacks`` — ``onNewSolution`` lifts the active child's
    solutions and later children are never started).

    Input semantics: every child receives the same container inputs (like
    ``Alternatives``); the container tries children in order and advances
    only when the active child is exhausted — fed everything available and
    produced nothing. This is what keeps ``can_compute`` finite: children
    are advanced past exactly once each, and the first solution stops the
    container.
    """

    stage_kind = "fallbacks"

    def __init__(self, name=None, params=None):
        super().__init__(name, params)
        self._current = 0  # index of the child being tried

    def can_compute(self) -> bool:
        if self._has_unfed_external():
            return True
        if self.solutions:
            self._current = None
            return False
        if self._current is None or self._current >= len(self.children):
            self._current = None
            return False
        return self.children[self._current].can_compute()

    def compute(self) -> None:
        if self.solutions:
            self._current = None  # first solution wins — stop scheduling
            return
        self._feed_all()  # all children share the same inputs (MTC fallbacks)
        while self._current is not None and self._current < len(self.children):
            child = self.children[self._current]
            if not child.can_compute():
                self._current += 1  # exhausted / no computable input -> next
                continue
            child.run_compute()
            self._propagate_outputs(child, self._current)
            if child.solutions:
                self._current = None  # done: first solution wins
            elif not child.can_compute():
                self._current += 1  # failed with all its input -> next child
            return
        self._current = None

    def _on_child_solution(self, child: Stage, sol: Solution) -> None:
        if sol is not None:
            self._lift_child_solution(sol)

    def reset(self) -> None:
        self._current = 0
        super().reset()


class IndependentComponents(ParallelContainerBase):
    """Children act on disjoint joint groups (bimanual work).

    Same input(s) reach every child and each child's solution lifts directly,
    like ``Alternatives``; the difference is intent — the children do NOT
    pass states to each other, so no adjacency validation applies between
    them. Lowest priority; the disjoint-set validation itself is deferred to
    the robot model at runtime.
    """

    stage_kind = "independent"

    def compute(self) -> None:
        self._feed_all()
        for i, child in enumerate(self.children):
            if child.can_compute():
                child.run_compute()
                self._propagate_outputs(child, i)

    def _on_child_solution(self, child: Stage, sol: Solution) -> None:
        if sol is not None:
            self._lift_child_solution(sol)
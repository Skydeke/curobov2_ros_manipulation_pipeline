"""Containers — compose stages into task graphs (MTC paper Sec. III-C / MTC's
``core/src/container.cpp`` semantics). A container is itself a ``Stage`` (nestable), and its *interface* is derived
from its boundary children:

- ``SerialContainer``  — children run in order; adjacent interface types are
  validated at init with MTC's exact ``connect()`` rules; a container
  solution exists only when *every* child produced one (costs accumulate). - ``Alternatives``     — same input(s) to all children; every child solution
  becomes a container solution."""

from __future__ import annotations

import time
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


from curobo_task_constructor.core.container_base import *

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

__all__ = [
    'ParallelContainerBase',
    'Alternatives',
    'Fallbacks',
    'IndependentComponents',
]

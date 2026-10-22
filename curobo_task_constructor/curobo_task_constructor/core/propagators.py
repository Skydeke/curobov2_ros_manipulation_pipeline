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
from curobo_task_constructor.core.stage_base import *

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
    """A propagator that may run forward (read start) or backward (read end). The containing serial container resolves the direction from its
    neighbours — what MTC's ``PropagatingEitherWay`` does in
    ``resolveInterface``: a stage whose output has to feed the *previous*
    sibling runs backward (e.g. the pick/place approach stages), otherwise it
    runs forward. Subclasses implement ``compute_forward`` and
    ``compute_backward``."""

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
            # Multi-attempt planning (MTC forwards num_planning_attempts to
            # MoveIt; cuRobo has no server-side retry, so the client loops):
            # every try is stored and published the moment it solves — like
            # failed tries fail live — so the panel walks attempt 1, 2, 3 as
            # they happen instead of only after the Nth try. Only the
            # cheapest try is promoted downstream; ranking still picks the
            # best seed while successful + failed rows sum to attempts.
            attempts = max(1, int(self.params.get("planning_attempts", 3)))
            tried = []
            # One allowance window for the whole attempt loop (see
            # RobotInterface.allowance_scope): the first OFF sends,
            # nested plan() calls compose silently, the exit restores
            # ON — 2 round trips per leg instead of 2 per attempt.
            with self.robot.allowance_scope([req]):
                for _ in range(attempts):
                    try:
                        result = self._timed_plan(req)
                    except Exception as exc:  # ServiceError etc.
                        self._fail(start, None, f"plan call failed: {exc}")
                        continue
                    if not result.success:
                        self._fail(start, None, result.message or "plan failed")
                        continue
                    self._note_plan_attempt()
                    end = self.make_end_state(start, result, raw=result)
                    if end is None:
                        continue  # already recorded (e.g. straightness gate)
                    sol = Solution(
                        start, end, trajectory=result.trajectory,
                        cost=self._cost_of(result),
                        comment=self._comment(req, result),
                        response=result.raw, plan_request=req)
                    self._consider(sol)
                    tried.append(sol)
            if not tried:
                continue
            best = min(tried, key=lambda s: s.cost)
            self._promote_forward(start, best.end, best)

    def commit_result(self, start: InterfaceState, req: PlanRequest,
                      result: PlanResult) -> None:
        """Commit the result of a batched solve for (start, req)."""
        self._note_plan_attempt()
        if not result.success:
            self._fail(start, None, result.message or "plan failed")
            return
        end = self.make_end_state(start, result, raw=result)
        if end is not None:
            self.send_forward(start, end, trajectory=result.trajectory,
                              cost=self._cost_of(result),
                              comment=self._comment(req, result),
                              response=result.raw, plan_request=req)

    def _comment(self, req: PlanRequest, result: PlanResult) -> str:
        """Solution comment. Hook for stages that resolve several candidates
        and want the winner recorded — without it a multi-candidate goalset is
        indistinguishable from a single-pose one in the task statistics."""
        return ""

    def compute(self) -> None:
        if self.prepare():
            self.solve_pending()

    def _cost_of(self, result: PlanResult) -> float:
        """Ranking cost of a solve.

        Delegates to ``stages._util.cost_of`` so every motion stage ranks on
        the same term (MTC ``cost::PathLength`` on the returned waypoints by
        default) and a stage can select a different term through its ``cost``
        param. Imported lazily: ``stages`` depends on ``core``, so a top-level
        import here would be circular.
        """
        from curobo_task_constructor.stages._util import cost_of

        return cost_of(result, self.params)

__all__ = [
    'GeneratorStage',
    'PropagatingStage',
    'BackwardPropagatingStage',
    'PropagatingEitherWay',
    'ConnectingStage',
    'TrajectoryStage',
]

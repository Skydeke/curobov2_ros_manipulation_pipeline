"""Connect — solve a trajectory joining a known start and end interface state
(MTC ``Connecting``; the "move to pick" / "move to place" stages).

A connector does not write either interface — its two neighbours already
published the states it joins — so its solutions are emitted directly (only
the enclosing container sees them, via the on-solution hook) and are free of
a push direction. Every (start, end) pair is tried; the whole path is one
multi-goalset ``PlanRequest`` ending at the end state's joint configuration.
"""

from __future__ import annotations

from curobo_task_constructor.core.registry import register_stage
from curobo_task_constructor.core.robot import GoalsetSpec
from curobo_task_constructor.core.stage import ConnectingStage, Solution
from curobo_task_constructor.core.state import InterfaceState
from curobo_task_constructor.stages._util import full_request


@register_stage("connect")
class Connect(ConnectingStage):
    def compute(self) -> None:
        starts, self.start_pull = self.start_pull, []
        ends, self.end_pull = self.end_pull, []
        for s in starts:
            for e in ends:
                self._connect_pair(s, e)

    def _connect_pair(self, s: InterfaceState, e: InterfaceState) -> None:
        allowed = set()
        if s.scene is not None:
            allowed.update(s.scene.all_allowed_links())
        if e.scene is not None:
            allowed.update(e.scene.all_allowed_links())
        goalset = GoalsetSpec(
            target_joint_positions=list(getattr(e.joint_state, "position", []) or []),
            allowed_collisions=sorted(allowed))
        req = full_request(self.robot, s.joint_state, [goalset], self.params)
        try:
            result = self.robot.plan(req)
        except Exception as exc:  # ServiceError etc.
            self._fail(s, e, f"plan call failed: {exc}")
            return
        if not result.success:
            self._fail(s, e, result.message or "connect plan failed")
            return
        sol = Solution(
            start=s, end=e, trajectory=result.trajectory,
            cost=self._cost_of(result),
            comment=f"connect (n={len(result.trajectory) if result.trajectory else 0} wp)",
            response=result.raw, plan_request=req)
        self._emit(sol)

    def _cost_of(self, result) -> float:
        if result.cost != float("inf"):
            return float(result.cost)
        return float(len(result.trajectory)) if result.trajectory else 0.0
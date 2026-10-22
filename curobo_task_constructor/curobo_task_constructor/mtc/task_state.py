
from __future__ import annotations

from typing import Any, Optional

#: Introspection namespace default (MTC task ns; absolute default kept for compat).
#: MTC uses RELATIVE topics ``description/statistics/solution`` in the task's
#: namespace plus service ``get_solution_<task_id>``. This package keeps the
#: absolute default ``/curobo_task_constructor/...`` but honours ``Task(ns)``:
#: ``Task(ns="my_ns")`` publishes ``my_ns/description`` etc. and serves
#: ``my_ns/get_solution_<task_id>`` — see ``Task._ns_base``.

from curobo_task_constructor.mtc.constants import *
from curobo_task_constructor.mtc.ros_runtime import *
from curobo_task_constructor.mtc.planners import *
from curobo_task_constructor.mtc.containers import *
from curobo_task_constructor.mtc.introspection_support import *
from curobo_task_constructor.mtc.introspection import *

class _TaskStateMixin:
    def publish(self, solution: Any) -> None:
        """Publish ONE solution (MTC ``Task.publish`` Python binding). MTC binds ``Task.publish`` to *exactly*
        ``self.introspection().publishSolution(*solution)``
        (``core/python/bindings/src/core.cpp:481``), documented as
        "Publish the given solution to the ROS topic ``solution``" — nothing
        else. The RViz side subscribes to that topic, fetches the payload and
        renders the trajectory, so publishing the message IS how the solution
        is shown."""
        self.last_published = solution
        self.introspection().publishSolution(solution)

    def publishAllSolutions(self, wait: bool = True) -> None:
        """Publish every ranked solution (MTC ``Task.publishAllSolutions``).

        MTC pauses for a keypress between solutions when ``wait``; a ROS
        node's thread must never block on stdin, so they publish
        back-to-back here (see ``Introspection.publishAllSolutions``).
        """
        self.introspection().publishAllSolutions(wait)

    def rank(self) -> list:
        """Ranked solutions, cheapest first (MTC solution ordering)."""
        if self._executor is None:
            return []
        return self._executor.rank()

    def reset(self) -> None:
        # MTC Task::reset: signal introspection FIRST, then reset the stages.
        if self._introspection is not None:
            self._introspection.reset()
        if self._executor is not None:
            self._executor.reset()
        self._solutions = []
    # -- results -------------------------------------------------------
    @property
    def solutions(self) -> list:
        return list(self._solutions)

    @property
    def failures(self) -> list:
        """Failed attempts across all stages (MTC Task.failures)."""
        if self._executor is None:
            return []
        out = []
        for stage in self._executor.root.subtree_stages():
            out.extend(stage.failures)
        return out

    def best(self) -> Any:
        return self._solutions[0] if self._solutions else None

    def numSolutions(self) -> int:
        """Number of ranked solutions (MTC ``Task::numSolutions``)."""
        return len(self._solutions)

    def printState(self) -> None:
        """Print the task state to stdout (MTC ``Task::printState``).

        Reports the solution count and, per stage, the successful/failed
        attempt counts — the same shape the rviz panel renders per row.
        """
        print(
            f"task '{self.task_id or self.name}': "
            f"{len(self._solutions)} solution(s)"
        )
        if self._executor is not None:
            for stage in self._executor.root.subtree_stages():
                print(
                    f"  '{stage.name}': {len(stage.solutions)} ok / "
                    f"{len(stage.failures)} failed"
                )

    def explainFailure(self) -> bool:
        """Print the last failure per stage, True when any exist.

        Mirrors ``Stage::explainFailure``: the same per-stage messages the
        action server logs when a task produces no complete solution.
        """
        if self._executor is None:
            print("not planned yet")
            return False
        found = False
        for stage in self._executor.root.subtree_stages():
            if stage.failures:
                print(f"  stage '{stage.name}': " f"{stage.failures[-1].message}")
                found = True
        return found

    @property
    def executor(self):
        return self._executor

    def describe(self) -> dict:
        if self._executor is None:
            return {
                "task_id": self.task_id or self.name,
                "valid": False,
                "comment": "not planned yet",
                "stage_count": len(self._stages),
            }
        return self._executor.describe()

__all__ = [
    '_TaskStateMixin',
]

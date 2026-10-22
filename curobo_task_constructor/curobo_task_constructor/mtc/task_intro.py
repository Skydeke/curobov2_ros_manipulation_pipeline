
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

class _TaskIntrospectionMixin:
    def _register_solution(self, obj: Any) -> int:
        """Global solution id without touching introspection endpoints.

        The ids live in the EXECUTOR's registry (MTC
        ``Introspection::id_solution_bimap_``), which needs no node,
        publishers or service — so serializing a solution (the execute
        path) never creates endpoints on a dark task. ``Introspection``
        delegates here too, so enabled and dark tasks number identically.
        """
        ex = self._executor
        return ex.register_solution(obj) if ex is not None else 0

    def _solution_from_id(self, solution_id: int) -> Any:
        ex = self._executor
        return ex.solutionFromId(solution_id) if ex is not None else None

    def _on_get_solution(self, request, response):
        """Serve GetSolution (MTC ``Introspection::getSolution``)."""
        obj = self._solution_from_id(getattr(request, "solution_id", 0))
        if obj is None:
            return response
        try:
            from curobo_task_constructor import msg_convert
            response.solution = msg_convert.solution_to_msg(
                self._solution_dict(obj))
        except Exception:
            pass
        return response

    def _solution_dict(self, obj: Any) -> Optional[dict]:
        """solution_to_dict-shaped dict for a solution or failure."""
        if self._executor is None:
            return None
        from curobo_task_constructor.core.stage import Solution, StageFailure
        if isinstance(obj, Solution):
            return self._executor.solution_to_dict(obj)
        if isinstance(obj, StageFailure):
            stg = self._failure_stage(obj)
            info = {
                # The failure's GLOBAL id (not its per-stage failure_id),
                # so the panel can join it to the statistics row that
                # reports it.
                "id": self._register_solution(obj),
                "cost": float("inf"),
                "comment": str(getattr(obj, "message", "") or ""),
                "stage_id": getattr(stg, "stage_id", 0)
                if stg is not None else 0,
                "planner_id": "",
            }
            return {"task_id": self._executor.task_id, "start_scene": [],
                    "sub_solution": [{"info": info, "sub_solution_id": []}],
                    "sub_trajectories": []}
        return None

    def _failure_stage(self, failure: Any) -> Any:
        # The recording stage rides on the failure (MTC: failures live in
        # the stage that made them). Identity fallback only: dataclass
        # __eq__ matches VALUE-equal failures from sibling stages sharing
        # one input (same message, same failure_id sequence), which used to
        # attribute every such failure to the first stage in tree order.
        direct = getattr(failure, "stage", None)
        if direct is not None:
            return direct
        if self._executor is None:
            return None
        for stg in self._executor.root.subtree_stages():
            if any(f is failure for f in stg.failures):
                return stg
        return None

    # ``publishTaskDescription`` / ``publishTaskState`` / ``publishSolution``
    # live on Introspection (MTC: ``IntrospectionPrivate`` owns the publishers
    # and every publish call). Task only keeps the fill helpers they need.

    def _markers_for_object(self, obj: Any) -> dict:
        """(stage_id, global_solution_id) -> marker list for one sol/failure."""
        from curobo_task_constructor.core.stage import Solution
        out: dict = {}
        if isinstance(obj, Solution):
            leaves = self._executor.flatten_leaves(obj) \
                if self._executor is not None else [obj]
            for leaf in leaves:
                stage = getattr(leaf, "stage", None)
                key = (int(getattr(stage, "stage_id", 0) or 0),
                       self._register_solution(leaf))
                out[key] = self._attempt_markers(leaf, success=True)
        else:
            stg = self._failure_stage(obj)
            key = (int(getattr(stg, "stage_id", 0) or 0),
                   self._register_solution(obj))
            out[key] = self._failure_markers(obj)
        return out

    def _attempt_markers(self, sol: Any, success: bool) -> list:
        """Debug markers for one attempt (MTC stage start/goal frames)."""
        try:
            from curobo_task_constructor import msg_convert
            from curobo_task_constructor.core.geom import Pose3
        except Exception:
            return []
        out = []
        try:
            req = getattr(sol, "plan_request", None)
            goalsets = list(getattr(req, "goalsets", None) or [])
            positions = [msg_convert._pose_to_dict(p)
                         for gs in goalsets
                         for p in (getattr(gs, "poses", None) or [])]
            positions = [p for p in positions if p]
            if positions:
                out.append(msg_convert.sphere_list_marker(
                    "candidates", 0, positions, 0.02,
                    (0.3, 0.6, 1.0, 0.8)))
            traj = list(getattr(sol, "trajectory", None) or [])
            if traj and self.robot is not None:
                fk = getattr(self.robot, "fk_batch", None)
                poses = fk(traj) if callable(fk) else []
                if poses:
                    first = msg_convert._pose_to_dict(poses[0])
                    last = msg_convert._pose_to_dict(poses[-1])
                    if first:
                        out.append(msg_convert.sphere_marker(
                            "start", 1, first, 0.025, (0.0, 1.0, 0.0, 1.0)))
                    if last:
                        color = (0.0, 1.0, 0.0, 1.0) if success \
                            else (1.0, 0.0, 0.0, 1.0)
                        out.append(msg_convert.sphere_marker(
                            "end", 2, last, 0.025, color))
        except Exception:
            pass
        return out

    def _failure_markers(self, failure: Any) -> list:
        """Red marker at the state a failed attempt started from."""
        try:
            from curobo_task_constructor import msg_convert
        except Exception:
            return []
        try:
            state = getattr(failure, "from_state", None)
            js = getattr(state, "joint_state", None) if state else None
            if js is None or self.robot is None:
                return []
            pose = self.robot.fk(js)
            pose_dict = msg_convert._pose_to_dict(pose)
            if pose_dict is None:
                return []
            return [msg_convert.sphere_marker(
                "failed_at", 0, pose_dict, 0.025, (1.0, 0.0, 0.0, 1.0))]
        except Exception:
            return []

    # -- spec conversion ----------------------------------------------

__all__ = [
    '_TaskIntrospectionMixin',
]

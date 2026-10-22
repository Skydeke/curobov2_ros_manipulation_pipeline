
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

class Introspection:
    """MTC ``Task::introspection()`` (``core::Introspection``) surface. OWNS the publishers and the ``get_solution`` service, and implements
    MTC's contract exactly:

    ==========================  ============================================
    ``publishTaskDescription``  publish TaskDescription
    ``publishTaskState``        publish TaskStatistics
    ``publishSolution(s)``      publish ONE Solution message (nothing else)
    ``publishAllSolutions(w)``  publish every top-level Solution
    ``registerSolution(s)``     assign the global solution id
    ``solutionId(s)``           retrieve or assign it
    ``solutionFromId(id)``      the object named by that id
    ``reset()``                 publish the empty description, clear ids
    ``getSolution(req, res)``   the ``get_solution`` service callback
    ==========================  ============================================

    Rendezvous and QoS are MTC's verbatim: ``TaskDescription`` QoS(2),
    ``TaskStatistics`` QoS(1) and ``Solution`` QoS(1), all transient_local,
    and the service ``get_solution_<task_id>``. Node handling: MTC's ``IntrospectionPrivate`` creates its OWN node
    (``introspection_<task_id>`` in the task's namespace) on a shared
    ref-counted executor it spins itself; that is what isolates every task's
    publishers and service."""

    def __init__(self, task: "Task"):
        self._task = task

    @property
    def task(self) -> "Task":
        return self._task

    @property
    def task_id(self) -> str:
        return self._task.task_id or self._task.name

    def get_solution_service_name(self) -> str:
        """``<ns>/get_solution_<task_id>`` (MTC ``GET_SOLUTION_SERVICE"_"+id``)."""
        return f"{self._task._ns_base()}/get_solution_{self.task_id}"

    # -- lifecycle (MTC Task::enableIntrospection) ----------------------
    def enableIntrospection(self, enabled: bool = True) -> None:
        """MTC ``Task::enableIntrospection``.

        Disabling releases the introspection object — MTC resets its
        ``unique_ptr`` (task.cpp:159-174) — which drops its node from the
        shared executor and stops answering the panel.
        """
        task = self._task
        if enabled:
            task._introspection_enabled = True
        else:
            task._introspection_enabled = False
            self._release()
            task._introspection = None

    def node(self):
        """This introspection's own node (MTC ``IntrospectionPrivate::node_``)."""
        return self._task._introspection_node

    def _ensure_publishers(self) -> bool:
        """Create the publishers + this task's GetSolution service. What ``IntrospectionPrivate``'s CONSTRUCTOR does in MTC
        (introspection.cpp:112-131): own node, publishers, service,
        ``indicateReset()``, ``resetMaps()``. MTC gives each Introspection its OWN node."""
        task = self._task
        if not task._introspection_enabled:
            return False
        if task._srv_get_solution is not None:
            return True
        try:
            from curobo_task_constructor_interfaces.msg import (
                Solution,
                TaskDescription,
                TaskStatistics,
            )
            from curobo_task_constructor_interfaces.srv import GetSolution
        except Exception:
            return False

        if task._introspection_node is None:
            task._introspection_node = _shared_introspection_node()
            if task._introspection_node is None:
                return False

        node = task._introspection_node
        try:
            # Shared handles: every task in this namespace publishes the same
            # three rendezvous topics (default /curobo_task_constructor/*, or
            # <Task ns>/* when constructed with Task(ns=...)).
            task._pub_desc = _shared_publisher(
                node, TaskDescription, task.description_topic(), 2)
            task._pub_stat = _shared_publisher(
                node, TaskStatistics, task.statistics_topic(), 1)
            task._pub_sol = _shared_publisher(
                node, Solution, task.solution_topic(), 1)
            # Per-task service (MTC's get_solution_<task_id>) on the shared
            # node, created once per task object. Replacement is
            # deterministic (see _replace_service): this pipeline reuses
            # labels across cycles ("setup_scene" every cycle), so a new
            # Task must take over the name instead of depending on
            # garbage-collection timing.
            if task._srv_get_solution is None:
                task._srv_get_solution = _replace_service(
                    node, GetSolution, self.get_solution_service_name(),
                    self.getSolution)
        except Exception:
            return False
        # Send the reset indication as early as possible, like MTC's
        # constructor does, so a panel that joins late sees a defined state.
        self.reset()
        return True

    def _release(self) -> None:
        """Drop this task's publishers/service (MTC ``~IntrospectionPrivate``).

        The NODE and the three publishers are shared and outlive any one
        task, so only the handle this task owns is torn down: its
        ``get_solution_<task_id>`` service stops answering (removed from the
        shared registry only if it is still ours — a successor task may
        already have taken the name). The publisher refs are plain aliases
        of the shared handles — deliberately NOT destroyed here, or every
        task handover would kill the topics for its successor.
        """
        task = self._task
        handle = task._srv_get_solution
        task._pub_desc = None
        task._pub_stat = None
        task._pub_sol = None
        task._srv_get_solution = None
        if handle is None:
            return
        node = task._introspection_node
        if node is None:
            return
        with _service_lock():
            try:
                name = self.get_solution_service_name()
            except Exception:
                return
            if _SERVICE_HANDLES.get(name) is handle:
                _SERVICE_HANDLES.pop(name, None)
                try:
                    node.destroy_service(handle)
                except Exception:
                    pass

    # -- global solution ids (MTC id_solution_bimap_) -------------------
    def solutionId(self, obj: Any) -> int:
        """Retrieve or assign the global id (MTC ``Introspection::solutionId``)."""
        ex = self._task._executor
        return ex.register_solution(obj) if ex is not None else 0

    def registerSolution(self, obj: Any) -> None:
        """Assign the global id (MTC ``Introspection::registerSolution``)."""
        self.solutionId(obj)

    def solutionFromId(self, solution_id: int) -> Any:
        """The object named by a global id, or None (MTC)."""
        ex = self._task._executor
        return ex.solutionFromId(solution_id) if ex is not None else None

    def stageId(self, stage: Any) -> int:
        """Id of a stage (MTC ``Introspection::stageId``)."""
        return int(getattr(stage, "stage_id", 0) or 0)

    def getSolution(self, request: Any, response: Any) -> bool:
        """The GetSolution service callback (MTC ``Introspection::getSolution``)."""
        return self._task._on_get_solution(request, response)

    # -- reset (MTC Introspection::reset) ------------------------------
    def reset(self) -> None:
        """Signal a task reset (MTC ``Introspection::reset``).

        ``indicateReset`` publishes the EMPTY task description — the "this
        task is gone" signal the panel uses — then ``resetMaps`` drops the
        global solution ids, so an id from the previous plan is never served
        again. MTC leaves the STAGE reset to ``Task::reset``; the same split
        holds, and ``reset_solution_ids`` is what makes it safe to call this
        from ``setup()``.
        """
        self._indicate_reset()
        ex = self._task._executor
        if ex is not None:
            ex.reset_solution_ids()

    def _indicate_reset(self) -> None:
        if self._task._pub_desc is None:
            return
        try:
            from curobo_task_constructor_interfaces.msg import TaskDescription
        except Exception:
            return
        msg = TaskDescription()
        msg.task_id = str(self.task_id)
        try:
            self._task._pub_desc.publish(msg)
        except Exception:
            pass

    # -- publish* (MTC Introspection::publish*) ------------------------
    def publishTaskDescription(self) -> None:
        """Publish TaskDescription (MTC fillTaskDescription + publish)."""
        task = self._task
        if not self._ensure_publishers() or task._pub_desc is None:
            return
        try:
            from curobo_task_constructor import msg_convert
        except Exception:
            return
        ex = task._executor
        if ex is None:
            return
        from curobo_task_constructor_interfaces.msg import TaskDescription
        msg = TaskDescription()
        msg.task_id = str(self.task_id)
        stages = list(ex.root.subtree_stages())
        # MTC fillTaskDescription (introspection.cpp:290-292) resolves
        # desc.parent_id through a parent lookup, with the task wrapper
        # itself at id 0 (introspection.cpp:148: "root is task having
        # ID = 0") and the root container at id 1. Getting this wrong is
        # NOT cosmetic: a panel that receives parent_id == id for every
        # stage treats the LAST one as the root and drops the whole tree,
        # which is exactly how a fully-populated task renders as a single
        # row. Self-parented roots are never emitted (MTC has none).
        stage_ids = {}
        for stage in stages:
            try:
                stage_ids[id(stage)] = self.stageId(stage)
            except Exception:
                continue
        children_of: dict = {}
        for stage in stages:
            for child in (getattr(stage, "children", None) or []):
                children_of.setdefault(id(stage), set()).add(id(child))
        child_ids: set = set()
        for kids in children_of.values():
            child_ids.update(kids)
        for stage in stages:
            sid = stage_ids.get(id(stage))
            if sid is None:
                continue
            if id(stage) in child_ids:
                owner = next(
                    (pid for pid, kids in children_of.items()
                     if id(stage) in kids), None)
                parent_id = stage_ids.get(owner, 0) if owner else 0
            else:
                parent_id = 0  # tree root: parent is the (unpublished) task
            try:
                msg.stages.append(msg_convert.stage_description_to_msg(
                    sid, parent_id,
                    getattr(stage, "name", ""),
                    msg_convert.interface_flags(stage),
                    self._stage_properties(stage)))
            except Exception:
                continue
        try:
            task._pub_desc.publish(msg)
        except Exception:
            pass

    def _stage_properties(self, stage: Any) -> list:
        """One stage's Property messages (MTC fillTaskDescription's loop)."""
        try:
            from curobo_task_constructor import msg_convert
        except Exception:
            return []
        try:
            stage_type = stage.stage_type()
        except Exception:
            stage_type = ""
        allowed = _DISPLAY_PROPERTIES.get(
            str(stage_type or ""), ("group", "timeout"))
        out = []
        params = getattr(stage, "params", None) or {}
        for key in sorted(params):
            if key not in allowed:
                continue
            try:
                out.append(msg_convert.property_to_msg(key, params[key]))
            except Exception:
                continue
        return out

    def publishTaskState(self) -> None:
        """Publish TaskStatistics (MTC ``publishTaskState``)."""
        if not self._ensure_publishers() or self._task._pub_stat is None:
            return
        ex = self._task._executor
        if ex is None:
            return
        from curobo_task_constructor_interfaces.msg import (
            StageStatistics,
            TaskStatistics,
        )
        msg = TaskStatistics()
        msg.task_id = str(self.task_id)
        for stage in ex.root.subtree_stages():
            stat = StageStatistics()
            stat.id = self.stageId(stage)
            # MTC fillStageStatistics: GLOBAL ids for both collections.
            stat.solved = [self.solutionId(s) for s in stage.solutions]
            stat.failed = [self.solutionId(f) for f in stage.failures]
            stat.num_failed = len(stage.failures)
            stat.total_compute_time = float(getattr(stage, "compute_time", 0.0))
            msg.stages.append(stat)
        try:
            self._task._pub_stat.publish(msg)
        except Exception:
            pass

    def publishSolution(self, solution: Any) -> None:
        """Publish ONE solution/failure (MTC ``Introspection::publishSolution``). This is the entire responsibility: the Solution message goes to the
        ``solution`` topic. RENDERING it is the consumer's job — in MTC the
        RViz Display subscribes to this topic, and our panel does the
        equivalent."""
        if not self._ensure_publishers() or self._task._pub_sol is None:
            return
        d = self._task._solution_dict(solution)
        if d is None:
            return
        try:
            from curobo_task_constructor import msg_convert
        except Exception:
            return
        try:
            msg = msg_convert.solution_to_msg(d)
            markers = self._task._markers_for_object(solution)
            for sub in list(msg.sub_solution) + list(msg.sub_trajectory):
                info = getattr(sub, "info", None)
                if info is None:
                    continue
                key = (int(getattr(info, "stage_id", 0)),
                       int(getattr(info, "id", 0)))
                for marker in markers.get(key, []):
                    info.markers.append(marker)
            self._task._pub_sol.publish(msg)
        except Exception:
            pass

    def publishAllSolutions(self, wait: bool = True) -> None:
        """Publish every top-level solution (MTC ``publishAllSolutions``).

        MTC blocks on ``getchar()`` between solutions when ``wait`` — a
        console convenience that would hang a ROS node's thread, so the
        flag is accepted and documented as a no-op here (API parity).
        """
        for solution in self._task.rank():
            if not self._ensure_publishers():
                return
            self.publishSolution(solution)

__all__ = [
    'Introspection',
]

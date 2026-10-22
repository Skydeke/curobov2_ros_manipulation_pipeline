
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
from curobo_task_constructor.mtc.task_model import *
from curobo_task_constructor.mtc.task_intro import *
from curobo_task_constructor.mtc.task_remote import *
from curobo_task_constructor.mtc.task_state import *

class Task(_TaskModelMixin, _TaskIntrospectionMixin, _TaskRemoteMixin, _TaskStateMixin):
    """MTC ``core.Task`` equivalent: ordered stage list, local plan/publish.

    Mirrors ``moveit.task_constructor.core.Task``: ``init()``,
    ``plan(max_solutions=0)``, ``publish(solution)``,
    ``execute(solution)``. ``setProperty`` stores the same task-level
    ``group`` / ``eef`` / ``ik_frame`` bag the old pipeline sets.
    """

    def __init__(
        self, robot: Any = None, name: str = "task", task_id: Optional[str] = None,
        ns: str = "", introspection: bool = True,
    ):
        # MTC call shape Task(ns="", introspection=True): a lone string first
        # arg is the namespace, not a robot (Task("my_ns")).
        if isinstance(robot, str) and ns == "" and name == "task":
            ns, robot = robot, None
        self._ns = str(ns or "")
        self._name = name
        self.robot = robot
        self.task_id = task_id or name
        self._stages: list = []
        self._executor = None
        self._solutions: list = []
        self.last_published: Any = None
        # MTC's Task ctor (task.cpp:96-105): "enable introspection by
        # default, but only if ros::init() was called".
        self._introspection_enabled = bool(introspection) and _ros_ok()
        #: Task-level properties (MTC ``Task::setProperty`` / ``properties``).
        #: The old pipeline sets ``group`` / ``eef`` / ``ik_frame`` here and
        #: stages inherit them; the cuRobo builders carry the same values on
        #: each stage instead, so this bag is stored for API parity and
        #: inspection (``describe()`` reports it) rather than consumed.
        from curobo_task_constructor.mtc.stages import _PropertyMap
        self._properties: dict = _PropertyMap()
        #: Set by preempt(); the next plan() returns failure immediately.
        self._preempted = False
        #: The introspection OWN node (MTC ``IntrospectionPrivate::node_``),
        #: created on first use and destroyed by ``Introspection._release``.
        self._introspection_node: Any = None
        #: Publisher/service handles on that node (MTC
        #: ``IntrospectionPrivate`` members).
        self._pub_desc = None
        self._pub_stat = None
        self._pub_sol = None
        self._srv_get_solution = None
        #: MTC ``Task::introspection_`` object, created on first use.
        self._introspection: Optional[Introspection] = None

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def ns(self) -> str:
        """Task namespace (MTC ``Task(ns)`` ctor arg)."""
        return self._ns

    def _ns_base(self) -> str:
        """Introspection rendezvous base: Task ns or the default."""
        ns = (self._ns or "").strip()
        if not ns:
            return DEFAULT_TASK_NS
        return "/" + ns.strip("/") if "://" not in ns else ns.rstrip("/")

    def description_topic(self) -> str:
        return f"{self._ns_base()}/task_description"

    def statistics_topic(self) -> str:
        return f"{self._ns_base()}/task_statistics"

    def solution_topic(self) -> str:
        return f"{self._ns_base()}/solution"

    def execute_action_name(self) -> str:
        return f"{self._ns_base()}/execute_task_solution"

    # -- construction -------------------------------------------------
    def add(self, *args: Any) -> None:
        """Append stage(s) to the task's top-level container (MTC add)."""
        for stage in args:
            self._stages.append(stage)

    def insert(self, stage: Any, before: int = -1) -> Any:
        """Insert stage before given index (MTC ``Task.insert``)."""
        self._stages.insert(before if before >= 0 else len(self._stages), stage)
        return stage

    def remove(self, stage: Any) -> None:
        """Remove a child stage by index or instance (MTC Task.remove)."""
        if isinstance(stage, int):
            del self._stages[stage]
        else:
            self._stages.remove(stage)

    def clear(self) -> None:
        """Remove all stages from the task (MTC Task.clear)."""
        self._stages = []
        self.reset()

    def __len__(self) -> int:
        return len(self._stages)

    def __getitem__(self, key):
        """Child stage by index or by name (MTC findChild)."""
        if isinstance(key, str):
            for child in self._stages:
                if getattr(child, "name", None) == key:
                    return child
            raise IndexError(f"no stage named {key!r}")
        return self._stages[key]

    def __iter__(self):
        return iter(self._stages)

    def attach_robot(self, robot: Any) -> None:
        self.robot = robot

    def setProperty(self, name: str, value: Any) -> None:
        """Set a task-level property (MTC ``Task::setProperty``).

        The old pipeline sets ``group`` / ``eef`` / ``ik_frame`` here for
        stages to inherit. Stored verbatim; stages built by the cuRobo
        builders already carry their own copies, so this is configuration
        bookkeeping with the same call shape as the old API.
        """
        self._properties[str(name)] = value

    @property
    def properties(self) -> dict:
        """Task-level property bag (MTC ``Task::properties``, read-only)."""
        return self._properties

    def enableIntrospection(self, enabled: bool = True) -> None:
        """MTC ``Task::enableIntrospection``. Enabling creates the introspection object; disabling releases it —
        MTC resets its ``unique_ptr`` (task.cpp:159-174) — so a disabled
        task's ``get_solution_<task_id>`` service stops answering and its
        node leaves the shared executor. The flag is set UNCONDITIONALLY (MTC assigns
        ``introspection_enabled_`` first): disabling a fresh task with no
        object yet still sticks, so a later ``plan()`` cannot silently
        re-enable it."""
        if enabled:
            if self._introspection is None:
                self._introspection_enabled = True
                self._introspection = Introspection(self)
        else:
            self._introspection_enabled = False
            if self._introspection is not None:
                introspection = self._introspection
                self._introspection = None
                introspection._release()

    def getRobotModel(self) -> Any:
        """Get the robot model (MTC ``Task.getRobotModel``). MTC returns a ``moveit::core::RobotModel``, because MTC runs the
        kinematics IN-PROCESS against it — IK (``compute_ik.cpp:252``),
        Cartesian interpolation (``cartesian_path.cpp:105``), joint models
        and collision checks are all client-side. cuRobo runs none of that here: the URDF, the collision-sphere model
        and the kinematics live in ``curobo_server``, so the client-side
        object playing that role is the ``RobotInterface``."""
        return self.robot

    def setRobotModel(self, robot_model: Any) -> None:
        """Set the robot model (MTC ``Task.setRobotModel``).

        MTC accepts a ``RobotModelConstPtr``; here it is the
        ``RobotInterface`` (anything ``init()`` can drive), so the
        curobo-shaped call is ``attach_robot`` — this is MTC's spelling
        for it.
        """
        self.attach_robot(robot_model)

    def to_spec(self):
        import curobo_task_constructor.stages  # noqa: F401 (populate registry)
        from curobo_task_constructor.graph.spec import StageSpec

        children = []
        for st in self._stages:
            to_spec = getattr(st, "to_spec", None)
            if to_spec is None:
                raise TypeError(f"stage {st!r} is not an MTC wrapper stage")
            children.append(to_spec())
        return StageSpec(
            stage_type="serial",
            name=self.name,
            container_type="serial",
            children=children,
            params_yaml="",
        )

    # -- lifecycle (MTC Task::init/plan/execute/publish) ---------------
    def init(self) -> None:
        """Build and initialize the task (MTC ``Task::init``). MTC does, in this order (task.cpp:201-229):
          #. require a robot model (else ``throw``);
          #."""
        if self.robot is None:
            raise RuntimeError("Task has no robot: pass Task(robot) or attach_robot()")
        from curobo_task_constructor.executor import TaskExecutor

        spec = self.to_spec()
        ex = TaskExecutor(spec, self.robot, task_id=self.task_id or self.name)
        try:
            ex.base_scene = ex.build_base_scene()
        except Exception:
            from curobo_task_constructor.core.state import SceneDiff

            ex.base_scene = SceneDiff()
        self._executor = ex
        if not ex.init():
            self._solutions = []

        # MTC Task::init steps 3 + 4, gated on the flag like MTC gates on
        # its introspection_ member: a dark task gets stages with no hooks
        # and no description publish. init() must NOT reach for the enabling
        # introspection() getter here, or every plan would force-enable a
        # task the caller explicitly darkened.
        if not self._introspection_enabled:
            return
        introspect = self.introspection()
        from curobo_task_constructor.core.stage import chain_hook
        for stg in ex.root.subtree_stages():
            # MTC publishes TaskStatistics per compute pass and Solution
            # messages ONLY via explicit publish() (every demo ends with
            # task.publish(task.solutions[0]); nothing auto-publishes during
            # planning — planner-internal attempts are invisible there too).
            # Same here: per-try statistics stream via on_progress below,
            # while solutions go out solely through explicit publish calls.
            # The on_solution/on_failure/on_considered hooks stay available
            # for consumers and tests; Task itself subscribes to none of
            # them, so planning never emits Solution messages on its own.
            stg.on_progress = chain_hook(
                getattr(stg, "on_progress", None),
                lambda _stage, _self=self: _self.introspection().publishTaskState())
        introspect.publishTaskDescription()

    class _Preempted(Exception):
        pass

    def preempt(self) -> None:
        """Interrupt current planning (MTC Task.preempt).

        Takes effect between compute passes; an in-flight planner call
        runs to completion first.
        """
        self._preempted = True

    def plan(self, max_solutions: int = 0, progress_callback=None) -> MoveItErrorCode:
        """Reset, init, and plan (MTC ``Task.plan``). ``max_solutions`` (0 = unlimited) bounds the number of solutions
        collected. Returns MoveItErrorCode (SUCCESS/FAILURE-PLANNING_FAILED)
        exactly like MTC; truthy on success so ``assert task.plan()`` keeps
        working."""
        # NOTE (cuRobo backend limit): MTC plan(max_solutions) collects up to N
        # solutions. Our executor stops at the first complete root solution
        # (re-solving the trailing chain per reconnect once wedged the
        # server's CUDA-graph capture), so max_solutions only bounds the
        # compute loop; use fallbacks/alternatives for variant coverage.
        self.reset()
        if self._preempted:
            self._preempted = False
            return MoveItErrorCode(MoveItErrorCode.FAILURE)
        # A fresh plan burst must not trust a planner belief formed before a
        # remote execute (or a sibling node) mutated the shared
        # /curobo_server planner. Forget it here; the adapter's skip still
        # removes the redundant repeats WITHIN this burst.
        self._invalidate_planner_session()
        self.init()
        if self._executor is None or not self._executor._valid:
            return MoveItErrorCode(MoveItErrorCode.PLANNING_FAILED)
        ex = self._executor
        # MTC Task::plan (task.cpp:243-272) is: init() + compute loop (no
        # description publish, no hook install here - init() did both).
        # The per-pass statistics publish, like init's hooks, only exists
        # on an enabled task: the loop must not conjure endpoints onto a
        # dark task via the enabling introspection() getter.
        enabled = self._introspection is not None

        def _guarded_progress():
            if self._preempted:
                raise Task._Preempted()
            if progress_callback is not None:
                progress_callback()

        def _progress():
            _guarded_progress()
            # MTC Task::plan publishes task state after every compute pass.
            if self._introspection is not None:
                self._introspection.publishTaskState()

        try:
            ok = ex.plan(
                max_iterations=max_solutions, progress_callback=_progress
            )
        except Task._Preempted:
            self._solutions = []
            return MoveItErrorCode(MoveItErrorCode.PREEMPTED)
        # The per-solution publishing happens on the stage hooks installed
        # above (MTC Stage::onNewSolution -> publishSolution), so the panel
        # follows the plan live rather than only at the end.
        self._solutions = ex.rank()
        if enabled and self._introspection is not None:
            self._introspection.publishTaskState()
        if ok and self._solutions:
            return MoveItErrorCode(MoveItErrorCode.SUCCESS)
        return MoveItErrorCode(MoveItErrorCode.PLANNING_FAILED)

    def execute(self, solution: Any = None):
        """Send given solution for execution (MTC ``Task.execute``). Exactly like MoveIt: the full Solution message goes to the
        ``execute_task_solution`` action (served by the task-constructor
        node next to the curobo_server), which drives its sub-trajectories
        in order with no replanning. Like MTC's ``Task::execute`` (task.cpp:282-339) this is BLOCKING and
        returns a MoveItErrorCode: it creates its own private node and
        executor for the action round trip (``execute_solution_node_`` in
        C++), so no caller has to supply — or keep spinning — a node."""
        sol = solution if solution is not None else self.best()
        if sol is None:
            raise RuntimeError("Task.execute(): no solution")
        try:
            import rclpy
        except Exception:
            rclpy = None
        try:
            if rclpy is None or not rclpy.ok():
                # Offline path (no ROS): drive directly through the executor.
                if self._executor is None:
                    raise RuntimeError("Task.execute() before plan()")
                results = self._executor.execute(sol)
                ok = bool(results) and all(r.success for r in results)
                return MoveItErrorCode(
                    MoveItErrorCode.SUCCESS if ok else MoveItErrorCode.CONTROL_FAILED)
            return self._execute_remote(sol)
        finally:
            # The drive ran segments under their cached plans' planners (remote
            # execute server, or the local executor) and may have left the
            # shared /curobo_server on a different one; the next plan burst
            # must re-assert its planner, so forget the belief here too.
            self._invalidate_planner_session()

__all__ = [
    'Task',
]

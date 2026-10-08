"""MTC-shaped local task API over cuRobo (``core`` namespace).

Mirrors ``moveit.task_constructor.core`` so ``cartesian.py``-style scripts run
with only the import changed::

    from curobo_task_constructor.mtc import core, stages

    task = core.Task(robot)
    task.name = "cartesian"
    task.add(stages.CurrentState("current state"))
    ...

``Task.plan()`` runs **locally in-process** (MTC C++ semantics): it converts
the added wrapper stages to a ``StageSpec`` tree and drives ``TaskExecutor``
directly against the injected ``RobotInterface``. The action server in
``node.py`` is a thin remote front for the same path.
"""

from __future__ import annotations

from typing import Any, Optional

PLANNER_CLASSIC = 0
PLANNER_JOINT_SPACE = 5


class SolutionHandle:
    """A reference to a stored solution for ``Task.execute()``.

    In the remote case (planned via the Task action server), the
    solution is stored server-side and addressed by ``task_id`` +
    ``solution_index``. This handle carries those fields so
    ``Task.execute(solution)`` can send the right ExecuteTaskSolution
    goal — the same role a MoveIt ``Solution`` object plays in
    ``task.execute(*task.solutions().front())``.
    """

    def __init__(self, task_id: str, solution_index: int = 0, remote: bool = True):
        self.task_id = str(task_id)
        self.solution_index = int(solution_index)
        self.remote = bool(remote)


def roscpp_init(name: str, *args, **kwargs) -> None:
    """MTC ``py_binding_tools.roscpp_init`` equivalent: init rclpy if present."""
    try:
        import rclpy

        if not rclpy.ok():
            rclpy.init(args=None)
    except Exception:
        pass


class CartesianPrecision:
    """Precision for Cartesian interpolation (MTC CartesianPrecision)."""

    def __init__(
        self,
        translational: float = 0.001,
        rotational: float = 0.01,
        max_resolution: float = 0.01,
    ):
        self.translational = float(translational)
        self.rotational = float(rotational)
        self.max_resolution = float(max_resolution)

    def __str__(self) -> str:
        return (
            f"CartesianPrecision(translational={self.translational}, "
            f"rotational={self.rotational}, "
            f"max_resolution={self.max_resolution})"
        )


class PlannerInterface:
    """Abstract base class for planning algorithms (MTC PlannerInterface).

    ``max_velocity_scaling_factor`` / ``max_acceleration_scaling_factor``
    are accepted for compatibility; the cuRobo backend times its own
    trajectories, so they ride along but do not change the solve.
    """

    wire_name = None

    def __init__(self):
        self.max_velocity_scaling_factor = 1.0
        self.max_acceleration_scaling_factor = 1.0
        self.planner_id = ""

    def planner_key(self):
        key = getattr(self, "wire_name", None)
        return None if key is None else key

    def extra_params(self) -> dict:
        return {}


class CartesianPath(PlannerInterface):
    """Linear interpolation between Cartesian poses (MTC CartesianPath).

    Fails unless ``min_fraction`` of the path is feasible. cuRobo has no
    Cartesian solver: a line is expressed as whole-path axis holds, and
    ``min_fraction >= 1.0`` keeps the stage's straightness gate on while a
    smaller fraction disables it (the free-space opt-out).
    """

    wire_name = "classic"

    def __init__(
        self,
        step_size: float = 0.01,
        min_fraction: float = 1.0,
        precision: CartesianPrecision = None,
    ):
        super().__init__()
        self.step_size = float(step_size)
        self.min_fraction = float(min_fraction)
        self.precision = precision or CartesianPrecision()
        self.max_cartesian_speed = 0.0
        self.cartesian_speed_limited_link = ""

    def planner_key(self) -> int:
        # Straight-line holds are honoured only by the classic planner.
        return PLANNER_CLASSIC


class JointInterpolationPlanner(PlannerInterface):
    """Linear interpolation between joint-space poses (MTC
    JointInterpolationPlanner). Fails on collision along the path; no
    obstacle avoidance.
    """

    wire_name = "joint_space"

    def __init__(self, max_step: float = 0.1):
        super().__init__()
        self.max_step = float(max_step)

    def planner_key(self) -> int:
        return PLANNER_JOINT_SPACE


class PipelinePlanner(PlannerInterface):
    """Plan using a planning pipeline (MTC PipelinePlanner).

    ``num_planning_attempts`` is honoured: when a stage is built with this
    planner and no explicit attempt count, the stage plans that many times
    and keeps the cheapest result. ``pipeline``/``planner_id`` select the
    server backend; ``None`` leaves the server default.
    """

    def __init__(
        self,
        pipeline: str = "ompl",
        planner_id: str = "",
        num_planning_attempts: int = 1,
    ):
        super().__init__()
        self.pipeline = str(pipeline)
        self.planner_id = str(planner_id)
        self.num_planning_attempts = int(num_planning_attempts)
        self.goal_joint_tolerance = 1e-4
        self.goal_position_tolerance = 1e-4
        self.goal_orientation_tolerance = 1e-3
        self.workspace_parameters = None

    @property
    def wire_name(self):
        return None


class MultiPlanner(PlannerInterface):
    """Run alternative planners in sequence, first solution wins (MTC
    MultiPlanner). A stage built with this planner solves with the first
    entry; the list itself is carried for inspection.
    """

    def __init__(self, *planners):
        super().__init__()
        self._planners = list(planners)

    def add(self, *planners) -> None:
        """Insert one or more planners."""
        self._planners.extend(planners)

    def clear(self) -> None:
        """Remove all planners."""
        self._planners.clear()

    def __len__(self) -> int:
        return len(self._planners)

    def __getitem__(self, index):
        return self._planners[index]

    @property
    def wire_name(self):
        if not self._planners:
            return None
        first = self._planners[0]
        return getattr(first, "wire_name", None)

    @property
    def num_planning_attempts(self):
        for planner in self._planners:
            attempts = getattr(planner, "num_planning_attempts", None)
            if attempts is not None:
                return attempts
        return None


class ContainerBase:
    """MTC container stage: an ordered child list with a container type.

    Mirrors ``moveit.task_constructor.core`` containers (SerialContainer /
    Fallbacks / Alternatives): ``add()`` children, then ``to_spec()`` for the
    executor or the wire. ``Task`` itself wraps its children in a serial root,
    so a task is built exactly like ``cartesian.py`` builds stages.
    """

    container_type = ""

    def __init__(self, name: str = ""):
        self.name = name or self.container_type
        self._children: list = []

    def add(self, *args: Any) -> None:
        """Insert stage(s) at the end of the children list (MTC add)."""
        for stage in args:
            self._children.append(stage)

    def insert(self, stage: Any, before: int = -1) -> Any:
        """Insert a stage before the given index (MTC ``ContainerBase.insert``)."""
        self._children.insert(before if before >= 0 else len(self._children), stage)
        return stage

    def remove(self, stage: Any) -> Any:
        """Remove a child stage by index or instance (MTC remove)."""
        if isinstance(stage, int):
            return self._children.pop(stage)
        self._children.remove(stage)
        return stage

    def clear(self) -> None:
        """Remove all stages from the container (MTC clear)."""
        self._children = []

    def __len__(self) -> int:
        return len(self._children)

    def __getitem__(self, key):
        """Child stage by index or by name (MTC findChild)."""
        if isinstance(key, str):
            for child in self._children:
                if getattr(child, "name", None) == key:
                    return child
            raise IndexError(f"no stage named {key!r}")
        return self._children[key]

    def __iter__(self):
        return iter(self._children)

    def to_spec(self):
        from curobo_task_constructor.graph.spec import StageSpec

        return StageSpec(
            stage_type="",
            name=self.name,
            container_type=self.container_type,
            children=[c.to_spec() for c in self._children],
            params_yaml="",
        )


class SerialContainer(ContainerBase):
    container_type = "serial"


class Fallbacks(ContainerBase):
    container_type = "fallbacks"


class Alternatives(ContainerBase):
    container_type = "alternatives"


class IndependentComponents(ContainerBase):
    container_type = "independent"


class Merger(ContainerBase):
    """Plan parallel sub-tasks on disjoint joint groups and merge (MTC
    Merger). Same composition contract as independent components: every
    child sees the same input, solutions lift directly."""

    container_type = "merger"


class Task:
    """MTC ``core.Task`` equivalent: ordered stage list, local plan/publish.

    Mirrors ``moveit.task_constructor.core.Task``: ``init()``,
    ``plan(max_solutions=0)``, ``publish(solution)``,
    ``execute(solution)``. ``setProperty`` stores the same task-level
    ``group`` / ``eef`` / ``ik_frame`` bag the old pipeline sets.
    """

    def __init__(
        self, robot: Any = None, name: str = "task", task_id: Optional[str] = None
    ):
        self._name = name
        self.robot = robot
        self.task_id = task_id or name
        self._stages: list = []
        self._executor = None
        self._solutions: list = []
        self.last_published: Any = None
        #: Task-level properties (MTC ``Task::setProperty`` / ``properties``).
        #: The old pipeline sets ``group`` / ``eef`` / ``ik_frame`` here and
        #: stages inherit them; the cuRobo builders carry the same values on
        #: each stage instead, so this bag is stored for API parity and
        #: inspection (``describe()`` reports it) rather than consumed.
        self._properties: dict = {}
        #: rclpy node used by publish() for the RViz trajectory topic
        #: (``attach_node``). Without one, publish() only records.
        self._node: Any = None
        #: Publish gate (MTC ``enableIntrospection``): publish() reaches RViz
        #: only when enabled (default). Local planning itself stays silent;
        #: nothing is published unless publish() is called.
        self._introspection_enabled = True
        #: Set by preempt(); the next plan() returns False immediately.
        self._preempted = False

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

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

    def attach_node(self, node: Any) -> None:
        """Attach an rclpy node so ``publish()`` reaches RViz.

        MTC ``Task.publish`` shows the solution in RViz; here that means
        publishing the trajectory for the ``CuroboTrajectoryDisplay``.
        Without a node (plain unit tests), ``publish()`` only records.
        """
        self._node = node

    def enableIntrospection(self, enabled: bool = True) -> None:
        """Gate RViz publishing (MTC enableIntrospection).

        publish()/publishAllSolutions() reach RViz only when enabled
        (default) and a node is attached. Planning itself never publishes.
        """
        self._introspection_enabled = bool(enabled)

    def getRobotModel(self) -> Any:
        """Get the robot model (MTC ``Task.getRobotModel``).

        cuRobo uses a RobotInterface rather than a MoveIt RobotModel;
        returns the interface's robot config or None.
        """
        if self.robot is None:
            return None
        return getattr(self.robot, "robot_config", self.robot)

    def setRobotModel(self, robot_model: Any) -> None:
        """Set the robot model (MTC ``Task.setRobotModel``).

        cuRobo uses a RobotInterface; attaches it via ``attach_robot``.
        """
        self.attach_robot(robot_model)

    def introspection(self) -> Any:
        """Access introspection object (MTC ``Task.introspection``).

        Returns the executor's introspection surface (describe/statistics).
        """
        if self._executor is None:
            return None
        return self._executor

    def loadRobotModel(self, robot_description: str = "robot_description") -> None:
        """Load robot model from given ROS parameter (MTC ``Task.loadRobotModel``).

        cuRobo's equivalent: the robot config is already loaded in the
        RobotInterface; this is a no-op for API parity.
        """
        pass

    def setCostTerm(self, *args, **kwargs) -> None:
        """Specify a CostTerm for calculation of stage costs (MTC ``Task.setCostTerm``).

        cuRobo stages carry their own cost terms via ``setCost``; this
        stores the task-level term for inspection.
        """
        if args:
            self._properties["cost_term"] = args[0]

    # -- spec conversion ----------------------------------------------
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
        """Build and initialize the task (MTC Task.init).

        Returns None (MTC parity). Check ``describe()`` for validity.
        """
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

    class _Preempted(Exception):
        pass

    def preempt(self) -> None:
        """Interrupt current planning (MTC Task.preempt).

        Takes effect between compute passes; an in-flight planner call
        runs to completion first.
        """
        self._preempted = True

    def plan(self, max_solutions: int = 0, progress_callback=None) -> bool:
        """Reset, init, and plan (MTC ``Task.plan``).

        ``max_solutions`` (0 = unlimited) bounds the number of solutions
        collected. Returns True on success (MTC returns MoveItErrorCode;
        cuRobo uses bool for the Python API).
        """
        # NOTE (MTC divergence): MTC plan(max_solutions) collects up to N
        # solutions. Our executor stops at the first complete root solution
        # (re-solving the trailing chain per reconnect once wedged the
        # server's CUDA-graph capture), so max_solutions only bounds the
        # compute loop; use fallbacks/alternatives for variant coverage.
        self.reset()
        if self._preempted:
            self._preempted = False
            return False
        self.init()
        if self._executor is None or not self._executor._valid:
            return False
        ex = self._executor

        def _guarded_progress():
            if self._preempted:
                raise Task._Preempted()
            if progress_callback is not None:
                progress_callback()

        try:
            ok = ex.plan(
                max_iterations=max_solutions, progress_callback=_guarded_progress
            )
        except Task._Preempted:
            self._solutions = []
            return False
        self._solutions = ex.rank()
        return bool(ok and self._solutions)

    def execute(self, solution: Any = None):
        """Send given solution for execution (MTC ``Task.execute``).

        MTC sends the solution to the ``move_group`` node via the
        ``execute_task_solution`` action. cuRobo does the same: the
        solution is sent to the task-constructor node's
        ``ExecuteTaskSolution`` action, which drives the stored plan
        through curobo's SendTrajectory.

        The action client is created on the node supplied with
        ``attach_node()``. The caller owns that node and its ROS executor;
        this class never creates or spins a node/executor for action I/O.

        Two paths:
        - **Local** (planned in-process): the executor drives the
          solution directly against the RobotInterface. Returns True on
          success.
        - **Remote** (planned via the Task action server): the solution
          handle carries ``task_id`` + ``solution_index`` and the goal
          is sent to the task-constructor node's ExecuteTaskSolution
          action, which replays the stored plan with no replanning.
          Returns an rclpy Future that resolves to True/False on
          completion (the caller's executor spins it — calling
          ``spin_until_future_complete`` from within a callback would
          deadlock the executor).
        """
        sol = solution if solution is not None else self.best()
        if sol is None:
            raise RuntimeError("Task.execute(): no solution")
        # Remote path: the solution handle carries task_id + solution_index
        # (the plan happened on the task-constructor node, not in-process).
        task_id = getattr(sol, "task_id", None) or self.task_id
        solution_index = getattr(sol, "solution_index", 0)
        if getattr(sol, "remote", False) or self._executor is None:
            return self._execute_remote(task_id, solution_index)
        # Local execution through the executor (same as MTC's local
        # move_group call).
        results = self._executor.execute(sol)
        return all(r.success for r in results)

    def _execute_remote(self, task_id: str, solution_index: int = 0) -> bool:
        """Execute a stored task solution synchronously through the ROS action.

        The action client is bound to the node supplied with
        :meth:`attach_node`. ROS callbacks/futures are serviced exclusively
        by that node's existing executor. This method never creates a node,
        executor, thread, or calls ``rclpy.spin*``.

        The caller must ensure that the attached node's executor is already
        spinning while this blocking call waits for the action to complete.
        In particular, do not call this method from a callback running on a
        single-threaded executor, because blocking that callback would prevent
        the action response from being processed.
        """
        try:
            from rclpy.action.client import ActionClient
            from curobo_task_constructor_interfaces.action import (
                ExecuteTaskSolution,
            )
        except Exception:
            raise RuntimeError(
                "Task.execute() remote path requires rclpy and "
                "curobo_task_constructor_interfaces"
            )

        if self._node is None:
            raise RuntimeError(
                "Task.execute() remote path requires an attached ROS node; "
                "call task.attach_node(node) first"
            )

        if getattr(self, "_execute_client", None) is None:
            self._execute_client = ActionClient(
                self._node,
                ExecuteTaskSolution,
                "/curobo_task_constructor/execute_task_solution",
            )

        self._execute_client.wait_for_server()

        goal = ExecuteTaskSolution.Goal()
        goal.task_id = task_id
        goal.solution_index = int(solution_index)
        goal.stage_id = 0xFFFFFFFF  # NO_STAGE: whole chain
        goal.attempt_id = 0

        # rclpy's action API is asynchronous; result() makes this wrapper
        # blocking while the application's existing ROS executor continues
        # to service the ActionClient callbacks.
        goal_handle = self._execute_client.send_goal_async(goal).result()

        if not goal_handle or not goal_handle.accepted:
            return False

        action_result = goal_handle.get_result_async().result()
        result = getattr(action_result, "result", action_result)
        return bool(getattr(result, "success", False))

    def publish(self, solution: Any) -> None:
        """MTC ``Task.publish`` equivalent: show the solution in RViz.

        Records the solution and, when introspection is enabled (default)
        and a node is attached (``attach_node``), publishes its trajectory
        as ``trajectory_msgs/JointTrajectory`` for the
        ``CuroboTrajectoryDisplay`` — the same topic the rviz panel
        publishes a selected solution on. Planning itself never publishes.
        """
        self.last_published = solution
        if self._node is None or not self._introspection_enabled:
            return
        try:
            from curobo_task_constructor.viz import (
                publish_solution_trajectory,
            )

            publish_solution_trajectory(self._node, solution)
        except Exception:
            pass

    def publishAllSolutions(self, wait: bool = True) -> None:
        """Publish every ranked solution (MTC publishAllSolutions).

        Only the last published trajectory stays visible on the display;
        the panel lists them all once a task description arrives.
        """
        for solution in self.rank():
            self.publish(solution)

    def rank(self) -> list:
        """Ranked solutions, cheapest first (MTC solution ordering)."""
        if self._executor is None:
            return []
        return self._executor.rank()

    def reset(self) -> None:
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

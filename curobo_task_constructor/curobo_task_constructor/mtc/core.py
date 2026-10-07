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


def roscpp_init(name: str, *args, **kwargs) -> None:
    """MTC ``py_binding_tools.roscpp_init`` equivalent: init rclpy if present."""
    try:
        import rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
    except Exception:
        pass


class CartesianPath:
    """MTC ``core.CartesianPath`` solver object (cuRobo: straight-line cost)."""

    wire_name = "classic"

    def __init__(self, max_cartesian_speed: float = 0.1,
                 cartesian_speed_limited_link: str = ""):
        self.max_cartesian_speed = float(max_cartesian_speed)
        self.cartesian_speed_limited_link = str(cartesian_speed_limited_link or "")

    def planner_key(self) -> int:
        # Straight-line holds are honoured only by the classic planner.
        return PLANNER_CLASSIC

    def extra_params(self) -> dict:
        params: dict = {}
        if self.cartesian_speed_limited_link:
            params["link"] = self.cartesian_speed_limited_link
        return params


class JointInterpolationPlanner:
    """MTC ``core.JointInterpolationPlanner`` solver object."""

    wire_name = "joint_space"

    def planner_key(self) -> int:
        return PLANNER_JOINT_SPACE

    def extra_params(self) -> dict:
        return {}


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

    def add(self, stage: Any) -> Any:
        self._children.append(stage)
        return stage

    def __len__(self) -> int:
        return len(self._children)

    def to_spec(self):
        from curobo_task_constructor.graph.spec import StageSpec
        return StageSpec(stage_type="", name=self.name,
                         container_type=self.container_type, children=[
                             c.to_spec() for c in self._children],
                         params_yaml="")


class SerialContainer(ContainerBase):
    container_type = "serial"


class Fallbacks(ContainerBase):
    container_type = "fallbacks"


class Alternatives(ContainerBase):
    container_type = "alternatives"


class IndependentComponents(ContainerBase):
    container_type = "independent"


class Task:
    """MTC ``core.Task`` equivalent: ordered stage list, local plan/publish."""

    def __init__(self, robot: Any = None, name: str = "task",
                 task_id: Optional[str] = None):
        self.name = name
        self.robot = robot
        self.task_id = task_id or name
        self._stages: list = []
        self._executor = None
        self._solutions: list = []
        self.last_published: Any = None
        #: rclpy node used by publish() for the RViz trajectory topic
        #: (``attach_node``). Without one, publish() only records.
        self._node: Any = None

    # -- construction -------------------------------------------------
    def add(self, stage: Any) -> Any:
        self._stages.append(stage)
        return stage

    def insert(self, stage: Any, index: int = -1) -> Any:
        self._stages.insert(index if index >= 0 else len(self._stages), stage)
        return stage

    def attach_robot(self, robot: Any) -> None:
        self.robot = robot

    def attach_node(self, node: Any) -> None:
        """Attach an rclpy node so ``publish()`` reaches RViz.

        MTC ``Task.publish`` shows the solution in RViz; here that means
        publishing the trajectory for the ``CuroboTrajectoryDisplay``.
        Without a node (plain unit tests), ``publish()`` only records.
        """
        self._node = node

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
        return StageSpec(stage_type="serial", name=self.name,
                         container_type="serial", children=children,
                         params_yaml="")

    # -- lifecycle (MTC Task::init/plan/execute/publish) ---------------
    def plan(self, max_iterations: int = 0, progress_callback=None) -> bool:
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
        if not ex.init():
            self._executor = ex
            self._solutions = []
            return False
        ok = ex.plan(max_iterations=max_iterations,
                     progress_callback=progress_callback)
        self._executor = ex
        self._solutions = ex.rank()
        return bool(ok and self._solutions)

    def execute(self, solution: Any = None) -> list:
        if self._executor is None:
            raise RuntimeError("Task.execute() before plan()")
        sol = solution if solution is not None else self.best()
        if sol is None:
            raise RuntimeError("Task.execute(): no solution")
        return self._executor.execute(sol)

    def publish(self, solution: Any) -> None:
        """MTC ``Task.publish`` equivalent: show the solution in RViz.

        Records the solution and, when a node is attached (``attach_node``),
        publishes its trajectory as ``trajectory_msgs/JointTrajectory`` for
        the ``CuroboTrajectoryDisplay`` — the same topic the rviz panel
        publishes a selected solution on.
        """
        self.last_published = solution
        if self._node is None:
            return
        try:
            from curobo_task_constructor.viz import (
                publish_solution_trajectory,
            )
            publish_solution_trajectory(self._node, solution)
        except Exception:
            pass

    def reset(self) -> None:
        if self._executor is not None:
            self._executor.reset()
        self._solutions = []

    # -- results -------------------------------------------------------
    @property
    def solutions(self) -> list:
        return list(self._solutions)

    def best(self) -> Any:
        return self._solutions[0] if self._solutions else None

    @property
    def executor(self):
        return self._executor

    def describe(self) -> dict:
        if self._executor is None:
            return {"task_id": self.task_id or self.name, "valid": False,
                    "comment": "not planned yet", "stage_count": len(self._stages)}
        return self._executor.describe()

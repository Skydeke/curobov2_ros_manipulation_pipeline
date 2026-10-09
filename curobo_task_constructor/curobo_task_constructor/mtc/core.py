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

#: Introspection namespace default (MTC task ns; absolute default kept for compat).
#: MTC uses RELATIVE topics ``description/statistics/solution`` in the task's
#: namespace plus service ``get_solution_<task_id>``. This package keeps the
#: absolute default ``/curobo_task_constructor/...`` but honours ``Task(ns)``:
#: ``Task(ns="my_ns")`` publishes ``my_ns/description`` etc. and serves
#: ``my_ns/get_solution_<task_id>`` — see ``Task._ns_base``.
DEFAULT_TASK_NS = "/curobo_task_constructor"
DESCRIPTION_TOPIC = f"{DEFAULT_TASK_NS}/task_description"
STATISTICS_TOPIC = f"{DEFAULT_TASK_NS}/task_statistics"
SOLUTION_TOPIC = f"{DEFAULT_TASK_NS}/solution"
GET_SOLUTION_SERVICE = f"{DEFAULT_TASK_NS}/get_solution"
#: ExecuteTaskSolution rendezvous. MTC uses the RELATIVE action name
#: ``execute_task_solution`` on its own private node
#: (``moveit_task_constructor_executor_<this>``, ``task.cpp:282-288``), and
#: ``Task::execute`` waits **0.5 s** for that server before returning
#: ``MoveItErrorCode::FAILURE`` — an execution backend that is not up must
#: fail the call, never hang the caller.
EXECUTE_ACTION = "/curobo_task_constructor/execute_task_solution"
#: MTC ``wait_for_action_server(0.5s)``.
EXECUTE_WAIT_SEC = 0.5

PLANNER_CLASSIC = 0
PLANNER_JOINT_SPACE = 5


class MoveItErrorCode:
    """Minimal ``moveit_msgs/MoveItErrorCodes`` equivalent (MTC ``plan/execute`` return).

    ``Task.plan/execute`` return this (not bool) so callers can check
    ``.val == MoveItErrorCode.SUCCESS`` exactly like MoveIt. Truthiness
    follows MTC success (``bool(code)`` is ``val == SUCCESS``) so existing
    ``assert task.plan()`` / ``if task.execute():`` call sites keep working.
    """

    SUCCESS = 1
    FAILURE = 99999
    PLANNING_FAILED = -1
    INVALID_MOTION_PLAN = -3
    CONTROL_FAILED = -6
    PREEMPTED = -5

    def __init__(self, val: int = SUCCESS):
        self.val = int(val)

    def __bool__(self) -> bool:
        return self.val == MoveItErrorCode.SUCCESS

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, MoveItErrorCode):
            return self.val == other.val
        if isinstance(other, int):
            return self.val == other
        return NotImplemented

    def __repr__(self) -> str:
        return f"MoveItErrorCode(val={self.val})"


def _ros_ok() -> bool:
    """MTC's ``rclcpp::ok()`` — is a ROS context initialized?

    MTC enables introspection at Task construction only when it is
    (task.cpp:102-104). Same gate here: without rclpy up, a Task plans and
    executes locally and publishes nothing, rather than half-creating an
    introspection node that cannot spin.
    """
    try:
        import rclpy

        return bool(rclpy.ok())
    except Exception:
        return False


class _IntrospectionExecutor:
    """Shared, ref-counted spinning executor (MTC ``IntrospectionExecutor``).

    MTC gives each ``Introspection`` ITS OWN node
    (``introspection_<task_id>`` in the task's namespace) on one static
    executor that spins a thread while any of those nodes is alive and stops
    when the last one goes away (introspection.cpp:82-110). That is why the
    MTC API has no ``attach_node``: the caller never supplies a node, and the
    panel is fed regardless.

    Ported verbatim in spirit here so curobo needs no such call either. The
    node is ours, the executor is ours, and it shuts down with the last
    introspection.
    """

    _executor: Any = None
    _thread: Any = None
    _count: int = 0
    _lock: Any = None

    @classmethod
    def _lock_(cls):
        import threading

        if cls._lock is None:
            cls._lock = threading.Lock()
        return cls._lock

    @classmethod
    def add_node(cls, node: Any) -> None:
        with cls._lock_():
            if cls._executor is None:
                from rclpy.executors import SingleThreadedExecutor

                cls._executor = SingleThreadedExecutor()
            cls._executor.add_node(node)
            if cls._count == 0:
                import threading

                # The ONE and ONLY spin() in this package, and it is on an
                # Executor: start the shared executor's spin loop exactly
                # once, at the 0 -> 1 transition. Everything else that needs
                # serviced uses spin_once() on an executor that ONLY
                # the calling thread drives - sharing an already-spinning
                # executor across threads is what raises "Executor is
                # already spinning" and, downstream, took the planner down.
                # NEVER call spin() on a node, and never add a second spin()
                # to this executor.
                cls._thread = threading.Thread(
                    target=cls._executor.spin, daemon=True)
                cls._thread.start()
            cls._count += 1

    @classmethod
    def remove_node(cls, node: Any) -> None:
        with cls._lock_():
            if cls._executor is None:
                return
            cls._executor.remove_node(node)
            cls._count -= 1
            if cls._count > 0:
                return
            # Last node gone: cancel, join, drop (MTC does exactly this).
            executor, thread = cls._executor, cls._thread
            cls._executor = None
            cls._thread = None
            cls._count = 0
        executor.shutdown()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)


#: The one introspection node in this process (MTC's per-Introspection
#: ``node_``, made process-shared because a Task per motion would otherwise
#: leak a node per motion — see ``Introspection._ensure_publishers``).
_INTROSPECTION_NODE: Any = None
_INTROSPECTION_NODE_LOCK: Any = None


def _private_action_executor() -> Any:
    """A PRIVATE (node, executor) pair for one action round trip.

    The executor is fresh, has no ``spin()`` running on it anywhere, and is
    to be serviced by the CALLING thread with ``spin_once()``
    only. It is structurally impossible to get the shared introspection
    executor from here: that one has a background ``spin()``, and driving it
    from two threads is what raised "Executor is already spinning".
    """
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    node = rclpy.create_node("curobo_task_constructor_executor")
    executor = SingleThreadedExecutor()
    assert executor is not _IntrospectionExecutor._executor, (
        "_private_action_executor must never return the shared executor")
    executor.add_node(node)
    return node, executor


def _shared_introspection_node() -> Any:
    """Create (once) and return the shared introspection node.

    ``None`` when rclpy is not up (plain unit tests): publication then fails
    closed, exactly as MTC's introspection stays off without ``rclcpp::ok()``.
    """
    global _INTROSPECTION_NODE, _INTROSPECTION_NODE_LOCK
    import threading

    if _INTROSPECTION_NODE_LOCK is None:
        _INTROSPECTION_NODE_LOCK = threading.Lock()
    with _INTROSPECTION_NODE_LOCK:
        if _INTROSPECTION_NODE is not None:
            return _INTROSPECTION_NODE
        try:
            import rclpy

            if not rclpy.ok():
                return None
            node = rclpy.create_node("curobo_task_constructor_introspection")
            _IntrospectionExecutor.add_node(node)
        except Exception:
            return None
        _INTROSPECTION_NODE = node
        return node


class _DoneFuture:

    def __init__(self, result: Any = None):
        self._result = result
        self._callbacks: list = []
        if result is not None:
            self._fire()

    def set_result(self, result: Any) -> "_DoneFuture":
        self._result = result
        self._fire()
        return self

    def _fire(self) -> None:
        callbacks, self._callbacks = self._callbacks, []
        for callback in callbacks:
            try:
                callback(self)
            except Exception:
                pass

    def add_done_callback(self, callback) -> None:
        if self._result is not None:
            try:
                callback(self)
            except Exception:
                pass
        else:
            self._callbacks.append(callback)

    def result(self) -> Any:
        result = self._result
        while isinstance(result, _DoneFuture):
            result = result._result
        return result


def _result_ok(res: Any) -> bool:
    """ExecuteTaskSolution result -> success bool (MTC error_code check).

    MTC reports ``moveit_msgs/MoveItErrorCodes`` (SUCCESS == 1); accept
    that as well as a plain ``success`` bool for duck-typed results.
    """
    code = getattr(res, "error_code", None)
    if code is not None:
        val = getattr(code, "val", code)
        try:
            return int(val) == 1
        except (TypeError, ValueError):
            pass
    res = getattr(res, "result", res)
    code = getattr(res, "error_code", None)
    if code is not None:
        val = getattr(code, "val", code)
        try:
            return int(val) == 1
        except (TypeError, ValueError):
            pass
        return False
    return bool(getattr(res, "success", False))


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

    Configuration follows MTC exactly: ``setProperty``/``setTimeout``/
    ``setMaxVelocityScalingFactor``/``setMaxAccelerationScalingFactor`` write
    the planner's property bag (``properties()``), which takes precedence
    over stage properties. The cuRobo backend times its own trajectories,
    so the scaling factors ride along but do not change the solve.
    """

    wire_name = None

    def __init__(self):
        self.max_velocity_scaling_factor = 1.0
        self.max_acceleration_scaling_factor = 1.0
        self.planner_id = ""
        self._planner_properties: dict = {}

    def properties(self) -> dict:
        """Planner property bag (MTC ``PlannerInterface::properties``)."""
        return self._planner_properties

    def setProperty(self, name: str, value: Any) -> None:
        """Set a planner property (MTC ``PlannerInterface::setProperty``)."""
        self._planner_properties[str(name)] = value
        if name == "num_planning_attempts":
            try:
                self.num_planning_attempts = int(value)
            except (TypeError, ValueError):
                pass

    def setTimeout(self, timeout: float) -> None:
        """Set the planning timeout (MTC ``PlannerInterface::setTimeout``)."""
        self.setProperty("timeout", float(timeout))

    def setMaxVelocityScalingFactor(self, factor: float) -> None:
        """Set max velocity scaling (MTC, stored for compatibility)."""
        self.max_velocity_scaling_factor = float(factor)
        self.setProperty("max_velocity_scaling_factor", float(factor))

    def setMaxAccelerationScalingFactor(self, factor: float) -> None:
        """Set max acceleration scaling (MTC, stored for compatibility)."""
        self.max_acceleration_scaling_factor = float(factor)
        self.setProperty("max_acceleration_scaling_factor", float(factor))

    def init(self, robot_model: Any = None) -> None:
        """Initialize pipelines (MTC ``PlannerInterface::init``).

        cuRobo planners need no initialization; no-op for API parity.
        """

    def getPlannerId(self) -> str:
        """Name of the planner (MTC ``PlannerInterface::getPlannerId``)."""
        name = getattr(self, "wire_name", None)
        return str(name) if isinstance(name, str) else type(self).__name__

    def planner_key(self):
        key = getattr(self, "wire_name", None)
        return None if key is None else key

    def extra_params(self) -> dict:
        params = dict(self._planner_properties)
        params.pop("timeout", None)
        return params


def _looks_like_node(obj: Any) -> bool:
    """Is ``obj`` an rclpy-style node (MTC's first PipelinePlanner arg)?"""
    if obj is None or isinstance(obj, str):
        return False
    return callable(getattr(obj, "create_client", None)) and callable(
        getattr(obj, "get_name", None)
    )


#: demo_ws planning pipelines -> the cuRobo planner that backs them.
#:
#: The demo configures ``pipeline="ompl"`` (+ a per-planner id such as
#: ``RRTConnectkConfigDefault``). cuRobo has no OMPL: its equivalent of a
#: sampling planner is the JOINT_SPACE motion generator, and its equivalent
#: of a constrained/consecutive planner is CLASSIC, which is also the only
#: one that honours Cartesian axis holds. The mapping is explicit and total,
#: so an unknown pipeline fails loudly instead of silently planning with
#: whatever the server had selected last.
PIPELINE_PLANNERS = {
    "classic": PLANNER_CLASSIC,
    "cartes": PLANNER_CLASSIC,
    "cartesian": PLANNER_CLASSIC,
    "pilz": PLANNER_CLASSIC,
    "pilz_industrial_motion_planner": PLANNER_CLASSIC,
    "mpc": 1,
    "batch": 2,
    "joint": PLANNER_JOINT_SPACE,
    "joint_space": PLANNER_JOINT_SPACE,
    "ompl": PLANNER_JOINT_SPACE,
    "chomp": PLANNER_JOINT_SPACE,
    "retarget": 6,
}


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

    def setStepSize(self, step_size: float) -> None:
        self.step_size = float(step_size)

    def setPrecision(self, precision: CartesianPrecision) -> None:
        self.precision = precision

    def setMinFraction(self, min_fraction: float) -> None:
        self.min_fraction = float(min_fraction)

    def setIKFrame(self, *args) -> None:
        """Set the IK frame (MTC ``CartesianPath::setIKFrame``).

        Accepts a link name (stored); pose forms are accepted and reduced
        to their link where available.
        """
        for arg in args:
            if isinstance(arg, str):
                self.setProperty("ik_frame", arg)
                return
        if args:
            self.setProperty("ik_frame", args[-1])

    def getPlannerId(self) -> str:
        return "CartesianPath"

    def planner_key(self) -> int:
        # Straight-line holds are honoured only by the classic planner.
        return PLANNER_CLASSIC

    def extra_params(self) -> dict:
        params = dict(self._planner_properties)
        params.pop("timeout", None)
        params.setdefault("step_size", self.step_size)
        params.setdefault("min_fraction", self.min_fraction)
        return params


class JointInterpolationPlanner(PlannerInterface):
    """Linear interpolation between joint-space poses (MTC
    JointInterpolationPlanner). Fails on collision along the path; no
    obstacle avoidance.
    """

    wire_name = "joint_space"

    def __init__(self, max_step: float = 0.1):
        super().__init__()
        self.max_step = float(max_step)

    def setMaxStep(self, max_step: float) -> None:
        self.max_step = float(max_step)

    def getPlannerId(self) -> str:
        return "JointInterpolationPlanner"

    def planner_key(self) -> int:
        return PLANNER_JOINT_SPACE

    def extra_params(self) -> dict:
        params = dict(self._planner_properties)
        params.pop("timeout", None)
        params.setdefault("max_step", self.max_step)
        return params


class PipelinePlanner(PlannerInterface):
    """Plan using a planning pipeline (MTC PipelinePlanner).

    ``num_planning_attempts`` is honoured: when a stage is built with this
    planner and no explicit attempt count, the stage plans that many times
    and keeps the cheapest result. ``pipeline``/``planner_id`` select the
    server backend; ``None`` leaves the server default. ``node`` is accepted
    first for MTC call-shape parity (``PipelinePlanner(node, pipeline,
    planner_id)``) and otherwise unused — planning goes through the robot
    interface's services.
    """

    def __init__(
        self,
        node: Any = None,
        pipeline: str = "ompl",
        planner_id: str = "",
        num_planning_attempts: int = 1,
    ):
        # MTC call shape is (node, pipeline, planner_id); the node-less form
        # is (pipeline[, planner_id[, attempts]]) where the FIRST arg is a
        # pipeline NAME string. Anything non-string first (node, None,
        # dummy test namespace) is the MTC form — the node is kept as-is
        # (unused by the backend) so PipelinePlanner(dummy_node, "ompl", id)
        # preserves the configured pipeline/planner id.
        if isinstance(node, str):
            name, second, third = node, pipeline, planner_id
            node = None
            pipeline = name
            if isinstance(second, str) and not second.lstrip("-").isdigit():
                planner_id = second
                if isinstance(third, int):
                    num_planning_attempts = third
            else:
                planner_id = ""
                try:
                    num_planning_attempts = int(second)
                except (TypeError, ValueError):
                    pass
        super().__init__()
        self.node = node
        self.pipeline = str(pipeline)
        self.planner_id = str(planner_id)
        self.num_planning_attempts = int(num_planning_attempts)
        self.goal_joint_tolerance = 1e-4
        self.goal_position_tolerance = 1e-4
        self.goal_orientation_tolerance = 1e-3
        self.workspace_parameters = None

    def setPlannerId(self, pipeline_name: str, planner_id: str) -> bool:
        """Set the planner id for a pipeline (MTC, single-pipeline: set)."""
        self.pipeline = str(pipeline_name)
        self.planner_id = str(planner_id)
        return True

    def getPlannerId(self) -> str:
        return self.planner_id or self.pipeline or "PipelinePlanner"

    @property
    def wire_name(self):
        return None

    def planner_key(self):
        """The cuRobo planner this pipeline selects (MTC planner resolution).

        ``None`` only when the pipeline is unknown — see ``PIPELINE_PLANNERS``
        for why an unknown pipeline must not fall through silently.
        """
        key = PIPELINE_PLANNERS.get(str(self.pipeline).strip().lower())
        if key is None:
            raise ValueError(
                f"unknown planning pipeline {self.pipeline!r} "
                f"(planner_id={self.planner_id!r}); expected one of "
                f"{sorted(PIPELINE_PLANNERS)}. cuRobo has no OMPL/Pilz "
                f"backends - see PIPELINE_PLANNERS for the mapping."
            )
        return key

    def extra_params(self) -> dict:
        """Planner config as stage params (visible in the panel properties).

        ``num_planning_attempts`` is folded into the stage's
        ``planning_attempts`` by ``_MotionBase``; it is echoed here under the
        MTC spelling so the configured value is inspectable either way.
        """
        params = dict(self._planner_properties)
        params.pop("timeout", None)
        params.setdefault("pipeline", self.pipeline)
        if self.planner_id:
            params.setdefault("planner_id", self.planner_id)
        return params


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


def load_planner_configs(path_or_dict: Any) -> dict:
    """Per-group planner configs (``moveit_wrapper_task_constructor/params/jaco2.yaml`` shape).

    Input ``{planning_groups: [...], <group>: {planning_time,
    max_velocity_scaling_factor, max_acceleration_scaling_factor,
    planning_attempts, planner_id, planning_pipeline_id}}`` or a YAML file
    path with the same content. Empty ``planner_id`` means the pipeline
    default (MTC: OMPL default from ``ompl_planning.yaml``).

    Returns ``{group: PipelinePlanner}`` with timeout/attempts/scaling
    applied — the per-stage lookup ``planners[stage.group]`` replaces the
    single shared planner object in demo scripts.
    """
    if isinstance(path_or_dict, str):
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"load_planner_configs: pyyaml needed for {path_or_dict!r}: {exc!r}")
        with open(path_or_dict, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
    else:
        cfg = dict(path_or_dict or {})
    groups = list(cfg.get("planning_groups", []))
    out: dict = {}
    for group in groups:
        gc = dict(cfg.get(group, {}) or {})
        pipeline = str(gc.get("planning_pipeline_id", "ompl") or "ompl")
        planner_id = str(gc.get("planner_id", "") or "")
        attempts = gc.get("planning_attempts", gc.get("num_planning_attempts", 1))
        try:
            attempts = int(attempts)
        except (TypeError, ValueError):
            attempts = 1
        planner = PipelinePlanner(
            None, pipeline, planner_id, num_planning_attempts=attempts)
        if gc.get("planning_time") is not None:
            try:
                planner.setTimeout(float(gc["planning_time"]))
            except (TypeError, ValueError):
                pass
        if gc.get("max_velocity_scaling_factor") is not None:
            try:
                planner.setMaxVelocityScalingFactor(
                    float(gc["max_velocity_scaling_factor"]))
            except (TypeError, ValueError):
                pass
        if gc.get("max_acceleration_scaling_factor") is not None:
            try:
                planner.setMaxAccelerationScalingFactor(
                    float(gc["max_acceleration_scaling_factor"]))
            except (TypeError, ValueError):
                pass
        out[str(group)] = planner
    return out


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


#: StageDescription properties published per stage type.
#:
#: MTC's fillTaskDescription publishes each stage's PropertyMap, which holds
#: DECLARED properties only (stage.cpp declares timeout/marker_ns/... on the
#: base; move_to.cpp declares group/ik_frame/goal/path_constraints; ...).
#: Everything else (planner wire keys, attempt budgets, cost terms, solver
#: extras) is solve-time config: it rides params_yaml to the executor, never
#: the panel. Publishing the full params dict drowned the properties pane,
#: so this allowlist keeps the MTC-declared set per stage type. curobo-only
#: stages keep their actual config keys (they are few and meaningful).
_DISPLAY_PROPERTIES = {
    "current_state": ("require_not_attached", "require_attached", "timeout"),
    "fixed_state": ("goal", "timeout"),
    "move_to": ("group", "goal", "ik_frame", "path_constraints", "timeout"),
    "move_relative": ("group", "ik_frame", "axis", "distance", "rotation",
                      "joint_offsets", "min_distance", "max_distance",
                      "path_constraints", "timeout"),
    "connect": ("merge_mode", "max_distance", "path_constraints", "timeout"),
    "compute_ik": ("group", "eef", "max_solutions", "target_pose", "ik_frame",
                   "ignore_collisions", "timeout"),
    "generate_pose": ("pose", "timeout"),
    "cartesian_path": ("goal", "hold", "check_straightness", "timeout"),
    "modify_scene": ("add", "remove", "remove_all", "attach", "attach_link",
                     "detach", "detach_all", "allow_collisions", "timeout"),
}


#: Process-global GetSolution services by rendezvous name, served on the
#: shared introspection node. MTC never collides (its task_id embeds
#: hostname_pid_this-pointer), but this pipeline reuses labels across cycles
#: ("setup_scene" every cycle, "pick" every pick), so a new Task must
#: deterministically TAKE OVER the name: depending on garbage-collection
#: timing, a bare create_service would otherwise raise (failing
#: _ensure_publishers closed, silencing the whole task — no description,
#: no statistics, no solutions, panel never updates) or leave the STALE
#: task answering ids from a reset registry (wrong/empty payloads).
_SERVICE_HANDLES: dict = {}
_SERVICE_LOCK: Any = None


def _service_lock():
    """Process lock guarding _SERVICE_HANDLES."""
    global _SERVICE_LOCK
    import threading

    if _SERVICE_LOCK is None:
        _SERVICE_LOCK = threading.Lock()
    return _SERVICE_LOCK


def _replace_service(node: Any, srv_type: Any, name: str, callback: Any):
    """Create a named service, destroying any live predecessor first."""
    with _service_lock():
        old = _SERVICE_HANDLES.get(name)
        if old is not None:
            try:
                node.destroy_service(old)
            except Exception:
                pass
            _SERVICE_HANDLES.pop(name, None)
        handle = node.create_service(srv_type, name, callback)
        _SERVICE_HANDLES[name] = handle
        return handle


class Introspection:
    """MTC ``Task::introspection()`` (``core::Introspection``) surface.

    OWNS the publishers and the ``get_solution`` service, and implements
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
    and the service ``get_solution_<task_id>``.

    Node handling: MTC's ``IntrospectionPrivate`` creates its OWN node
    (``introspection_<task_id>`` in the task's namespace) on a shared
    ref-counted executor it spins itself; that is what isolates every task's
    publishers and service. A pybind11 binding gets that for free by calling
    into the same class. This Python core cannot: the caller owns the node
    and its executor, and a second executor spinning the same context is a
    data race. So the publishers are cached ON THE ATTACHED NODE and shared
    by every Task that attaches it, the service stays per TASK (MTC's
    naming), and ``setup()`` is called at every publication point so a Task
    that attaches late still gets its service. Both halves — C++ and Python
    — therefore present the identical MTC API surface.
    """

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
        """Create the publishers + this task's GetSolution service.

        What ``IntrospectionPrivate``'s CONSTRUCTOR does in MTC
        (introspection.cpp:112-131): own node, publishers, service,
        ``indicateReset()``, ``resetMaps()``.

        MTC gives each Introspection its OWN node. That is safe there because
        a Task is long-lived; our pipeline builds a fresh Task for every
        motion of every cycle, so a node per Task leaks a node per motion —
        and two nodes registering a rosout publisher for the same logger name
        produce rclcpp's "Publisher already registered for node name" warning
        and pin the publishers after the owner is gone.

        So: the node is PROCESS-SHARED (one per process, on MTC's shared
        executor), while the publishers are shared per node AND the
        ``get_solution_<task_id>`` service stays PER TASK — MTC's naming, MTC's
        isolation, one node total.
        """
        task = self._task
        if not task._introspection_enabled:
            return False
        if task._pub_desc is not None and task._srv_get_solution is not None:
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
            from rclpy.qos import (
                DurabilityPolicy,
                HistoryPolicy,
                QoSProfile,
                ReliabilityPolicy,
            )

            # MTC introspection.cpp:118-126 rendezvous, verbatim (depths 2/1/1,
            # transient_local). Topics are ns-aware: default
            # /curobo_task_constructor/*, or <Task ns>/* when constructed
            # with Task(ns=...).
            for attr, msg_type, topic_fn, depth in (
                ("_pub_desc", TaskDescription, task.description_topic, 2),
                ("_pub_stat", TaskStatistics, task.statistics_topic, 1),
                ("_pub_sol", Solution, task.solution_topic, 1),
            ):
                if getattr(task, attr) is None:
                    setattr(task, attr, node.create_publisher(
                        msg_type, topic_fn(),
                        QoSProfile(
                            depth=depth, history=HistoryPolicy.KEEP_LAST,
                            reliability=ReliabilityPolicy.RELIABLE,
                            durability=DurabilityPolicy.TRANSIENT_LOCAL)))
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

        The NODE is shared and outlives any one task, so only the handles
        this task owns are torn down: its ``get_solution_<task_id>`` service
        stops answering (removed from the shared registry only if it is
        still ours — a successor task may already have taken the name) and
        its publisher handles are released.
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
        """Publish ONE solution/failure (MTC ``Introspection::publishSolution``).

        This is the entire responsibility: the Solution message goes to the
        ``solution`` topic. RENDERING it is the consumer's job — in MTC the
        RViz Display subscribes to this topic, and our panel does the
        equivalent. Nothing is published anywhere else from here.
        """
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


class Task:
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
        """MTC ``Task::enableIntrospection``.

        Enabling creates the introspection object; disabling releases it —
        MTC resets its ``unique_ptr`` (task.cpp:159-174) — so a disabled
        task's ``get_solution_<task_id>`` service stops answering and its
        node leaves the shared executor.

        The release has to happen HERE, on the Task: dropping the reference
        alone leaks the node, its executor slot, and with it the shared
        spinning thread, because ``Introspection`` never learns it was
        disowned.
        """
        if enabled:
            if self._introspection is None:
                self._introspection_enabled = True
                self._introspection = Introspection(self)
        elif self._introspection is not None:
            introspection = self._introspection
            self._introspection_enabled = False
            self._introspection = None
            introspection._release()

    def getRobotModel(self) -> Any:
        """Get the robot model (MTC ``Task.getRobotModel``).

        MTC returns a ``moveit::core::RobotModel``, because MTC runs the
        kinematics IN-PROCESS against it — IK (``compute_ik.cpp:252``),
        Cartesian interpolation (``cartesian_path.cpp:105``), joint models
        and collision checks are all client-side.

        cuRobo runs none of that here: the URDF, the collision-sphere model
        and the kinematics live in ``curobo_server``, so the client-side
        object playing that role is the ``RobotInterface``. It is returned
        by identity (never a copy): a ``None`` result means "no interface
        yet", which is the state that makes ``init()`` raise.
        """
        return self.robot

    def setRobotModel(self, robot_model: Any) -> None:
        """Set the robot model (MTC ``Task.setRobotModel``).

        MTC accepts a ``RobotModelConstPtr``; here it is the
        ``RobotInterface`` (anything ``init()`` can drive), so the
        curobo-shaped call is ``attach_robot`` — this is MTC's spelling
        for it.
        """
        self.attach_robot(robot_model)

    def introspection(self) -> Any:
        """Access introspection object (MTC ``Task.introspection``).

        MTC's ``Task::introspection()`` (task.cpp:176-180) calls
        ``enableIntrospection(true)`` first, so reaching for it both creates
        and enables it — which is exactly what the ``Task.publish`` Python
        binding relies on (``core/python/bindings/src/core.cpp:481``). Created
        on first use and owned by the Task.
        """
        if self._introspection is None:
            self.enableIntrospection(True)
        return self._introspection

    def loadRobotModel(self, robot_description: str = "robot_description") -> None:
        """Point this task at the ROBOT DESCRIPTOR (curobo's loadRobotModel).

        MTC reads the ``robot_description`` parameter into a
        ``RobotModelLoader`` and throws when that fails
        (``task.cpp:138-144``), because without the model there are no
        kinematics to solve with. cuRobo has no client-side kinematics to
        load — but the core still reads two things from the robot
        DESCRIPTOR itself:

        * ``kinematics.cspace.joint_names`` — the canonical cspace order
          that ``CuroboServerInterface.normalize_joint_state`` reorders
          ``/joint_states`` into before anything reaches the server. This
          is the one piece of client-side model knowledge, and it is
          load-bearing: without it every start state is shifted by one
          joint (joint_4 reads joint_3's value, the finger reads
          joint_7's) and plans fail against limits that are not violated.
        * ``named_joint_configs`` — resolved by the ``FixedState`` stage.

        So this resolves the descriptor and gives it to the interface,
        which is the work that actually has to happen client-side. It
        never fabricates a request: a caller that follows MTC's
        ``task.loadRobotModel(node)`` pattern therefore still ends up with
        the joint order, which a silent no-op would have dropped.
        """
        path = self._resolve_robot_descriptor(robot_description)
        if path is None:
            return
        if self.robot is None:
            # MTC's loadRobotModel builds the loader; here the robot is a
            # RobotInterface, and it needs a node (which the introspection
            # owns). Without one there is nothing to point at.
            node = self._introspection_node
            if node is None:
                try:
                    import rclpy

                    node = rclpy.create_node(f"introspection_{self.task_id}")
                    _IntrospectionExecutor.add_node(node)
                except Exception:
                    node = None
            if node is None:
                return
            from curobo_task_constructor.robot.curobo import (
                CuroboServerInterface,
            )
            self.attach_robot(CuroboServerInterface(node,
                                                   robot_config_path=path))
            return
        # Interface already attached: give it the descriptor it was built
        # without, and re-derive the joint order (once).
        if getattr(self.robot, "_robot_config_path", None):
            return
        self.robot._robot_config_path = str(path)
        try:
            from curobo_task_constructor.core.robot_config import (
                canonical_joint_order,
            )
            self.robot._joint_order = canonical_joint_order(path)
        except Exception:
            self.robot._joint_order = None

    def _resolve_robot_descriptor(self, robot_description: Any) -> Any:
        """Descriptor path for ``loadRobotModel``, or None when unknown.

        Order: an explicit filesystem path in the argument wins; else the
        robot parameter on our introspection node (or the caller's, if one
        is still attached); else None (a caller with no descriptor anywhere
        gets the server default, exactly as before).
        """
        if isinstance(robot_description, str) and (
                "/" in robot_description or robot_description.endswith((".yml", ".yaml"))):
            return robot_description
        for node in (self._introspection_node, getattr(self, "_node", None)):
            if node is None:
                continue
            for name in ("robot_config_path", "robot"):
                try:
                    if not node.has_parameter(name):
                        continue
                    value = node.get_parameter(name).value
                except Exception:
                    continue
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    def setCostTerm(self, *args, **kwargs) -> None:
        """Specify a CostTerm for calculation of stage costs (MTC ``Task.setCostTerm``).

        cuRobo stages carry their own cost terms via ``setCost``; this
        stores the task-level term for inspection.
        """
        if args:
            self._properties["cost_term"] = args[0]

    # -- introspection plumbing (MTC Introspection-backed) --------------
    def _register_solution(self, obj: Any) -> int:
        """Global solution id (delegates to ``Introspection::solutionId``)."""
        return self.introspection().solutionId(obj)

    def _solution_from_id(self, solution_id: int) -> Any:
        return self.introspection().solutionFromId(solution_id)

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
        """Build and initialize the task (MTC ``Task::init``).

        MTC does, in this order (task.cpp:201-229):
          #. require a robot model (else ``throw``);
          #. build + init the stage tree, resolving the GENERATE interface;
          #. hand the introspection instance to every stage so
             ``Stage::onNewSolution`` can publish live;
          #. publish the TaskDescription ("first time publish task").

        This mirrors all four. In particular the description publish lives
        HERE, not in ``plan()`` — MTC's ``plan()`` merely calls ``init()``
        and then runs the compute loop, so a caller that calls ``init()``
        alone (as demo_grasp's mtc_manager.cpp:46 does) still gets the task
        onto the panel.
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

        # MTC Task::init step 3 + 4: give the stages the introspection
        # instance (their onNewSolution publishes during compute), then
        # publish the task description once.
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
        """Reset, init, and plan (MTC ``Task.plan``).

        ``max_solutions`` (0 = unlimited) bounds the number of solutions
        collected. Returns MoveItErrorCode (SUCCESS/FAILURE-PLANNING_FAILED)
        exactly like MTC; truthy on success so ``assert task.plan()`` keeps
        working.

        Like MTC, every compute pass publishes the task statistics
        (``publishTaskState``) and every new solution/failure is published
        as found, when introspection is live (node attached + enabled).
        """
        # NOTE (cuRobo backend limit): MTC plan(max_solutions) collects up to N
        # solutions. Our executor stops at the first complete root solution
        # (re-solving the trailing chain per reconnect once wedged the
        # server's CUDA-graph capture), so max_solutions only bounds the
        # compute loop; use fallbacks/alternatives for variant coverage.
        self.reset()
        if self._preempted:
            self._preempted = False
            return MoveItErrorCode(MoveItErrorCode.FAILURE)
        self.init()
        if self._executor is None or not self._executor._valid:
            return MoveItErrorCode(MoveItErrorCode.PLANNING_FAILED)
        ex = self._executor
        # MTC Task::plan (task.cpp:243-272) is: init() + compute loop (no
        # description publish, no hook install here - init() did both).
        introspect = self.introspection()

        def _guarded_progress():
            if self._preempted:
                raise Task._Preempted()
            if progress_callback is not None:
                progress_callback()

        def _progress():
            _guarded_progress()
            # MTC Task::plan publishes task state after every compute pass.
            introspect.publishTaskState()

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
        introspect.publishTaskState()
        if ok and self._solutions:
            return MoveItErrorCode(MoveItErrorCode.SUCCESS)
        return MoveItErrorCode(MoveItErrorCode.PLANNING_FAILED)

    def execute(self, solution: Any = None):
        """Send given solution for execution (MTC ``Task.execute``).

        Exactly like MoveIt: the full Solution message goes to the
        ``execute_task_solution`` action (served by the task-constructor
        node next to the curobo_server), which drives its sub-trajectories
        in order with no replanning.

        Like MTC's ``Task::execute`` (task.cpp:282-339) this is BLOCKING and
        returns a MoveItErrorCode: it creates its own private node and
        executor for the action round trip (``execute_solution_node_`` in
        C++), so no caller has to supply — or keep spinning — a node. A
        preempt request cancels the goal, as upstream.

        Without rclpy up (plain unit tests, offline scripts) the solution
        drives directly through the local executor instead and a
        MoveItErrorCode returns (truthy on success).
        """
        sol = solution if solution is not None else self.best()
        if sol is None:
            raise RuntimeError("Task.execute(): no solution")
        try:
            import rclpy
        except Exception:
            rclpy = None
        if rclpy is None or not rclpy.ok():
            # Offline path (no ROS): drive directly through the executor.
            if self._executor is None:
                raise RuntimeError("Task.execute() before plan()")
            results = self._executor.execute(sol)
            ok = bool(results) and all(r.success for r in results)
            return MoveItErrorCode(
                MoveItErrorCode.SUCCESS if ok else MoveItErrorCode.CONTROL_FAILED)
        return self._execute_remote(sol)

    def _execute_remote(self, solution: Any):
        """Send a full Solution message through the ExecuteTaskSolution action.

        MTC's ``Task::execute`` (task.cpp:282-330) drives the action on a
        PRIVATE node and a LOCAL single-threaded executor that the CALLING
        thread services with ``spin_once()``, waits 0.5 s for the action
        server and then blocks for the result, honouring ``preempt()`` as a
        goal cancel. Same shape here.

        Reusing the shared introspection executor instead would put a
        background ``spin()`` and this thread's spins on ONE executor, which
        raises "Executor is already spinning" - and that exception made the
        orchestrator believe the drive had failed while the trajectory was
        actually starting, so it issued a fresh plan mid-execution and took
        the server down with a CUDA illegal memory access.
        """
        import rclpy
        from rclpy.action.client import ActionClient
        from rclpy.executors import SingleThreadedExecutor
        from curobo_task_constructor_interfaces.action import (
            ExecuteTaskSolution,
        )
        from curobo_task_constructor import msg_convert

        if isinstance(solution, dict):
            d = solution
        elif self._executor is not None:
            d = self._executor.solution_to_dict(solution)
        else:
            raise RuntimeError(
                "Task.execute(): cannot serialize solution without executor")
        goal = ExecuteTaskSolution.Goal()
        goal.solution = msg_convert.solution_to_msg(d)

        # Private node + LOCAL executor, driven by THIS thread only (see the
        # spin invariant in _IntrospectionExecutor.add_node). Returned by a
        # factory so nothing can ever hand _execute_remote the shared, already
        # spinning executor again.
        node, executor = _private_action_executor()
        try:
            # Ns-aware rendezvous (MTC relative action name; default keeps the
            # absolute /curobo_task_constructor/... for compat).
            client = ActionClient(
                node, ExecuteTaskSolution, self.execute_action_name())

            # Bounded wait (MTC: 0.5s) - an execution backend that is not up
            # must fail the call, never hang the caller's thread.
            if not client.wait_for_server(
                    timeout_sec=rclpy.duration.Duration(
                        seconds=EXECUTE_WAIT_SEC)):
                return MoveItErrorCode(MoveItErrorCode.FAILURE)

            goal_future = client.send_goal_async(goal)
            if not self._spin_until(executor, goal_future):
                return MoveItErrorCode(MoveItErrorCode.FAILURE)
            goal_handle = goal_future.result()
            if goal_handle is None or not goal_handle.accepted:
                return MoveItErrorCode(MoveItErrorCode.FAILURE)

            result_future = goal_handle.get_result_async()
            while not result_future.done():
                if self._preempted:
                    cancel_future = client.async_cancel_goal(goal_handle)
                    self._preempted = False
                    self._spin_until(executor, cancel_future)
                    return MoveItErrorCode(MoveItErrorCode.PREEMPTED)
                # rclpy has no spin_some() - that is a C++ rclcpp method
                # (MTC's Task::execute calls executor.spin_some()). The
                # rclpy equivalent for "service this once, then check" is
                # spin_once(), which is safe here because this executor is
                # private and driven by this thread only.
                executor.spin_once(timeout_sec=0.01)
            raw = result_future.result()
            # Mirror MTC: surface the server's MoveItErrorCodes value, default
            # CONTROL_FAILED when the wrapper only yields a bool.
            code = getattr(getattr(raw, "result", raw), "error_code", None)
            val = getattr(code, "val", None)
            if val is not None:
                try:
                    return MoveItErrorCode(int(val))
                except (TypeError, ValueError):
                    pass
            ok = _result_ok(raw)
            return MoveItErrorCode(
                MoveItErrorCode.SUCCESS if ok else MoveItErrorCode.CONTROL_FAILED)
        finally:
            try:
                executor.remove_node(node)
                node.destroy_node()
            except Exception:
                pass

    @staticmethod
    def _spin_until(executor, future) -> bool:
        """Spin ``executor`` until ``future`` completes; False on timeout."""
        import time as _time

        deadline = _time.monotonic() + EXECUTE_WAIT_SEC
        while not future.done() and _time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.01)
        return future.done()

    def publish(self, solution: Any) -> None:
        """Publish ONE solution (MTC ``Task.publish`` Python binding).

        MTC binds ``Task.publish`` to *exactly*
        ``self.introspection().publishSolution(*solution)``
        (``core/python/bindings/src/core.cpp:481``), documented as
        "Publish the given solution to the ROS topic ``solution``" — nothing
        else. The RViz side subscribes to that topic, fetches the payload and
        renders the trajectory, so publishing the message IS how the solution
        is shown. No second publish on another topic happens here: that would
        be the consumer's job, done twice.
        """
        self.last_published = solution
        if not self._introspection_enabled:
            return
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

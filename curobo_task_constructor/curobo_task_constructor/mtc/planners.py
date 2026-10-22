
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
#:
#: ``cartes``/``cartesian`` map to CARTESIAN (the MTC ``solvers::CartesianPath``
#: backend, SetPlanner=3) — the straight-line planner — NOT to CLASSIC. A
#: ``PipelinePlanner(node, "cartes")`` therefore means "plan linearly", the
#: same job MTC's CartesianPath solver does.
PIPELINE_PLANNERS = {
    "classic": PLANNER_CLASSIC,
    "cartes": PLANNER_CARTESIAN,
    "cartesian": PLANNER_CARTESIAN,
    "cartesian_path": PLANNER_CARTESIAN,
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
    """Straight-line Cartesian planning (MTC ``solvers::CartesianPath``). A faithful port of MoveIt Task Constructor's ``solvers::CartesianPath``. MTC's solver interpolates a straight EEF path from the start to the goal
    at ``step_size``, IK's + collision-checks every waypoint, and reports the
    fraction of the path it reached."""

    wire_name = "cartesian"

    def __init__(
        self,
        step_size: float = 0.01,
        precision: CartesianPrecision = None,
    ):
        super().__init__()
        self.step_size = float(step_size)
        self.precision = precision or CartesianPrecision()
        #: Max Cartesian speed cap (m/s); 0 = unlimited (MTC
        #: ``max_cartesian_speed``). Carried for parity; the cuRobo backend
        #: times its own trajectory.
        self.max_cartesian_speed = 0.0
        #: Link whose speed the cap applies to (MTC
        #: ``cartesian_speed_limited_link``).
        self.cartesian_speed_limited_link = ""

    def setStepSize(self, step_size: float) -> None:
        """Set the step size between consecutive waypoints (m)."""
        self.step_size = float(step_size)
        self.setProperty("step_size", self.step_size)

    def setPrecision(self, precision: CartesianPrecision) -> None:
        """Set the precision of the linear path."""
        self.precision = precision
        self.setProperty("precision", precision)

    def setIKFrame(self, *args) -> None:
        """Set the frame to move linearly (MTC ``CartesianPath::setIKFrame``). Accepts a link name, or ``(pose[, link])``. Pose-only forms are
        reduced to a stored property; a trailing link string wins."""
        for arg in args:
            if isinstance(arg, str):
                self.setProperty("ik_frame", arg)
                return
        if args:
            self.setProperty("ik_frame", args[-1])

    def setJumpThreshold(self, *args) -> None:
        """Deprecated no-op (MTC ``CartesianPath::setJumpThreshold``).

        MTC deprecated the jump threshold: the linear interpolator no longer
        aborts on a joint-space jump. Accepted for call-site parity.
        """

    def setMaxVelocityScaling(self, *args) -> None:
        """Deprecated alias for ``setMaxVelocityScalingFactor`` (MTC)."""
        if args:
            self.setMaxVelocityScalingFactor(args[0])

    def setMaxAccelerationScaling(self, *args) -> None:
        """Deprecated alias for ``setMaxAccelerationScalingFactor`` (MTC)."""
        if args:
            self.setMaxAccelerationScalingFactor(args[0])

    def getPlannerId(self) -> str:
        return "CartesianPath"

    def planner_key(self) -> int:
        # Straight-line planning is its own backend (SetPlanner CARTESIAN).
        return PLANNER_CARTESIAN

    def extra_params(self) -> dict:
        params = dict(self._planner_properties)
        params.pop("timeout", None)
        params.setdefault("step_size", self.step_size)
        # Translational precision (m): the tolerance under which an interpolated
        # waypoint counts as ON the line. ``self.precision`` is the source of
        # truth even when ``setPrecision`` stored the CartesianPrecision object
        # in the property bag (MTC parity), so coerce it back to a float here.
        params["precision"] = float(self.precision.translational)
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
    """Plan using a planning pipeline (MTC PipelinePlanner). ``num_planning_attempts`` is honoured: when a stage is built with this
    planner and no explicit attempt count, the stage plans that many times
    and keeps the cheapest result. ``pipeline``/``planner_id`` select the
    server backend; ``None`` leaves the server default."""

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
    """Per-group planner configs (``moveit_wrapper_task_constructor/params/jaco2.yaml`` shape). Input ``{planning_groups: [...], <group>: {planning_time,
    max_velocity_scaling_factor, max_acceleration_scaling_factor,
    planning_attempts, planner_id, planning_pipeline_id}}`` or a YAML file
    path with the same content. Empty ``planner_id`` means the pipeline
    default (MTC: OMPL default from ``ompl_planning.yaml``)."""
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

__all__ = [
    'CartesianPrecision',
    'PlannerInterface',
    '_looks_like_node',
    'PIPELINE_PLANNERS',
    'CartesianPath',
    'JointInterpolationPlanner',
    'PipelinePlanner',
    'MultiPlanner',
    'load_planner_configs',
]

"""MTC-shaped stage wrappers (``stages`` namespace). Mirrors ``moveit.task_constructor.stages`` constructor/setter shape::

    move = stages.MoveRelative("x +0.2", cartesian)
    move.group = "arm"
    move.setDirection(Vector3Stamped(...))

Method names match MTC (``setGoal``/``setDirection``/``attachObject``/…);
each wrapper converts itself to a ``StageSpec`` node (same ``params_yaml``
schema the registry stages parse), so the wire format never changes and
``Task.plan()`` can drive the existing ``TaskExecutor`` locally. Planners may
be solver objects (``core.CartesianPath``), wire strings (``"classic"`` /
``"joint_space"``) or ``SetPlanner`` ints — the string form is what the pick
pipeline's trees carry on the wire."""

from __future__ import annotations

import math
from typing import Any



from curobo_task_constructor.mtc.stage_util import *

class _PropertyMap(dict):
    """Stage PropertyMap (MTC): a dict with ``configureInitFrom``.

    Property inheritance is resolved at authoring time in cuRobo (builders
    carry explicit values), so this only records the declaration for
    inspection.

    Calling it returns itself, so both the Python-binding shape
    (``stage.properties``) and the C++ shape
    (``stage.properties().configureInitFrom(...)``) work.
    """

    def __call__(self) -> "_PropertyMap":
        return self

    def configureInitFrom(self, source: Any, names=None) -> None:
        """Declare init-from sources (MTC ``PropertyMap::configureInitFrom``).

        ``source`` is a PropertyInitializerSource flag (PARENT/INTERFACE);
        ``names`` limits it to listed properties (empty = all).
        """
        self.setdefault("_init_from", []).append(
            {"source": source, "names": list(names or [])}
        )


class _StageBase:
    stage_type = ""
    is_generator = False

    def __init__(self, name: str = "", planner: Any = None):
        self._name = name or self.stage_type
        self.planner = planner
        self.group: str = ""
        self.params: _PropertyMap = _PropertyMap()
        self._timeout: float = 0.0
        self._marker_ns: str = ""
        self._forwarded_properties: list = []

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def timeout(self) -> float:
        """Maximally allowed time [s] per computation step (MTC)."""
        return self._timeout

    @timeout.setter
    def timeout(self, value: float) -> None:
        self._timeout = float(value)

    @property
    def marker_ns(self) -> str:
        """Namespace for any markers associated to the stage (MTC)."""
        return self._marker_ns

    @marker_ns.setter
    def marker_ns(self, value: str) -> None:
        self._marker_ns = str(value)

    @property
    def forwarded_properties(self) -> list:
        """Set of properties forwarded from input to output InterfaceState (MTC)."""
        return self._forwarded_properties

    @property
    def properties(self) -> dict:
        """PropertyMap of the stage (MTC, read-only).

        cuRobo stores params in a PropertyMap dict; ``configureInitFrom``
        declarations ride along for inspection.
        """
        return self.params

    def setProperty(self, name: str, value: Any) -> None:
        """Set a stage property (MTC ``Stage::setProperty``)."""
        self.params[str(name)] = value

    def setGroup(self, group: str) -> None:
        """Set the planning group (MTC ``Stage::setGroup``)."""
        self.group = str(group)

    def setTimeout(self, timeout: float) -> None:
        """Set the planning timeout (MTC ``Stage::setTimeout``)."""
        self.timeout = float(timeout)

    @property
    def solutions(self) -> list:
        """Successful Solutions of the stage (MTC, read-only).

        Populated after plan(); empty before planning.
        """
        return []

    @property
    def failures(self) -> list:
        """Failed Solutions of the stage (MTC, read-only).

        Populated after plan(); empty before planning.
        """
        return []

    def setCostTerm(self, *args, **kwargs) -> None:
        """Specify a CostTerm for calculation of stage costs (MTC).

        Accepts a ranking-term string (cuRobo ``setCost``) or a CostTerm
        object (MTC ``cost::PathLength``/``LinkMotion``/``Clearance``/...),
        which maps onto the closest cuRobo ranking term; the original is
        kept under ``cost_term`` for inspection.
        """
        if not args:
            return
        term = args[0]
        if isinstance(term, str):
            known = getattr(_MotionBase, "COST_TERMS", ())
            if known and term not in known:
                raise ValueError(f"unknown cost term {term!r}; "
                                 f"expected one of {list(known)}")
            self.params["cost"] = term
            return
        name = type(term).__name__
        mapped = "path_length" if "PathLength" in name else "auto"
        self.params["cost"] = mapped
        # Kept off params (params must stay YAML-serializable for the
        # StageSpec); the backend ranks on the mapped term.
        self._cost_term_obj = term

    def init(self, robot_model: Any = None) -> None:
        """Initialize the stage once before planning (MTC ``Stage.init``).

        cuRobo stages are initialized by the executor; this is a no-op
        for API parity.
        """
        pass

    def reset(self) -> None:
        """Reset the Stage (MTC ``Stage.reset``).

        cuRobo stages are reset by the executor; this is a no-op.
        """
        pass

    def _planner_params(self) -> dict:
        """Planner-derived stage params (MTC: the planner config wins over
        stage defaults, so it is what reaches the wire and the panel)."""
        out: dict = {}
        wire = _planner_name(self.planner)
        if wire is not None:
            out["planner"] = wire
        extra = getattr(self.planner, "extra_params", None)
        if callable(extra):
            try:
                out.update(extra() or {})
            except Exception:
                pass
        # Which planner actually backs this stage (MTC getPlannerId), so the
        # panel shows the planner behind each row instead of guessing from
        # the wire key.
        get_id = getattr(self.planner, "getPlannerId", None)
        if callable(get_id) and self.planner is not None:
            try:
                planner_id = get_id()
            except Exception:
                planner_id = ""
            if planner_id:
                out.setdefault("planner_id", str(planner_id))
        return out

    def _stage_timeout(self) -> float:
        """Configured timeout [s] (0 = unbounded).

        MTC resolves the timeout on the planner (``planner->setTimeout``) and
        the stage (``stage->setTimeout``); the stage's explicit value wins,
        else the planner's. Both are propagated: an unset timeout is the
        single most common cause of an MTC stage silently planning forever,
        so it is never left implicit.
        """
        if self._timeout > 0:
            return self._timeout
        props = getattr(self.planner, "properties", None)
        if callable(props) and self.planner is not None:
            try:
                return float(props().get("timeout", 0.0) or 0.0)
            except Exception:
                return 0.0
        return 0.0

    def _common_params(self) -> dict:
        params = dict(self.params)
        params.update(self._planner_params())
        if self.group and "group" not in params:
            params["group"] = self.group
        timeout = self._stage_timeout()
        if timeout:
            params.setdefault("timeout", timeout)
        return params

    def to_spec(self):
        from curobo_task_constructor.graph.spec import StageSpec
        return StageSpec(stage_type=self.stage_type, name=self.name,
                         container_type="", children=[],
                         params_yaml=_dump_params(self._common_params()))

    def to_dict(self) -> dict:
        return self.to_spec().to_dict()


class _MotionBase(_StageBase):
    """Motion stages carry planning control (MTC ``num_planning_attempts``).

    Ranking is explicit, never a hidden default: every motion stage declares
    its cost term (MTC ``cost::PathLength`` et al) via ``setCost``. The
    builders in the pick pipeline set ``path_length`` on each motion leg;
    generators cost 0 and containers accumulate their children, so neither
    carries a term.
    """

    #: Ranking terms understood by ``stages._util.cost_of``.
    COST_TERMS = ("auto", "path_length", "waypoints", "solver_cost", "inf")

    def __init__(self, name: str = "", planner: Any = None,
                 planning_attempts: int = None):
        super().__init__(name, planner)
        # MTC configures attempts on the planner (PipelinePlanner.
        # num_planning_attempts), not the stage: an explicit count wins,
        # else the planner's, else 3.
        self.planning_attempts = (
            None if planning_attempts is None else int(planning_attempts))

    def setCost(self, term: str) -> None:
        """Declare the ranking term explicitly (default ``auto``)."""
        if term not in self.COST_TERMS:
            raise ValueError(f"unknown cost term {term!r}; "
                             f"expected one of {list(self.COST_TERMS)}")
        self.params["cost"] = term

    def _common_params(self) -> dict:
        params = super()._common_params()
        attempts = self.planning_attempts
        if attempts is None:
            attempts = getattr(self.planner, "num_planning_attempts", None)
        params["planning_attempts"] = int(
            attempts if attempts is not None else 3)
        return params

__all__ = [
    '_PropertyMap',
    '_StageBase',
    '_MotionBase',
]

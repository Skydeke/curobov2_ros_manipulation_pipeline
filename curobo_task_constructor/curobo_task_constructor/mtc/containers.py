
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

__all__ = [
    'ContainerBase',
    'SerialContainer',
    'Fallbacks',
    'Alternatives',
    'IndependentComponents',
    'Merger',
]

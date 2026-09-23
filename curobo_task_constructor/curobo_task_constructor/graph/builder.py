"""Build a stage tree from a declarative ``StageSpec`` (Sec. 1/2 of the plan).

The wire format is generic: containers are created from
``spec.container_type`` (an open map below), leaf/wrapper stages from
``spec.stage_type`` through the open ``STAGE_REGISTRY``. The builder never
needs to know a stage type by name — a stage declares children by accepting
them (a ``ContainerStage`` via ``add``, a wrapper such as ``ComputeIK`` via
``set_child``); anything else with children that cannot take them is a
validation error at build time.

The root of a task may be a container (the common case — the StageSpec root
of Task.action is normally a ``SerialContainer``) or a single stage.
"""

from __future__ import annotations

from typing import Optional

from curobo_task_constructor.core.container import (
    Alternatives,
    ContainerStage,
    Fallbacks,
    IndependentComponents,
    SerialContainer,
)
from curobo_task_constructor.core.registry import create_stage
from curobo_task_constructor.core.stage import InitStageError, Stage
from curobo_task_constructor.graph.spec import StageSpec, params_from_yaml

__all__ = ["CONTAINERS", "build_tree", "stage_from_spec"]

#: container_type -> ContainerStage subclass. Open (anyone may add entries),
#: mirroring the STAGE_REGISTRY openness for leaf stages.
CONTAINERS: dict = {
    "serial": SerialContainer,
    "alternatives": Alternatives,
    "fallbacks": Fallbacks,
    "independent": IndependentComponents,
}


def stage_from_spec(spec: StageSpec) -> Stage:
    """Instantiate one spec node (no children attached)."""
    name = spec.name or None
    params = params_from_yaml(spec.params_yaml)
    if spec.container_type:
        try:
            cls = CONTAINERS[spec.container_type]
        except KeyError:
            raise ValueError(
                f"unknown container_type {spec.container_type!r}; "
                f"known: {sorted(CONTAINERS)}") from None
        return cls(name=name, params=params)
    return create_stage(spec.stage_type, name=name, params=params)


def build_tree(spec: StageSpec, parent: Optional[Stage] = None) -> Stage:
    """Recursively build the stage tree described by ``spec``.

    Raises ``ValueError``/``InitStageError`` for malformed graphs — the same
    errors the executor surfaces as ``TaskDescription.valid == false`` (the
    task is rejected before any solve).
    """
    spec.validate()
    stage = stage_from_spec(spec)
    if parent is not None:
        stage.parent = parent

    is_wrapper = not spec.container_type and hasattr(stage, "set_child")
    if is_wrapper and len(spec.children) > 1:
        raise InitStageError(
            stage.name,
            f"wrapper stage '{spec.stage_type}' accepts exactly one child, "
            f"got {len(spec.children)}")

    for child_spec in spec.children:
        child = build_tree(child_spec, parent=stage)
        if isinstance(stage, ContainerStage):
            stage.add(child)
        elif is_wrapper:
            stage.set_child(child)
        else:
            raise InitStageError(
                stage.name,
                f"stage type '{spec.stage_type}' cannot take children "
                "(only containers and set_child wrappers can)")
    return stage


def build_from_yaml(text: str) -> Stage:
    """Convenience: parse a StageSpec YAML document and build its tree."""
    return build_tree(StageSpec.from_yaml(text))
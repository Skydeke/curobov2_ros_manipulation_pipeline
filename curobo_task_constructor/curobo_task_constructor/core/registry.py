"""The open stage registry (MTC's ``StageRegistry`` equivalent).

Task graphs reference stage types by string name (``StageSpec.stage_type``),
resolved here — never through a closed enum. Adding a capability is exactly
"write a ``Stage`` subclass, decorate it, done":

    @register_stage("move_to")
    class MoveTo(stages.PropagatingEitherWay): ...

That is the whole extension surface; the executor, the graph builder and the
ROS wiring never need to change for a new stage type. This mirrors the
`CostManager`/`BaseCost.register` extension idiom cuRoboV2 already uses for
its cost terms.
"""

from __future__ import annotations

from typing import Callable, Type

from curobo_task_constructor.core.stage import Stage

STAGE_REGISTRY: dict = {}


def register_stage(name: str) -> Callable[[Type[Stage]], Type[Stage]]:
    """Class decorator registering a stage type under ``name``."""

    def deco(cls: Type[Stage]) -> Type[Stage]:
        if not (isinstance(cls, type) and issubclass(cls, Stage)):
            raise TypeError(f"{cls!r} is not a Stage subclass")
        if name in STAGE_REGISTRY:
            raise ValueError(f"stage type {name!r} already registered by "
                             f"{STAGE_REGISTRY[name]}")
        STAGE_REGISTRY[name] = cls
        cls._registry_name = name
        return cls

    return deco


def get_stage_class(stage_type: str) -> Type[Stage]:
    try:
        return STAGE_REGISTRY[stage_type]
    except KeyError:
        known = ", ".join(sorted(STAGE_REGISTRY))
        raise KeyError(f"unknown stage type {stage_type!r}; "
                       f"registered: {known or '<none>'}") from None


def create_stage(stage_type: str, name: str = None, params: dict = None) -> Stage:
    """Instantiate a registered stage. ``name``/``params`` are optional —
    callers usually let the graph builder supply them from the StageSpec."""
    cls = get_stage_class(stage_type)
    kwargs = {}
    if name is not None:
        kwargs["name"] = name
    if params is not None:
        kwargs["params"] = params
    return cls(**kwargs)
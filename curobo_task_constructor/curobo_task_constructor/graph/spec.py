"""Declarative task description — the ``StageSpec`` tree.

One node per graph element, generic across every registered stage type.
Stage-specific parameters ride in ``params_yaml`` (an opaque string the
stage class itself parses) — that openness is what keeps the format stable
while the stage catalog grows. Planning is local (in-process), so this
tree never crosses the wire; introspection describes stages with
StageDescription/Property messages instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:  # keep the core importable without yaml in exotic envs
    import yaml
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class StageSpec:
    stage_type: str
    name: str = ""
    container_type: str = ""  # "" | serial | alternatives | fallbacks | independent
    children: list = field(default_factory=list)  # list[StageSpec]
    params_yaml: str = ""
    #: Interface flags (MTC StageDescription.flags bits); 0 when unknown
    #: (authoring time) — the node fills live flags from resolved stages.
    flags: int = 0

    # -- format converters --------------------------------------------
    def to_dict(self) -> dict:
        return {
            "stage_type": self.stage_type,
            "name": self.name,
            "container_type": self.container_type,
            "children": [c.to_dict() for c in self.children],
            "params_yaml": self.params_yaml,
            "flags": int(self.flags),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StageSpec":
        return cls(
            stage_type=d.get("stage_type", ""),
            name=d.get("name", ""),
            container_type=d.get("container_type", ""),
            children=[cls.from_dict(c) for c in d.get("children", []) or []],
            params_yaml=d.get("params_yaml", ""),
            flags=int(d.get("flags", 0) or 0),
        )

    def to_yaml(self) -> str:
        if yaml is None:
            raise RuntimeError("PyYAML is required to serialize a StageSpec")
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_yaml(cls, text: str) -> "StageSpec":
        if yaml is None:
            raise RuntimeError("PyYAML is required to parse a StageSpec")
        return cls.from_dict(yaml.safe_load(text) or {})

    # -- flat pre-order traversal ---------------------------------------
    # (kept for tree utilities; the tree itself never crosses the wire —
    # introspection describes stages with StageDescription/Property).
    def _preorder(self) -> list:
        """This tree in pre-order (root first, parents before children)."""
        out = []

        def visit(node: "StageSpec") -> None:
            out.append(node)
            for child in node.children:
                visit(child)

        visit(self)
        return out

    @property
    def is_container(self) -> bool:
        return bool(self.container_type)

    def validate(self) -> None:
        """Structural checks independent of the registry."""
        if self.is_container and not self.children:
            raise ValueError(f"container '{self.name or self.stage_type}' "
                             "requires at least one child")
        if not self.is_container and self.stage_type == "":
            raise ValueError("stage_type must be non-empty")
        for child in self.children:
            child.validate()


def params_from_yaml(params_yaml: str) -> dict:
    """Parse a stage's opaque params string into a dict."""
    if not params_yaml or not params_yaml.strip():
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to parse stage params")
    data = yaml.safe_load(params_yaml)
    return data if isinstance(data, dict) else {}
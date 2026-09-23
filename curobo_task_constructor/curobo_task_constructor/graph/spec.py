"""Declarative task description — the ``StageSpec`` tree.

Mirror of ``curobo_task_constructor_interfaces/msg/StageSpec.msg``: one node
per graph element, generic across every registered stage type. Stage-specific
parameters ride in ``params_yaml`` (an opaque string the stage class itself
parses) — that openness is what keeps the wire format stable while the stage
catalog grows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

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

    # -- format converters --------------------------------------------
    def to_dict(self) -> dict:
        return {
            "stage_type": self.stage_type,
            "name": self.name,
            "container_type": self.container_type,
            "children": [c.to_dict() for c in self.children],
            "params_yaml": self.params_yaml,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StageSpec":
        return cls(
            stage_type=d.get("stage_type", ""),
            name=d.get("name", ""),
            container_type=d.get("container_type", ""),
            children=[cls.from_dict(c) for c in d.get("children", []) or []],
            params_yaml=d.get("params_yaml", ""),
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

    # -- ROS message conversion -----------------------------------------
    # The wire format is FLAT (StageSpec[] with MTC-style id/parent_id wiring;
    # parent_id == id marks the root — see StageSpec.msg). The python-side tree
    # keeps ``children``; only the msg boundary flattens.
    @classmethod
    def from_msg_list(cls, stages: list) -> "StageSpec":
        """Re-link a flat ``StageSpec[]`` wire list into a tree.

        The list must be in pre-order (a stage's parent always precedes it,
        the root is element 0 with ``parent_id == id``); this mirrors what
        ``to_msg_list`` emits and matches how moveit_task_constructor_msgs
        serializes StageDescription.
        """
        nodes: dict = {}
        roots: list = []
        for idx, s in enumerate(stages):
            sid = int(s.id)
            if sid in nodes:
                raise ValueError(f"duplicate stage id {sid} in task spec")
            nodes[sid] = cls(
                stage_type=str(s.stage_type),
                name=str(s.name),
                container_type=str(s.container_type),
                children=[],
                params_yaml=str(s.params_yaml),
            )
            pid = int(s.parent_id)
            if pid == sid:
                roots.append(nodes[sid])
            elif pid not in nodes:
                raise ValueError(
                    f"stage {sid} ('{s.name}') references parent {pid} that "
                    "has not appeared yet — task spec must be in pre-order "
                    "(root first, parents before children)")
            else:
                nodes[pid].children.append(nodes[sid])
        if len(roots) != 1:
            raise ValueError(
                f"task spec must have exactly one root stage "
                f"(parent_id == id), found {len(roots)}")
        if roots[0] is not nodes[0]:
            raise ValueError("task spec root must be the first element")
        return roots[0]

    def to_msg_list(self, msg_cls: Optional[Any] = None) -> list:
        """Flatten this tree into a list of wire messages, pre-order: the
        root is element 0 (parent_id == id marks it) and every other stage's
        parent_id points at an earlier element. ``msg_cls`` default: a
        duck-typed stand-in so the core stays ROS-free."""
        out = []

        def visit(node: "StageSpec", parent_id: int) -> None:
            stub = _StageSpecStub(node)
            stub.id = len(out)
            stub.parent_id = parent_id
            if msg_cls is None:
                msg = stub
            else:
                # The real ROS message constructor starts blank, so copy every
                # field: without this the wire carries empty strings and
                # consumers get blank stages (rviz Stage/Type columns render
                # empty, and a re-sent goal fails validate()).
                msg = msg_cls()
                msg.id = stub.id
                msg.parent_id = stub.parent_id
                msg.stage_type = stub.stage_type
                msg.name = stub.name
                msg.container_type = stub.container_type
                msg.params_yaml = stub.params_yaml
            out.append(msg)
            for child in node.children:
                visit(child, msg.id)

        visit(self, 0)
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


class _StageSpecStub:
    """Duck-typed StageSpec stand-in for the ROS-free core/tests."""

    def __init__(self, spec: StageSpec):
        self.id = None
        self.parent_id = None
        self.stage_type = spec.stage_type
        self.name = spec.name
        self.container_type = spec.container_type
        self.params_yaml = spec.params_yaml
        self.children = [_StageSpecStub(c) for c in spec.children]


def params_from_yaml(params_yaml: str) -> dict:
    """Parse a stage's opaque params string into a dict."""
    if not params_yaml or not params_yaml.strip():
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to parse stage params")
    data = yaml.safe_load(params_yaml)
    return data if isinstance(data, dict) else {}
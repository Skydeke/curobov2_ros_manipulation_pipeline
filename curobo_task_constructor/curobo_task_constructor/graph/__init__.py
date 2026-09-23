"""Declarative task graph: StageSpec -> executable stage tree.

- ``builder`` — the registry-based ``build_tree`` that turns a StageSpec
  (from YAML, a ROS message or a test fixture) into the concrete stage tree.
- ``spec`` — the StageSpec dataclass itself.
"""

from curobo_task_constructor.graph.builder import build_tree  # noqa: F401
from curobo_task_constructor.graph.spec import (  # noqa: F401
    StageSpec,
    params_from_yaml,
)
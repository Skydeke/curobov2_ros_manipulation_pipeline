"""CurrentState — MTC's seed generator: the robot's live state.

Reads the current joint state from the robot interface and emits it on both
interfaces with the task's base scene. Optional declarative predicates port
the pick&place task's ``PredicateFilter`` use-case without shipping Python
callables through the StageSpec wire format:

- ``require_not_attached: ["object"]``  — refuse when the object is already
  attached (MTC's "object already attached and cannot be picked" filter).
- ``require_attached: ["name"]``       — require the object IS attached.
"""

from __future__ import annotations

from curobo_task_constructor.core.registry import register_stage
from curobo_task_constructor.core.stage import GeneratorStage
from curobo_task_constructor.core.state import InterfaceState


@register_stage("current_state")
class CurrentState(GeneratorStage):
    def compute(self) -> None:
        joint = self.robot.get_current_joint_state()
        scene = self._base_scene  # set by init()
        # The attached-object set is server state, not part of the task's
        # declarative base scene: an object may already be attached when the
        # task starts (the pick&place PredicateFilter rejection case).
        attached = set(self.robot.get_attached_objects() or [])
        if scene.attached_object:
            attached.add(scene.attached_object)
        for obj in self.params.get("require_not_attached", []) or []:
            if obj in attached:
                self._fail(None, None,
                           f"object '{obj}' is already attached and cannot be picked")
                return
        for obj in self.params.get("require_attached", []) or []:
            if obj not in attached:
                self._fail(None, None, f"object '{obj}' is not attached")
                return
        self.spawn(InterfaceState(joint_state=joint, scene=scene),
                   comment="current state")

    def init(self, base_scene, robot) -> None:
        super().init(base_scene, robot)
        self._base_scene = base_scene
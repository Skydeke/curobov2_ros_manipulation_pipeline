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
from curobo_task_constructor.mtc.stage_wrappers_base import *
from curobo_task_constructor.mtc.stage_wrappers_motion import *

class GeneratePose(_StageBase):
    """Spawn one configured target pose (MTC GeneratePose).

    The pose rides ``meta["target_pose"]`` for an enclosing ComputeIK,
    like the old demo_ws ``ik_movement`` generator.
    """

    stage_type = "generate_pose"
    is_generator = True

    def __init__(self, name: str = "target pose"):
        super().__init__(name, None)
        self._monitored_stage: Any = None

    def setPose(self, pose: Any) -> None:
        """Set the pose to spawn (PoseStamped/dict/Pose)."""
        pose_dict = _pose_dict_of(pose)
        if pose_dict is None:
            raise TypeError(f"GeneratePose.setPose: unsupported {type(pose)}")
        self.params["pose"] = pose_dict

    def setMonitoredStage(self, stage: Any) -> None:
        """Set the stage to monitor (MTC ``GeneratePose``).

        Recorded for API parity; seeding always reads the live state.
        """
        self._monitored_stage = stage


class ComputeIK(_StageBase):
    """Turn a generator child's candidate poses into joint states (MTC ComputeIK).

    Wraps one generator (MTC ``ComputeIK(name, generator)``); the child
    nests under this stage in ``to_spec`` so the builder installs it via
    ``set_child``.
    """

    stage_type = "compute_ik"

    def __init__(self, name: str = "IK", generator: Any = None):
        super().__init__(name, None)
        self._children: list = []
        #: IK reference frame (MTC ik_frame). Carried for API compatibility.
        self.ik_frame = None
        self._ignore_collisions = False
        if generator is not None:
            self._children.append(generator)

    def setMaxIKSolutions(self, count: int) -> None:
        """Cap the IK solutions (MTC ``ComputeIK``; maps to max_solutions)."""
        self.params["max_solutions"] = int(count)

    def setTargetPose(self, pose: Any) -> None:
        """Set the IK target hint (MTC ``ComputeIK``; recorded)."""
        pose_dict = _pose_dict_of(pose)
        if pose_dict is None:
            raise TypeError(
                f"ComputeIK.setTargetPose: unsupported {type(pose)}")
        self.params["target_pose"] = pose_dict

    def setIKFrame(self, frame: Any) -> None:
        """Set the IK frame (MTC ``ComputeIK``; recorded)."""
        self.ik_frame = frame
        self.params["ik_frame"] = str(frame)

    def setIgnoreCollisions(self, ignore: bool) -> None:
        """Ignore collisions in IK (MTC ``ComputeIK``; recorded)."""
        self._ignore_collisions = bool(ignore)
        self.params["ignore_collisions"] = bool(ignore)

    def to_spec(self):
        from curobo_task_constructor.graph.spec import StageSpec
        return StageSpec(stage_type=self.stage_type, name=self.name,
                         container_type="", children=[
                             c.to_spec() for c in self._children],
                         params_yaml=_dump_params(self._common_params()))


class ModifyPlanningScene(_StageBase):
    """Scene mutation without moving the robot (MTC ModifyPlanningScene). MTC verbs with MTC names: ``attachObject(name, link)``,
    ``detachObject(name, link)``, ``allowCollisions``, ``addObject``,
    ``removeObject``. The link arguments are accepted for compatibility;
    the server attaches to the flange implicitly."""

    stage_type = "modify_scene"

    def __init__(self, name: str = "modify planning scene"):
        super().__init__(name, None)

    def addObject(self, spec: dict) -> None:
        self.params["add"] = dict(spec)

    def removeObject(self, name: str) -> None:
        self.params["remove"] = str(name)

    def remove_all(self) -> None:
        self.params["remove_all"] = True

    def attachObject(self, name: str, link: str = None) -> None:
        params = {"attach": str(name)}
        if link is not None:
            params["attach_link"] = str(link)
        self.params.update(params)

    def detachObject(self, name: str, link: str = None) -> None:
        params = {"detach": str(name)}
        if link is not None:
            params["attach_link"] = str(link)
        self.params.update(params)

    def detach_all(self) -> None:
        self.params["detach_all"] = True

    def allowCollisions(self, object_name: str, links=None,
                        enabled: bool = True) -> None:
        """Allow (or forbid) collisions between object and links.

        ``links`` may be one link name, a list of names, or a joint-model
        group (MTC passes the group: link names are read off it),
        mirroring MTC's (object, group, allowed) form.
        """
        if isinstance(links, str):
            links = [links]
        elif links is not None and not isinstance(links, (list, tuple)):
            links = _group_link_names(links)
        self.params["allow_collisions"] = {
            "object": str(object_name),
            "links": [str(link) for link in (links or [])],
            "enabled": bool(enabled),
        }

__all__ = [
    'GeneratePose',
    'ComputeIK',
    'ModifyPlanningScene',
]

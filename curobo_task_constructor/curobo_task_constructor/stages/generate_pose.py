"""GeneratePose — emit one configured target pose (MTC ``GeneratePose``).

The pose (``params["pose"]``, flat dict) is emitted as a zero-length
solution whose ``meta["target_pose"]`` the enclosing ``ComputeIK`` wrapper
turns into a joint-space state — the same contract ``GenerateGraspPose``
fulfils per sampled candidate. The joint state seeds from the live robot
state; ``monitored_stage`` is accepted for MTC call-shape parity and
recorded, not consumed (seeding always reads the live state).
"""

from __future__ import annotations

from curobo_task_constructor.core.geom import Pose3
from curobo_task_constructor.core.registry import register_stage
from curobo_task_constructor.core.stage import GeneratorStage
from curobo_task_constructor.core.state import InterfaceState


@register_stage("generate_pose")
class GeneratePose(GeneratorStage):
    @staticmethod
    def _as_pose3(pose) -> "Pose3 | None":
        """Flat dict / Pose3 / Pose-like -> Pose3 (None when unparseable)."""
        try:
            if isinstance(pose, Pose3):
                return pose.copy()
            if isinstance(pose, dict):
                return Pose3(
                    [float(pose.get("x", 0.0)), float(pose.get("y", 0.0)),
                     float(pose.get("z", 0.0))],
                    [float(pose.get("qx", 0.0)), float(pose.get("qy", 0.0)),
                     float(pose.get("qz", 0.0)), float(pose.get("qw", 1.0))])
            return Pose3.from_any(pose)
        except Exception:
            return None

    def init(self, base_scene, robot) -> None:
        super().init(base_scene, robot)
        self._base_scene = base_scene

    def compute(self) -> None:
        pose = self.params.get("pose")
        if pose is None:
            self._fail(None, None, "generate_pose requires 'pose'")
            return
        target = self._as_pose3(pose)
        if target is None:
            self._fail(None, None, "generate_pose 'pose' is not a pose")
            return
        try:
            joints = self.robot.get_current_joint_state()
        except Exception as exc:
            self._fail(None, None, f"no current joint state: {exc}")
            return
        state = InterfaceState(
            joint_state=joints,
            scene=self._base_scene,
            meta={"target_pose": target})
        self.spawn(state, cost=0.0, comment="target pose")

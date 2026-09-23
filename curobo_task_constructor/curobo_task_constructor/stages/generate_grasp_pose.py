"""GenerateGraspPose — sample pre-grasp pose candidates around the object
(MTC ``GenerateGraspPose``; Sec. IV-A angle-step sampling).

An open, axis-aligned pre-grasp pose is rotated around the object's up axis
in ``angle_delta`` steps. Each candidate is emitted as a zero-length
solution whose ``meta["target_pose"]`` (plus the pre-grasp joint config)
the enclosing ``ComputeIK`` wrapper turns into a joint-space state.

Geometric sampling is the default; a future ``gpd_topic`` param could point
at ``/grasp_pose_detection/generate_grasps`` without touching this class's
contract — candidates just need to appear in ``meta["target_pose"]``.
"""

from __future__ import annotations

from curobo_task_constructor.core.geom import (
    Pose3,
    quat_from_axis_angle,
    quat_multiply,
    quat_rotate_vector,
)
from curobo_task_constructor.core.registry import register_stage
from curobo_task_constructor.core.stage import GeneratorStage
from curobo_task_constructor.core.state import InterfaceState, make_joint_state


@register_stage("generate_grasp_pose")
class GenerateGraspPose(GeneratorStage):
    def compute(self) -> None:
        obj = self.params.get("object")
        if not obj:
            self._fail(None, None, "generate_grasp_pose requires 'object'")
            return
        obj_pose = self.robot.get_object_pose(obj)
        if obj_pose is None:
            self._fail(None, None, f"object '{obj}' not found in scene")
            return
        obj_pose = Pose3.from_any(obj_pose)
        joints = self._pre_grasp_joints()
        if joints is None:
            self._fail(None, None, "pre-grasp joint config could not be resolved")
            return

        angle_delta = float(self.params.get("angle_delta", 0.2618))
        steps = self._num_steps(angle_delta)
        offset = float(self.params.get("approach_offset", 0.05))
        # ``mode``: "grasp" (MTC GenerateGraspPose) samples the pre-grasp ring
        # below the flipped object; "place" (MTC GeneratePlacePose) samples it
        # above the object — the place container of pick&place uses the same
        # sampling machinery, only mirrored.
        mode = str(self.params.get("mode", "grasp"))
        offset_sign = -1.0 if mode == "place" else 1.0
        # Flip the object's z axis so the effector approaches downward.
        base_q = quat_multiply(obj_pose.orientation,
                               quat_from_axis_angle([1.0, 0.0, 0.0], 3.141592653589793))

        made = 0
        for k in range(steps):
            angle = k * angle_delta
            if self.params.get("angle_end") is not None and \
                    angle > float(self.params["angle_end"]):
                break
            q_rot = quat_from_axis_angle([0.0, 0.0, 1.0], angle)
            orientation = quat_multiply(q_rot, base_q)
            d = quat_rotate_vector(q_rot, [0.0, 0.0, offset_sign * offset])
            pose = Pose3([obj_pose.position[0] + d[0],
                          obj_pose.position[1] + d[1],
                          obj_pose.position[2] + d[2]],
                         orientation)
            state = InterfaceState(
                joint_state=joints,
                scene=self._base_scene,
                meta={"target_pose": pose,
                      "grasp_object": obj,
                      "angle": angle})
            self.spawn(state, cost=0.0, comment=f"candidate angle={angle:.3f}")
            made += 1
        if not made:
            self._fail(None, None, "generate_grasp_pose produced no candidates")

    # ------------------------------------------------------------------
    def _num_steps(self, angle_delta: float) -> int:
        if self.params.get("num_steps") is not None:
            return max(1, int(self.params["num_steps"]))
        if self.params.get("angle_end") is not None:
            return int(float(self.params["angle_end"]) / angle_delta) + 1
        return max(1, int(round(2.0 * 3.141592653589793 / angle_delta)))

    def _pre_grasp_joints(self):
        goal = self.params.get("pre_grasp") or {}
        robot = self.robot
        if "name" in goal:
            cfg = robot.get_named_joint_config(goal["name"])
            base = dict(zip(getattr(robot.get_current_joint_state(), "name", []),
                            getattr(robot.get_current_joint_state(), "position", [])))
            if getattr(cfg, "names", None):
                base.update(cfg.as_dict())
                names = list(getattr(robot.get_current_joint_state(), "name", []) or [])
                positions = [base[n] for n in names]
            else:
                names = list(getattr(robot.get_current_joint_state(), "name", []) or [])
                positions = list(getattr(cfg, "positions", [])) or \
                    [base[n] for n in names]
            return make_joint_state(getattr(robot, "joint_state_cls", None),
                                    names, positions)
        joints = goal.get("joints")
        if joints is not None:
            names = list(getattr(robot.get_current_joint_state(), "name", []) or [])
            return make_joint_state(getattr(robot, "joint_state_cls", None),
                                    names, [float(j) for j in joints])
        return None

    def init(self, base_scene, robot) -> None:
        super().init(base_scene, robot)
        self._base_scene = base_scene
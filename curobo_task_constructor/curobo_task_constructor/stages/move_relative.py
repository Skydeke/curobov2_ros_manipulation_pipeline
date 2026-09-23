"""MoveRelative — propagate along an axis by a distance sampled in a range
(MTC ``MoveRelative``; the approach/lift/lower/retreat stages of pick&place).

Anchor the effector at ``robot.fk`` of the current joints, translate the
effector pose along ``axis`` by each sampled distance, and solve one
whole-task request per sample. Backward (approach-style): the END state is
known — IK the shifted pose for the START state, then plan start→end.
"""

from __future__ import annotations

import math

from curobo_task_constructor.core.geom import Pose3, quat_rotate_vector, pose_to_any
from curobo_task_constructor.core.registry import register_stage
from curobo_task_constructor.core.robot import GoalsetSpec
from curobo_task_constructor.core.stage import PropagatingEitherWay
from curobo_task_constructor.core.state import InterfaceState
from curobo_task_constructor.stages._util import full_request


@register_stage("move_relative")
class MoveRelative(PropagatingEitherWay):
    def __init__(self, name=None, params=None, direction="auto"):
        # declarative support: a YAML spec may pin the direction
        if direction == "auto" and params and params.get("direction"):
            direction = params["direction"]
        super().__init__(name, params, direction=direction)

    # ------------------------------------------------------------------
    def _delta_world(self, pose_link: Pose3, distance: float) -> list:
        axis = self.params.get("axis") or {}
        xyz = axis.get("xyz", [0.0, 0.0, 1.0])
        norm = math.sqrt(sum(v * v for v in xyz))
        unit = [v / norm for v in xyz] if norm else [0.0, 0.0, 1.0]
        scaled = [distance * v for v in unit]
        if axis.get("frame", "hand") == "hand":
            return quat_rotate_vector(pose_link.orientation, scaled)
        return scaled

    def _sample_distances(self) -> list:
        if self.params.get("distance") is not None:
            return [float(self.params["distance"])]
        lo = float(self.params.get("min_distance", 0.0))
        hi = float(self.params.get("max_distance", lo))
        n = max(1, int(self.params.get("num_samples", 3)))
        if n == 1 or hi <= lo:
            return [(lo + hi) / 2.0]
        return [lo + (hi - lo) * k / (n - 1) for k in range(n)]

    # ------------------------------------------------------------------
    def compute_forward(self, state: InterfaceState) -> None:
        link = self.params.get("link")
        pose = Pose3.from_any(self.robot.fk(state.joint_state, link))
        allowed = state.scene.all_allowed_links() if state.scene else []
        made = 0
        for d in self._sample_distances():
            target = Pose3(
                [p + q for p, q in zip(pose.position, self._delta_world(pose, d))],
                pose.orientation)
            goal = GoalsetSpec(poses=[pose_to_any(target, getattr(self.robot, "pose_cls", None))],
                               allowed_collisions=allowed)
            req = full_request(self.robot, state.joint_state, [goal], self.params)
            try:
                result = self.robot.plan(req)
            except Exception as exc:
                self._fail(state, None, f"plan call failed: {exc}")
                continue
            if not result.success:
                self._fail(state, None, result.message or "move_relative plan failed")
                continue
            end = state.clone(joint_state=result.last_state)
            self.send_forward(state, end, trajectory=result.trajectory,
                              cost=self._cost_of(result),
                              comment=f"{self.name} d={d:.3f}",
                              response=result.raw, plan_request=req)
            made += 1
        if not made:
            self._fail(state, None, "move_relative produced no solution")

    def compute_backward(self, state: InterfaceState) -> None:
        link = self.params.get("link")
        pose = Pose3.from_any(self.robot.fk(state.joint_state, link))
        allowed = state.scene.all_allowed_links() if state.scene else []
        made = 0
        for d in self._sample_distances():
            # START is `d` back along the axis from the known END pose.
            start_pose = Pose3(
                [p - q for p, q in zip(pose.position, self._delta_world(pose, d))],
                pose.orientation)
            seed = self.robot.ik(start_pose)
            if seed is None:
                continue
            goal = GoalsetSpec(target_joint_positions=list(
                getattr(state.joint_state, "position", []) or []),
                allowed_collisions=allowed)
            req = full_request(self.robot, seed, [goal], self.params)
            try:
                result = self.robot.plan(req)
            except Exception as exc:
                self._fail(state, None, f"plan call failed: {exc}")
                continue
            if not result.success or not result.trajectory:
                self._fail(state, None, result.message or "move_relative backward failed")
                continue
            start_state = state.clone(joint_state=result.trajectory[0])
            self.send_backward(start_state, state, trajectory=result.trajectory,
                               cost=self._cost_of(result),
                               comment=f"{self.name} d={d:.3f} (bwd)",
                               response=result.raw, plan_request=req)
            made += 1
        if not made:
            self._fail(state, None, "move_relative backward produced no solution")

    def _cost_of(self, result) -> float:
        if result.cost != float("inf"):
            return float(result.cost)
        return float(len(result.trajectory)) if result.trajectory else 0.0
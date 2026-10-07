"""MoveRelative — propagate along an axis by a distance sampled in a range
(MTC ``MoveRelative``; the approach/lift/lower/retreat stages of pick&place).

Three direction modes (MTC ``setDirection`` parity, all carried in params):

- translation (default): anchor the effector at ``robot.fk`` of the current
  joints, translate along ``axis`` by each sampled distance. Backward
  (approach-style): the END state is known — IK the shifted pose for the
  START state, then plan start→end.
- rotation: ``rotation: {axis: [x,y,z], angle: <rad>, frame: hand|world}``
  rotates the tool orientation about the axis, optionally combined with a
  translation when both ``axis``/``distance`` and ``rotation`` are present
  (MTC ``TwistStamped`` may carry linear + angular velocity together).
- joint offsets: ``joint_offsets: {name: delta}`` adds the deltas onto the
  start joints and plans a joint-space goal (MTC ``setDirection(dict)``).
"""

from __future__ import annotations

import math

from curobo_task_constructor.core.geom import (
    Pose3,
    pose_to_any,
    quat_from_axis_angle,
    quat_multiply,
    quat_rotate_vector,
)
from curobo_task_constructor.core.registry import register_stage
from curobo_task_constructor.core.robot import GoalsetSpec
from curobo_task_constructor.core.stage import PropagatingEitherWay
from curobo_task_constructor.core.state import InterfaceState
from curobo_task_constructor.stages._util import (
    axis_holds,
    cost_of,
    full_request,
    goalset_for_scene,
)


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

    def _rotation_of(self) -> tuple:
        """(axis_xyz, angle_rad, frame) or (None, 0.0, frame) when absent."""
        rot = self.params.get("rotation") or {}
        axis = rot.get("axis")
        angle = float(rot.get("angle", 0.0) or 0.0)
        if not axis or angle == 0.0:
            return None, 0.0, str(rot.get("frame", "hand"))
        return [float(v) for v in axis], angle, str(rot.get("frame", "hand"))

    def _joint_offsets(self) -> dict:
        off = self.params.get("joint_offsets") or {}
        return {str(k): float(v) for k, v in dict(off).items()}

    def _apply_rotation(self, orientation: list, frame: str = "hand") -> list:
        axis, angle, rot_frame = self._rotation_of()
        if axis is None:
            return list(orientation)
        dq = quat_from_axis_angle(axis, angle)
        if (frame or rot_frame or "hand").lower() == "hand":
            # tool-local rotation (MTC TwistStamped expressed in the hand
            # frame, the cartesian.py "rz +45°" case)
            return quat_multiply(orientation, dq)
        return quat_multiply(dq, orientation)

    def _sample_distances(self) -> list:
        if self.params.get("distance") is not None:
            return [float(self.params["distance"])]
        lo = float(self.params.get("min_distance", 0.0))
        hi = float(self.params.get("max_distance", lo))
        n = max(1, int(self.params.get("num_samples", 3)))
        if n == 1 or hi <= lo:
            return [(lo + hi) / 2.0]
        return [lo + (hi - lo) * k / (n - 1) for k in range(n)]

    def _target_for(self, pose: Pose3, distance: float) -> Pose3:
        """Target tool pose for one sampled distance + configured rotation."""
        translated = [p + q for p, q in zip(
            pose.position, self._delta_world(pose, distance))]
        axis = self.params.get("axis") or {}
        oriented = self._apply_rotation(
            pose.orientation, str(axis.get("frame", "hand")))
        return Pose3(translated, oriented)

    # ------------------------------------------------------------------
    def _compute_joint_offsets_forward(self, state: InterfaceState) -> bool:
        """Joint-delta mode (MTC ``setDirection(dict)``). Returns True if
        this mode owned the compute (emitted or failed, caller returns)."""
        offsets = self._joint_offsets()
        if not offsets:
            return False
        names = list(getattr(state.joint_state, "name", []) or [])
        positions = list(getattr(state.joint_state, "position", []) or [])
        by_name = dict(zip(names, positions))
        for joint, delta in offsets.items():
            if joint not in by_name:
                self._fail(state, None,
                           f"move_relative joint offset for unknown joint {joint!r}")
                return True
            by_name[joint] += delta
        goal = [by_name[n] for n in names] if names else list(by_name.values())
        goalset = goalset_for_scene(state.scene, joint_positions=goal)
        req = full_request(self.robot, state.joint_state, [goalset], self.params)
        try:
            self._note_plan_attempt()
            result = self.robot.plan(req)
        except Exception as exc:
            self._fail(state, None, f"plan call failed: {exc}")
            return True
        if not result.success:
            self._fail(state, None, result.message or "move_relative plan failed")
            return True
        end = state.clone(joint_state=result.last_state)
        self.send_forward(state, end, trajectory=result.trajectory,
                          cost=self._cost_of(result),
                          comment=f"{self.name} joint offset",
                          response=result.raw, plan_request=req)
        return True

    def compute_forward(self, state: InterfaceState) -> None:
        if self._compute_joint_offsets_forward(state):
            return
        link = self.params.get("link")
        pose = Pose3.from_any(self.robot.fk(state.joint_state, link))
        allowed = state.scene.all_allowed_links() if state.scene else []
        made = 0
        for d in self._sample_distances():
            target = self._target_for(pose, d)
            goal = GoalsetSpec(poses=[pose_to_any(target, getattr(self.robot, "pose_cls", None))],
                               allowed_collisions=allowed,
                               trajectory_constraints=self._hold(pose, target))
            req = full_request(self.robot, state.joint_state, [goal], self.params)
            try:
                self._note_plan_attempt()
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

    def _compute_joint_offsets_backward(self, state: InterfaceState) -> bool:
        offsets = self._joint_offsets()
        if not offsets:
            return False
        names = list(getattr(state.joint_state, "name", []) or [])
        positions = list(getattr(state.joint_state, "position", []) or [])
        by_name = dict(zip(names, positions))
        start_positions = [by_name.get(n, 0.0) - offsets.get(n, 0.0) for n in names]
        goalset = goalset_for_scene(
            state.scene, joint_positions=list(positions))
        req = full_request(self.robot, None, [goalset], self.params)
        # Seed the solve from the offset-back joints by issuing the request
        # with that start pose: rebuild with explicit start.
        from curobo_task_constructor.core.state import JointStateStub
        _ = JointStateStub  # keep import local-free; stub unused in ROS path
        req.start_pose = self._joint_state_with(names, start_positions)
        try:
            self._note_plan_attempt()
            result = self.robot.plan(req)
        except Exception as exc:
            self._fail(state, None, f"plan call failed: {exc}")
            return True
        if not result.success or not result.trajectory:
            self._fail(state, None, result.message or "move_relative backward failed")
            return True
        start_state = state.clone(joint_state=result.trajectory[0])
        self.send_backward(start_state, state, trajectory=result.trajectory,
                           cost=self._cost_of(result),
                           comment=f"{self.name} joint offset (bwd)",
                           response=result.raw, plan_request=req)
        return True

    def _inverse_orientation(self, orientation: list) -> list:
        from curobo_task_constructor.core.geom import quat_conjugate
        axis, angle, rot_frame = self._rotation_of()
        if axis is None:
            return list(orientation)
        dq = quat_from_axis_angle(axis, -angle)
        if str(rot_frame).lower() == "hand":
            return quat_multiply(orientation, dq)
        return quat_multiply(dq, orientation)

    def _joint_state_with(self, names, positions):
        cls = getattr(self.robot, "joint_state_cls", None)
        if cls is None:
            from curobo_task_constructor.core.state import JointStateStub
            return JointStateStub(names=list(names), positions=list(positions))
        js = cls()
        js.name = list(names)
        js.position = [float(v) for v in positions]
        return js

    def compute_backward(self, state: InterfaceState) -> None:
        if self._compute_joint_offsets_backward(state):
            return
        link = self.params.get("link")
        pose = Pose3.from_any(self.robot.fk(state.joint_state, link))
        allowed = state.scene.all_allowed_links() if state.scene else []
        made = 0
        for d in self._sample_distances():
            # START is `d` back along the axis from the known END pose,
            # with the rotation inverted.
            start_pose = Pose3(
                [p - q for p, q in zip(pose.position, self._delta_world(pose, d))],
                self._inverse_orientation(pose.orientation))
            seed = self.robot.ik(start_pose)
            if seed is None:
                continue
            goal = GoalsetSpec(target_joint_positions=list(
                getattr(state.joint_state, "position", []) or []),
                allowed_collisions=allowed,
                trajectory_constraints=self._hold(start_pose, pose))
            req = full_request(self.robot, seed, [goal], self.params)
            try:
                self._note_plan_attempt()
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

    def _hold(self, start_pose: Pose3, target: Pose3) -> list:
        """int8[6] axis holds for one sampled segment, or [] for none.

        ``MoveRelative`` exists to interpolate a straight line, so by default it
        derives the whole-path holds exactly like ``CartesianPath`` does: the
        offset changes only the axes the motion actually moves along, and every
        axis the start and the target agree on is pinned. A ``hold`` param
        overrides it (pass ``[0, 0, 0, 0, 0, 0]`` for a free-space relative
        move). Only the classic planner reads the field, so this is a no-op on
        any other planner.
        """
        explicit = self.params.get("hold")
        if explicit is not None:
            hold = [int(c) for c in (explicit or [])]
            return hold if len(hold) == 6 else []
        return axis_holds(start_pose, [target],
                          pos_tol=float(self.params.get("pos_tol", 0.005)),
                          rot_tol=float(self.params.get("rot_tol", 0.05)))

    def _cost_of(self, result) -> float:
        return cost_of(result, self.params)
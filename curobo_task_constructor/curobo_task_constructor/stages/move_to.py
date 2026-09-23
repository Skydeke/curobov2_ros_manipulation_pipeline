"""MoveTo — propagate to a named / pose / joint-space goal (MTC ``MoveTo``).

A forward move plans one whole-task ``PlanRequest`` with a single goalset
(reaching the goal). Because the pick&place graph places a ``Connect``
upstream of containers whose first child must *write* its start, ``MoveTo``
is a ``PropagatingEitherWay``: the containing container may resolve it to run
backward (read an already-known end, plan the motion to it, emit the start).
"""

from __future__ import annotations

from curobo_task_constructor.core.registry import register_stage
from curobo_task_constructor.core.robot import GoalsetSpec
from curobo_task_constructor.core.stage import PropagatingEitherWay
from curobo_task_constructor.core.state import InterfaceState
from curobo_task_constructor.stages._util import (
    full_request,
    goalset_for_scene,
    pose_from_params,
)


@register_stage("move_to")
class MoveTo(PropagatingEitherWay):
    def __init__(self, name=None, params=None, direction="auto"):
        # declarative support: a YAML spec may pin the direction
        if direction == "auto" and params and params.get("direction"):
            direction = params["direction"]
        super().__init__(name, params, direction=direction)

    # ------------------------------------------------------------------
    # Goal resolution
    # ------------------------------------------------------------------
    def _goal_positions(self, start: InterfaceState) -> list:
        """Joint-space goal aligned to the start state's joint-name order."""
        goal = self.params.get("goal") or {}
        names = getattr(start.joint_state, "name", None)
        if "name" in goal:
            cfg = self.robot.get_named_joint_config(goal["name"])
            return self._merge_named(cfg, names, start.joint_state)
        joints = goal.get("joints")
        if joints is not None:
            return [float(j) for j in joints]
        return None

    def _merge_named(self, cfg, names, start_joint_state) -> list:
        """Named config merged onto the *start* state's joints.

        Like MTC: a partial named config (e.g. a gripper group's ``open``/
        ``close`` naming only ``finger_joint``) overrides just those joints of
        the planning-scene state at solve time — the arm must not move when
        the gripper opens or closes mid-task.
        """
        base = dict(zip(getattr(start_joint_state, "name", []),
                        getattr(start_joint_state, "position", [])))
        if getattr(cfg, "names", None):
            base.update(cfg.as_dict())
            return [base[n] for n in (names or cfg.names)]
        return [float(p) for p in getattr(cfg, "positions", [])]

    def _goal_pose(self):
        goal = self.params.get("goal") or {}
        if "pose" in goal:
            return pose_from_params(goal["pose"], self.robot)
        return None

    def _allowed_collisions(self, scene):
        return scene.all_allowed_links() if scene is not None else []

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def compute_forward(self, state: InterfaceState) -> None:
        joint_goal = self._goal_positions(state)
        pose_goal = self._goal_pose() if joint_goal is None else None
        if joint_goal is None and pose_goal is None:
            self._fail(state, None,
                       "move_to goal must be one of name/joints/pose")
            return
        if pose_goal is not None:
            goalset = GoalsetSpec(poses=[pose_goal],
                                  allowed_collisions=self._allowed_collisions(state.scene))
        else:
            goalset = GoalsetSpec(
                target_joint_positions=joint_goal,
                allowed_collisions=self._allowed_collisions(state.scene))
        req = full_request(self.robot, state.joint_state, [goalset], self.params)
        try:
            result = self.robot.plan(req)
        except Exception as exc:  # ServiceError etc.
            # repr, not str: rclpy futures can fail with an empty-str
            # exception (CancelledError/StopIteration), which str() would
            # silently swallow into "plan call failed: ".
            self._fail(state, None, f"plan call failed: {exc!r}")
            return
        if not result.success:
            self._fail(state, None, result.message or "move_to plan failed")
            return
        end = state.clone(joint_state=result.last_state)
        self.send_forward(state, end, trajectory=result.trajectory,
                          cost=self._cost_of(result), comment=self._comment(),
                          response=result.raw, plan_request=req)

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------
    def compute_backward(self, state: InterfaceState) -> None:
        """The end state is known; plan the motion and emit its start.

        The trajectory's first waypoint becomes the start state; the Connect
        upstream of us solves the actual path to it.
        """
        req = full_request(
            self.robot, None,
            [GoalsetSpec(target_joint_positions=list(
                getattr(state.joint_state, "position", []) or []),
                allowed_collisions=self._allowed_collisions(state.scene))],
            self.params)
        try:
            result = self.robot.plan(req)
        except Exception as exc:
            self._fail(state, None, f"plan call failed: {exc!r}")
            return
        if not result.success or not result.trajectory:
            self._fail(state, None, result.message or "move_to backward failed")
            return
        start = state.clone(joint_state=result.trajectory[0])
        self.send_backward(start, state, trajectory=result.trajectory,
                           cost=self._cost_of(result), comment=self._comment(),
                           response=result.raw, plan_request=req)

    def _comment(self) -> str:
        return f"move_to {self.params.get('goal', {})}"

    def _cost_of(self, result) -> float:
        if result.cost != float("inf"):
            return float(result.cost)
        return float(len(result.trajectory)) if result.trajectory else 0.0
"""MoveTo — propagate to a named / pose / joint-space goal (MTC ``MoveTo``).

A forward move plans one whole-task ``PlanRequest`` with a single goalset
(reaching the goal). Because the pick&place graph places a ``Connect``
upstream of containers whose first child must *write* its start, ``MoveTo``
is a ``PropagatingEitherWay``: the containing container may resolve it to run
backward (read an already-known end, plan the motion to it, emit the start).
"""

from __future__ import annotations

from curobo_task_constructor.core.registry import register_stage
from curobo_task_constructor.core.stage import PropagatingEitherWay
from curobo_task_constructor.core.state import InterfaceState
from curobo_task_constructor.stages._util import (
    cost_of,
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
        if isinstance(joints, dict):
            # Sparse name -> position, merged onto the start state: the
            # "close the fingers without moving the arm" goal. Same semantics
            # as a partial named config, but it does not require the robot
            # descriptor to carry an entry per gripper value.
            return self._merge_sparse(joints, names, start.joint_state)
        if joints is not None:
            return [float(j) for j in joints]
        return None

    @staticmethod
    def _merge_sparse(values: dict, names, start_joint_state) -> list:
        base = dict(zip(getattr(start_joint_state, "name", []),
                        getattr(start_joint_state, "position", [])))
        base.update({k: float(v) for k, v in values.items()})
        order = list(names or base.keys())
        return [base[n] for n in order if n in base]

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

    def _goal_poses(self, start=None):
        """Every candidate pose goal, in the order the caller listed them.

        ``goal.poses`` is a LIST of pose dicts. All candidates go into ONE
        goalset, and the server resolves the set inside a single
        ``plan_pose()`` call (ClassicPlanner: a segment with N>1 poses becomes a
        cuRobo goalset solve) and reports the winner via
        ``selected_goal_index``. This is the framework's cheapest fan-out: one
        service round-trip, one GPU batch, N IK+trajopt seeds attempted in
        parallel, best one stitched into the solution — the direct equivalent of
        the reference pipeline's ``ComputeIK``+``Connect`` multi-seed solve, and
        the reason ``max_goalset`` must be > 1 on the server.

        ``goal.pose`` is sugar for a one-element list. ``goal.point`` (MTC
        PointStamped: position goal, orientation kept) resolves against the
        start state's tool orientation, so it needs ``start``.
        """
        from curobo_task_constructor.core.geom import Pose3
        goal = self.params.get("goal") or {}
        if "poses" in goal:
            return [pose_from_params(p, self.robot) for p in (goal["poses"] or [])]
        if "point" in goal and start is not None:
            try:
                anchor = Pose3.from_any(
                    self.robot.fk(start.joint_state,
                                  self.params.get("link")))
            except Exception:
                return []
            pt = goal["point"]
            target = anchor.copy()
            target.position = [float(pt.get("x", target.position[0])),
                               float(pt.get("y", target.position[1])),
                               float(pt.get("z", target.position[2]))]
            from curobo_task_constructor.core.geom import pose_to_any
            return [pose_to_any(target, getattr(self.robot, "pose_cls", None))]
        single = self._goal_pose()
        return [single] if single is not None else []

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _attempts(self) -> int:
        return max(1, int(self.params.get("planning_attempts", 3)))

    def _try_plan(self, req, state, fail_msg):
        """One planner call: None on failure (recorded), result on success."""
        try:
            result = self._timed_plan(req)
        except Exception as exc:  # ServiceError etc.
            # repr, not str: rclpy futures can fail with an empty-str
            # exception (CancelledError/StopIteration), which str() would
            # silently swallow into "plan call failed: ".
            self._fail(state, None, f"plan call failed: {exc!r}")
            return None
        if not result.success:
            self._fail(state, None, result.message or fail_msg)
            return None
        self._note_plan_attempt()
        return result

    def compute_forward(self, state: InterfaceState) -> None:
        joint_goal = self._goal_positions(state)
        pose_goals = [] if joint_goal is not None else self._goal_poses(state)
        if joint_goal is None and not pose_goals:
            self._fail(state, None,
                       "move_to goal must be one of name/joints/pose/poses/point")
            return
        if pose_goals:
            goalset = goalset_for_scene(state.scene, poses=pose_goals)
        else:
            goalset = goalset_for_scene(state.scene, joint_positions=joint_goal)
        req = full_request(self.robot, state.joint_state, [goalset], self.params)
        # Every try is stored and published the moment it solves (like failed
        # tries fail live); only the cheapest is promoted downstream.
        from curobo_task_constructor.core.stage import Solution
        tried = []
        for _ in range(self._attempts()):
            result = self._try_plan(req, state, "move_to plan failed")
            if result is None:
                continue
            end = state.clone(joint_state=result.last_state)
            sol = Solution(
                state, end, trajectory=result.trajectory,
                cost=self._cost_of(result), comment=self._comment(result),
                response=result.raw, plan_request=req)
            self._consider(sol)
            tried.append(sol)
        if not tried:
            # Every failed try already recorded itself above (MTC: each
            # failed plan is a failed solution); nothing more to report.
            return
        best = min(tried, key=lambda s: s.cost)
        self._promote_forward(state, best.end, best)

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
            [goalset_for_scene(
                state.scene,
                joint_positions=list(getattr(state.joint_state, "position", [])
                                     or []))],
            self.params)
        from curobo_task_constructor.core.stage import Solution
        tried = []
        for _ in range(self._attempts()):
            result = self._try_plan(req, state, "move_to backward failed")
            if result is None:
                continue
            if not result.trajectory:
                self._fail(state, None, "move_to backward failed")
                continue
            start = state.clone(joint_state=result.trajectory[0])
            sol = Solution(
                start, state, trajectory=result.trajectory,
                cost=self._cost_of(result), comment=self._comment(result),
                response=result.raw, plan_request=req)
            self._consider(sol)
            tried.append(sol)
        if not tried:
            # Every failed try already recorded itself (see _try_plan).
            return
        best = min(tried, key=lambda s: s.cost)
        self._promote_backward(best.start, state, best)

    def _comment(self, result=None) -> str:
        goal = self.params.get("goal", {})
        picked = ""
        if result is not None and getattr(result, "selected_goal_index", None):
            # Which candidate the server's goalset solve actually picked. This
            # is the whole point of the fan-out, so surface it: without it a
            # multi-candidate goalset is indistinguishable from a single one in
            # the solution statistics.
            picked = f" -> candidate {list(result.selected_goal_index)}"
        return f"move_to {goal}{picked}"

    def _cost_of(self, result) -> float:
        return cost_of(result, self.params)
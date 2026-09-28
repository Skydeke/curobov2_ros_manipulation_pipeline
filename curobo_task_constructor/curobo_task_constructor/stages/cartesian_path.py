"""CartesianPath — a straight-line whole-task solve (the port of the reference
pipeline's ``MoveRelative`` + ``CartesianPath`` solver).

Why this stage exists
---------------------
On cuRobo a straight line is not a solve MODE, it is a COST: a goalset carries
``trajectory_constraints`` (``int8[6]`` = ``[theta_x, theta_y, theta_z, x, y,
z]``) which the server turns into ``ToolPoseCriteria`` and scores on every
non-terminal waypoint. Holding every axis on which the start and the goal agree
therefore leaves the optimizer no choice but the segment between them. That is
the whole mechanism, and it is what replaces MoveIt's
``computeCartesianPath(step=0.01, min_fraction=0.95, jump_threshold=5.0)``.

Three goal forms, matching the three ways a reference Cartesian stage is
written:

- ``goal.pose``      absolute tool pose (the descend onto a grasp)
- ``goal.poses``     N candidate poses in ONE goalset — the server resolves the
                     set inside a single ``plan_pose()`` call and reports the
                     winner through ``selected_goal_index``. The hold is the
                     intersection over all candidates, so it stays correct for
                     whichever one wins.
- ``goal.relative``  ``{x, y, z, frame: world|hand}`` offset from FK(start),
                     keeping the start orientation. The literal
                     ``MoveRelative`` port: a translation along a direction with
                     the effector orientation frozen. This is what a retreat /
                     lift should be, because it is anchored to the pose the arm
                     ACTUALLY reached rather than to the pose that was asked
                     for.

FORWARD ONLY
------------
The hold vector is derived from the start pose, which is not known until the
plan runs, so a backward (end-seeded) cartesian move would need a two-pass
plan. Place this stage after a stage that writes a joint state.

Rounding out the MoveIt contract
--------------------------------
cuRobo's hold is a soft cost and the wire has no ``fraction`` / ``jump`` knob,
so the reference's partial-path guarantee is reproduced client-side by
``check_straightness``: FK the whole trajectory in one call and fail the stage
when the tool strays further than ``straightness_tol`` from the segment. A
failure is not fatal — it propagates to the enclosing ``Fallbacks``, which is
exactly how the reference pipeline reached the next grasp candidate after
rejecting a Cartesian solve.
"""

from __future__ import annotations

import math
from typing import Optional

from curobo_task_constructor.core.geom import (
    Pose3,
    pose_to_any,
    quat_rotate_vector,
)
from curobo_task_constructor.core.registry import register_stage
from curobo_task_constructor.core.robot import GoalsetSpec, PlanRequest
from curobo_task_constructor.core.stage import TrajectoryStage
from curobo_task_constructor.core.state import InterfaceState
from curobo_task_constructor.stages._util import (
    axis_holds,
    cost_of,
    full_request,
    pose_from_params,
)

#: Default lateral tolerance for the straightness gate. The reference solver
#: used a 5 mm jump threshold on a 1 cm interpolation step, i.e. "do not cut a
#: corner"; 10 mm of bow over the whole segment is the same order of magnitude.
DEFAULT_STRAIGHTNESS_TOL = 0.01


def _segment_deviation(pose: Pose3, a: Pose3, b: Pose3) -> float:
    """Distance from ``pose.position`` to the segment a->b.

    Zero when the segment is degenerate (start and goal coincide) — nothing to
    bow out of the way of.
    """
    ax, ay, az = a.position
    dx, dy, dz = (b.position[i] - a.position[i] for i in range(3))
    denom = dx * dx + dy * dy + dz * dz
    if denom < 1e-18:
        return 0.0
    t = ((pose.position[0] - ax) * dx + (pose.position[1] - ay) * dy
         + (pose.position[2] - az) * dz) / denom
    t = max(0.0, min(1.0, t))
    px, py, pz = ax + t * dx, ay + t * dy, az + t * dz
    return math.sqrt((pose.position[0] - px) ** 2 + (pose.position[1] - py) ** 2
                     + (pose.position[2] - pz) ** 2)


@register_stage("cartesian_path")
class CartesianPath(TrajectoryStage):
    """One straight-line whole-task solve with whole-path axis holds."""

    def __init__(self, name=None, params=None):
        super().__init__(name, params)
        # id(InterfaceState) -> (FK'd start pose, candidate target poses).
        # Lets the post-solve straightness gate measure deviation from the
        # REQUESTED line (not the achieved one) even when a container has
        # deferred the solve into a batch. Keyed on the start state, which
        # ``make_end_state`` does receive.
        self._pending_goals: dict = {}

    def reset(self) -> None:
        super().reset()
        self._pending_goals = {}

    # ------------------------------------------------------------------
    # Goal resolution
    # ------------------------------------------------------------------
    def _link(self) -> Optional[str]:
        return self.params.get("link")

    def _start_pose(self, start: InterfaceState) -> Optional[Pose3]:
        try:
            return Pose3.from_any(
                self.robot.fk(start.joint_state, self._link()))
        except Exception as exc:  # noqa: BLE001 - ServiceError etc.
            self._fail(start, None,
                       f"cartesian_path FK of the start failed: {exc!r}")
            return None

    def _targets(self, start_pose: Pose3) -> list:
        """Resolve the goal form into a list of candidate target poses."""
        goal = self.params.get("goal") or {}
        if "poses" in goal:
            return [pose_from_params(p, self.robot) for p in (goal["poses"] or [])]
        if "pose" in goal:
            return [pose_from_params(goal["pose"], self.robot)]
        rel = goal.get("relative")
        if rel is not None:
            delta = [float(rel.get("x", 0.0)), float(rel.get("y", 0.0)),
                     float(rel.get("z", 0.0))]
            if str(rel.get("frame", "world")).lower() == "hand":
                delta = quat_rotate_vector(start_pose.orientation, delta)
            target = start_pose.copy()
            for i in range(3):
                target.position[i] += delta[i]
            # Orientation frozen: MoveRelative interpolates a direction and
            # leaves the effector angle alone for the whole move.
            return [pose_to_any(target, getattr(self.robot, "pose_cls", None))]
        return []

    def _hold(self, start_pose: Pose3, targets: list) -> list:
        explicit = self.params.get("hold")
        if explicit:
            hold = [int(c) for c in explicit]
            return hold if len(hold) == 6 else []
        return axis_holds(start_pose, targets,
                          pos_tol=float(self.params.get("pos_tol", 0.005)),
                          rot_tol=float(self.params.get("rot_tol", 0.05)))

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def build_plan_request(self, start: InterfaceState) -> Optional[PlanRequest]:
        start_pose = self._start_pose(start)
        if start_pose is None:
            return None
        targets = self._targets(start_pose)
        if not targets:
            self._fail(start, None,
                       "cartesian_path requires goal.pose, goal.poses or "
                       "goal.relative")
            return None
        goalset = GoalsetSpec(
            poses=targets,
            allowed_collisions=(start.scene.all_allowed_links()
                                if start.scene is not None else []),
            trajectory_constraints=self._hold(start_pose, targets))
        req = full_request(self.robot, start.joint_state, [goalset], self.params)
        self._pending_goals[id(start)] = (start_pose, targets)
        return req

    def make_end_state(self, start: InterfaceState, result, raw=None):
        end = start.clone(joint_state=result.last_state)
        straight = self._straight_enough(start, end, result)
        self._pending_goals.pop(id(start), None)
        return end if straight else None

    def _comment(self, req: PlanRequest, result) -> str:
        # The resolved holds and the winning candidate index, so a
        # multi-candidate descent is legible in the task statistics.
        gs = req.goalsets[0] if req.goalsets else None
        parts = []
        if gs is not None:
            parts.append(f"hold={list(gs.trajectory_constraints)}")
            parts.append(f"candidates={len(gs.poses)}")
        picked = list(getattr(result, "selected_goal_index", None) or [])
        if picked:
            parts.append(f"winner={picked}")
        return "cartesian_path " + " ".join(parts)

    # ------------------------------------------------------------------
    # Straightness gate (MoveIt min_fraction / jump_threshold analogue)
    # ------------------------------------------------------------------
    def _straight_enough(self, start: InterfaceState, end: InterfaceState,
                         result) -> bool:
        """Reject a solve that bowed off the straight segment.

        cuRobo's hold is a soft cost: the solver will leave the line when that
        is cheaper than staying on it. FK the whole trajectory in ONE call and
        fail the stage when the worst waypoint is further than
        ``straightness_tol`` from the requested start->goal segment, so a bowed
        path is rejected and the enclosing Fallbacks gets to try the next
        variant — which is what the reference pipeline's rejected Cartesian
        solve did.
        """
        if not self.params.get("check_straightness", True):
            return True
        tol = float(self.params.get("straightness_tol",
                                    DEFAULT_STRAIGHTNESS_TOL))
        traj = result.trajectory or []
        if len(traj) < 2:
            return True
        start_pose, targets = self._pending_goals.get(id(start), (None, []))
        if start_pose is None or not targets:
            return True
        try:
            poses = self.robot.fk_batch(traj, self._link())
        except Exception as exc:  # noqa: BLE001
            # The trajectory itself solved; do not discard a good plan over a
            # failed measurement round-trip.
            self._log_debug(f"straightness check skipped: {exc!r}")
            return True
        if len(poses) != len(traj):
            return True
        # Deviation is measured against the *requested* goal of the candidate
        # the server reports as the winner, not against the achieved end pose:
        # a path that consistently undershoots is still straight.
        b = Pose3.from_any(targets[self._winner(result)])
        worst = max(_segment_deviation(Pose3.from_any(p), start_pose, b)
                    for p in poses)
        if worst > tol:
            self._fail(start, end,
                       f"cartesian_path bowed {worst:.4f} m off the straight "
                       f"segment (tolerance {tol:.4f} m)")
            return False
        return True

    @staticmethod
    def _winner(result) -> int:
        picked = list(getattr(result, "selected_goal_index", None) or [])
        return int(picked[0]) if picked else 0

    def _log_debug(self, msg: str) -> None:  # pragma: no cover - logging
        node = getattr(self.robot, "_node", None)
        if node is not None:
            node.get_logger().debug(msg)

    def _cost_of(self, result) -> float:
        return cost_of(result, self.params)

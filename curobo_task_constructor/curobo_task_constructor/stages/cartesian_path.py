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

Straightness is the optimizer's job
-----------------------------------
The hold is a soft cost: cuRobo is free to leave the line when staying on it is
more expensive (an obstacle on the segment, a better IK branch). That is the
desired behaviour — the reference's ``min_fraction`` / ``jump_threshold``
rejection is deliberately NOT reproduced client-side. A bowed solve is the
optimizer's answer, not a failure to fall through on.
"""

from __future__ import annotations

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


@register_stage("cartesian_path")
class CartesianPath(TrajectoryStage):
    """One straight-line whole-task solve with whole-path axis holds."""

    def __init__(self, name=None, params=None):
        super().__init__(name, params)

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
        return full_request(self.robot, start.joint_state, [goalset], self.params)

    def make_end_state(self, start: InterfaceState, result, raw=None):
        return start.clone(joint_state=result.last_state)

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

    def _cost_of(self, result) -> float:
        return cost_of(result, self.params)

"""MultiToolCartesianPath — a straight-line simultaneous per-tool solve.

One multi-tool goalset carries parallel ``tool_frames`` / ``tool_poses`` (and no
``poses``), so the server plans every named tool frame to its absolute pose in
a single ``plan_pose`` solve. The goalset also carries whole-path axis holds
per tool (``tool_trajectory_constraints``, a flat ``6 * N`` list in wire order
``[theta_x, theta_y, theta_z, x, y, z]`` per tool), which the CARTESIAN
planner turns into one ``ToolPoseCriteria`` per frame and scores on every
non-terminal waypoint. That is what keeps every tool's descent on the
Cartesian segment instead of swinging through free space.

``goal.tools`` is a list of ``{frame: <link>, pose: {x,y,z,qx,qy,qz,qw}}``
entries, as built by the MTC wrapper ``stages.MultiToolCartesianPath``.

The legacy single-tool ``trajectory_constraints`` wire field is deliberately
left empty by this stage: the per-tool field already covers every tool, and
sending both would apply the first tool's criteria twice.
"""

from __future__ import annotations

from curobo_task_constructor.core.geom import Pose3
from curobo_task_constructor.core.registry import register_stage
from curobo_task_constructor.core.robot import GoalsetSpec
from curobo_task_constructor.core.stage import Solution
from curobo_task_constructor.core.state import InterfaceState
from curobo_task_constructor.stages._util import (
    PLANNER_KEYS,
    axis_holds,
    carry_missing_joints,
    cost_of,
    full_request,
)
from curobo_task_constructor.stages.multi_tool_move_to import MultiToolMoveTo


@register_stage("multi_tool_cartesian_path")
class MultiToolCartesianPath(MultiToolMoveTo):
    """One goalset reaching N named tool frames along a Cartesian line."""

    def __init__(self, name=None, params=None, direction="auto"):
        super().__init__(name, params, direction=direction)

    # ------------------------------------------------------------------
    # Goal resolution
    # ------------------------------------------------------------------
    def _holds(self, state: InterfaceState, tools: list) -> list:
        """Per-tool whole-path holds as a flat ``6 * N`` list.

        Each tool's group is the wire order ``[theta_x, theta_y, theta_z, x,
        y, z]`` (1 = hold). Explicit ``holds`` win; otherwise the legacy
        ``hold`` vector pins the first tool (all other groups zero), and
        otherwise every tool's hold is derived from its own FK start and goal.
        """
        count = len(tools)
        if "holds" in self.params:
            explicit = self.params.get("holds") or []
            if explicit and all(isinstance(group, (list, tuple))
                                for group in explicit):
                groups = [[int(c) for c in group] for group in explicit]
                if len(groups) != count or any(len(group) != 6
                                               for group in groups):
                    raise ValueError(
                        f"{type(self).__name__} {self.name!r}: holds must be "
                        f"{count} groups of 6 ints [theta_x, theta_y, "
                        f"theta_z, x, y, z], got {groups!r}")
                return [c for group in groups for c in group]
            flat = [int(c) for c in explicit]
            if len(flat) != 6 * count:
                raise ValueError(
                    f"{type(self).__name__} {self.name!r}: holds must be "
                    f"{count} groups of 6 ints or a flat {6 * count}-int list "
                    f"[theta_x, theta_y, theta_z, x, y, z] per tool, got "
                    f"{len(flat)}")
            return flat
        legacy = self.params.get("hold")
        if legacy:
            hold = [int(c) for c in legacy]
            if len(hold) != 6:
                raise ValueError(
                    f"{type(self).__name__} {self.name!r}: hold must have 6 "
                    f"ints [theta_x, theta_y, theta_z, x, y, z], got "
                    f"{len(hold)}")
            # Backward-compatible single-tool hold: the first tool gets it,
            # every other tool gets an all-zero (unconstrained) group.
            return hold + [0] * 6 * (count - 1)
        holds = []
        for frame, goal in tools:
            try:
                start_pose = Pose3.from_any(
                    self.robot.fk(state.joint_state, frame))
            except Exception as exc:  # noqa: BLE001 - ServiceError etc.
                raise ValueError(
                    f"{type(self).__name__} {self.name!r}: FK of the start for "
                    f"{frame!r} failed: {exc!r}") from exc
            holds.extend(axis_holds(
                start_pose, [goal],
                pos_tol=float(self.params.get("pos_tol", 0.005)),
                rot_tol=float(self.params.get("rot_tol", 0.05))))
        return holds

    def _goalset(self, state: InterfaceState) -> GoalsetSpec:
        tools = self._tools()
        holds = self._holds(state, tools)
        self._last_holds = [holds[i:i + 6]
                            for i in range(0, len(holds), 6)]
        return GoalsetSpec(
            tool_frames=[frame for frame, _ in tools],
            tool_poses=[pose for _, pose in tools],
            allowed_collisions=(state.scene.all_allowed_links()
                                if state.scene is not None else []),
            tool_trajectory_constraints=holds,
        )

    def _planner(self):
        """The Cartesian backend (SetPlanner CARTESIAN, not CLASSIC)."""
        return PLANNER_KEYS["cartesian"]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def compute_forward(self, state: InterfaceState) -> None:
        try:
            goalset = self._goalset(state)
        except ValueError as exc:
            self._fail(state, None, str(exc))
            return
        req = full_request(self.robot, state.joint_state, [goalset],
                           self.params, planner=self._planner(),
                           scene=state.scene)
        last_error = "multi_tool_cartesian_path plan failed"
        for _ in range(self._attempts()):
            try:
                with self.robot.allowance_scope([req]):
                    result = self._timed_plan(req)
            except Exception as exc:  # ServiceError etc.
                last_error = f"plan call failed: {exc!r}"
                continue
            if not result.success:
                last_error = result.message or (
                    "multi_tool_cartesian_path plan failed")
                continue
            self._note_plan_attempt()
            end = state.clone(joint_state=carry_missing_joints(
                state.joint_state, result.last_state))
            sol = Solution(
                state, end, trajectory=result.trajectory,
                cost=cost_of(result, self.params), comment=self._comment(result),
                response=result.raw, plan_request=req)
            self._consider(sol)
            self._promote_forward(state, end, sol)
            return
        self._fail(state, None, last_error)

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------
    def compute_backward(self, state: InterfaceState) -> None:
        """Absolute per-tool poses have no start to solve for: forward only."""
        self._fail(state, None,
                   "multi_tool_cartesian_path is forward-only (per-tool goals "
                   "are absolute poses, not a joint-space end state)")

    def _comment(self, result=None) -> str:
        tools = (self.params.get("goal") or {}).get("tools") or []
        frames = [str(t.get("frame", "")) for t in tools
                  if isinstance(t, dict)]
        picked = ""
        if result is not None and getattr(result, "selected_goal_index", None):
            picked = f" -> candidate {list(result.selected_goal_index)}"
        return (f"multi_tool_cartesian_path tools={frames} "
                f"holds={list(getattr(self, '_last_holds', []))}{picked}")


__all__ = ["MultiToolCartesianPath"]

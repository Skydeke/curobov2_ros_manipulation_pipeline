"""MultiToolMoveTo — propagate to simultaneous per-tool pose goals.

A multi-tool goal is ONE goalset carrying parallel ``tool_frames`` /
``tool_poses`` (and no ``poses``): the server plans every named tool frame to
its absolute pose in a single ``plan_pose`` solve. Bimanual is the two-tool
case; the list is N-tool. The stage therefore never calls FK for a single
link — the goals are already absolute poses.

``goal.tools`` is a list of ``{frame: <link>, pose: {x,y,z,qx,qy,qz,qw}}``
entries, as built by the MTC wrapper ``stages.MultiToolMoveTo``.
"""

from __future__ import annotations

from curobo_task_constructor.core.registry import register_stage
from curobo_task_constructor.core.robot import GoalsetSpec
from curobo_task_constructor.core.stage import PropagatingEitherWay, Solution
from curobo_task_constructor.core.state import InterfaceState
from curobo_task_constructor.stages._util import (
    PLANNER_KEYS,
    carry_missing_joints,
    cost_of,
    full_request,
    planner_key,
    pose_from_params,
)


@register_stage("multi_tool_move_to")
class MultiToolMoveTo(PropagatingEitherWay):
    """One goalset reaching N named tool frames at once (multi-tool; 2 = bimanual)."""

    def __init__(self, name=None, params=None, direction="auto"):
        # declarative support: a YAML spec may pin the direction
        if direction == "auto" and params and params.get("direction"):
            direction = params["direction"]
        super().__init__(name, params, direction=direction)

    # ------------------------------------------------------------------
    # Goal resolution
    # ------------------------------------------------------------------
    def _tools(self) -> list:
        """``[(frame, Pose-like), ...]`` from ``goal.tools``.

        Validates the shape up front so a malformed stage fails with a clear
        message instead of sending an empty or mismatched goalset.
        """
        goal = self.params.get("goal") or {}
        tools = goal.get("tools")
        if not isinstance(tools, list) or not tools:
            raise ValueError(
                f"{type(self).__name__} {self.name!r}: goal.tools must be a "
                "non-empty list of {frame, pose} entries")
        out = []
        for i, tool in enumerate(tools):
            if not isinstance(tool, dict):
                raise ValueError(
                    f"{type(self).__name__} {self.name!r}: goal.tools[{i}] must be "
                    "a dict with 'frame' and 'pose'")
            frame = str(tool.get("frame") or "").strip()
            if not frame:
                raise ValueError(
                    f"{type(self).__name__} {self.name!r}: goal.tools[{i}] has no "
                    "'frame'")
            pose = tool.get("pose")
            if not isinstance(pose, dict):
                raise ValueError(
                    f"{type(self).__name__} {self.name!r}: goal.tools[{i}] has no "
                    "'pose' dict")
            out.append((frame, pose_from_params(pose, self.robot)))
        frames = [frame for frame, _ in out]
        if len(set(frames)) != len(frames):
            raise ValueError(
                f"{type(self).__name__} {self.name!r}: goal.tools repeats a tool "
                f"frame ({frames}); every frame may appear once")
        return out

    def _goalset(self, state: InterfaceState) -> GoalsetSpec:
        tools = self._tools()
        return GoalsetSpec(
            tool_frames=[frame for frame, _ in tools],
            tool_poses=[pose for _, pose in tools],
            allowed_collisions=(state.scene.all_allowed_links()
                                if state.scene is not None else []),
        )

    def _planner(self):
        """Planner key for the request (CLASSIC unless the caller overrides).

        The server's multi-tool path lives in the classic planner's
        ``plan_pose``; selecting another planner by default would route the
        request somewhere that ignores ``tool_frames``.
        """
        planner = planner_key(self.params)
        return PLANNER_KEYS["classic"] if planner is None else planner

    def _attempts(self) -> int:
        return max(1, int(self.params.get("planning_attempts", 3)))

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
        last_error = "multi_tool_move_to plan failed"
        for _ in range(self._attempts()):
            try:
                with self.robot.allowance_scope([req]):
                    result = self._timed_plan(req)
            except Exception as exc:  # ServiceError etc.
                last_error = f"plan call failed: {exc!r}"
                continue
            if not result.success:
                last_error = result.message or "multi_tool_move_to plan failed"
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
                   "multi_tool_move_to is forward-only (per-tool goals are "
                   "absolute poses, not a joint-space end state)")

    def _comment(self, result=None) -> str:
        tools = (self.params.get("goal") or {}).get("tools") or []
        frames = [str(t.get("frame", "")) for t in tools
                  if isinstance(t, dict)]
        picked = ""
        if result is not None and getattr(result, "selected_goal_index", None):
            picked = f" -> candidate {list(result.selected_goal_index)}"
        return f"multi_tool_move_to tools={frames}{picked}"


__all__ = ["MultiToolMoveTo"]

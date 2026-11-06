"""MultiToolWaypoints — follow ordered per-tool Cartesian paths together.

One multi-tool waypoint goalset carries parallel ``tool_frames`` and a flat
``tool_waypoints`` list in **waypoint-major** order::

    [wp0 tool0, wp0 tool1, ..., wp1 tool0, wp1 tool1, ...]

That is the native layout both the Python bridge and the C++ runtime consume,
so the stage flattens the natural tool-major input (one ordered pose list per
tool) into it. Every tool must provide the same number of waypoints ``K``;
``waypoint_count`` carries ``K``.

Like :class:`~curobo_task_constructor.stages.cartesian_waypoints.CartesianWaypoints`
this is FULL pose tracking, not axis holds: no ``trajectory_constraints`` and
no ``tool_trajectory_constraints`` are ever sent, and the stage selects the
MULTIPOINT planner (``SetPlanner.MULTIPOINT`` = 4). ``goal.tools`` is a list of
``{frame: <link>, waypoints: [<pose>, ...]}`` entries (``pose``/``poses`` are
accepted as aliases for a single / list target).
"""

from __future__ import annotations

from curobo_task_constructor.core.registry import register_stage
from curobo_task_constructor.core.robot import GoalsetSpec
from curobo_task_constructor.core.state import InterfaceState
from curobo_task_constructor.stages._util import (
    PLANNER_KEYS,
    pose_from_params,
    waypoint_goalset_options,
)
from curobo_task_constructor.stages.multi_tool_move_to import MultiToolMoveTo


def _as_waypoint_list(value, *, frame: str, index: int, stage: str) -> list:
    """Normalize one tool's goal (single pose / list) to a pose list."""
    if isinstance(value, dict) and {"x", "y", "z"} <= set(value):
        value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(
            f"{stage}: goal.tools[{index}] ({frame!r}) needs a non-empty "
            "'waypoints' list of ordered poses")
    out = []
    for j, pose in enumerate(value):
        if not isinstance(pose, dict) or not {"x", "y", "z"} <= set(pose):
            raise ValueError(
                f"{stage}: goal.tools[{index}] ({frame!r}) waypoints[{j}] must "
                "be a pose dict {x,y,z,qx,qy,qz,qw}")
        out.append(pose)
    return out


@register_stage("multi_tool_waypoints")
class MultiToolWaypoints(MultiToolMoveTo):
    """One goalset following ordered poses for every named tool frame."""

    # ------------------------------------------------------------------
    # Goal resolution
    # ------------------------------------------------------------------
    def _tools(self) -> list:
        """``[(frame, [Pose-like, ...]), ...]`` from ``goal.tools``.

        The tool list may also be a ``{frame: [poses]}`` mapping (the shape the
        MTC wrapper's ``setWaypoints`` accepts), so YAML-authored tasks have a
        compact spelling.
        """
        goal = self.params.get("goal") or {}
        tools = goal.get("tools")
        if isinstance(tools, dict):
            tools = [{"frame": frame, "waypoints": poses}
                     for frame, poses in tools.items()]
        if not isinstance(tools, list) or not tools:
            raise ValueError(
                f"{type(self).__name__} {self.name!r}: goal.tools must be a "
                "non-empty list of {frame, waypoints} entries")
        stage = f"{type(self).__name__} {self.name!r}"
        out = []
        for i, tool in enumerate(tools):
            if isinstance(tool, (list, tuple)) and len(tool) == 2:
                tool = {"frame": tool[0], "waypoints": tool[1]}
            if not isinstance(tool, dict):
                raise ValueError(
                    f"{stage}: goal.tools[{i}] must be a dict with 'frame' and "
                    "'waypoints'")
            frame = str(tool.get("frame") or tool.get("link") or "").strip()
            if not frame:
                raise ValueError(
                    f"{stage}: goal.tools[{i}] has no 'frame'")
            raw = tool.get("waypoints")
            if raw is None:
                raw = tool.get("poses", tool.get("pose"))
            out.append((frame, _as_waypoint_list(
                raw, frame=frame, index=i, stage=stage)))
        frames = [frame for frame, _ in out]
        if len(set(frames)) != len(frames):
            raise ValueError(
                f"{stage}: goal.tools repeats a tool frame ({frames}); every "
                "frame may appear once")
        return [(frame, [pose_from_params(pose, self.robot) for pose in poses])
                for frame, poses in out]

    def _flatten(self) -> tuple:
        """Return ``(tool_frames, waypoint-major poses, K)``."""
        tools = self._tools()
        counts = {len(waypoints) for _, waypoints in tools}
        if len(counts) != 1:
            per_tool = {frame: len(waypoints) for frame, waypoints in tools}
            raise ValueError(
                f"{type(self).__name__} {self.name!r}: every tool needs the "
                f"same number of ordered waypoints, got {per_tool}")
        count = counts.pop()
        flat = [waypoints[index]
                for index in range(count)
                for _, waypoints in tools]
        return [frame for frame, _ in tools], flat, count

    def _goalset(self, state: InterfaceState) -> GoalsetSpec:
        frames, tool_waypoints, count = self._flatten()
        # Deliberately NO trajectory_constraints / tool_trajectory_constraints:
        # waypoint mode tracks full poses, it does not hold axes.
        return GoalsetSpec(
            tool_frames=frames,
            tool_waypoints=tool_waypoints,
            waypoint_count=count,
            allowed_collisions=(state.scene.all_allowed_links()
                                if state.scene is not None else []),
            **waypoint_goalset_options(self.params),
        )

    def _planner(self):
        """The ordered-waypoint backend (MULTIPOINT = 4)."""
        return PLANNER_KEYS["multipoint"]

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------
    def compute_backward(self, state: InterfaceState) -> None:
        """Ordered per-tool poses have no start to solve for: forward only."""
        self._fail(state, None,
                   "multi_tool_waypoints is forward-only (ordered waypoints are "
                   "absolute poses, not a joint-space end state)")

    def _comment(self, result=None) -> str:
        goal = self.params.get("goal") or {}
        tools = goal.get("tools") or []
        frames = []
        count = 0
        if isinstance(tools, dict):
            for frame, poses in tools.items():
                frames.append(str(frame))
                count = max(count, len(poses) if isinstance(poses, list) else 1)
        else:
            for tool in tools:
                if isinstance(tool, dict):
                    frame = tool.get("frame") or tool.get("link") or ""
                    raw = tool.get("waypoints")
                    if raw is None:
                        raw = tool.get("poses", tool.get("pose"))
                    count = max(count, len(raw) if isinstance(raw, list) else 1)
                elif isinstance(tool, (list, tuple)) and len(tool) == 2:
                    frame, raw = tool
                    count = max(count, len(raw) if isinstance(raw, list) else 1)
                else:
                    frame = ""
                frames.append(str(frame))
        picked = ""
        if result is not None and getattr(result, "selected_goal_index", None):
            picked = f" -> candidate {list(result.selected_goal_index)}"
        return (f"multi_tool_waypoints tools={frames} waypoints={count}{picked}")


__all__ = ["MultiToolWaypoints"]

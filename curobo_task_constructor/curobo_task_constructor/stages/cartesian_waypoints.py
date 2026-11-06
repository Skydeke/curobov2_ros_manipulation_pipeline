"""CartesianWaypoints — follow an ordered single-tool Cartesian path.

A waypoint goalset is ONE goalset carrying ``waypoints`` (``K`` absolute tool
poses, the start excluded) plus the path/final tolerances. Unlike
``cartesian_path`` there are no axis holds: the server's MULTIPOINT planner
(``SetPlanner.MULTIPOINT`` = 4) follows the ordered poses directly through
``plan_pose_waypoints`` — positions interpolate linearly between waypoints and
orientations use SLERP, so the whole path is tracked rather than derived from
an axis-hold cost.

``goal.waypoints`` is a non-empty list of ``{x,y,z,qx,qy,qz,qw}`` poses. An
optional ``goal.link`` / ``link`` names the tool frame; when omitted the
server's default tool frame is used (the single-tool waypoint wire form has no
per-tool frame list, so it must not masquerade as a multi-tool goal).

This stage is forward-only: the first waypoint is planned from the state's
actual ``FK(start)``.
"""

from __future__ import annotations

from typing import Optional

from curobo_task_constructor.core.registry import register_stage
from curobo_task_constructor.core.robot import GoalsetSpec
from curobo_task_constructor.core.state import InterfaceState
from curobo_task_constructor.stages._util import (
    PLANNER_KEYS,
    pose_from_params,
    waypoint_goalset_options,
)
from curobo_task_constructor.stages.multi_tool_move_to import MultiToolMoveTo


@register_stage("cartesian_waypoints")
class CartesianWaypoints(MultiToolMoveTo):
    """One goalset following ``K`` ordered tool poses (MULTIPOINT)."""

    # ------------------------------------------------------------------
    # Goal resolution
    # ------------------------------------------------------------------
    def _link(self) -> Optional[str]:
        goal = self.params.get("goal") or {}
        link = goal.get("link") or self.params.get("link")
        return str(link).strip() if link else None

    def _waypoints(self) -> list:
        """Validate and return the ordered ``goal.waypoints`` poses."""
        goal = self.params.get("goal") or {}
        waypoints = goal.get("waypoints")
        if not isinstance(waypoints, list) or not waypoints:
            raise ValueError(
                f"{type(self).__name__} {self.name!r}: goal.waypoints must be "
                "a non-empty list of ordered {x,y,z,qx,qy,qz,qw} poses (the "
                "start pose is excluded)")
        out = []
        for i, pose in enumerate(waypoints):
            if not isinstance(pose, dict) or not {"x", "y", "z"} <= set(pose):
                raise ValueError(
                    f"{type(self).__name__} {self.name!r}: goal.waypoints[{i}] "
                    "must be a pose dict {x,y,z,qx,qy,qz,qw}")
            out.append(pose_from_params(pose, self.robot))
        return out

    def _goalset(self, state: InterfaceState) -> GoalsetSpec:
        waypoints = self._waypoints()
        link = self._link()
        return GoalsetSpec(
            waypoints=waypoints,
            # Only name a frame when the caller asked for one; the single-tool
            # wire form reaches the default tool frame otherwise.
            tool_frames=[link] if link else [],
            waypoint_count=len(waypoints),
            allowed_collisions=(state.scene.all_allowed_links()
                                if state.scene is not None else []),
            **waypoint_goalset_options(self.params),
        )

    def _planner(self):
        """The ordered-waypoint backend.

        A waypoint goalset is only consumed by MULTIPOINT; selecting any other
        planner would silently ignore ``waypoints``, so the stage always picks
        it (an explicit ``planner`` param cannot redirect the wire shape).
        """
        return PLANNER_KEYS["multipoint"]

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------
    def compute_backward(self, state: InterfaceState) -> None:
        """Ordered poses have no start to solve for: forward only."""
        self._fail(state, None,
                   "cartesian_waypoints is forward-only (ordered waypoints are "
                   "absolute poses, not a joint-space end state)")

    def _comment(self, result=None) -> str:
        picked = ""
        if result is not None and getattr(result, "selected_goal_index", None):
            picked = f" -> candidate {list(result.selected_goal_index)}"
        link = self._link()
        frame = f" link={link}" if link else ""
        return (f"cartesian_waypoints waypoints="
                f"{len((self.params.get('goal') or {}).get('waypoints') or [])}"
                f"{frame}{picked}")


__all__ = ["CartesianWaypoints"]

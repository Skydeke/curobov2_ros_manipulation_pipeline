#!/usr/bin/env python3
"""
Classic trajectory planner (v2: MotionPlanner.plan_pose).

v2 notes:
- MotionGen → MotionPlanner (wired via SinglePlanner._shared_motion_planner).
- MotionGenPlanConfig is gone: per-call params are kwargs on plan_pose().
- Pose → ToolPose, wrapped in GoalToolPose.
- PoseCostMetric was removed upstream in v2. Trajectory-axis constraints
  (`trajectory_constraints`) are re-wired via `ToolPoseCriteria` (held along
  the whole path) — see SinglePlanner._apply_pose_constraints.
"""

from curobo.types import JointState, Pose, GoalToolPose

from .single_planner import SinglePlanner


class ClassicPlanner(SinglePlanner):
    """
    Classic motion generation planner.

    Uses cuRobo's MotionPlanner to generate a complete collision-free trajectory
    from start to goal, which is then executed in open-loop.

    - Takes a single waypoint segment (`goalsets` with exactly 1 entry).
    - That segment may hold N candidate poses: cuRobo resolves the set inside
      one plan_pose() call (via its goalset path) and the winner is reported
      via `selected_goal_index`.
    - Generates a full trajectory in one shot via plan_pose().
    - Executes the trajectory as-is (no post-processing).
    """

    def get_planner_name(self) -> str:
        return "Classic Motion Generation"

    def _plan_trajectory(
        self,
        start_state: JointState,
        goal_request,
        config: dict,
    ):
        """
        Generate trajectory using MotionPlanner.plan_pose().

        Args:
            start_state: Initial joint configuration.
            goal_request: TrajectoryGeneration request (uses goalsets).
            config: Dict with keys:
                - max_attempts (default 1)
                - timeout (default 5.0)
                - time_dilation_factor (default 0.5)

        Returns:
            MotionPlannerResult-like object.
        """
        num_goalset = max(
            (len(getattr(g, 'poses', [])) for g in getattr(goal_request, 'goalsets', [])),
            default=0,
        )
        # Node validation rejects empty goalsets for classic; be defensive since
        # plan() forwards invalid requests straight here.
        if num_goalset == 0:
            self.node.get_logger().warn(
                f"{self.get_planner_name()}: goalsets must contain exactly 1 "
                f"segment with >= 1 pose for classic planning, got empty request"
            )
            return None

        # v2: Cartesian axis constraints via ToolPoseCriteria (held along the
        # whole path). Reset afterwards since the MotionPlanner is shared.
        applied = self._apply_pose_constraints(goal_request)

        try:
            if num_goalset > 1:
                # Goalset solve: cuRobo's _plan_pose_goalset path resolves all
                # N candidates inside ONE plan_pose() call and reports the
                # winner via result.goalset_index.
                self.node.get_logger().info(
                    f"{self.get_planner_name()}: planning to candidate set "
                    f"({num_goalset} poses) resolved within a single solve"
                )
                goal = self._build_goal_segments(goal_request)[0]
            else:
                # Single fixed waypoint — today's single-goal plan.
                goal = Pose.from_list([
                    goal_request.goalsets[0].poses[0].position.x,
                    goal_request.goalsets[0].poses[0].position.y,
                    goal_request.goalsets[0].poses[0].position.z,
                    goal_request.goalsets[0].poses[0].orientation.w,
                    goal_request.goalsets[0].poses[0].orientation.x,
                    goal_request.goalsets[0].poses[0].orientation.y,
                    goal_request.goalsets[0].poses[0].orientation.z,
                ])
                goal = GoalToolPose.from_poses({self.motion_planner.tool_frames[0]: goal})

            max_attempts = config.get('max_attempts', 1)

            self.node.get_logger().info(f"Planning with max_attempts={max_attempts}")

            result = self.motion_planner.plan_pose(
                goal,
                start_state,
                max_attempts=max_attempts,
            )
        finally:
            if applied:
                self._reset_pose_criteria()

        self._selected_goal_indexes = [self._select_goal_index(result)]
        return result

    # _process_trajectory(): default (no-op) from SinglePlanner is fine.
    # execute() / cancel(): inherited from SinglePlanner.

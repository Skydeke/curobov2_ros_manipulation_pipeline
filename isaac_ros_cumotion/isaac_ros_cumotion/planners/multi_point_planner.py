"""
Multi-point trajectory planner (v2: MotionPlanner.plan_pose / plan_cspace).

v2 notes:
- PoseCostMetric is removed; whole-path axis constraints
  (``Goalset.trajectory_constraints``) are re-wired via ToolPoseCriteria and
  applied to the whole plan.  Per-segment ``trajectories_contraints`` remain
  unsupported (a single criteria spans the plan); those goalset segments that
  set it will be warned once.
- MotionGenPlanConfig is gone; per-call params become kwargs on plan_pose().
- Kunz-Stilman retiming path is dropped (v2 exposes retiming differently and
  the v1 code was already falling back to raw stacked segments in practice).
  v2 users that need smoother blending can switch to a single goalset call.

Per-segment allowed collisions:
- A segment may carry ``allowed_collisions`` (link names) to disable the
  corresponding collision spheres for the duration of that segment's solve
  (re-enabled afterwards, exception-safe). Used e.g. to allow the gripper
  fingers to contact the grasped object during a grasp approach/lift segment.

Per-segment joint targets:
- A goalset segment that carries ``target_joint_positions`` (and no/empty
  ``poses``) is dispatched to ``plan_cspace()`` (joint-space planning).
  A segment that carries ``poses`` is dispatched to ``plan_pose()``
  (Cartesian planning). The two may be interleaved freely. After each
  segment the end-state is projected back onto the active-DOF joint names
  so the next segment's IK seed / start state matches the solver's action
  dimensions.
"""

import torch
from curobo.types import JointState, Pose, GoalToolPose
from curobo._src.state.state_joint_ops import stack_joint_states

from .single_planner import SinglePlanner


class MultiPointPlanner(SinglePlanner):
    """
    Multi-waypoint planner built on top of MotionPlanner.plan_pose().

    Plans each waypoint sequentially, then stacks interpolated segments into
    a single trajectory for open-loop execution.
    """

    def get_planner_name(self) -> str:
        return "Multi-Point Motion Generation"

    def _build_segment_goal(self, gset, start_state):
        """Build the solver goal and tag for one Goalset entry.

        Returns ``('joint', JointState)`` when ``target_joint_positions`` is
        non-empty, else ``('pose', GoalToolPose)``.  ``None`` is returned when
        the segment is empty (zero poses AND zero joint targets) — the node
        rejects such requests before planning, so this is a defensive backstop.
        """
        joint_targets = list(getattr(gset, 'target_joint_positions', None) or [])
        if joint_targets:
            # Short joint target (arm-only): keep the trailing end-effector DOF
            # (gripper) at its current/start value, mirroring JointSpacePlanner.
            robot_dof = self.motion_planner.kinematics.get_dof()
            if len(joint_targets) < robot_dof:
                start_tail = start_state.position[0][len(joint_targets):].cpu().tolist()
                joint_targets = joint_targets + start_tail
                self.node.get_logger().info(
                    f"Joint target shorter than robot DOF ({robot_dof}): padded "
                    f"trailing DOF with start-state values "
                    f"{[f'{x:.3f}' for x in start_tail]}"
                )
            goal_state = JointState.from_position(
                torch.tensor(
                    [joint_targets],
                    dtype=start_state.position.dtype,
                    device=start_state.position.device,
                ),
            )
            return ('joint', goal_state, len(joint_targets))

        poses = [
            [p.position.x, p.position.y, p.position.z,
             p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z]
            for p in gset.poses
        ]
        if not poses:
            return None
        tool_frame = self.motion_planner.tool_frames[0]
        pose = Pose.from_batch_list(poses)
        goal = GoalToolPose.from_poses({tool_frame: pose}, num_goalset=len(poses))
        return ('pose', goal, len(poses))

    def _plan_trajectory(
        self,
        start_state: JointState,
        goal_request,
        config: dict,
    ):
        goalsets = list(getattr(goal_request, 'goalsets', None) or [])
        if not goalsets:
            self.node.get_logger().warn(
                "MultiPointPlanner: goalsets is empty - nothing to plan to"
            )
            return None

        # Per-segment allowed collisions (toggle collision spheres around each solve).
        allowed_per_seg = [
            list(getattr(g, 'allowed_collisions', None) or []) for g in goalsets
        ]

        max_attempts = config.get('max_attempts', 1)

        # v2: per-segment ``trajectories_contraints`` (flattened per-waypoint
        # holds) are not expressible — a single ToolPoseCriteria spans the
        # whole plan.  Warn once if any segment tries to set them.
        for i, g in enumerate(goalsets):
            c = list(getattr(g, 'trajectories_contraints', None) or [])
            if c and any(x == 1 for x in c):
                self.node.get_logger().warn(
                    f"MultiPointPlanner: per-waypoint `trajectories_contraints` "
                    f"on segment {i} are not honoured in cuRobo v2 — use "
                    f"`trajectory_constraints` on segment 0 (whole path) instead."
                )
                break  # warn once

        # Hold the requested Cartesian axes along every segment; reset after
        # (the MotionPlanner is shared across planners).
        applied = self._apply_pose_constraints(goal_request)
        selected = []
        try:
            current_state = start_state.clone()
            combined_trajectory = None
            self._combined_trajectory = None
            last_result = None

            for i, gset in enumerate(goalsets):
                seg = self._build_segment_goal(gset, current_state)
                if seg is None:
                    self.node.get_logger().warn(
                        f"MultiPointPlanner: segment {i} has neither poses nor "
                        f"target_joint_positions — skipping"
                    )
                    continue

                kind, goal, _size = seg
                current_state.velocity[:] = 0.0
                current_state.acceleration[:] = 0.0

                # Per-segment allowed collisions: disable before solve, re-enable after.
                links = allowed_per_seg[i] if i < len(allowed_per_seg) else []
                if links:
                    self.motion_planner.disable_link_collision(links)
                    self.node.get_logger().info(
                        f"Segment {i} ({kind}): disabled collision spheres for {links}"
                    )

                try:
                    if kind == 'joint':
                        enable_graph_attempt = config.get('enable_graph_attempt', 1)
                        result = self.motion_planner.plan_cspace(
                            goal,
                            current_state.clone(),
                            max_attempts=max_attempts,
                            enable_graph_attempt=enable_graph_attempt,
                        )
                    else:
                        result = self.motion_planner.plan_pose(
                            goal,
                            current_state.clone(),
                            max_attempts=max_attempts,
                        )
                finally:
                    if links:
                        self.motion_planner.enable_link_collision(links)
                        self.node.get_logger().info(
                            f"Segment {i}: re-enabled collision spheres for {links}"
                        )

                selected.append(self._select_goal_index(result))

                if result is None:
                    self._selected_goal_indexes = selected
                    self.node.get_logger().error(
                        f"Failed to plan segment {i} ({kind}): no solution found"
                    )
                    return None

                if not result.success.item():
                    self._selected_goal_indexes = selected
                    status = getattr(result, 'status', None) or "unknown"
                    self.node.get_logger().error(
                        f"Failed to plan segment {i} ({kind}): {status}"
                    )
                    return result

                segment = result.get_interpolated_plan()

                if combined_trajectory is None:
                    combined_trajectory = segment
                else:
                    combined_trajectory = stack_joint_states(
                        combined_trajectory, segment.clone())

                # Build the next start state from the final waypoint of the segment.
                last_pos = segment.position[..., -1, :]
                while last_pos.ndim > 2:
                    last_pos = last_pos[0]
                if segment.joint_names is not None:
                    segment_end_js = JointState.from_position(
                        last_pos.clone(),
                        joint_names=segment.joint_names,
                    )
                    current_state = self.motion_planner.kinematics.get_active_js(segment_end_js)
                else:
                    current_state = JointState.from_position(
                        last_pos.clone(),
                        joint_names=start_state.joint_names,
                    )
                last_result = result

            self._selected_goal_indexes = selected
            self._combined_trajectory = combined_trajectory
            return last_result
        finally:
            if applied:
                self._reset_pose_criteria()

    def _process_trajectory(self, trajectory: JointState, config: dict) -> JointState:
        """Return the stacked multi-waypoint trajectory built in _plan_trajectory."""
        if self._combined_trajectory is not None:
            self.node.get_logger().info(
                f"MultiPointPlanner: returning combined trajectory with "
                f"{len(self._combined_trajectory.position)} waypoints"
            )
            return self._combined_trajectory

        self.node.get_logger().warn("_combined_trajectory not set, using single segment")
        return trajectory

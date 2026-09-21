#!/usr/bin/env python3
"""
Joint space trajectory planner (v2).

v2 notes:
- MotionGen.plan_single_js() → MotionPlanner.plan_cspace() (joint goal).
- MotionGenPlanConfig is gone; per-call params are kwargs on plan_cspace().
- v2 plan_cspace() no longer accepts timeout/time_dilation_factor/
  enable_graph/enable_opt — those tunables live on the trajopt YAML.
"""

import torch
from curobo.types import JointState

from .single_planner import SinglePlanner


class JointSpacePlanner(SinglePlanner):
    """
    Joint-space planner using MotionPlanner.plan_joint_state() (v2).

    Plans directly in joint space — no IK, no singularities to worry about.
    Ideal when the goal is already expressed as a joint configuration.
    """

    def get_planner_name(self) -> str:
        return "Joint Space Motion Generation"

    def _plan_trajectory(
        self,
        start_state: JointState,
        goal_request,
        config: dict,
    ):
        goalsets = list(getattr(goal_request, 'goalsets', None) or [])
        goal_joint_positions = list(getattr(goalsets[0], 'target_joint_positions', None) or []) \
            if goalsets else []
        if not goal_joint_positions:
            raise ValueError(
                "JointSpacePlanner requires a non-empty 'target_joint_positions' "
                "in goalsets[0] (the srv/action no longer expose a top-level "
                "joint array)."
            )

        robot_dof = self.motion_planner.kinematics.get_dof()
        if len(goal_joint_positions) > robot_dof:
            raise ValueError(
                f"Joint count mismatch: received {len(goal_joint_positions)} joints, "
                f"but robot has {robot_dof} DOF"
            )
        if len(goal_joint_positions) < robot_dof:
            # Short joint target (e.g. an arm-only MoveIt goal covering only the
            # manipulator's DOF): keep the trailing end-effector DOF (gripper)
            # at its current/start value instead of rejecting the request.
            start_tail = start_state.position[0][len(goal_joint_positions):].cpu().tolist()
            goal_joint_positions = goal_joint_positions + start_tail
            self.node.get_logger().info(
                f"Joint target shorter than robot DOF ({robot_dof}): padded "
                f"trailing DOF with start-state values {[f'{x:.3f}' for x in start_tail]}"
            )
        if any(not (-1e6 < x < 1e6) or x != x for x in goal_joint_positions):
            raise ValueError(
                f"Invalid joint positions (NaN/Inf): {goal_joint_positions}"
            )

        goal_state = JointState.from_position(
            torch.tensor(
                [goal_joint_positions],
                dtype=start_state.position.dtype,
                device=start_state.position.device,
            )
        )

        max_attempts = config.get('max_attempts', 1)
        enable_graph_attempt = config.get('enable_graph_attempt', 1)

        start_pos = start_state.position[0].cpu().tolist()
        goal_pos = list(goal_joint_positions)
        self.node.get_logger().info("Planning joint space trajectory:")
        self.node.get_logger().info(f"  Start: {[f'{x:.3f}' for x in start_pos]}")
        self.node.get_logger().info(f"  Goal:  {[f'{x:.3f}' for x in goal_pos]}")
        self.node.get_logger().info(
            f"  Config: max_attempts={max_attempts}, "
            f"enable_graph_attempt={enable_graph_attempt}"
        )

        # Contact allowance rides on the goalset (the SetLinkCollision service
        # is no longer used for this): disable the listed links' collision
        # spheres for the solve only, then re-enable (exception-safe).
        allowed = list(getattr(goalsets[0], 'allowed_collisions', None) or [])
        if allowed:
            self.motion_planner.disable_link_collision(allowed)
            self.node.get_logger().info(
                f"Disabled collision spheres for contact links: {allowed}"
            )

        try:
            result = self.motion_planner.plan_cspace(
                goal_state,
                start_state,
                max_attempts=max_attempts,
                enable_graph_attempt=enable_graph_attempt,
            )
        finally:
            if allowed:
                self.motion_planner.enable_link_collision(allowed)
                self.node.get_logger().info(
                    f"Re-enabled collision spheres for contact links: {allowed}"
                )

        # Per-segment insight metadata (one entry for this single segment;
        # goalset candidate is 0/N/A for a joint-space solve).
        seg_ok = False
        if result is not None:
            succ = result.success
            seg_ok = bool(succ.item()) if hasattr(succ, 'item') else bool(succ)
        seed_id = self._select_seed_index(result)
        self._selected_goal_indexes = [self._select_goal_index(result)]
        self._selected_seed_index = [seed_id]
        self._waypoint_status = [self._segment_reached(
            result, seed_id, self._waypoint_tolerance, seg_ok)]
        self._candidate_tally = self._tally_candidates(result)
        self._considered_rows = self._segment_considered_rows(
            result, 0, self._selected_goal_indexes[0], self._log_considered)
        return result

#!/usr/bin/env python3
"""
Joint space trajectory planner (v2).

v2 notes:
- MotionGen.plan_single_js() → MotionPlanner.plan_cspace() (joint goal).
- MotionGenPlanConfig is gone; per-call params are kwargs on plan_cspace().
- v2 plan_cspace() no longer accepts timeout/time_dilation_factor/
  enable_graph/enable_opt — those tunables live on the trajopt YAML.
"""

import math

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

    def _wrap_goal_to_start_branch(
        self, goal_pos: list, start_pos: list
    ) -> list:
        """Normalize periodic (continuous) joint goals onto the start branch.

        cuRobo's URDF parser converts URDF ``continuous`` joints (which have NO
        position limits) into revolute joints with limits [-2π, 2π], but it does
        NOT treat them as periodic. A goal like 0.288 rad given while the joint
        reads -4.71 rad is therefore planned as a ~286° sweep instead of the
        short -73.6° path, making the arm spin in a circle (and, for a
        physically-wrapping joint, even report ending at the start reading).

        For every joint whose position limit spans at least a full revolution
        (a converted continuous joint), this rewrites the goal as the equivalent
        ``goal + 2πk`` value (within the declared limits) that requires the
        SHORTEST linear sweep from the CURRENT reading, so the planner always
        takes the shortest path. Because the joint is physically periodic, all
        such values describe the same configuration -- but cuRobo interpolates
        linearly in the non-periodic range, so the branch that minimizes
        ``|goal - start|`` is what produces the minimal physical rotation.
        Returns the (possibly adjusted) goal list.
        """
        try:
            limits = self.motion_planner.kinematics.get_joint_limits()
            pos_limits = limits.position  # [2, dof] rows [min, max]
        except Exception as e:
            self.node.get_logger().warn(
                f"Cannot read joint limits for continuity detection "
                f"({e}) — skipping goal wrap."
            )
            return goal_pos

        if len(start_pos) != len(goal_pos):
            return goal_pos
        if pos_limits.shape[1] < len(goal_pos):
            pos_limits = pos_limits[:, : len(goal_pos)]

        wrapped = list(goal_pos)
        for i, g in enumerate(goal_pos):
            low = float(pos_limits[0, i])
            high = float(pos_limits[1, i])
            if high - low < 2.0 * math.pi:
                continue  # finite joint — no wrap needed
            s = start_pos[i]
            # Branch indices that keep goal+2πk inside the declared limits.
            k_min = math.ceil((low - g) / (2.0 * math.pi))
            k_max = math.floor((high - g) / (2.0 * math.pi))
            best = None  # (linear_sweep, value)
            for k in range(k_min, k_max + 1):
                candidate = g + 2.0 * math.pi * k
                sweep = abs(candidate - s)
                if best is None or sweep < best[0]:
                    best = (sweep, candidate)
            if best is not None and best[0] > 1e-9:
                wrapped[i] = best[1]
        if wrapped != list(goal_pos):
            self.node.get_logger().info(
                "Joint-space goal wrapped to shortest path for periodic "
                f"(continuous) joints: goal {[f'{x:.3f}' for x in goal_pos]} -> "
                f"{[f'{x:.3f}' for x in wrapped]}"
            )
        return wrapped

    def _plan_trajectory(
        self,
        start_state: JointState,
        goal_request,
        config: dict,
    ):
        if not hasattr(goal_request, 'target_joint_positions'):
            raise ValueError(
                "JointSpacePlanner requires 'target_joint_positions' in the request."
            )

        goal_joint_positions = goal_request.target_joint_positions
        if not goal_joint_positions:
            raise ValueError("target_joint_positions is empty.")

        robot_dof = self.motion_planner.kinematics.get_dof()
        if len(goal_joint_positions) != robot_dof:
            raise ValueError(
                f"Joint count mismatch: received {len(goal_joint_positions)} joints, "
                f"but robot has {robot_dof} DOF"
            )
        if any(not (-1e6 < x < 1e6) or x != x for x in goal_joint_positions):
            raise ValueError(
                f"Invalid joint positions (NaN/Inf): {goal_joint_positions}"
            )

        start_pos = start_state.position[0].cpu().tolist()
        goal_pos = self._wrap_goal_to_start_branch(
            list(goal_joint_positions), start_pos)

        goal_state = JointState.from_position(
            torch.tensor(
                [goal_pos],
                dtype=start_state.position.dtype,
                device=start_state.position.device,
            )
        )

        max_attempts = config.get('max_attempts', 1)
        enable_graph_attempt = config.get('enable_graph_attempt', 1)

        self.node.get_logger().info("Planning joint space trajectory:")
        self.node.get_logger().info(f"  Start: {[f'{x:.3f}' for x in start_pos]}")
        self.node.get_logger().info(f"  Goal:  {[f'{x:.3f}' for x in goal_pos]}")
        self.node.get_logger().info(
            f"  Config: max_attempts={max_attempts}, "
            f"enable_graph_attempt={enable_graph_attempt}"
        )

        return self.motion_planner.plan_cspace(
            goal_state,
            start_state,
            max_attempts=max_attempts,
            enable_graph_attempt=enable_graph_attempt,
        )

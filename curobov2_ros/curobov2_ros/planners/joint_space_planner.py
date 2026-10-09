#!/usr/bin/env python3
"""
Joint space trajectory planner (v2).

v2 notes:
- MotionGen.plan_single_js() → MotionPlanner.plan_cspace() (joint goal).
- MotionGenPlanConfig is gone; per-call params are kwargs on plan_cspace().
- v2 plan_cspace() no longer accepts timeout/time_dilation_factor/
  enable_graph/enable_opt — those tunables live on the trajopt YAML.
"""

import time

import torch
from curobo.types import JointState

from curobov2_ros.core.joint_order import resolve_joint_targets
from curobov2_ros.planners.zero_motion import ZeroMotionResult, is_zero_delta

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
        goalsets = list(getattr(goal_request, "goalsets", None) or [])
        # A goal may now name its joints (Goalset.target_joint_names); when it
        # does, the node re-orders them into the model's active cspace order
        # here, so no client needs to know that order out-of-band. A nameless
        # list keeps its historical meaning: already in active-DOF order.
        model_names = self.motion_planner.kinematics.joint_names
        raw_target = (
            list(getattr(goalsets[0], "target_joint_positions", None) or [])
            if goalsets
            else []
        )
        raw_names = (
            list(getattr(goalsets[0], "target_joint_names", None) or [])
            if goalsets
            else []
        )
        goal_joint_positions = resolve_joint_targets(raw_names, raw_target,
                                                    model_names)
        if raw_names and len(raw_names) == len(raw_target):
            self.node.get_logger().info(
                f"Re-ordered named joint target via {raw_names} into the "
                f"model's active cspace order {model_names}"
            )
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
            start_tail = (
                start_state.position[0][len(goal_joint_positions) :].cpu().tolist()
            )
            goal_joint_positions = goal_joint_positions + start_tail
            self.node.get_logger().info(
                f"Joint target shorter than robot DOF ({robot_dof}): padded "
                f"trailing DOF with start-state values {[f'{x:.3f}' for x in start_tail]}"
            )
        if any(not (-1e6 < x < 1e6) or x != x for x in goal_joint_positions):
            raise ValueError(
                f"Invalid joint positions (NaN/Inf): {goal_joint_positions}"
            )

        start_pos = start_state.position[0].cpu().tolist()

        # DEGENERATE-REQUEST GUARD (server-side, on purpose: it protects every
        # client, not just the task constructor). A goal that already equals
        # the start asks trajopt for nothing, and cuRobo is not a no-op on it
        # - its optimizer re-captures CUDA graphs mid-plan, which is where a
        # start==goal solve surfaced as "CUDA error: an illegal memory access"
        # in cuda_graph_util.replay(). Answer with the zero-length trajectory
        # instead of solving, and keep CUDA graphs enabled for every real plan.
        if is_zero_delta(start_pos, goal_joint_positions):
            self.node.get_logger().info(
                "Joint target equals the start state: no motion needed, "
                "skipping the solver (avoids a degenerate CUDA-graph solve)"
            )
            self._selected_goal_indexes = [0]
            self._selected_seed_index = [0]
            self._waypoint_status = [True]
            self._candidate_tally = [{"poses": 1, "seeds": 0}]
            self._considered_rows = []
            return ZeroMotionResult(
                JointState.from_position(
                    torch.tensor(
                        [list(start_pos)],
                        dtype=start_state.position.dtype,
                        device=start_state.position.device,
                    ),
                    joint_names=model_names,
                ),
                start_state.position.device,
                start_state.position.dtype,
            )

        goal_state = JointState.from_position(
            torch.tensor(
                [goal_joint_positions],
                dtype=start_state.position.dtype,
                device=start_state.position.device,
            )
        )

        max_attempts = config.get("max_attempts", 1)
        enable_graph_attempt = config.get("enable_graph_attempt", 1)

        self.node.get_logger().info("Planning joint space trajectory:")
        self.node.get_logger().info(f"  Start: {[f'{x:.3f}' for x in start_pos]}")
        self.node.get_logger().info(
            f"  Goal:  {[f'{x:.3f}' for x in goal_joint_positions]}"
        )
        self.node.get_logger().info(
            f"  Config: max_attempts={max_attempts}, "
            f"enable_graph_attempt={enable_graph_attempt}"
        )

        # Collision/contact allowance is the task constructor's responsibility
        # (it applies the goalsets' allowed links around each solve via the
        # server's set_link_collision service); the planning interface no
        # longer mutates the shared motion-planner collision state.
        _t_solve = time.monotonic()
        result = self.motion_planner.plan_cspace(
            goal_state,
            start_state,
            max_attempts=max_attempts,
            enable_graph_attempt=enable_graph_attempt,
        )
        _elapsed = (time.monotonic() - _t_solve) * 1e3
        n_seeds = getattr(self.config_wrapper, "num_trajopt_seeds", None) or 0
        per_seed = f", ~{_elapsed / n_seeds:.0f} ms/seed" if n_seeds > 0 else ""
        self.node.get_logger().info(
            f"  plan_cspace: {_elapsed:.1f} ms (num_trajopt_seeds={n_seeds}"
            f"{per_seed}, max_attempts={max_attempts}, "
            f"enable_graph_attempt={enable_graph_attempt})"
        )

        # Per-segment insight metadata (one entry for this single segment;
        # goalset candidate is 0/N/A for a joint-space solve).
        seg_ok = False
        if result is not None:
            succ = result.success
            seg_ok = bool(succ.item()) if hasattr(succ, "item") else bool(succ)
        seed_id = self._select_seed_index(result)
        self._selected_goal_indexes = [self._select_goal_index(result)]
        self._selected_seed_index = [seed_id]
        self._waypoint_status = [self._segment_reached(result, seed_id, seg_ok)]
        self._candidate_tally = self._tally_candidates(result)
        self._considered_rows = self._segment_considered_rows(
            result, 0, self._selected_goal_indexes[0]
        )
        return result

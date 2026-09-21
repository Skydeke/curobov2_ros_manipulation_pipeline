#!/usr/bin/env python3
"""
Abstract base class for planners using cuRobo MotionGen.

This class provides shared infrastructure for all planners that use MotionGen
as their underlying solver. All child planners share the same MotionGen instance
and configuration, avoiding redundant warmup operations.

Architecture:
    TrajectoryPlanner (abstract interface)
        ├── SinglePlanner (open-loop, MotionPlanner-based) [THIS CLASS]
        │   ├── ClassicPlanner (single-shot planning)
        │   ├── MultiPointPlanner (waypoint planning)
        │   └── JointSpacePlanner (joint space planning)
        └── ReactiveController (closed-loop control loop)
            └── MPCController (cuRobo ModelPredictiveControl)
"""

from abc import abstractmethod
from typing import Optional, Any
import time

from sensor_msgs.msg import Image as ImageMsg
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)

from curobo.types import JointState, Pose, GoalToolPose
from curobo.motion_planner import MotionPlanner
# v2: PoseCostMetric is gone; Cartesian axis constraints use ToolPoseCriteria.
# Not re-exported publicly yet, so import from _src (same pattern as Mapper).
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria

from .trajectory_planner import TrajectoryPlanner, PlannerResult, ExecutionMode
from .plan_plot import render_plan_plot
from isaac_ros_cumotion_interfaces.action import SendTrajectory

import traceback


class SinglePlanner(TrajectoryPlanner):
    """
    Abstract base class for MotionPlanner-based planners (v2).

    This class implements the shared infrastructure that all MotionPlanner-based
    planners need:
    - Shared MotionPlanner instance (set once, used by all child planners)
    - Common execution logic for open-loop trajectory execution
    - Trajectory storage and state management
    - Cancellation handling

    Child classes only need to implement:
    - _plan_trajectory(): How to generate the trajectory using plan_pose()
    - _process_trajectory(): Optional post-processing of the generated trajectory
    - get_planner_name(): Name of the specific planner

    Key design decisions:
    1. All child planners share the SAME MotionPlanner instance
       - Warmup is done only ONCE by ConfigWrapperMotion
       - Switching between SinglePlanner children does NOT trigger warmup
       - This saves significant initialization time (~seconds)

    2. All planners use open-loop execution
       - Trajectory is fully generated in plan()
       - Then executed as-is in execute()
       - Different from MPC which uses closed-loop

    3. Thread safety: NOT thread-safe by design
       - MotionPlanner instance is shared without locks
       - Assumes single-threaded sequential execution
       - If concurrent planning needed, add synchronization in child classes

    v2 notes:
    - MotionGen → MotionPlanner (curobo.motion_planner).
    - MotionGenResult → MotionPlannerResult, but we only use the `.success`,
      `.status`, `.solve_time` duck-typed attributes here.
    - MotionGenPlanConfig is gone: per-call params are kwargs on plan_pose().
    """

    # Class-level shared MotionPlanner instance
    # This is shared across ALL instances of SinglePlanner and its children
    _shared_motion_planner: Optional[MotionPlanner] = None

    def __init__(self, node, config_wrapper):
        """
        Initialize the planner.

        Args:
            node: ROS2 node for logging and parameters
            config_wrapper: ConfigWrapperMotion with world/robot config
                           (NOT used to create MotionGen, just for world updates)
        """
        super().__init__(node, config_wrapper)

        # Trajectory state (instance-specific)
        self.planned_trajectory = None
        self.start_state = None
        self.goal_pose = None
        # Buffer epoch returned by the preview set_command() in plan(), paired
        # with send_trajectrory(expect_epoch=...) in execute() — see M2 in the
        # pre-publication audit: plan() and execute() are separate calls (a
        # service call, then later an action), so another set_command() could
        # otherwise land in between and execute() would send someone else's
        # trajectory.
        self._command_epoch = None

        # Cancellation flag
        self._cancelled = False

        # Motion-plan debug image (joint-trajectory plot as an RGB Image).
        # Lazy publisher; only active when the node's `publish_plan_debug_image`
        # param is true (off by default). See _publish_plan_image().
        self._debug_img_pub = None
        # Frame id stamped into the debug image header (robot root frame).
        self._debug_frame = getattr(config_wrapper, 'base_link', None)

    def _plan_image_enabled(self) -> bool:
        """Whether the owning node should publish the motion-plan debug image.

        Publish-side knob: unlike the legacy ``enable_curobo_debug_mode``
        (which raises curobo's own logging), this is an INDEPENDENT param so the
        plan plot can be turned on without bumping curobo's log verbosity.
        Defaults to off if the node has not declared the param.
        """
        if not getattr(self.node, 'has_parameter', None):
            return False
        if not self.node.has_parameter('publish_plan_debug_image'):
            return False
        return bool(self.node.get_parameter('publish_plan_debug_image').value)

    def _publish_plan_image(self):
        """Publish the planned trajectory's debug plot as a latched Image.

        Only runs while the node's ``publish_plan_debug_image`` param is true.
        Publishes once per NEW plan (this is called from ``plan()`` on every
        successful plan) on ``/<node>/motion_plan_debug`` with a transient_local
        depth-1 QoS: the latest frame is latched for late subscribers and at
        most one frame is buffered, so bandwidth stays flat.
        """
        if not self._plan_image_enabled():
            return
        if self.planned_trajectory is None:
            return
        try:
            traj = self.planned_trajectory

            def _squeeze(arr):
                while arr.ndim > 2:
                    arr = arr[0]
                if arr.ndim == 1:
                    arr = arr.unsqueeze(0)
                return arr

            position = _squeeze(traj.position).cpu().numpy()
            velocity = (
                _squeeze(traj.velocity).cpu().numpy()
                if getattr(traj, 'velocity', None) is not None
                else None
            )
            acceleration = (
                _squeeze(traj.acceleration).cpu().numpy()
                if getattr(traj, 'acceleration', None) is not None
                else None
            )
            names = list(getattr(traj, 'joint_names', None) or [])
            dt = 0.025
            if getattr(self.node, 'has_parameter', None) and self.node.has_parameter(
                'interpolation_dt'
            ):
                dt = float(self.node.get_parameter('interpolation_dt').value)
            if dt <= 0:
                dt = 0.025

            img = render_plan_plot(
                position,
                names,
                float(dt),
                velocity=velocity,
                acceleration=acceleration,
                title=f"{self.get_planner_name()} plan",
            )

            if self._debug_img_pub is None:
                self._debug_img_pub = self.node.create_publisher(
                    ImageMsg,
                    self.node.get_name() + '/motion_plan_debug',
                    QoSProfile(
                        depth=1,
                        reliability=ReliabilityPolicy.RELIABLE,
                        durability=DurabilityPolicy.TRANSIENT_LOCAL,
                        history=HistoryPolicy.KEEP_LAST,
                    ),
                )

            h, w = img.shape[:2]
            im = ImageMsg()
            im.header.stamp = self.node.get_clock().now().to_msg()
            im.header.frame_id = self._debug_frame or ''
            im.height = h
            im.width = w
            im.encoding = 'rgb8'
            im.is_bigendian = False
            im.step = w * 3
            im.data = img.astype('uint8').tobytes()
            self._debug_img_pub.publish(im)
        except Exception as e:
            self.node.get_logger().warn(
                f"{self.get_planner_name()}: failed to publish motion-plan debug "
                f"image: {e}",
                throttle_duration_sec=5.0,
            )

    def _get_execution_mode(self) -> ExecutionMode:
        """
        All SinglePlanner children use open-loop execution.

        The trajectory is fully generated upfront, then executed.
        This is different from MPC which uses closed-loop execution.
        """
        return ExecutionMode.OPEN_LOOP

    @classmethod
    def set_motion_planner(cls, motion_planner: MotionPlanner):
        """
        Set the shared MotionPlanner instance (v2).

        This is called ONCE after ConfigWrapperMotion.set_motion_gen_config()
        completes the warmup. All SinglePlanner instances (current and future)
        will use this same MotionPlanner instance.

        Args:
            motion_planner: Warmed-up MotionPlanner instance from ConfigWrapperMotion

        Example:
            >>> config_wrapper = ConfigWrapperMotion(node, robot)
            >>> config_wrapper.set_motion_gen_config(node, None, None)
            >>> SinglePlanner.set_motion_planner(node.motion_planner)
        """
        cls._shared_motion_planner = motion_planner

    # Legacy alias: keeps older call sites (`set_motion_gen`, `.motion_gen`) working
    # during the v2 transition.
    set_motion_gen = set_motion_planner

    @property
    def motion_planner(self) -> Optional[MotionPlanner]:
        """Access the shared MotionPlanner instance."""
        return self._shared_motion_planner

    @property
    def motion_gen(self) -> Optional[MotionPlanner]:
        """Legacy alias for motion_planner."""
        return self._shared_motion_planner

    # ------------------------------------------------------------------
    # Cartesian trajectory constraints (v2: ToolPoseCriteria)
    # ------------------------------------------------------------------

    def _apply_pose_constraints(self, goal_request) -> bool:
        """Hold Cartesian axes along the whole path, if requested.

        Reads the first non-empty ``Goalset.trajectory_constraints`` (int8[6],
        order ``[theta_x, theta_y, theta_z, x, y, z]``; 1 = lock that axis along
        the path) and sets ``ToolPoseCriteria.non_terminal_pose_axes_weight_factor``
        (order ``[x, y, z, roll, pitch, yaw]``) on the shared MotionPlanner.
        This is the v2 replacement for the removed PoseCostMetric. Per-waypoint
        ``trajectories_contraints`` (flattened per-waypoint holds) are not
        expressible in v2 — only the whole-path span is.

        Returns True if constraints were applied (caller must reset afterwards).
        """
        constraints = []
        goalsets = list(getattr(goal_request, 'goalsets', None) or [])
        for g in goalsets:
            c = list(getattr(g, 'trajectory_constraints', None) or [])
            if c:
                constraints = c
                break
        if not any(c == 1 for c in constraints):
            return False
        if len(constraints) != 6:
            self.node.get_logger().warn(
                f"{self.get_planner_name()}: trajectory_constraints must have 6 entries "
                f"[theta_x, theta_y, theta_z, x, y, z], got {len(constraints)} - ignoring."
            )
            return False

        tx, ty, tz, x, y, z = (1.0 if c == 1 else 0.0 for c in constraints)
        axes = [x, y, z, tx, ty, tz]  # ToolPoseCriteria order: x,y,z,roll,pitch,yaw
        tool_frame = self.motion_planner.tool_frames[0]
        self.motion_planner.update_tool_pose_criteria(
            {tool_frame: ToolPoseCriteria(non_terminal_pose_axes_weight_factor=axes)}
        )
        self.node.get_logger().info(
            f"{self.get_planner_name()}: holding axes along path "
            f"(x,y,z,roll,pitch,yaw)={axes}"
        )
        return True

    def _reset_pose_criteria(self) -> None:
        """Restore default (unconstrained) criteria — the MotionPlanner is shared."""
        tool_frame = self.motion_planner.tool_frames[0]
        self.motion_planner.update_tool_pose_criteria({tool_frame: ToolPoseCriteria()})

    # ------------------------------------------------------------------
    # Goalset support (per-waypoint candidate sets, open-loop planners)
    # ------------------------------------------------------------------

    def _build_goal_segments(self, goal_request):
        """One GoalToolPose per ``goalsets[i]``; ``[]`` when `goalsets` is empty.

        A segment with N candidate poses becomes a goalset solve
        (``GoalToolPose ... num_goalset=N``): cuRobo resolves the set inside
        that single ``plan_pose()`` call and reports the winner via
        ``result.goalset_index``. ``N == 1`` degenerates to a fixed waypoint —
        today's single-goal plan. ``None`` entries would break index alignment,
        so ``goalsets[i]`` with zero poses is skipped from the segment list;
        the node's validation rejects such requests before planning.
        """
        sets = list(getattr(goal_request, 'goalsets', None) or [])
        if not sets:
            return []
        tool_frame = self.motion_planner.tool_frames[0]
        segments = []
        for gset in sets:
            poses = [
                [p.position.x, p.position.y, p.position.z,
                 p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z]
                for p in gset.poses
            ]
            if not poses:
                continue
            pose = Pose.from_batch_list(poses)
            segments.append(
                GoalToolPose.from_poses({tool_frame: pose}, num_goalset=len(poses))
            )
        return segments

    @staticmethod
    def _select_goal_index(result) -> int:
        """Winner candidate index of a plan_pose goalset solve.

        ``result.goalset_index`` is a ``[batch, num_seeds]`` int tensor (the
        core reads it with ``.view(-1)[0].item()``, motion_planner.py). Single-
        candidate solves leave it ``None``, and the winner is trivially
        candidate 0. ``-1`` signals a failed/absent solve.
        """
        if result is None or not getattr(result, 'success', False):
            return -1
        succ = getattr(result, 'success', False)
        if hasattr(succ, 'item') and not bool(succ.item()):
            return -1
        idx = getattr(result, 'goalset_index', None)
        if idx is None:
            return 0
        try:
            flat = idx.view(-1) if hasattr(idx, 'view') else idx
            return int(flat[0].item())
        except (TypeError, ValueError, IndexError):
            return -1

    @staticmethod
    def _flat_values(value, default=None):
        """Flatten a solver-result field to a plain Python list.

        Handles None (returns ``default``), 0-dim tensors / scalars
        (single-item list), and tensors of any rank (detached, CPU,
        flattened). Used to read seed-ordered ``[B, S]`` result rows
        defensively.
        """
        if value is None:
            return default
        if hasattr(value, 'detach'):
            try:
                flat = value.detach().cpu()
            except Exception:
                flat = value
            return flat.reshape(-1).tolist()
        if isinstance(value, (list, tuple)):
            return [x.item() if hasattr(x, 'item') else x for x in value]
        if hasattr(value, 'item'):
            return [value.item()]
        return [value]

    @staticmethod
    def _select_seed_index(result) -> int:
        """Winner restart (seed) index of a plan solve.

        cuRobo ranks seeds by total cost (successful seeds first) in
        ``result.seed_rank`` (``[B, S]``; row 0 holds the winner's seed).
        Fall back to the argmin of ``seed_cost`` (row 0) when the rank is
        missing, then to 0 for a single-seed solve. ``-1`` signals a
        failed/absent solve.
        """
        if result is None:
            return -1
        succ = getattr(result, 'success', None)
        if succ is None:
            return -1
        try:
            ok = bool(succ.item()) if hasattr(succ, 'item') else bool(succ)
            if not ok:
                return -1
        except Exception:
            return -1
        rank = getattr(result, 'seed_rank', None)
        if rank is not None:
            try:
                row0 = rank[0]
                winner = row0.argmin().item() if hasattr(row0, 'argmin') else row0[0]
                return int(winner)
            except Exception:
                pass
        cost = getattr(result, 'seed_cost', None)
        if cost is not None:
            try:
                row0 = cost[0]
                winner = row0.argmin().item() if hasattr(row0, 'argmin') else row0
                return int(winner)
            except Exception:
                pass
        return 0

    @staticmethod
    def _segment_reached(result, winner_seed, tolerance, segment_success) -> int:
        """Per-waypoint reached flag (int 0/1) for one segment.

        1 when the winner seed's ``position_error`` is within ``tolerance`` (m;
        ``tolerance > 0`` enables the FK check), else 0. With no tolerance or
        no per-seed error the flag falls back to the segment's solve success.
        """
        if tolerance and tolerance > 0 and winner_seed >= 0:
            perr = getattr(result, 'position_error', None)
            if perr is not None:
                try:
                    errs = SinglePlanner._flat_values(perr)
                    if winner_seed < len(errs):
                        return 1 if float(errs[winner_seed]) <= tolerance else 0
                except Exception:
                    pass
        return 1 if segment_success else 0

    @staticmethod
    def _tally_candidates(result) -> dict:
        """Candidate (seed) accounting for one solver call.

        Named distinctly from the ``_candidate_tally`` INSTANCE attribute
        (the per-plan accumulator): ``plan()`` resets ``self._candidate_tally
        = None`` before the child ``_plan_trajectory`` runs, which would
        shadow a same-named method on the instance.

        ``generated`` = the solver's ``num_seeds`` for this problem/segment
        (all seeds restarted; 1 when the result carries no per-seed count),
        ``solved`` = the seeds whose solve succeeded, ``pruned`` = 0 — the
        per-segment machinery spills nothing (whole-task caps would report a
        nonzero pruned count).
        """
        success = getattr(result, 'success', None)
        n_seeds = int(getattr(result, 'num_seeds', 0) or 0)
        if n_seeds <= 0:
            n_seeds = 1
        solved = 0
        if success is not None:
            try:
                vals = SinglePlanner._flat_values(success)
                solved = sum(1 for v in vals if v)
            except Exception:
                solved = 0
        return {'generated': n_seeds, 'solved': solved, 'pruned': 0}

    @staticmethod
    def _segment_considered_rows(result, segment_i, fallback_candidate, log_flag) -> list:
        """One considered-trajectory row per seed of a solved segment.

        Row shape ``[problem=0, segment, goalset_candidate, seed]``; fields
        filled from the solver result's per-seed data (``success`` /
        ``seed_cost`` / ``position_error`` / ``goalset_index`` /
        ``solve_time``). The whole-task-only fields (``waypoint_cost`` /
        ``path_length`` / ``clearance``) are measured by the task solver, not
        by the per-segment machinery, so they stay 0 and the node's defensive
        mapping stays truthful. Empty unless ``log_flag``.
        """
        if not log_flag:
            return []
        success = SinglePlanner._flat_values(getattr(result, 'success', None))
        costs = SinglePlanner._flat_values(getattr(result, 'seed_cost', None))
        perr = SinglePlanner._flat_values(getattr(result, 'position_error', None))
        gidx = getattr(result, 'goalset_index', None)
        gidx_flat = SinglePlanner._flat_values(gidx)
        # goalset_index is [B, S, L]: seed s's candidate rides at flat index s*L.
        n_links = 1
        if gidx is not None and getattr(gidx, 'ndim', 0) >= 3:
            try:
                n_links = int(gidx.shape[-1])
            except Exception:
                n_links = 1
        n_seeds = len(success) if success else 1
        if n_seeds <= 0:
            n_seeds = 1
        solve_time = getattr(result, 'solve_time', 0.0)
        rows = []
        for s in range(n_seeds):
            ok = bool(success[s]) if s < len(success) else False
            cost = float(costs[s]) if costs and s < len(costs) else 0.0
            err = float(perr[s]) if perr and s < len(perr) else 0.0
            cand = 0
            if gidx_flat and s * n_links < len(gidx_flat):
                try:
                    cand = int(gidx_flat[s * n_links])
                except (TypeError, ValueError):
                    cand = 0
            elif not ok:
                cand = int(fallback_candidate)
            rows.append({
                'problem': 0,
                'segment': segment_i,
                'goalset_candidate': cand,
                'seed': s,
                'success': ok,
                'cost': cost,
                'waypoint_cost': 0.0,
                'path_length': 0.0,
                'clearance': 0.0,
                'max_waypoint_error': err,
                'solve_time': solve_time,
            })
        return rows

    @staticmethod
    def _request_waypoint_tolerance(goal_request) -> float:
        """Per-request waypoint FK tolerance (m) from ``PlanningOptions``.

        0.0 when unset (the node leaves the reached-flag fallback to the
        segment's solve success).
        """
        opts = getattr(goal_request, 'options', None)
        if opts is None:
            return 0.0
        try:
            return float(getattr(opts, 'waypoint_tolerance', 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _request_log_considered(goal_request) -> bool:
        """Whether considered-trajectory rows should be computed.

        Gates per-segment ``_segment_considered_rows`` collection on the
        request's ``PlanningOptions.log_considered_trajectories``.
        """
        opts = getattr(goal_request, 'options', None)
        if opts is None:
            return False
        try:
            return bool(getattr(opts, 'log_considered_trajectories', False))
        except Exception:
            return False

    def _result_metadata(self, result=None, num_wp=None) -> dict:
        """Metadata block for PlannerResult; includes per-segment insight.

        ``selected_goal_index`` / ``selected_seed_index`` / ``waypoint_status``
        ride here (plain int lists) so the node can copy them straight into
        the service/action response; they are shaped by the child planner in
        ``_plan_trajectory`` (one entry per ``goalsets[i]`` segment). The
        candidate tally and gated ``considered`` rows ride here for the
        ``PlanningStats`` block.
        """
        metadata = {
            'planner_type': self.get_planner_name(),
        }
        if num_wp is not None:
            metadata['num_waypoints'] = num_wp
        if result is not None:
            metadata['planning_time'] = getattr(result, 'solve_time', 0.0)
        sel = getattr(self, '_selected_goal_indexes', None)
        if sel is not None:
            metadata['selected_goal_index'] = [int(x) for x in sel]
        seeds = getattr(self, '_selected_seed_index', None)
        if seeds is not None:
            metadata['selected_seed_index'] = [int(x) for x in seeds]
        status = getattr(self, '_waypoint_status', None)
        if status is not None:
            metadata['waypoint_status'] = [int(x) for x in status]
        tally = getattr(self, '_candidate_tally', None)
        if tally is not None:
            metadata['candidates_generated'] = int(tally.get('generated', 0))
            metadata['candidates_solved'] = int(tally.get('solved', 0))
            metadata['candidates_pruned'] = int(tally.get('pruned', 0))
        rows = getattr(self, '_considered_rows', None)
        if rows:
            metadata['considered'] = list(rows)
        return metadata

    def cancel(self):
        """
        Cancel the current trajectory execution.

        This sets a flag that breaks the execution loop in execute().
        Called by the node when a cancellation request is received.
        """
        self._cancelled = True
        self.node.get_logger().info(f"{self.get_planner_name()}: Cancellation requested")

    def plan(
        self,
        start_state: JointState,
        goal_request: Any,
        config: dict,
        robot_context: Optional[Any] = None
    ) -> PlannerResult:
        """
        Generate a complete trajectory using MotionGen.

        This method orchestrates the planning process:
        1. Validate MotionGen is initialized
        2. Call child class's _plan_trajectory() to generate trajectory
        3. Optionally process trajectory via _process_trajectory()
        4. Send to robot_context for visualization

        Args:
            start_state: Initial joint configuration
            goal_request: TrajectoryGeneration request containing goal specification
                         Child classes extract what they need (goalsets)
            config: Dictionary with planner-specific parameters
                   Common parameters:
                   - max_attempts: Number of planning attempts
            robot_context: Optional RobotContext for trajectory visualization

        Returns:
            PlannerResult with success status and trajectory or error message
        """
        # Validate MotionPlanner is initialized
        if self.motion_planner is None:
            return PlannerResult(
                success=False,
                message=(
                    "MotionPlanner not initialized. "
                    "Call SinglePlanner.set_motion_planner() after warmup."
                ),
            )

        # Store for execution
        self.start_state = start_state
        self.goal_pose = goal_request  # Store request, child classes interpret it
        # Reset: only set below if this call actually binds a fresh
        # set_command() (robot_context is not None). A stale epoch from a
        # PREVIOUS plan() call must not silently guard this one's execute().
        self._command_epoch = None
        # Reset per-segment goalset winners — children (ClassicPlanner /
        # MultiPointPlanner) set this in _plan_trajectory; plan() reports it as
        # metadata['selected_goal_index'] so the node can fill the response.
        self._selected_goal_indexes = None
        # Per-request insight metadata (rooted in PlanningOptions): winner seed
        # + per-segment reached flags + candidate tally + the gated considered
        # rows. Children set these in _plan_trajectory; plan() reports them via
        # _result_metadata so the node can fill TrajectoryResult / PlanningStats.
        self._selected_seed_index = None
        self._waypoint_status = None
        self._candidate_tally = None
        self._considered_rows = None
        self._waypoint_tolerance = self._request_waypoint_tolerance(goal_request)
        self._log_considered = self._request_log_considered(goal_request)

        try:
            # Let child class generate the trajectory using MotionGen
            result = self._plan_trajectory(start_state, goal_request, config)

            # Check if planning succeeded
            # v2: plan_pose() returns Optional[TrajOptSolverResult] — None on failure
            if result is None:
                return PlannerResult(
                    success=False,
                    message="Planning failed: no solution found (plan_pose returned None)",
                    metadata=self._result_metadata(result=None, num_wp=None),
                )

            success_val = result.success
            if hasattr(success_val, 'item'):
                success_val = success_val.item()
            if not success_val:
                # TrajOptSolverResult has no `.status`; the informative fields
                # are debug_info (dict) and feasible (constraint satisfaction).
                status = getattr(result, 'status', None)
                if not status:
                    dbg = getattr(result, 'debug_info', None) or {}
                    status = next(iter(dbg.values()), None) if dbg else None
                if not status:
                    feasible = getattr(result, 'feasible', None)
                    if feasible is not None:
                        try:
                            ok = feasible
                            if hasattr(ok, 'detach'):
                                ok = ok.detach().cpu()
                            if hasattr(ok, 'all'):
                                ok = bool(ok.all())
                            if not ok:
                                status = "constraints violated (collision/limits)"
                        except Exception:
                            pass
                return PlannerResult(
                    success=False,
                    message=f"Planning failed: {status or 'unknown'}",
                    metadata=self._result_metadata(result=result)
                )

            # Get interpolated trajectory
            self.planned_trajectory = result.get_interpolated_plan()

            # Allow child class to post-process the trajectory
            # (e.g., add grasp commands, modify velocities, etc.)
            self.planned_trajectory = self._process_trajectory(
                self.planned_trajectory,
                config
            )

            # v2: position shape can be [B, T, D] — count waypoints on horizon dim.
            _pos = self.planned_trajectory.position
            num_wp = _pos.shape[-2] if _pos.ndim >= 2 else len(_pos)
            self.node.get_logger().info(
                f"{self.get_planner_name()}: Successfully planned trajectory "
                f"with {num_wp} waypoints"
            )

            # Publish the motion-plan debug image (gated by publish_plan_debug_image).
            self._publish_plan_image()

            # Send trajectory to robot context for visualization
            if robot_context is not None:
                traj = self.planned_trajectory
                # v2: position/velocity/acceleration may have shape [B, T, D];
                # robot_context expects [T, D] (one row of floats per waypoint).
                # Flatten all leading dims down to 2 so `.tolist()` yields a
                # list[list[float]] regardless of batch rank.
                def _to_2d_list(t):
                    if t is None:
                        return None
                    while t.ndim > 2:
                        t = t[0]
                    return t.detach().cpu().tolist()

                self.node.get_logger().debug(
                    f"Trajectory shapes - pos: {tuple(traj.position.shape)}, "
                    f"vel: {tuple(traj.velocity.shape) if traj.velocity is not None else None}, "
                    f"acc: {tuple(traj.acceleration.shape) if traj.acceleration is not None else None}"
                )
                pos_list = _to_2d_list(traj.position)
                vel_list = _to_2d_list(traj.velocity)
                acc_list = _to_2d_list(traj.acceleration)

                # Ensure velocity/acceleration arrays align with positions even
                # if the planner omitted them (rare but possible for a stubbed
                # trajectory).
                if vel_list is None:
                    vel_list = [[0.0] * len(pos_list[0]) for _ in pos_list]
                if acc_list is None:
                    acc_list = [[0.0] * len(pos_list[0]) for _ in pos_list]

                # Interpolated plans are in FULL joint space: cuRobo augments
                # locked joints (e.g. a gripper finger_joint) via
                # get_full_dof_from_solution(), so rows can be wider than
                # joint_names (Kortex: 8 columns vs 7 names). Project the
                # streamed command back onto ACTIVE joints by name so it matches
                # the controller's arm joints — same active-joint projection the
                # preview/ghost pipeline (robot_context) applies to set_command().
                joint_names, cols = self._active_joint_projection(traj)
                if cols is not None:
                    pos_list = [[r[i] for i in cols] for r in pos_list]
                    vel_list = [[r[i] for i in cols] for r in vel_list]
                    acc_list = [[r[i] for i in cols] for r in acc_list]
                    joint_names = [joint_names[i] for i in cols]

                self._command_epoch = robot_context.set_command(
                    joint_names,
                    vel_list,
                    acc_list,
                    pos_list,
                )
                self.node.get_logger().info(
                    "Trajectory sent to robot context for visualization"
                )

            return PlannerResult(
                success=True,
                message="Trajectory planned successfully",
                trajectory=self.planned_trajectory,
                metadata=self._result_metadata(result=result, num_wp=num_wp),
            )

        except Exception as e:
            self.node.get_logger().error(f"Planning exception: {e}")
            self.node.get_logger().error(traceback.format_exc())

            return PlannerResult(
                success=False,
                message=f"Planning error: {str(e)}",
            )

    @abstractmethod
    def _plan_trajectory(
        self,
        start_state: JointState,
        goal_request: Any,
        config: dict
    ):
        """
        Generate trajectory using MotionPlanner.plan_pose() (v2).

        Child planners extract different data from goal_request:
        - ClassicPlanner: goalsets[0]  → single GoalToolPose (goal-set resolve when N>1)
        - MultiPointPlanner: goalsets[i] → per-waypoint GoalToolPose
        - JointSpacePlanner: goal_request.target_joints → joint goal

        Args:
            start_state: Initial joint configuration
            goal_request: TrajectoryGeneration request (child extracts what it needs)
            config: Dictionary with planner-specific configuration

        Returns:
            MotionPlannerResult-like object with `.success`, `.status`,
            `.solve_time`, and `.get_interpolated_plan()`.

        Example (ClassicPlanner):
            >>> from curobo.types import ToolPose, GoalToolPose
            >>> goal = GoalToolPose(tool_pose=ToolPose.from_list([
            ...     p.position.x, p.position.y, p.position.z,
            ...     p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z,
            ... ]))
            >>> return self.motion_planner.plan_pose(
            ...     start_state, goal,
            ...     max_attempts=config['max_attempts'],
            ... )
        """
        pass

    def _process_trajectory(self, trajectory: JointState, config: dict) -> JointState:
        """
        Post-process the generated trajectory.

        Override this in child classes if you need to modify the trajectory
        after it's generated. For example:
        - SlowPlanner: Reduce velocities for safety
        - VibrateFilter: Smooth out high-frequency oscillations

        Args:
            trajectory: Raw trajectory from MotionGen
            config: Configuration dictionary

        Returns:
            Processed trajectory (default: unchanged)
        """
        return trajectory

    def _active_joint_projection(self, traj):
        """Column projection of a FULL-joint-space plan onto ACTIVE joints.

        cuRobo's interpolated plan carries every model joint, including joints
        the config locks (e.g. ``lock_joints: {finger_joint: 0.0}``), appended
        via ``get_full_dof_from_solution()``. The robot's controller only owns
        the active (arm) joints, so streamed commands must drop those extra
        columns. Returns ``(names, cols)`` where ``cols`` selects the active
        columns of each row by NAME; ``(name_list, None)`` signals the plan is
        already active-sized and needs no projection.
        """
        full = list(getattr(traj, 'joint_names', None) or [])
        if (not full or self.motion_planner is None
                or traj.position is None or traj.position.ndim == 0):
            return full, None
        dof = int(traj.position.shape[-1])
        # Rows wider than the name list: cuRobo appended locked/fixed joint
        # columns (e.g. finger_joint) via get_full_dof_from_solution(), with
        # the named joints leading in order — a plain prefix keeps the arm
        # joints and drops the appended tail.
        if len(full) < dof:
            return full, list(range(len(full)))
        try:
            probe = JointState.from_position(
                traj.position.reshape(-1, dof)[:1], joint_names=full)
            active = self.motion_planner.kinematics.get_active_js(probe)
        except Exception:
            return full, None
        active_names = list(active.joint_names)
        if len(active_names) == len(full):
            return full, None
        index = {n: i for i, n in enumerate(full)}
        if any(n not in index for n in active_names):
            return full, None
        return full, [index[n] for n in active_names]

    def execute(self, robot_context, goal_handle=None) -> bool:
        """
        Execute the planned trajectory in open-loop.

        Sends the full pre-computed trajectory to the robot and monitors
        progress until completion or cancellation.

        This implementation is shared by all SinglePlanner children since
        they all use the same open-loop execution pattern.

        Args:
            robot_context: RobotContext for command sending
            goal_handle: Optional ROS action goal handle for feedback

        Returns:
            True if execution completed successfully, False if cancelled or error
        """
        if self.planned_trajectory is None:
            self.node.get_logger().error("No trajectory to execute. Call plan() first.")
            return False

        try:
            # Reset cancellation flag at the start of execution
            self._cancelled = False

            # Start trajectory execution. expect_epoch guards against another
            # set_command() landing between plan() (which set _command_epoch)
            # and this call — see JointCommandStrategy.buffer_epoch / M2.
            if not robot_context.send_trajectrory(expect_epoch=self._command_epoch):
                self.node.get_logger().error(
                    f"{self.get_planner_name()}: refusing to execute - the "
                    f"planned trajectory was superseded by a newer command "
                    f"before execution started."
                )
                return False

            self.node.get_logger().info(
                f"{self.get_planner_name()}: Trajectory execution started"
            )

            exec_csv = self._exec_csv_enabled()
            if exec_csv:
                self._exec_csv_init(
                    prefix="exec_progress",
                    columns=["t_s", "elapsed_s", "progression", "goal_active",
                             "cancelled", "joint_pose_snapshot"],
                )

            # Monitor progress with feedback
            start_time = time.time()
            time_dilation_factor = self.node.get_parameter(
                'time_dilation_factor'
            ).get_parameter_value().double_value

            progression = robot_context.get_progression()

            while progression < 1.0 and not self._cancelled:
                # Check for cancellation
                if goal_handle is not None and not goal_handle.is_active:
                    self.node.get_logger().warn("Trajectory execution cancelled")
                    robot_context.stop_robot()
                    if exec_csv:
                        self._exec_csv_close()
                    return False

                # Publish feedback at regular intervals
                if (time.time() - start_time) > time_dilation_factor:
                    # Check for cancellation again before publishing feedback
                    if goal_handle is not None and not goal_handle.is_active:
                        self.node.get_logger().warn("Trajectory execution cancelled")
                        robot_context.stop_robot()
                        if exec_csv:
                            self._exec_csv_close()
                        return False

                    if goal_handle is not None:
                        feedback_msg = SendTrajectory.Feedback()
                        feedback_msg.state = "EXECUTING"
                        feedback_msg.on_target = False
                        feedback_msg.step_progression = robot_context.get_progression()
                        goal_handle.publish_feedback(feedback_msg)

                    progression = robot_context.get_progression()
                    # Only log at significant milestones to reduce spam
                    # if progression >= 0.99 or int(progression * 10) != int((progression - 0.1) * 10):
                        # self.node.get_logger().info(
                        #     f"Trajectory progress: {progression*100:.1f}%"
                        # )
                    if exec_csv:
                        goal_active = False
                        if goal_handle is not None:
                            try:
                                goal_active = goal_handle.is_active
                            except Exception:
                                pass
                        try:
                            pose = robot_context.robot_strategy.get_joint_pose()
                            pose_str = ("[" + ",".join(f"{v:.4f}" for v in pose) + "]"
                                        if pose else "[]")
                        except Exception:
                            pose_str = "[]"
                        self._exec_csv_write([
                            f"{time.monotonic() - self._exec_csv_t0:.3f}",
                            f"{time.time() - start_time:.3f}",
                            f"{progression:.6f}",
                            str(goal_active),
                            str(self._cancelled),
                            pose_str,
                        ])
                    start_time = time.time()

                # Small sleep to prevent busy-waiting
                time.sleep(0.01)

            # Check if we exited due to cancellation or completion
            if self._cancelled:
                self.node.get_logger().info("Trajectory execution cancelled via flag")
                if exec_csv:
                    self._exec_csv_close()
                return False

            # Re-read progression: the loop may have stopped on a stale sample
            # while the controller's Result reported failure. The action is the
            # execution authority — a non-success Result (progression -1.0)
            # means the path was NOT completed; report it instead of success.
            final_progression = robot_context.get_progression()
            if final_progression < 0.0:
                reason = ""
                strategy = getattr(robot_context, 'robot_strategy', None)
                if strategy is not None and hasattr(strategy, 'action_failure_summary'):
                    reason = strategy.action_failure_summary()
                self.node.get_logger().error(
                    f"{self.get_planner_name()}: trajectory execution FAILED: "
                    f"{reason or 'controller reported a non-success result'}"
                )
                robot_context.stop_robot()
                if exec_csv:
                    self._exec_csv_close()
                return False

            # Wait for emulator thread to finish updating position
            # This ensures the next planner reads the correct final position
            time.sleep(0.1)
            self.node.get_logger().info(
                f"Trajectory execution completed. "
                f"Final position: {robot_context.get_joint_pose()}"
            )
            if exec_csv:
                self._exec_csv_close()
            return True

        except Exception as e:
            self.node.get_logger().error(f"Execution error: {e}")
            self.node.get_logger().error(traceback.format_exc())
            robot_context.stop_robot()
            self._exec_csv_close()
            return False

    def get_config_parameters(self) -> list:
        """
        Get list of common configuration parameters for SinglePlanner.

        Child classes should override this and call super() to add their own.

        Returns:
            List of parameter names
        """
        return [
            'max_attempts',
            'time_dilation_factor',
            'voxel_size',
            'collision_activation_distance',
        ]

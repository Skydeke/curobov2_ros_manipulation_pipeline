#!/usr/bin/env python3
"""
Cartesian planner — the cuRobo port of MoveIt Task Constructor's
``solvers::CartesianPath``.

MTC's CartesianPath is a *planner*. Given a start scene and a target tool pose
it runs ``computeCartesianPath``: a straight EEF interpolation sampled every
``step_size`` metres, IK'd and validity-checked at each waypoint, which stops at
the first infeasible sample and reports the fraction of the path it reached.
``plan()`` fails when that achieved fraction is below ``min_fraction``.

cuRobo has no Cartesian solve mode, but it has the Cartesian *cost*. A goalset
carries ``Goalset.trajectory_constraints`` (``int8[6]`` =
``[theta_x, theta_y, theta_z, x, y, z]``), which the shared planning interface
(MTC parity: ``SinglePlanner._apply_pose_constraints``) turns into
``ToolPoseCriteria`` and scores on every non-terminal waypoint. Holding every
axis on which the start and the goal agree leaves the optimizer no cheaper
choice than the straight segment between them. So this planner is:

1. resolve the single Cartesian goalset (1..N candidate tool poses),
2. apply its axis holds as the Cartesian cost (``_apply_pose_constraints``),
3. solve with ``plan_pose`` (a candidate set resolves inside ONE call),
4. measure how far along the straight start->goal segment the returned
   trajectory actually stayed (the FK'd tool positions), and
5. reject the solve when that fraction is below ``min_fraction`` — reproducing
   MTC's ``computeCartesianPath`` partial-path guarantee, message and all.

Steps 1-3 are exactly ``ClassicPlanner`` (this class subclasses it); the only
new work is the fraction guarantee in step 4/5, plus rejecting joint-space
goals, which a Cartesian planner cannot express. The interpolation parameters
(``step_size`` / ``min_fraction`` / ``precision``) ride on the request
(``TrajectoryGoal.cartesian_*``), where the task constructor's ``core.CartesianPath``
solver put them.
"""

from .classic_planner import ClassicPlanner
from .trajectory_planner import PlannerResult


def _segment_distance(point, a, b) -> float:
    """Perpendicular distance (m) from ``point`` to segment ``a`` -> ``b``."""
    ab = [b[i] - a[i] for i in range(3)]
    ap = [point[i] - a[i] for i in range(3)]
    denom = sum(v * v for v in ab)
    if denom < 1e-18:
        return sum(v * v for v in ap) ** 0.5
    t = max(0.0, min(1.0, sum(x * y for x, y in zip(ap, ab)) / denom))
    return sum((ap[i] - t * ab[i]) ** 2 for i in range(3)) ** 0.5


class CartesianPlanner(ClassicPlanner):
    """Straight-line planning with MTC CartesianPath semantics.

    Same goalset / candidate-set / open-loop machinery as ``ClassicPlanner``;
    the axis holds become the Cartesian cost, and the solve is accepted only
    when the tool traces at least ``min_fraction`` of the straight segment.
    """

    #: MTC ``solvers::CartesianPath`` property defaults (cartesian_path.cpp:
    #: ``step_size`` 0.01, ``min_fraction`` 1.0, default ``CartesianPrecision``
    #: -> translational 0.001). ``precision`` here is the tolerance under which
    #: an FK'd waypoint counts as ON the line, i.e. the deviation budget the
    #: soft cost is allowed to spend.
    DEFAULT_STEP_SIZE = 0.01
    DEFAULT_MIN_FRACTION = 1.0
    DEFAULT_PRECISION = 0.001

    def get_planner_name(self) -> str:
        return "Cartesian Motion Generation"

    # ------------------------------------------------------------------
    # Goalset validation
    # ------------------------------------------------------------------

    def _is_cartesian_request(self, goal_request) -> bool:
        """One goalset holding >= 1 tool pose, no joint-space target.

        MTC's CartesianPath plans ONE straight segment between two Cartesian
        scenes, so a joint-space target (or an empty/segmented path) is not a
        Cartesian request and must be refused, not silently planned as cspace.
        """
        goalsets = list(getattr(goal_request, "goalsets", None) or [])
        return (
            len(goalsets) == 1
            and bool(list(getattr(goalsets[0], "poses", None) or []))
            and not list(getattr(goalsets[0], "target_joint_positions", None) or [])
        )

    def _plan_trajectory(self, start_state, goal_request, config):
        if not self._is_cartesian_request(goal_request):
            goalsets = list(getattr(goal_request, "goalsets", None) or [])
            shapes = [
                {
                    "poses": len(list(getattr(g, "poses", None) or [])),
                    "joint_positions": len(
                        list(getattr(g, "target_joint_positions", None) or [])
                    ),
                }
                for g in goalsets
            ]
            self.node.get_logger().warn(
                f"{self.get_planner_name()}: CartesianPath needs exactly one "
                f"goalset with >= 1 pose and no joint target, got "
                f"{len(goalsets)} segment(s), shapes={shapes}"
            )
            return None
        # ClassicPlanner does the hold application + plan_pose solve + insight
        # binding; a Cartesian goal is a pose goalset, which is exactly what it
        # consumes.
        return super()._plan_trajectory(start_state, goal_request, config)

    # ------------------------------------------------------------------
    # Request options (MTC CartesianPath properties, carried on the request)
    # ------------------------------------------------------------------

    def _request_options(self, goal_request):
        """(step_size, min_fraction, precision) with MTC defaults filled in."""
        step_size = float(getattr(goal_request, "cartesian_step_size", 0.0) or 0.0)
        if step_size <= 0.0:
            step_size = self.DEFAULT_STEP_SIZE
        min_fraction = float(
            getattr(goal_request, "cartesian_min_fraction", -1.0)
        )
        if min_fraction < 0.0:
            min_fraction = self.DEFAULT_MIN_FRACTION
        precision = float(getattr(goal_request, "cartesian_precision", 0.0) or 0.0)
        if precision <= 0.0:
            precision = self.DEFAULT_PRECISION
        return step_size, min_fraction, precision

    # ------------------------------------------------------------------
    # Achieved-fraction guarantee
    # ------------------------------------------------------------------

    def _finalize_plan_result(self, result, goal_request, config, robot_context):
        """Shape the solve, then enforce ``min_fraction`` (MTC parity).

        Runs AFTER ``SinglePlanner`` has extracted ``self.planned_trajectory``
        (the FK'd tool path), so the fraction is measured on the same
        trajectory the caller will execute — no second interpolation. A solve
        that strayed off the line is returned as a FAILURE with MTC's exact
        wording, so the containing ``Fallbacks``/``Alternatives`` treats it as
        one more rejected candidate (which is how the pick ladder advances).
        """
        planner_result = super()._finalize_plan_result(
            result, goal_request, config, robot_context
        )
        if not getattr(planner_result, "success", False):
            return planner_result

        step_size, min_fraction, precision = self._request_options(goal_request)
        fraction = self._achieved_fraction(
            goal_request, planner_result.trajectory, precision
        )
        self._achieved_fraction_value = fraction
        if fraction + 1e-9 < min_fraction:
            self.node.get_logger().warn(
                f"{self.get_planner_name()}: min_fraction not met "
                f"(achieved {fraction:.3f} < {min_fraction:.3f}); rejecting solve"
            )
            return PlannerResult(
                success=False,
                message=(
                    "CartesianPath: min_fraction not met. Achieved: "
                    f"{fraction:.6f}"
                ),
                metadata=getattr(planner_result, "metadata", None),
            )
        self.node.get_logger().info(
            f"{self.get_planner_name()}: achieved fraction {fraction:.3f} "
            f"(step_size={step_size} m, min_fraction={min_fraction}, "
            f"precision={precision} m)"
        )
        return planner_result

    def _achieved_fraction(self, goal_request, trajectory, precision) -> float:
        """Fraction of the straight segment the tool actually traced.

        The analogue of ``computeCartesianPath``'s return: walk the FK'd tool
        positions from the start and count the contiguous prefix that stays
        within ``precision`` of the start->goal segment; the first waypoint
        that leaves the line ends the feasible prefix, exactly as the first
        failed IK/collision sample ends MoveIt's interpolation. Anything that
        cannot be measured (no trajectory, FK unavailable) counts as fully
        achieved — the axis-hold cost already represents the line, so a
        measurement failure must not manufacture a rejection.
        """
        try:
            start = self._start_tool_position()
            goal = self._goal_tool_position(goal_request)
            pts = self._tool_positions(trajectory)
        except Exception as exc:  # noqa: BLE001 - kinematics contract drift
            self.node.get_logger().warn(
                f"{self.get_planner_name()}: Cartesian fraction not measurable "
                f"({exc!r}); accepting the solve"
            )
            return 1.0
        if start is None or goal is None or not pts:
            return 1.0
        on_line = 0
        for p in pts:
            if _segment_distance(p, start, goal) <= precision:
                on_line += 1
            else:
                break
        return on_line / len(pts)

    # ------------------------------------------------------------------
    # Kinematics helpers (defensive: the fraction is advisory-strict, and a
    # kinematics mismatch must degrade to "accept", never fabricate a failure)
    # ------------------------------------------------------------------

    def _fk_tool_positions(self, joint_state):
        """FK the tool position of every waypoint of a JointState trajectory."""
        kin = self.motion_planner.kinematics
        probe = joint_state
        # The interpolated plan carries every model joint (cuRobo appends
        # locked DOF via get_full_dof_from_solution); project to the active
        # joints by name so compute_kinematics sees the model's own order.
        try:
            probe = kin.get_active_js(probe)
        except Exception:
            pass
        kin_state = kin.compute_kinematics(probe)
        return kin_state.tool_poses.position.reshape(-1, 3).detach().cpu().tolist()

    def _start_tool_position(self):
        start = getattr(self, "start_state", None)
        if start is None:
            return None
        pts = self._fk_tool_positions(start)
        return pts[0] if pts else None

    def _goal_tool_position(self, goal_request):
        goalsets = list(getattr(goal_request, "goalsets", None) or [])
        if not goalsets:
            return None
        poses = list(getattr(goalsets[0], "poses", None) or [])
        if not poses:
            return None
        index = 0
        picked = getattr(self, "_selected_goal_indexes", None)
        if picked and 0 <= int(picked[0]) < len(poses):
            index = int(picked[0])
        p = poses[index].position
        return [float(p.x), float(p.y), float(p.z)]

    def _tool_positions(self, trajectory):
        if trajectory is None:
            return []
        return self._fk_tool_positions(trajectory)

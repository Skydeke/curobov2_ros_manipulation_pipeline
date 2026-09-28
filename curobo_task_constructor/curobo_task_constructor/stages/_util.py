"""Shared helpers for the motion stages."""

from __future__ import annotations

import math

from curobo_task_constructor.core.geom import (
    Pose3,
    pose_to_any,
    quat_rotation_angle,
)
from curobo_task_constructor.core.robot import GoalsetSpec, PlanRequest, PlanningOptionsSpec

#: SetPlanner mirror (CLASSIC=0, MPC=1, BATCH=2, JOINT_SPACE=5, RETARGET=6).
#: MULTIPOINT=4 was removed together with the old MultiPointPlanner server
#: planner (the enum constant still exists in SetPlanner.srv for ABI).
PLANNER_KEYS = {
    "classic": 0, "mpc": 1, "batch": 2,
    "joint_space": 5, "retarget": 6,
}


def planner_key(params: dict):
    """Resolve the ``planner`` param to a SetPlanner int (None = default)."""
    planner = params.get("planner")
    if planner is None or isinstance(planner, int):
        return planner
    try:
        return PLANNER_KEYS[str(planner).lower()]
    except KeyError:
        raise ValueError(f"unknown planner {planner!r}; expected one of "
                         f"{sorted(PLANNER_KEYS)}") from None


def planning_options(params: dict) -> PlanningOptionsSpec:
    return PlanningOptionsSpec(
        num_seeds=int(params.get("num_seeds", 0) or 0),
        waypoint_tolerance=float(params.get("waypoint_tolerance", 0.0) or 0.0),
        exact_joints=list(params.get("exact_joints", []) or []),
        log_considered_trajectories=bool(params.get("log_considered", False)),
    )


def pose_from_params(cfg: dict, robot) -> object:
    """Build a Pose-like from a params dict ``{x,y,z,qx,qy,qz,qw}``."""
    p = cfg if isinstance(cfg, dict) else {}
    pose = Pose3([float(p.get("x", 0.0)), float(p.get("y", 0.0)),
                  float(p.get("z", 0.0))],
                 [float(p.get("qx", 0.0)), float(p.get("qy", 0.0)),
                  float(p.get("qz", 0.0)), float(p.get("qw", 1.0))])
    return pose_to_any(pose, getattr(robot, "pose_cls", None))


def goalset_for_scene(scene, *, poses=None, joint_positions=None) -> GoalsetSpec:
    return GoalsetSpec(
        poses=list(poses or []),
        target_joint_positions=list(joint_positions or []),
        allowed_collisions=scene.all_allowed_links() if scene is not None else [],
    )


def full_request(robot, start_joints, goalsets, params, planner=None) -> PlanRequest:
    return PlanRequest(
        start_pose=start_joints,
        goalsets=goalsets,
        options=planning_options(params),
        planner=planner if planner is not None else planner_key(params),
    )


#: ``Goalset.trajectory_constraints`` layout (``Goalset.msg``): 1 = hold that
#: axis along the whole path. Orientation first, then position — matching the
#: wire order, NOT cuRobo's ``ToolPoseCriteria`` order (the server re-orders it
#: in ``SinglePlanner._apply_pose_constraints``).
CONSTRAINT_ORDER = "theta_x theta_y theta_z x y z"


def axis_holds(start_pose, goal_poses, pos_tol: float = 0.005,
               rot_tol: float = 0.05) -> list:
    """``int8[6]`` holds that pin a path onto its straight segment.

    cuRobo scores a *non-terminal* waypoint's held-axis error against the GOAL
    pose in the world frame (``curobo/_src/cost/wp_tool_pose.py``, the
    unprojected branch: the reference pose is the goal and
    ``project_distance_to_goal`` stays at its ``False`` default because
    ``_apply_pose_constraints`` never sets it). So an axis can be HELD only
    where the start and the goal already agree: holding an axis they differ on
    would fight the very motion being planned.

    Consequences, both deliberate:

    - Orientation is all-or-nothing — a partial orientation hold is not
      expressible in the 6-slot vector, so it is held only when start and goal
      agree on the whole orientation (the usual case for a top-down grasp, where
      the tool does not rotate during the descent).
    - With several goal candidates, the hold is the INTERSECTION over all of
      them: an axis is held only if every candidate agrees with the start. That
      is the one vector that is correct for whichever candidate cuRobo ends up
      selecting out of the goalset, so it is applied unchanged afterwards
      (``_reset_pose_criteria`` restores the default criteria).
    """
    s = Pose3.from_any(start_pose)
    goals = [Pose3.from_any(g) for g in (goal_poses or [])]
    if not goals:
        return []
    hold_rot = 1 if all(
        quat_rotation_angle(s.orientation, g.orientation) <= rot_tol for g in goals
    ) else 0
    hold_pos = [
        1 if all(abs(s.position[i] - g.position[i]) <= pos_tol for g in goals) else 0
        for i in range(3)
    ]
    return [hold_rot] * 3 + hold_pos


def path_length_cost(trajectory, skip_tail: int = 0) -> float:
    """MTC ``cost::PathLength``: sum of per-waypoint joint-space L2 distance.

    Computed exactly from the waypoints that come back over the wire (each is a
    ``JointState`` in cspace order), so it is directly comparable to the
    reference pipeline's ``Connect`` cost term. Physically it also *rewards*
    straightness on a Cartesian move: a straight line in Cartesian space is a
    LONGER joint path than a shortcut around an obstacle, so minimising joint
    path length is what pushes the solver onto the line.

    ``skip_tail`` drops the trailing DOFs a pose goal leaves unconstrained (e.g.
    ``finger_joint``), so a gripper close does not register as arm motion.
    """
    pts = []
    for wp in (trajectory or []):
        pos = list(getattr(wp, "position", None) or [])
        pts.append(pos[:len(pos) - skip_tail] if skip_tail > 0 else pos)
    total = 0.0
    for a, b in zip(pts, pts[1:]):
        n = min(len(a), len(b))
        total += math.sqrt(sum((b[i] - a[i]) ** 2 for i in range(n)))
    return total


def cost_of(result, params: dict, start_joints=None) -> float:
    """Ranking cost of one solve, per the stage's ``cost`` param.

    Modes (``cost:`` in the stage params, default ``auto``):

    ``auto``         Prefer ``PlanResult.cost`` when a RobotInterface actually
                     populated it (the server's own opinion), else
                     ``path_length``. This is the default because it is
                     correct in both cases: today no adapter populates it (the
                     rclpy one deliberately leaves it ``inf``), so in practice
                     this is always path_length.
    ``path_length``  MTC ``cost::PathLength`` on the returned waypoints. It is
                     the reference pipeline's term, exactly computable from the
                     wire, and the only one comparable across planners (cuRobo's
                     ``seed_cost`` is a trajopt objective whose scale differs
                     per problem, so it must not be mixed with a length).
                     Physically it also *rewards* straightness on a Cartesian
                     move: a straight line is a longer joint path than a
                     shortcut, so minimising length is what pushes the solver
                     onto the line.
    ``waypoints``    Waypoint count — the previous proxy, kept for continuity.
    ``solver_cost``  The cheapest ``stats.considered[*].cost`` the server
                     reported. Needs ``log_considered: true`` to be populated.
    ``inf``          Never rank, never pick (keeps first-solution-wins).
    """
    mode = str((params or {}).get("cost", "auto"))
    if mode == "inf":
        return float("inf")
    if mode == "auto":
        if result.cost != float("inf"):
            return float(result.cost)
        mode = "path_length"
    if mode == "solver_cost":
        rows = getattr(getattr(result, "stats", None), "considered", None)
        costs = [float(getattr(r, "cost", float("inf"))) for r in (rows or [])]
        costs = [c for c in costs if c != float("inf")]
        if costs:
            return min(costs)
    elif mode == "waypoints":
        return float(len(result.trajectory)) if result.trajectory else 0.0
    skip_tail = int((params or {}).get("cost_skip_tail", 0) or 0)
    return path_length_cost(result.trajectory, skip_tail=skip_tail)
"""Shared helpers for the motion stages."""

from __future__ import annotations

from curobo_task_constructor.core.geom import Pose3, pose_to_any
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
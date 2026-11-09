"""Shared helpers for the motion stages."""

from __future__ import annotations

import math

from curobo_task_constructor.core.geom import (
    Pose3,
    pose_to_any,
    quat_rotation_angle,
)
from curobo_task_constructor.core.robot import GoalsetSpec, PlanRequest

#: SetPlanner mirror (CLASSIC=0, MPC=1, BATCH=2, CARTESIAN=3, MULTIPOINT=4,
#: JOINT_SPACE=5, RETARGET=6). MULTIPOINT is the ordered-waypoint backend:
#: the Python bridge selects ``plan_pose_waypoints`` for a goalset carrying
#: ``waypoints`` / ``tool_waypoints``; the native runtime has the same
#: planner (factory id ``multipoint``). CARTESIAN is the MTC
#: ``solvers::CartesianPath`` backend: it plans a straight line via the
#: axis-hold cost.
PLANNER_KEYS = {
    "classic": 0,
    "mpc": 1,
    "batch": 2,
    "cartesian": 3,
    "cartes": 3,
    "multipoint": 4,
    "waypoints": 4,
    "joint_space": 5,
    "retarget": 6,
}


def planner_key(params: dict):
    """Resolve the ``planner`` param to a SetPlanner int (None = default)."""
    planner = params.get("planner")
    if planner is None or isinstance(planner, int):
        return planner
    try:
        return PLANNER_KEYS[str(planner).lower()]
    except KeyError:
        raise ValueError(
            f"unknown planner {planner!r}; expected one of " f"{sorted(PLANNER_KEYS)}"
        ) from None


def joint_order(robot, joint_state) -> list:
    """Canonical cspace joint order for a joint goal.

    ``CuroboServerInterface`` derives this from the robot descriptor (or the
    named joint configs) and uses it to normalize every ``/joint_states``
    reading. Reusing it keeps a joint goal the same length and order as the
    start poses it is planned from: the server resolves a start pose by NAME,
    but takes ``target_joint_positions`` VERBATIM.
    """
    order = list(getattr(robot, "_joint_order", None) or [])
    if order:
        return order
    return list(getattr(joint_state, "name", []) or [])


def joint_positions_in_cspace(robot, joint_state) -> list:
    """Positions in the planner's canonical cspace order. ``/joint_states`` can carry MORE joints than the planner's cspace (the
    whole robot: arm, fingers, tips, base/PTU). A joint goal derived from that
    reading key-for-key produces a goal as long as the reading (16 values for
    an 8-DOF cspace) and the server rejects it with "q should have dof = 8,
    got 16"."""
    positions = list(getattr(joint_state, "position", []) or [])
    names = list(getattr(joint_state, "name", []) or [])
    order = list(getattr(robot, "_joint_order", None) or [])
    if not order or not names:
        return positions
    by_name = dict(zip(names, positions))
    return [by_name[n] for n in order if n in by_name]


def pose_from_params(cfg: dict, robot) -> object:
    """Build a Pose-like from a params dict ``{x,y,z,qx,qy,qz,qw}``."""
    p = cfg if isinstance(cfg, dict) else {}
    pose = Pose3(
        [float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("z", 0.0))],
        [
            float(p.get("qx", 0.0)),
            float(p.get("qy", 0.0)),
            float(p.get("qz", 0.0)),
            float(p.get("qw", 1.0)),
        ],
    )
    return pose_to_any(pose, getattr(robot, "pose_cls", None))


def goalset_for_scene(scene, *, poses=None, joint_positions=None) -> GoalsetSpec:
    return GoalsetSpec(
        poses=list(poses or []),
        target_joint_positions=list(joint_positions or []),
        allowed_collisions=scene.all_allowed_links() if scene is not None else [],
    )


def _joint_state_arrays(joint_state):
    """(names, positions) of a JointState-like, parallel and floored."""
    names = [str(n) for n in (getattr(joint_state, "name", None) or [])]
    positions = [float(p) for p in
                 (getattr(joint_state, "position", None) or [])]
    if len(positions) < len(names):
        names = names[: len(positions)]
    return names, positions


def measured_joint_state(robot):
    """The robot's newest measured joint state, or None if unavailable.

    A gripper-only stage may be planned without a ``current_state`` seed (a
    one-stage "open the jaws" task, the RViz panel's stage preview, or a
    chained task whose previous stage carries only its own group). The
    measured reading is the only source for the joints that stage does not
    name, so it must never be dropped just because the stage is instantaneous.
    """
    getter = getattr(robot, "get_current_joint_state", None)
    if not callable(getter):
        return None
    try:
        return getter()
    except Exception:  # no cached reading yet / offline mock
        return None


def gripper_controllers(robot) -> dict:
    """The descriptor's normalized ``gripper_controllers`` map, if exposed."""
    getter = getattr(robot, "gripper_controllers", None)
    if callable(getter):
        try:
            return dict(getter() or {})
        except Exception:  # pragma: no cover - defensive adapter
            return {}
    return {}


def gripper_joint_names(robot, gripper: str = "", joint: str = "") -> list:
    """Every joint one gripper command moves, in descriptor order.

    A parallel gripper (TIAGo's pal-gripper) has two finger joints driven by
    one ``FollowJointTrajectory`` command; a single-joint ``GripperCommand``
    has one. The task state must carry every commanded joint, otherwise the
    next leg's preview (and a state-only consumer) still sees the other jaw
    at its stale plan-time value.
    """
    cfgs = gripper_controllers(robot)
    cfg = cfgs.get(gripper) if gripper else None
    if cfg is None and cfgs:
        cfg = cfgs.get(next(iter(cfgs)))
    joints: list = []
    if cfg:
        joints = [str(j) for j in (cfg.get("joints") or []) if str(j)]
        if not joints:
            command_joint = str(cfg.get("command_joint") or "")
            if command_joint:
                joints = [command_joint]
    if not joints and joint:
        joints = [str(joint)]
    elif joint and str(joint) not in joints:
        # An explicit stage parameter still names a joint the state must carry.
        joints.append(str(joint))
    if not joints:
        resolver = getattr(robot, "gripper_joint", None)
        if callable(resolver):
            try:
                resolved = str(resolver(gripper) or "")
            except Exception:
                resolved = ""
            if resolved:
                joints = [resolved]
    return joints


def all_gripper_joints(robot) -> list:
    """Union of every joint that any descriptor gripper controller moves."""
    out: list = []
    for cfg in gripper_controllers(robot).values():
        joints = [str(j) for j in (cfg.get("joints") or []) if str(j)]
        if not joints and cfg.get("command_joint"):
            joints = [str(cfg["command_joint"])]
        for name in joints:
            if name not in out:
                out.append(name)
    resolver = getattr(robot, "gripper_joint", None)
    if callable(resolver):
        try:
            resolved = str(resolver("") or "")
        except Exception:
            resolved = ""
        if resolved and resolved not in out:
            out.append(resolved)
    return out


def merged_joint_state(robot, primary, fallback=None, overrides=None):
    """Merge joint states by NAME, primary wins, missing joints filled.

    Order is the planner cspace, then the descriptor's gripper joints, then
    whatever else ``primary``/``fallback`` carry. This gives a gripper-only
    stage a full robot state (arm from the previous stage or the live
    reading) without ever letting a stale live snapshot overwrite an arm
    value the task chain already planned.
    """
    primary = primary if primary is not None else fallback
    p_names, p_positions = _joint_state_arrays(primary)
    f_names, f_positions = _joint_state_arrays(fallback)
    override = {str(k): float(v) for k, v in (overrides or {}).items()}
    if not p_names and not f_names and not override:
        return primary

    order: list = []

    def add(name) -> None:
        name = str(name)
        if name and name not in order:
            order.append(name)

    for name in (getattr(robot, "_joint_order", None) or []):
        add(name)
    for name in all_gripper_joints(robot):
        add(name)
    for name in p_names:
        add(name)
    for name in f_names:
        add(name)
    for name in override:
        add(name)

    by_primary = dict(zip(p_names, p_positions))
    by_fallback = dict(zip(f_names, f_positions))
    names: list = []
    positions: list = []
    for name in order:
        if name in override:
            value = override[name]
        elif name in by_primary:
            value = by_primary[name]
        elif name in by_fallback:
            value = by_fallback[name]
        else:
            continue
        names.append(name)
        positions.append(float(value))
    if not names:
        return primary
    if names == p_names and positions == p_positions:
        return primary
    from curobo_task_constructor.core.state import make_joint_state
    cls = (type(primary) if primary is not None
           else getattr(robot, "joint_state_cls", None))
    return make_joint_state(cls, names, positions)


def gripper_command_for_joint_goal(robot, targets):
    """A gripper command for a joint goal the planner cspace does not own.

    TIAGo's fingers are ``lock_joints``, not planning DOFs: a sparse goal like
    ``{gripper_left_finger_joint: 0.045}`` cannot be a cuRobo trajectory. The
    old path projected it onto the cspace, leaving an arm-only goal equal to
    the start pose — a full-arm plan that did not move the jaws. When every
    named joint belongs to a descriptor gripper (and none is a planning DOF),
    route the goal to that gripper's controller instead. Robots whose cspace
    DOES contain the jaws (KATE/Kortex) fall through to normal planning.
    """
    if not isinstance(targets, dict) or not targets:
        return None
    goal = {str(k): float(v) for k, v in targets.items()}
    known = set(all_gripper_joints(robot))
    cspace = set(str(n) for n in (getattr(robot, "_joint_order", None) or []))
    if not known or not set(goal) <= known or set(goal) & cspace:
        return None
    values = list(goal.values())
    if max(values) - min(values) > 1e-9:
        # One GripperCommand position cannot express two different jaws.
        return None

    cfgs = gripper_controllers(robot)
    key = ""
    for name, cfg in cfgs.items():
        cfg_joints = {str(j) for j in (cfg.get("joints") or [])}
        if set(goal) <= cfg_joints:
            key = name
            break
    if not key:
        # No single controller owns every named joint. With exactly one entry
        # that entry is still the intended controller (a sparse goal may name
        # just the primary finger); with several, picking one silently would
        # leave another arm's jaws uncommanded — let the caller plan instead.
        if len(cfgs) != 1:
            return None
        key = next(iter(cfgs))
    command_joint = ""
    if key:
        command_joint = str(cfgs[key].get("command_joint") or "")
    if not command_joint:
        command_joint = next(iter(goal))
    return {
        "gripper": key,
        "joint": command_joint,
        "position": float(values[0]),
        "effort": 0.0,
        "max_effort": 0.0,
        "controller": "",
        "action_ns": "",
    }


def carry_missing_joints(previous, planned):
    """Keep joints the planner's trajectory omits (e.g. gripper fingers).

    The planner's cspace does not contain the jaws, so some responses carry
    only cspace names. The task state must still remember the commanded jaw
    value for the next leg; without this a pre-grasp leg planned right after
    ``ready`` would hand a closed finger state to descend/attach.
    """
    if planned is None:
        return previous
    if previous is None:
        return planned
    names = list(getattr(planned, "name", []) or [])
    positions = list(getattr(planned, "position", []) or [])
    prev_names = list(getattr(previous, "name", []) or [])
    prev_positions = list(getattr(previous, "position", []) or [])
    if not names or not prev_names:
        return planned
    present = set(names)
    changed = False
    for name, position in zip(prev_names, prev_positions):
        if name not in present:
            names.append(name)
            positions.append(float(position))
            changed = True
    if not changed:
        return planned
    try:
        from curobo_task_constructor.core.state import make_joint_state
        return make_joint_state(type(planned), names, positions)
    except Exception:
        return planned


def full_request(robot, start_joints, goalsets, params, planner=None,
                 scene=None) -> PlanRequest:
    """Build a PlanRequest, folding in the leg's CartesianPath interpolation
    options (MTC ``solvers::CartesianPath`` properties, put in the stage params
    by ``core.CartesianPath.extra_params``). A stage built with any other
    planner carries none of them, so the sentinels leave the server defaults
    in place — and the server ignores them unless the Cartesian planner is
    active. ``scene`` is the state's ``SceneDiff``: the leg is solved with the server
    reconciled to it (attach/detach previewed at ``start_joints``), so a leg
    after a grasp plans against the object already in the hand."""
    return PlanRequest(
        start_pose=start_joints,
        goalsets=goalsets,
        planner=planner if planner is not None else planner_key(params),
        cartesian_step_size=float(params.get("step_size", 0.0) or 0.0),
        cartesian_precision=float(params.get("precision", 0.0) or 0.0),
        scene=scene,
    )


#: ``Goalset.trajectory_constraints`` layout (``Goalset.msg``): 1 = hold that
#: axis along the whole path. Orientation first, then position — matching the
#: wire order, NOT cuRobo's ``ToolPoseCriteria`` order (the server re-orders it
#: in ``SinglePlanner._apply_pose_constraints``).
CONSTRAINT_ORDER = "theta_x theta_y theta_z x y z"

#: Waypoint solve options and their server defaults (``Goalset.msg``).
WAYPOINT_OPTION_DEFAULTS = {
    "path_position_tolerance": 0.05,
    "path_orientation_tolerance": 0.261799388,
    "final_position_tolerance": 0.005,
    "final_orientation_tolerance": 0.05,
    "track_orientation": True,
    "outer_iterations": 4,
}


def waypoint_goalset_options(params: dict) -> dict:
    """Validate and return the waypoint solve options for a goalset.

    The stage params use the wire field names (``path_position_tolerance``,
    ...), so a YAML-authored task and a ``GoalsetSpec`` carry the same keys.
    Every tolerance must be a finite, non-negative number; ``outer_iterations``
    must be a positive integer (the solver's own contract). A bad value raises
    ``ValueError`` so the stage fails with a clear comment instead of sending a
    request the server will reject.
    """
    params = params or {}
    out = {}
    for key in ("path_position_tolerance", "path_orientation_tolerance",
                "final_position_tolerance", "final_orientation_tolerance"):
        default = WAYPOINT_OPTION_DEFAULTS[key]
        try:
            value = float(params.get(key, default))
        except (TypeError, ValueError):
            raise ValueError(
                f"waypoint option {key!r} must be a number, got "
                f"{params.get(key)!r}") from None
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"waypoint option {key!r} must be a finite, positive number, "
                f"got {value!r}")
        if key.endswith("orientation_tolerance") and value > math.pi:
            raise ValueError(
                f"waypoint option {key!r} must be in (0, pi] radians, got "
                f"{value!r}")
        out[key] = value

    track = params.get("track_orientation",
                       WAYPOINT_OPTION_DEFAULTS["track_orientation"])
    if isinstance(track, str):
        track = track.strip().lower() not in ("", "0", "false", "no", "off")
    out["track_orientation"] = bool(track)

    outer = params.get("outer_iterations",
                       WAYPOINT_OPTION_DEFAULTS["outer_iterations"])
    if isinstance(outer, bool):
        raise ValueError(
            f"waypoint option 'outer_iterations' must be a positive integer, "
            f"got {outer!r}")
    try:
        outer = int(outer)
    except (TypeError, ValueError):
        raise ValueError(
            f"waypoint option 'outer_iterations' must be a positive integer, "
            f"got {params.get('outer_iterations')!r}") from None
    if outer < 1:
        raise ValueError(
            f"waypoint option 'outer_iterations' must be a positive integer, "
            f"got {outer}")
    out["outer_iterations"] = outer
    return out


def axis_holds(
    start_pose, goal_poses, pos_tol: float = 0.005, rot_tol: float = 0.05
) -> list:
    """``int8[6]`` holds that pin a path onto its straight segment. cuRobo scores a *non-terminal* waypoint's held-axis error against the GOAL
    pose in the world frame (``curobo/_src/cost/wp_tool_pose.py``, the
    unprojected branch: the reference pose is the goal and
    ``project_distance_to_goal`` stays at its ``False`` default because
    ``_apply_pose_constraints`` never sets it). So an axis can be HELD only
    where the start and the goal already agree: holding an axis they differ on
    would fight the very motion being planned."""
    s = Pose3.from_any(start_pose)
    goals = [Pose3.from_any(g) for g in (goal_poses or [])]
    if not goals:
        return []
    hold_rot = (
        1
        if all(
            quat_rotation_angle(s.orientation, g.orientation) <= rot_tol for g in goals
        )
        else 0
    )
    hold_pos = [
        1 if all(abs(s.position[i] - g.position[i]) <= pos_tol for g in goals) else 0
        for i in range(3)
    ]
    return [hold_rot] * 3 + hold_pos


def path_length_cost(trajectory, skip_tail: int = 0) -> float:
    """MTC ``cost::PathLength``: sum of per-waypoint joint-space L2 distance. Computed exactly from the waypoints that come back over the wire (each is a
    ``JointState`` in cspace order), so it is directly comparable to the
    reference pipeline's ``Connect`` cost term. Physically it also *rewards*
    straightness on a Cartesian move: a straight line in Cartesian space is a
    LONGER joint path than a shortcut around an obstacle, so minimising joint
    path length is what pushes the solver onto the line."""
    pts = []
    for wp in trajectory or []:
        pos = list(getattr(wp, "position", None) or [])
        pts.append(pos[: len(pos) - skip_tail] if skip_tail > 0 else pos)
    total = 0.0
    for a, b in zip(pts, pts[1:]):
        n = min(len(a), len(b))
        total += math.sqrt(sum((b[i] - a[i]) ** 2 for i in range(n)))
    return total


def cost_of(result, params: dict, start_joints=None) -> float:
    """Ranking cost of one solve, per the stage's ``cost`` param. Modes (``cost:`` in the stage params, default ``auto``):

    ``auto``         Prefer ``PlanResult.cost`` when a RobotInterface actually
                     populated it (the server's own opinion), else
                     ``path_length``. This is the default because it is
                     correct in both cases: today no adapter populates it (the
                     rclpy one deliberately leaves it ``inf``), so in practice
                     this is always path_length."""
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

"""TaskExecutor — MTC's ``Task::init()`` / ``plan()`` / ``execute()`` for cuRobo. Lifecycle (Sec. 3 of the plan):

1. ``build``    — turn the ``StageSpec`` tree into concrete stages via the open
                  ``STAGE_REGISTRY`` (graph.builder.build_tree)."""

from __future__ import annotations

from typing import Any, Optional

from curobo_task_constructor.core.container import GENERATE_INTERFACE
from curobo_task_constructor.core.robot import ObjectSpec, PlanResult, RobotInterface
from curobo_task_constructor.core.stage import InitStageError, Solution
from curobo_task_constructor.core.state import SceneDiff
from curobo_task_constructor.graph.builder import build_tree
from curobo_task_constructor.graph.spec import StageSpec


#: The start-agreement number the whole pipeline shares: 0.01 rad. Kept here
#: as the reference value because it is ALSO what the controller must enforce
#: before it accepts a replayed trajectory (``allowed_start_tolerance`` in
#: iki_kortex_moveit_config/config/moveit_controllers.yaml). The cuRobo
#: executor no longer compares endpoints itself — MTC ``Task::execute`` replays
#: the stored ``RobotTrajectory`` verbatim and lets the controller's start
#: tolerance reject a replay whose first point is not where the arm is. This
#: constant documents that shared number.

EXECUTE_CONTINUITY_TOLERANCE = 0.01


def _joint_positions(state) -> list:
    """Joint values out of a waypoint, whatever shape it arrived in.

    Waypoints reach here as JointState messages, as the raw responses they
    were built from, and as plain lists in tests, so all three are accepted.
    """
    positions = getattr(state, "position", state)
    return [float(x) for x in positions]


def _joint_state_to_dict(state) -> dict:
    """JointState-like -> ``{"name": [...], "position": [...]}``.

    The solution's start state travels with the display payload so a consumer
    can fill joints a planned trajectory does not name (the planner's cspace
    is arm-only, while the task state also carries the finger). Without it a
    gripper-only waypoint has no arm pose to stand on and a missing finger
    silently reads as the URDF default.
    """
    names = [str(n) for n in (getattr(state, "name", None) or [])]
    positions = [float(p) for p in (getattr(state, "position", None) or [])]
    if len(positions) < len(names):
        names = names[:len(positions)]
    return {"name": names, "position": positions}


def _max_joint_delta(a, b) -> float:
    """Largest per-joint difference between two waypoints, in rad.

    A missing trailing DOF (a 7-DOF arm group against a 8-joint arm) is
    compared only over the joints both carry, matching how
    ``_resolve_start_state`` pads a short start pose.
    """
    pa, pb = _joint_positions(a), _joint_positions(b)
    n = min(len(pa), len(pb))
    if n == 0:
        return 0.0
    return max(abs(x - y) for x, y in zip(pa[:n], pb[:n]))


def _align_waypoint_positions(wp, names) -> Optional[list]:
    """Waypoint values in ``names`` order, or None when it cannot be aligned.

    A plan response and a ``/joint_states`` reading can carry the SAME joints
    in a different order (or a different subset, e.g. a 16-joint reading next
    to an 8-joint cspace). Appending positions positionally would then paint
    the wrong joint — visible as a gripper that folds while the arm is still.
    """
    pos = _joint_positions(wp)
    wp_names = [str(n) for n in (getattr(wp, "name", None) or [])]
    if not wp_names or wp_names == names:
        return pos
    by_name = {n: i for i, n in enumerate(wp_names)}
    if any(n not in by_name for n in names):
        return None
    out = []
    for n in names:
        i = by_name[n]
        if i >= len(pos):
            return None
        out.append(pos[i])
    return out


def _replay_trajectory_dict(waypoints) -> Optional[dict]:
    """Pack a solved segment's waypoints for verbatim replay. MTC parity: ``Task::execute`` hands each sub-trajectory's stored
    ``RobotTrajectory`` to the controller and never re-derives it. The cuRobo
    adapter hands the same thing to the server as ``{"joint_names", "points",
    "velocities"}`` so ``SendTrajectory`` can stage and stream exactly these
    waypoints (``replay``) instead of re-matching a planning goal against the
    server's trajectory cache."""
    wps = list(waypoints or [])
    if not wps:
        return None
    names: list = []
    points: list = []
    velocities: list = []
    for wp in wps:
        if not names:
            names = [str(n) for n in (getattr(wp, "name", None) or [])]
        aligned = _align_waypoint_positions(wp, names) if names else None
        if aligned is None:
            raw = _joint_positions(wp)
            # Legacy positional layout: safe only while the width agrees.
            aligned = raw if len(raw) == len(names) else None
        if aligned is None:
            continue
        points.append(aligned)
        vel = getattr(wp, "velocity", None)
        velocities.append([float(v) for v in vel] if vel else [])
    return {"joint_names": names, "points": points, "velocities": velocities}


def _pose_to_dict(pose) -> Optional[dict]:
    """Any pose-like (dict, Pose3, ROS/stub Pose, PoseStamped) -> flat dict."""
    if pose is None:
        return None
    if isinstance(pose, dict):
        if {"x", "y", "z"} <= set(pose):
            return {"x": float(pose.get("x", 0.0)),
                    "y": float(pose.get("y", 0.0)),
                    "z": float(pose.get("z", 0.0)),
                    "qx": float(pose.get("qx", 0.0)),
                    "qy": float(pose.get("qy", 0.0)),
                    "qz": float(pose.get("qz", 0.0)),
                    "qw": float(pose.get("qw", 1.0))}
        return None
    stamped = getattr(pose, "pose", None)
    if stamped is not None and not hasattr(pose, "position"):
        return _pose_to_dict(stamped)
    pos = getattr(pose, "position", None)
    ori = getattr(pose, "orientation", None)
    if pos is None or ori is None:
        return None
    try:
        if isinstance(pos, (list, tuple)):
            x, y, z = (float(v) for v in list(pos)[:3])
        else:
            x, y, z = float(pos.x), float(pos.y), float(pos.z)
        if isinstance(ori, (list, tuple)):
            qx, qy, qz, qw = (float(v) for v in list(ori)[:4])
        else:
            qx, qy, qz, qw = (float(ori.x), float(ori.y),
                              float(ori.z), float(ori.w))
    except (TypeError, ValueError, AttributeError):
        return None
    return {"x": x, "y": y, "z": z, "qx": qx, "qy": qy, "qz": qz, "qw": qw}


def _pose_stub_from_dict(d) -> Any:
    """Flat pose dict -> duck-typed Pose stub (adapter parses it via Pose3)."""
    from curobo_task_constructor.core.geom import _PoseStub
    d = d or {}
    return _PoseStub(
        float(d.get("x", 0.0)), float(d.get("y", 0.0)),
        float(d.get("z", 0.0)), float(d.get("qx", 0.0)),
        float(d.get("qy", 0.0)), float(d.get("qz", 0.0)),
        float(d.get("qw", 1.0)))


def _object_spec_to_dict(spec) -> dict:
    pose = _pose_to_dict(getattr(spec, "pose", None))
    return {
        "name": str(getattr(spec, "name", "")),
        "shape": str(getattr(spec, "shape", None) or "cuboid"),
        "pose": pose,
        "dimensions": ([float(v) for v in (getattr(spec, "dimensions", None)
                                           or [])] or None),
        "mesh_path": getattr(spec, "mesh_path", None),
        "vertices": getattr(spec, "vertices", None),
        "triangles": getattr(spec, "triangles", None),
    }


def _object_spec_from_dict(d) -> Any:
    from curobo_task_constructor.core.state import ObjectSpec
    d = d or {}
    pose = d.get("pose")
    return ObjectSpec(
        name=str(d.get("name", "")),
        shape=str(d.get("shape", None) or "cuboid"),
        pose=_pose_stub_from_dict(pose) if pose else None,
        dimensions=d.get("dimensions"),
        mesh_path=d.get("mesh_path"),
        vertices=d.get("vertices"),
        triangles=d.get("triangles"),
    )


def _scene_ops_to_dict(ops) -> dict:
    """Materialization deltas -> MTC scene_diff-shaped dict.

    ``remove_all``/``detach_all`` have no MTC counterpart (a PlanningScene diff
    is name-keyed — there is no wildcard REMOVE), so they ride two extra
    boolean keys rather than being dropped. Dropping them is NOT harmless: the
    scene-setup task clears the world with ``remove_all`` (and releases any
    stale attach with ``detach_all``), and a name-keyed diff cannot say
    "everything", so the executing adapter would re-add onto the previous
    cycle's world — where every ``add_object`` is refused as a duplicate and
    the scene is left stale forever.
    """
    added, removed, attached, detached = [], [], None, None
    remove_all = detach_all = False
    for kind, payload in (ops or []):
        if kind == "add":
            added.append(_object_spec_to_dict(payload))
        elif kind == "remove":
            removed.append(str(payload))
        elif kind == "attach":
            attached = str(payload)
        elif kind == "detach":
            detached = None if payload is None else str(payload)
        elif kind == "remove_all":
            remove_all = True
        elif kind == "detach_all":
            detach_all = True
    return {"added": added, "removed": removed,
            "attached": attached, "detached": detached,
            "remove_all": remove_all, "detach_all": detach_all}


def _scene_ops_from_dict(d) -> list:
    """MTC scene_diff-shaped dict -> (kind, payload) op list, in order.

    Clear-all comes FIRST: ``detach_all`` then ``remove_all`` must precede the
    adds (and the named removes), or the world they clear would be the one just
    built in the same batch.
    """
    d = d or {}
    ops = []
    if d.get("detach_all"):
        ops.append(("detach_all", None))
    if d.get("remove_all"):
        ops.append(("remove_all", None))
    for name in (d.get("removed", None) or []):
        ops.append(("remove", str(name)))
    for spec in (d.get("added", None) or []):
        ops.append(("add", _object_spec_from_dict(spec)))
    if d.get("detached") is not None:
        ops.append(("detach", str(d["detached"])))
    if d.get("attached") is not None:
        ops.append(("attach", str(d["attached"])))
    return ops


def _plan_request_to_dict(req) -> Optional[dict]:
    """PlanRequest -> plain dicts (poses flat, joints float lists)."""
    if req is None:
        return None
    goalsets = []
    for gs in (getattr(req, "goalsets", None) or []):
        poses = []
        for p in (getattr(gs, "poses", None) or []):
            pd = _pose_to_dict(p)
            if pd is not None:
                poses.append(pd)
        goalsets.append({
            "poses": poses,
            "target_joints": [float(v) for v in (
                getattr(gs, "target_joint_positions", None) or [])],
            "allowed": [str(v) for v in (
                getattr(gs, "allowed_collisions", None) or [])],
            "holds": [int(v) for v in (
                getattr(gs, "trajectory_constraints", None) or [])],
        })
    start = getattr(req, "start_pose", None)
    return {
        "start": ({"name": [str(n) for n in (getattr(start, "name", []) or [])],
                   "position": [float(v) for v in (
                       getattr(start, "position", []) or [])]}
                  if start is not None else None),
        "goalsets": goalsets,
        "planner": getattr(req, "planner", None),
    }


def _plan_request_from_dict(d) -> Optional[Any]:
    """Plain dicts -> PlanRequest (poses as stubs the adapter parses)."""
    if not d:
        return None
    from curobo_task_constructor.core.robot import GoalsetSpec, PlanRequest
    from curobo_task_constructor.core.state import JointStateStub
    goalsets = []
    for gs in (d.get("goalsets", None) or []):
        goalsets.append(GoalsetSpec(
            poses=[_pose_stub_from_dict(p) for p in (gs.get("poses", None) or [])],
            target_joint_positions=[float(v) for v in (
                gs.get("target_joints", None) or [])],
            allowed_collisions=[str(v) for v in (gs.get("allowed", None) or [])],
            trajectory_constraints=[int(v) for v in (gs.get("holds", None) or [])],
        ))
    start = d.get("start")
    start_pose = None
    if start:
        start_pose = JointStateStub(dict(zip(
            [str(n) for n in (start.get("name", None) or [])],
            [float(v) for v in (start.get("position", None) or [])])))
    return PlanRequest(goalsets=goalsets, start_pose=start_pose,
                       planner=d.get("planner"),
                       cartesian_step_size=float(
                           d.get("cartesian_step_size", 0.0) or 0.0),
                       cartesian_precision=float(
                           d.get("cartesian_precision", 0.0) or 0.0))



def drive_solution_dict(robot, d, applied_ops=None,
                        progress_callback=None, cancel_event=None) -> tuple:
    """Drive a deserialized solution without an executor tree. The execute action server (another process, no stage tree) drives a
    received Solution this way: scene ops apply through ``robot`` in chain
    order and each motion segment replays its own VALIDATED waypoints via
    ``robot.execute(req, trajectory=...)`` — MTC ``Task::execute`` parity, the
    server streams the segment's stored trajectory verbatim (``replay``), so
    there is no cache lookup, no re-solve, and no way for a multi-solution pose
    goal to jump IK branch mid-chain. ``req`` is still sent for the planner
    switch and its ``allowed_collisions``."""
    from curobo_task_constructor.core.robot import PlanResult

    if applied_ops is None:
        applied_ops = []
    results = []
    segments = list((d or {}).get("sub_trajectories", None) or [])
    total = len(segments)

    def _apply(kind, payload) -> None:
        key = (kind, getattr(payload, "name", payload)
               if kind == "add" else payload)
        if key in applied_ops:
            return
        applied_ops.append(key)
        if kind == "add":
            robot.add_object(payload)
        elif kind == "remove":
            robot.remove_object(payload)
        elif kind == "remove_all":
            robot.remove_all_objects()
        elif kind == "detach_all":
            robot.detach_object(None)
        elif kind == "attach":
            robot.attach_object(payload)
        elif kind == "detach":
            robot.detach_object(payload)
        else:
            raise ValueError(f"unknown scene op kind {kind!r}")

    for index, seg in enumerate(segments):
        if cancel_event is not None and cancel_event.is_set():
            return results, None, True
        info = seg.get("info", {}) if isinstance(seg, dict) else {}
        diff = seg.get("scene_diff", {}) if isinstance(seg, dict) else {}
        for op in _scene_ops_from_dict(diff.get("before", {}) or {}):
            _apply(op[0], op[1])
        replay_goal = seg.get("replay_goal", {}) or {}
        gripper = (replay_goal.get("gripper_command")
                   if isinstance(replay_goal, dict) else None)
        if gripper:
            if progress_callback is not None:
                progress_callback(index, total, info)
            result = robot.gripper_command(**gripper)
            results.append(result)
            if not result.success:
                return results, index, False
            for op in _scene_ops_from_dict(diff.get("after", {}) or {}):
                _apply(op[0], op[1])
            continue
        req = _plan_request_from_dict(replay_goal)
        if req is None:
            for op in _scene_ops_from_dict(diff.get("after", {}) or {}):
                _apply(op[0], op[1])
            continue
        if progress_callback is not None:
            progress_callback(index, total, info)
        traj = seg.get("trajectory", {}) or {}
        points = list(traj.get("points", []) or [])
        if len(points) <= 1:
            # Trivial (no-motion) segment: the planner answers these without
            # solving (see CuroboServerInterface._trivial_goal), so there is
            # nothing to drive. Scene ops still apply (below); progress already
            # fired (above), so the panel still follows the chain.
            results.append(PlanResult(
                True, "trivial (no motion needed)"))
            for op in _scene_ops_from_dict(diff.get("after", {}) or {}):
                _apply(op[0], op[1])
            continue
        result = robot.execute(req, trajectory=traj)
        if not result.success:
            results.append(result)
            return results, index, False
        for op in _scene_ops_from_dict(diff.get("after", {}) or {}):
            _apply(op[0], op[1])
        results.append(result)
    if cancel_event is not None and cancel_event.is_set():
        return results, None, True
    return results, None, False

__all__ = [
    'EXECUTE_CONTINUITY_TOLERANCE',
    '_joint_positions',
    '_joint_state_to_dict',
    '_max_joint_delta',
    '_align_waypoint_positions',
    '_replay_trajectory_dict',
    '_pose_to_dict',
    '_pose_stub_from_dict',
    '_object_spec_to_dict',
    '_object_spec_from_dict',
    '_scene_ops_to_dict',
    '_scene_ops_from_dict',
    '_plan_request_to_dict',
    '_plan_request_from_dict',
    'drive_solution_dict',
]

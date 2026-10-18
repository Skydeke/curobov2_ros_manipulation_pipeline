"""TaskExecutor — MTC's ``Task::init()`` / ``plan()`` / ``execute()`` for cuRobo.

Lifecycle (Sec. 3 of the plan):

1. ``build``    — turn the ``StageSpec`` tree into concrete stages via the open
                  ``STAGE_REGISTRY`` (graph.builder.build_tree).
2. ``init()``   — ``init()`` every stage against the task's base scene and
                  ``resolve()`` the root with the GENERATE interface. Interface
                  adjacency inside every container is validated here; a task
                  that fails init is reported invalid (TaskDescription.valid
                  == false) and never computed.
3. ``plan()``   — drive the tree depth-first: generators seed first,
                  propagators extend, connectors run last (they are the
                  combinatorially expensive ones). Each motion leg reconciles
                  the shared server's scene to the ``SceneDiff`` it carries, so
                  a leg after a grasp plans with the object already attached
                  (MTC applies ``ModifyPlanningScene`` at compute time); the
                  pre-plan scene is restored when the burst ends. Because every
                  stage CONSUMES its pulls on compute and pushes are
                  keyed-deduped, the loop ``while any(s.can_compute())``
                  terminates on its own; ``max_iterations`` is only a
                  belt-and-braces guard.
4. ``rank()``   — full root solutions ranked by accumulated cost.
5. ``execute()``— apply the winning solution's scene deltas in chain order,
                  then re-solve + drive each motion segment (SendTrajectory).

Introspection (Sec. 7): ``describe()`` publishes the built TaskDescription and
``statistics()`` roll up per-stage StageStatistics + per-attempt SolutionInfo
records the action server publishes on the introspection topics.
"""

from __future__ import annotations

from typing import Any, Optional

from curobo_task_constructor.core.container import GENERATE_INTERFACE
from curobo_task_constructor.core.robot import ObjectSpec, PlanResult, RobotInterface
from curobo_task_constructor.core.stage import InitStageError, Solution
from curobo_task_constructor.core.state import SceneDiff
from curobo_task_constructor.graph.builder import build_tree
from curobo_task_constructor.graph.spec import StageSpec

__all__ = ["TaskExecutor", "EXECUTE_CONTINUITY_TOLERANCE"]

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


def _replay_trajectory_dict(waypoints) -> Optional[dict]:
    """Pack a solved segment's waypoints for verbatim replay.

    MTC parity: ``Task::execute`` hands each sub-trajectory's stored
    ``RobotTrajectory`` to the controller and never re-derives it. The cuRobo
    adapter hands the same thing to the server as ``{"joint_names", "points",
    "velocities"}`` so ``SendTrajectory`` can stage and stream exactly these
    waypoints (``replay``) instead of re-matching a planning goal against the
    server's trajectory cache. Returns ``None`` when there is no trajectory
    (a mutation leaf, whose drive is skipped upstream)."""
    wps = list(waypoints or [])
    if not wps:
        return None
    names: list = []
    points: list = []
    velocities: list = []
    for wp in wps:
        if not names:
            names = [str(n) for n in (getattr(wp, "name", None) or [])]
        points.append(_joint_positions(wp))
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


class TaskExecutor:
    """Build + init + plan + rank + execute a declarative task."""

    def __init__(self, spec: StageSpec, robot: RobotInterface,
                 base_scene: Optional[SceneDiff] = None,
                 task_id: Optional[str] = None):
        self.spec = spec
        self.robot = robot
        self.task_id = task_id or "task"
        # The task's starting world; every InterfaceState diff is relative to
        # this, so a freshly-built stage tree can be planned against any base
        # scene without re-querying the server.
        self.base_scene = base_scene if base_scene is not None else SceneDiff()
        self.root = build_tree(spec)
        self._valid = False
        self._init_error = ""
        self._applied_ops: list = []
        self._published_attempts = 0  # for SolutionInfo ids
        #: Global solution id registry (MTC ``Introspection::id_solution_bimap_``).
        #: ONE id space shared by everything that names a solution:
        #: ``SolutionInfo.id``, ``StageStatistics.solved``/``failed``, the
        #: ``sub_solution_id`` tree, and ``GetSolution``. Idempotent by
        #: identity, ids start at 1, reset per ``reset()``. The OBJECT is
        #: held alongside its key, so a freed solution's ``id()`` cannot be
        #: reused by a new one (which would alias two solutions onto one id)
        #: and so ``solutionFromId`` can look one up.
        self._solution_ids: dict = {}          # id(obj) -> global id
        self._solution_objects: dict = {}     # global id -> obj
        self._next_solution_id = 1

    # ------------------------------------------------------------------
    # Global solution ids (MTC Introspection::solutionId)
    # ------------------------------------------------------------------
    def register_solution(self, obj: Any) -> int:
        """Global id for a solution or failure (MTC ``solutionId``).

        Ids must come from HERE rather than from each Solution's
        per-stage ``solution_id``: the per-stage record id restarts at 0 in
        every stage, so three stages' first solutions all read 0 while three
        failures read 1, 2, 3. A client (the rviz panel) that joins those to
        the stage ids reported in the same message then asks GetSolution for
        an id that names a different solution and displays a random one.
        """
        key = id(obj)
        known = self._solution_ids.get(key)
        if known is not None:
            return known
        new_id = self._next_solution_id
        self._next_solution_id += 1
        self._solution_ids[key] = new_id
        self._solution_objects[new_id] = obj
        return new_id

    def global_solution_id(self, obj: Any) -> int:
        """Existing global id, or ``register_solution`` (MTC lookup-or-add)."""
        known = self._solution_ids.get(id(obj))
        return known if known is not None else self.register_solution(obj)

    def solutionFromId(self, solution_id: Any) -> Any:
        """The object named by a global id, or None (MTC ``solutionFromId``).

        Reverse of ``register_solution``; the ``GetSolution`` service needs it
        to turn the id the panel sends back into the Solution/StageFailure.
        """
        try:
            wanted = int(solution_id)
        except (TypeError, ValueError):
            return None
        if wanted < 1:
            return None
        return self._solution_objects.get(wanted)

    def reset_solution_ids(self) -> None:
        """Drop every global solution id (MTC ``Introspection::resetMaps``).

        Deliberately separate from :meth:`reset`: resetting the ID MAPS is
        something introspection does at its own lifecycle points (and from
        ``Task::reset``), while resetting the STAGES belongs to the task and
        must not happen mid-plan. Calling ``reset()`` from ``setup()`` ran
        the stage reset after ``init()`` had just resolved the tree, which
        left ``_valid`` False and made every live plan fail.
        """
        self._solution_ids = {}
        self._solution_objects = {}
        self._next_solution_id = 1

    # ------------------------------------------------------------------
    # Build / init
    # ------------------------------------------------------------------
    def init(self) -> bool:
        """Validate the whole tree against the base scene (adjacency checks
        included) and resolve the root's GENERATE interface.

        Returns True on success; failures are non-fatal and readable via
        ``describe()`` — mirroring TaskDescription.valid/comment.
        """
        for stage in self.root.subtree_stages():
            stage.reset()
        # Stable depth-first ids so introspection messages (Sec. 7) can key
        # SolutionInfo / StageStatistics to a stage across solves. MTC
        # numbers the (unpublished) task wrapper 0 and the root container 1
        # (introspection.cpp:148), so ids start at 1 here too.
        for idx, stage in enumerate(self.root.subtree_stages()):
            stage.stage_id = idx + 1
        try:
            self.root.init(self.base_scene, self.robot)
            self.root.resolve(*GENERATE_INTERFACE)
            self._valid = True
            self._init_error = ""
        except Exception as exc:  # InitStageError and friends
            self._valid = False
            self._init_error = str(exc)
        return self._valid

    def build_base_scene(self) -> SceneDiff:
        """Derive the task's base scene from the server's live world (empty
        diff when the robot interface has nothing to report).

        Each object is rebuilt from what the SERVER reports, not from a local
        mirror of this task's own add calls: ``get_scene_objects`` carries the
        type, pose, size and mesh path, so the base scene describes the world as
        it actually is — including objects another client put there. It used to
        label every entry ``shape="mesh"`` regardless of what it was, which made
        this reverse sync describe the world wrong for every non-mesh object.
        """
        obj_names = getattr(self.robot, "get_object_names", None)
        if obj_names is None:
            return SceneDiff()
        # Preferred: one call per object that returns the whole description.
        get_spec = getattr(self.robot, "get_object_spec", None)
        scene = SceneDiff()
        for name in obj_names() or []:
            if get_spec is not None:
                spec = get_spec(name)
                if spec is not None:
                    scene.objects_added[name] = spec
                continue
            # Fallback for an interface that only answers the older two questions.
            # The shape is left at ObjectSpec's default rather than guessed.
            pose = self.robot.get_object_pose(name)
            if pose is not None:
                scene.objects_added[name] = ObjectSpec(name=name, pose=pose)
        return scene

    # ------------------------------------------------------------------
    # Plan
    # ------------------------------------------------------------------
    def plan(self, max_iterations: int = 0, progress_callback=None) -> bool:
        """Run the compute loop until no stage can make progress.

        Returns True when at least one full root solution was found.
        ``max_iterations`` (0 = unlimited) guards against pathological graphs.
        ``progress_callback`` (MTC ``publishTaskState``) fires after every
        compute pass so introspection can stream per-stage rollups while
        planning instead of only at the end.

        The loop also stops the moment the FIRST complete root solution
        exists. Without that, a serial container sitting above a fallbacks /
        alternatives container keeps re-computing its trailing chain (e.g.
        return -> forbid -> open -> detach) for every upstream reconnect —
        pure waste that restarts already-solved stages and was observed to
        trigger a server-side gpu_lock wedge right after the pick task
        finished planning (the reconnect re-plan of the 'open' stage ~1.5 s
        after the first solve left the CUDA graph capture stuck, bricking
        gpu_lock until a node restart). The first full root solution is
        exactly what ``best()`` would have ranked first anyway — fallbacks
        already committed to its variant before the trailing chain was
        built — so nothing is lost by exiting early, and planning finishes
        sooner.
        """
        if not self._valid:
            return False
        iterations = 0
        try:
            while any(s.can_compute() for s in self.root.subtree_stages()):
                self.root.run_compute()
                iterations += 1
                if progress_callback is not None:
                    progress_callback()
                if self.root.solutions:
                    # First complete root solution: stop re-connecting / re-solving
                    # the trailing chain (see docstring). Return the first full
                    # solution — best() ranks it later.
                    break
                if max_iterations and iterations >= max_iterations:
                    break
        finally:
            # Planning previewed scene changes on the shared server (e.g. the
            # grasped object attached for the legs after the grasp). Restore
            # the pre-plan scene so execution materializes its scene ops in
            # chain order from the task's own base scene instead of inheriting
            # the last leg's preview.
            restore = getattr(self.robot, "end_plan_scene", None)
            if callable(restore):
                restore()
        return bool(self.root.solutions)

    def rank(self) -> list:
        """Full root solutions, best (lowest accumulated cost) first."""
        return sorted(self.root.solutions, key=lambda s: s.cost)

    def best(self, cost_threshold: Optional[float] = None) -> Optional[Solution]:
        """The best solution, or the first acceptable one under the cost
        threshold (MTC's equally-ranked acceptable solutions)."""
        for sol in self.rank():
            if cost_threshold is None or sol.cost <= float(cost_threshold):
                return sol
        return None

    def reset(self) -> None:
        """Best-effort reset so the same executor can re-plan (MTC Task::reset
        is cold-restart only; root.setCandidate early-returns unmodified)."""
        self._applied_ops = []
        for stage in self.root.subtree_stages():
            stage.reset()
        self.root._resolved = False
        self._valid = False
        # Global solution ids are per-plan (MTC resets its bimap per task
        # lifecycle): a re-plan re-issues ids from 1, so a stale id from the
        # previous plan is not served as if it still named a solution.
        self.reset_solution_ids()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def flatten_leaves(self, sol: Solution) -> list:
        """Leaf Solution segments of a composed solution, in chain order."""
        out: list = []

        def walk(s: Solution) -> None:
            if s.children:
                for ch in s.children:
                    walk(ch)
            else:
                out.append(s)

        walk(sol)
        return out

    def execute(self, sol: Solution, progress_callback=None,
                cancel_event=None) -> list:
        """Play back one solution: materialize each segment's scene delta on
        the curobo server in chain order, then drive each motion segment
        (SendTrajectory). MTC ``Task::execute`` parity: a segment is driven
        from the trajectory this ``Solution`` ALREADY carries — the caller
        passes the segment's PLANNED waypoints and the server streams them
        verbatim (``replay``), so there is no cache lookup, no re-solve, and a
        multi-solution pose goal cannot pick a different IK branch mid-chain
        than the one that was planned, collision-checked and displayed.
        Returns the list of drive results.

        ``progress_callback`` fires with each driven segment's stage name
        (MTC's active-stage highlight during playback): the node publishes
        it as action feedback so the panel can follow execution.
        ``cancel_event`` (threading.Event, MTC preempt) stops the chain
        between segments; the returned list is truncated like a failure.

        The chain STOPS at the first segment whose drive failed. The remaining
        segments each carry a ``start_pose`` baked from the PLANNED end state of
        their predecessor, so continuing past a failure drives requests anchored
        to poses the arm never reached. Start-state agreement (the arm actually
        being where a replay assumes) is the controller's job, enforced by
        ``allowed_start_tolerance`` — exactly where MoveIt puts it.

        The returned list is truncated at the failure, so the caller's
        zip-with-motion-leaves pairs each result with the leaf that produced
        it and reports the right stage name.
        """
        self._applied_ops = []
        results = []
        for leaf in self.flatten_leaves(sol):
            if cancel_event is not None and cancel_event.is_set():
                break
            for kind, payload in leaf.scene_ops or []:
                self._apply_op(kind, payload)
            if leaf.plan_request is None:
                continue
            if progress_callback is not None:
                progress_callback(getattr(leaf.stage, "name", ""))
            result = self.robot.execute(
                leaf.plan_request,
                trajectory=_replay_trajectory_dict(leaf.trajectory))
            if not result.success:
                results.append(result)
                break
            results.append(result)
        return results

    def _apply_op(self, kind: str, payload) -> None:
        key = (kind, payload.name if kind == "add" else payload)
        if key in self._applied_ops:
            return
        self._applied_ops.append(key)
        if kind == "add":
            self.robot.add_object(payload)
        elif kind == "remove":
            self.robot.remove_object(payload)
        elif kind == "remove_all":
            self.robot.remove_all_objects()
        elif kind == "detach_all":
            # Clear the attach unconditionally (name-agnostic, the server's
            # /detach_object Trigger): remove_all_objects alone drops the
            # obstacle list but not the attach, whose name-based disable the
            # checkers re-assert after the re-add.
            self.robot.detach_object(None)
        elif kind == "attach":
            self.robot.attach_object(payload)
        elif kind == "detach":
            self.robot.detach_object(payload)
        else:
            raise ValueError(f"unknown scene op kind {kind!r}")

    # ------------------------------------------------------------------
    # Introspection (Sec. 7)
    # ------------------------------------------------------------------
    def describe(self) -> dict:
        """TaskDescription payload: the built StageSpec tree + validity."""
        return {
            "task_id": self.task_id,
            "root": self.spec.to_dict(),
            "stage_count": len(self.root.subtree_stages()),
            "valid": self._valid,
            "comment": self._init_error,
        }

    def statistics(self) -> dict:
        """Per-stage StageStatistics + per-attempt SolutionInfo rollups.

        Shapes mirror ``curobo_task_constructor_interfaces``:
        stages: [{stage_id, stage_name, stage_type, attempt_count,
                  success_count, last_cost, total_compute_time}]
        attempts: [{stage_id, stage_name, solution_id, cost, success,
                    comment, planner_id}]

        ``last_cost`` is the lowest cost among the stage's solutions (not the
        last emitted — a multi-candidate goalset emits one solution per
        candidate, and the last one is not necessarily the cheapest).
        ``total_compute_time`` is in seconds (wall-clock, accumulated across
        every ``run_compute`` call on the stage).
        ``attempt_count`` is ``success_count + failure_count`` — the number of
        records the stage actually produced, which is exactly what the
        published ``StageStatistics`` shows and what a client can
        independently recount from ``solved`` + ``failed``. A count that
        disagrees with those columns is worse than no count.

        ``planner_calls`` is kept alongside it for the number that motivated
        the old field (planner calls, including attempts that produced no
        record — a discarded best-of-N seed, a generator pass). It is NOT
        reconcilable with solved/failed by design, so it lives under its own
        name rather than masquerading as the attempt count.
        """
        stages = []
        attempts = []
        for stg in self.root.subtree_stages():
            best_cost = min((s.cost for s in stg.solutions), default=float("inf"))
            stages.append({
                "stage_id": stg.stage_id,
                "stage_name": stg.name,
                "stage_type": stg.stage_type(),
                # Reconciles with success_count + failure_count below.
                "attempt_count": len(stg.solutions) + len(stg.failures),
                "success_count": len(stg.solutions),
                "failure_count": len(stg.failures),
                # Planner calls, NOT the record count (see docstring).
                "planner_calls": stg.attempt_count,
                "last_cost": best_cost,
                "total_compute_time": stg.compute_time,  # seconds
            })
            for sol in stg.solutions:
                attempts.append({
                    "stage_id": stg.stage_id,
                    "stage_name": stg.name,
                    # Global id: the same number the published SolutionInfo
                    # carries, so a client can join the two streams.
                    "solution_id": self.register_solution(sol),
                    "cost": sol.cost,
                    "success": True,
                    "comment": sol.comment,
                    "planner_id": self._planner_id(sol),
                })
            for fail in stg.failures:
                attempts.append({
                    "stage_id": stg.stage_id,
                    "stage_name": stg.name,
                    "solution_id": self.register_solution(fail),
                    "cost": float("inf"),
                    "success": False,
                    "comment": fail.message,
                    "planner_id": "",
                })
        return {"task_id": self.task_id, "stages": stages, "attempts": attempts}

    @staticmethod
    def _planner_id(sol: Solution) -> str:
        req = getattr(sol, "plan_request", None)
        planner = getattr(req, "planner", None)
        return str(planner) if planner is not None else ""

    # ------------------------------------------------------------------
    # Solution serialization (MTC Solution.toMsg / constructMotionPlan)
    # ------------------------------------------------------------------
    def solution_to_dict(self, sol: Solution) -> dict:
        """A solution as ROS-free dicts with MTC ``Solution`` field names.

        ``sub_solution`` mirrors the solution tree (one entry per node, with
        child solution ids); ``sub_trajectories`` carries one entry per
        MOTION leaf in chain order, each with the joint waypoints, the scene
        ops the executor applies before driving it, and the ``replay_goal``
        (the segment's planning goal) the execute server needs to replay the
        validated plan from the curobo_server trajectory cache instead of
        re-solving. ``start_scene`` is the task's base-scene objects.
        """
        sub_solutions = []

        def walk(s: Solution) -> None:
            sub_solutions.append({
                "info": self._solution_info_dict(s),
                "sub_solution_id": [
                    self.register_solution(c) for c in (s.children or [])],
            })
            for ch in (s.children or []):
                walk(ch)

        walk(sol)
        return {
            "task_id": self.task_id,
            "start_scene": [
                _object_spec_to_dict(spec)
                for spec in self.base_scene.objects_added.values()
            ],
            "sub_solution": sub_solutions,
            "sub_trajectories": self._chain_trajectory_dicts(sol),
        }

    def _chain_trajectory_dicts(self, sol: Solution) -> list:
        """One entry per motion leaf, in chain order, with scene ops folded
        in chain position.

        Mutation leaves (scene-only stages) carry no motion, so their ops
        flush into the nearest motion segment: preceding mutations apply
        BEFORE the segment drives (``before``), trailing mutations (e.g. a
        release task's detach) apply AFTER it (``after``). A scene-only
        chain yields one entry with no replay goal that only applies ops.
        """
        entries = []
        pending = []
        for leaf in self.flatten_leaves(sol):
            ops = list(getattr(leaf, "scene_ops", None) or [])
            if getattr(leaf, "plan_request", None) is None:
                pending.extend(ops)
                continue
            entry = self._leaf_trajectory_dict(leaf)
            entry["scene_diff"] = {
                "before": _scene_ops_to_dict(pending + ops),
                "after": _scene_ops_to_dict([]),
            }
            pending = []
            entries.append(entry)
        if pending:
            if entries:
                entries[-1]["scene_diff"]["after"] = _scene_ops_to_dict(
                    pending)
            else:
                entries.append({
                    # No motion leaf: ids stay 0 ("no solution"), never -1 —
                    # uint32 fields reject negatives in C serialization.
                    "info": {"id": 0, "cost": 0.0, "comment": "",
                             "stage_id": 0, "planner_id": ""},
                    "execution_info": {"controller_names": []},
                    "trajectory": {"joint_names": [], "points": []},
                    "scene_diff": {"before": _scene_ops_to_dict(pending),
                                   "after": _scene_ops_to_dict([])},
                    "replay_goal": None,
                })
        return entries

    def _solution_info_dict(self, sol: Solution) -> dict:
        """MTC SolutionInfo fields for one solution.

        ``id`` is the GLOBAL solution id (``register_solution``), exactly the
        number ``StageStatistics.solved``/``failed`` and ``GetSolution`` use —
        never the per-stage ``solution_id``, which restarts at 0 in every
        stage and would make a client join the wrong solution to a stage.
        """
        stage = getattr(sol, "stage", None)
        return {
            "id": self.register_solution(sol),
            "cost": float(getattr(sol, "cost", float("inf"))),
            "comment": str(getattr(sol, "comment", "") or ""),
            "stage_id": int(getattr(stage, "stage_id", 0) or 0),
            "planner_id": self._planner_id(sol),
        }

    def _leaf_trajectory_dict(self, leaf: Solution) -> dict:
        traj = list(getattr(leaf, "trajectory", None) or [])
        names = list(getattr(traj[0], "name", []) or []) if traj else []
        points, velocities = [], []
        for wp in traj:
            points.append(
                [float(v) for v in (getattr(wp, "position", []) or [])])
            vel = getattr(wp, "velocity", None) or []
            velocities.append([float(v) for v in vel])
        return {
            "info": self._solution_info_dict(leaf),
            "execution_info": {"controller_names": []},
            # MTC parity: the sub-trajectory carries the waypoints themselves so
            # execute can replay them verbatim. Velocities ride along when the
            # solver produced them (the wire JointState already carries them);
            # the open-loop streamer tapers/clamps them, so an empty list is a
            # benign "derive from positions".
            "trajectory": {
                "joint_names": [str(n) for n in names],
                "points": points,
                "velocities": velocities,
            },
            # Filled by _chain_trajectory_dicts (chain-position folding).
            "scene_diff": {"before": _scene_ops_to_dict([]),
                           "after": _scene_ops_to_dict([])},
            "replay_goal": _plan_request_to_dict(
                getattr(leaf, "plan_request", None)),
        }

    def execute_dict(self, d: dict, progress_callback=None,
                     cancel_event=None) -> tuple:
        """Drive a deserialized solution (see ``solution_to_dict``).

        Rebuilds each motion segment's PlanRequest + scene ops and drives
        them exactly like :meth:`execute`: MTC ``Task::execute`` parity, each
        segment is replayed from the validated trajectory the solution carries
        (never re-solved). Returns (results, failed_stage_name);
        ``failed_stage_name`` is "" on success and names the leaf stage (MTC
        reports per-sub-trajectory progress through the action feedback
        instead).
        """
        self._applied_ops = []
        # MTC maps sub-trajectory ids back to stage names through the task;
        # same here: names come from this executor's own stage tree.
        names = {stg.stage_id: stg.name
                 for stg in self.root.subtree_stages()}

        def _progress(index, total, info):
            if progress_callback is not None:
                stage_name = names.get(int((info or {}).get("stage_id", -1)
                                           or -1), f"segment_{index}")
                progress_callback(stage_name)

        results, failed_index, cancelled = drive_solution_dict(
            self.robot, d, self._applied_ops,
            progress_callback=_progress, cancel_event=cancel_event)
        if cancelled:
            return results, "<cancelled>"
        if failed_index is None:
            return results, ""
        segments = list((d or {}).get("sub_trajectories", None) or [])
        info = (segments[failed_index].get("info", {})
                if 0 <= failed_index < len(segments) else {})
        return results, names.get(int(info.get("stage_id", -1) or -1), "")


def drive_solution_dict(robot, d, applied_ops=None,
                        progress_callback=None, cancel_event=None) -> tuple:
    """Drive a deserialized solution without an executor tree.

    The execute action server (another process, no stage tree) drives a
    received Solution this way: scene ops apply through ``robot`` in chain
    order and each motion segment replays its own VALIDATED waypoints via
    ``robot.execute(req, trajectory=...)`` — MTC ``Task::execute`` parity, the
    server streams the segment's stored trajectory verbatim (``replay``), so
    there is no cache lookup, no re-solve, and no way for a multi-solution pose
    goal to jump IK branch mid-chain. ``req`` is still sent for the planner
    switch and its ``allowed_collisions``.

    ``progress_callback`` fires as ``(index, total, info)`` per driven
    motion segment (MTC's sub_id/sub_no feedback). ``applied_ops`` collects
    ``(kind, payload)`` dedup keys across calls (pass the server's
    per-goal list). Returns ``(results, failed_index, cancelled)``;
    ``failed_index`` is None on success.
    """
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
        req = _plan_request_from_dict(seg.get("replay_goal", {}) or {})
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
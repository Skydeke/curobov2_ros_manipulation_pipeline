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

__all__ = ["TaskExecutor", "drive_solution_dict", "EXECUTE_CONTINUITY_TOLERANCE"]

#: The start-agreement number the whole pipeline shares: 0.01 rad. Kept here
#: as the reference value because it is ALSO what the controller must enforce
#: before it accepts a replayed trajectory (``allowed_start_tolerance`` in
#: iki_kortex_moveit_config/config/moveit_controllers.yaml). The cuRobo
#: executor no longer compares endpoints itself — MTC ``Task::execute`` replays
#: the stored ``RobotTrajectory`` verbatim and lets the controller's start
#: tolerance reject a replay whose first point is not where the arm is. This
#: constant documents that shared number.

from curobo_task_constructor.executor_io import *
from curobo_task_constructor.executor_serialize import _ExecutorSerializationMixin

class TaskExecutor(_ExecutorSerializationMixin):
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
        """Run the compute loop until no stage can make progress. Returns True when at least one full root solution was found. ``max_iterations`` (0 = unlimited) guards against pathological graphs."""
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
        than the one that was planned, collision-checked and displayed. Returns the list of drive results."""
        self._applied_ops = []
        results = []
        for leaf in self.flatten_leaves(sol):
            if cancel_event is not None and cancel_event.is_set():
                break
            for kind, payload in leaf.scene_ops or []:
                self._apply_op(kind, payload)
            if getattr(leaf, "gripper_command", None):
                if progress_callback is not None:
                    progress_callback(getattr(leaf.stage, "name", ""))
                result = self.robot.gripper_command(**leaf.gripper_command)
                results.append(result)
                if not result.success:
                    break
                continue
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
        """Per-stage StageStatistics + per-attempt SolutionInfo rollups. Shapes mirror ``curobo_task_constructor_interfaces``:
        stages: [{stage_id, stage_name, stage_type, attempt_count,
                  success_count, last_cost, total_compute_time}]
        attempts: [{stage_id, stage_name, solution_id, cost, success,
                    comment, planner_id}]

        ``last_cost`` is the lowest cost among the stage's solutions (not the
        last emitted — a multi-candidate goalset emits one solution per
        candidate, and the last one is not necessarily the cheapest). ``total_compute_time`` is in seconds (wall-clock, accumulated across
        every ``run_compute`` call on the stage)."""
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

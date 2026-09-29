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
                  combinatorially expensive ones). Because every stage
                  CONSUMES its pulls on compute and pushes are keyed-deduped,
                  the loop ``while any(s.can_compute())`` terminates on its
                  own; ``max_iterations`` is only a belt-and-braces guard.
4. ``rank()``   — full root solutions ranked by accumulated cost.
5. ``execute()``— apply the winning solution's scene deltas in chain order,
                  then re-solve + drive each motion segment (SendTrajectory).

Introspection (Sec. 7): ``describe()`` publishes the built TaskDescription and
``statistics()`` roll up per-stage StageStatistics + per-attempt SolutionInfo
records the action server publishes on the introspection topics.
"""

from __future__ import annotations

from typing import Optional

from curobo_task_constructor.core.container import GENERATE_INTERFACE
from curobo_task_constructor.core.robot import ObjectSpec, PlanResult, RobotInterface
from curobo_task_constructor.core.stage import InitStageError, Solution
from curobo_task_constructor.core.state import SceneDiff
from curobo_task_constructor.graph.builder import build_tree
from curobo_task_constructor.graph.spec import StageSpec

__all__ = ["TaskExecutor", "EXECUTE_CONTINUITY_TOLERANCE"]

#: Largest per-joint difference (rad) tolerated between a segment's PLANNED
#: endpoint and the endpoint that was actually driven before the chain is
#: declared broken. See ``TaskExecutor._diverged_from_plan``.
#:
#: This is about chain CONTINUITY, not trajectory identity: a re-solve may
#: legitimately wobble through free space, but it has to still start and end
#: where the plan said, or the next segment's baked ``start_pose`` is wrong.
#: 0.05 rad is a little under 3 degrees - far above encoder/servo noise, far
#: below the fraction of a joint's range that choosing a different IK branch
#: moves (the failure this was added for was ~1.7 rad on one joint).
EXECUTE_CONTINUITY_TOLERANCE = 0.05


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
        # SolutionInfo / StageStatistics to a stage across solves.
        for idx, stage in enumerate(self.root.subtree_stages()):
            stage.stage_id = idx
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
        diff when the robot interface has nothing to report)."""
        obj_names = getattr(self.robot, "get_object_names", None)
        if obj_names is not None:
            scene = SceneDiff()
            for name in obj_names() or []:
                pose = self.robot.get_object_pose(name)
                if pose is not None:
                    scene.objects_added[name] = ObjectSpec(
                        name=name, shape="mesh", pose=pose)
            return scene
        return SceneDiff()

    # ------------------------------------------------------------------
    # Plan
    # ------------------------------------------------------------------
    def plan(self, max_iterations: int = 0) -> bool:
        """Run the compute loop until no stage can make progress.

        Returns True when at least one full root solution was found.
        ``max_iterations`` (0 = unlimited) guards against pathological graphs.

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
        while any(s.can_compute() for s in self.root.subtree_stages()):
            self.root.run_compute()
            iterations += 1
            if self.root.solutions:
                # First complete root solution: stop re-connecting / re-solving
                # the trailing chain (see docstring). Return the first full
                # solution — best() ranks it later.
                break
            if max_iterations and iterations >= max_iterations:
                break
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

    def execute(self, sol: Solution) -> list:
        """Play back one solution: materialize each segment's scene delta on
        the curobo server in chain order, then re-solve + drive each motion
        segment (SendTrajectory). Returns the list of drive results.

        The chain STOPS at the first segment that does not land where the plan
        said it would - either because the drive failed, or because the server
        did not use the cached plan and re-solved into a different trajectory
        (see ``_diverged_from_plan``). The remaining segments are not driven,
        because every one of them carries a ``start_pose`` baked from the
        PLANNED end state of its predecessor, so a chain that has already
        drifted is a chain whose remaining requests are anchored to poses the
        arm was never asked to be at. Driving them anyway is how one bad
        segment becomes a self-collision three segments later.

        The returned list is truncated at the failure, so the caller's
        zip-with-motion-leaves pairs each result with the leaf that produced
        it and reports the right stage name.
        """
        self._applied_ops = []
        results = []
        for leaf in self.flatten_leaves(sol):
            for kind, payload in leaf.scene_ops or []:
                self._apply_op(kind, payload)
            if leaf.plan_request is None:
                continue
            result = self.robot.execute(leaf.plan_request)
            if not result.success:
                results.append(result)
                break
            divergence = self._diverged_from_plan(leaf, result)
            if divergence is not None:
                results.append(PlanResult(
                    False, divergence, trajectory=result.trajectory))
                break
            results.append(result)
        return results

    @staticmethod
    def _diverged_from_plan(leaf: Solution, result: PlanResult):
        """Did the executed trajectory differ from the planned one?

        Only the two ENDPOINTS are compared, and that is deliberate. A
        re-solve from the same start to the same goal is free to take a
        different path through free space - trajopt is stochastic - and a
        different path is not a defect. What the chain depends on is only
        where each segment BEGINS and ENDS: the next segment's request
        ``start_pose`` is the planned end of this one. If either endpoint
        moved, the rest of the chain is anchored to poses that no longer
        describe where the arm is, and continuing is what turns a small
        mismatch into a runaway.

        Returns an error message, or None when the endpoints agree.

        The comparison is against ``leaf.trajectory``, the waypoints this
        segment produced at PLAN time, which the Solution carries. That is the
        trajectory the caller was shown and that collision checking accepted -
        so a mismatch here is precisely the execution/display divergence, and
        it is detectable without any protocol change.
        """
        planned = leaf.trajectory
        driven = result.trajectory
        if not planned or not driven:
            # Nothing to compare: a mutation leaf, or a result that carried no
            # trajectory. Not evidence of divergence.
            return None
        for label, want, got in (("start", planned[0], driven[0]),
                                 ("end", planned[-1], driven[-1])):
            delta = _max_joint_delta(want, got)
            if delta > EXECUTE_CONTINUITY_TOLERANCE:
                name = getattr(leaf.stage, "name", "<unnamed>")
                return (f"cache miss: '{name}' was re-solved and its {label} "
                        f"state moved {delta:.3f} rad from the plan "
                        f"(> {EXECUTE_CONTINUITY_TOLERANCE} rad); the rest of "
                        f"the chain is anchored to the planned trajectory and "
                        f"was not executed")
        return None

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
        """
        stages = []
        attempts = []
        for stg in self.root.subtree_stages():
            stages.append({
                "stage_id": stg.stage_id,
                "stage_name": stg.name,
                "stage_type": stg.stage_type(),
                "attempt_count": stg.attempt_count,
                "success_count": len(stg.solutions),
                "last_cost": (stg.solutions[-1].cost
                              if stg.solutions else float("inf")),
                "total_compute_time": stg.compute_time,
            })
            for sol in stg.solutions:
                attempts.append({
                    "stage_id": stg.stage_id,
                    "stage_name": stg.name,
                    "solution_id": sol.solution_id,
                    "cost": sol.cost,
                    "success": True,
                    "comment": sol.comment,
                    "planner_id": self._planner_id(sol),
                })
            for fail in stg.failures:
                attempts.append({
                    "stage_id": stg.stage_id,
                    "stage_name": stg.name,
                    "solution_id": -1,
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
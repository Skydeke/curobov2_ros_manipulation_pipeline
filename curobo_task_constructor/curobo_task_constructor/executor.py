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
from curobo_task_constructor.core.robot import ObjectSpec, RobotInterface
from curobo_task_constructor.core.stage import InitStageError, Solution
from curobo_task_constructor.core.state import SceneDiff
from curobo_task_constructor.graph.builder import build_tree
from curobo_task_constructor.graph.spec import StageSpec

__all__ = ["TaskExecutor"]


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
        segment (SendTrajectory). Returns the list of drive results."""
        self._applied_ops = []
        results = []
        for leaf in self.flatten_leaves(sol):
            for kind, payload in leaf.scene_ops or []:
                self._apply_op(kind, payload)
            if leaf.plan_request is not None:
                results.append(
                    self.robot.execute(leaf.plan_request))
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
"""ComputeIK — wrapper that turns a pose-generator child's candidates into
joint configurations (MTC ``ComputeIK : WrapperBase``).

The wrapped stage is any generator emitting candidate poses in
``meta["target_pose"]`` (``GenerateGraspPose`` by convention). ComputeIK
presents the *same* interface as its child — the pick&place graph relies on
this: the wrapper sits *after* a backward ``approach`` stage and *before* a
forward ``ModifyPlanningScene``, and because it writes both interfaces
(GENERATE), both neighbours connect cleanly.
"""

from __future__ import annotations

from curobo_task_constructor.core.geom import Pose3
from curobo_task_constructor.core.registry import register_stage
from curobo_task_constructor.core.stage import (
    InterfaceType,
    Solution,
    Stage,
    InitStageError,
)


@register_stage("compute_ik")
class ComputeIK(Stage):
    interface = InterfaceType.GENERATOR

    def __init__(self, name=None, params=None):
        super().__init__(name, params)
        self.children = []
        self._child_solutions = []
        self._processed = 0

    # -- construction --------------------------------------------------
    def set_child(self, child: Stage) -> None:
        """Install the pose-generator child (the builder calls this when the
        StageSpec carries one child)."""
        if len(self.children) >= 1:
            raise InitStageError(self.name, "compute_ik wraps exactly one child")
        child.parent = self
        self.children.append(child)

    # -- initialization ------------------------------------------------
    def init(self, base_scene, robot) -> None:
        super().init(base_scene, robot)
        if not self.children:
            raise InitStageError(self.name,
                                 "compute_ik requires one generator child")
        child = self.children[0]
        child.init(base_scene, robot)
        upstream = child.on_solution

        def _collect(sol, _child=child, _up=upstream):
            if _up is not None:
                _up(sol)
            self._child_solutions.append(sol)

        child.on_solution = _collect
        self._child = child

    def resolve(self, expected_start, expected_end=None) -> None:
        """Wrapper semantics: forward the container's expectation to the
        wrapped child and present the child's interface."""
        self._child.resolve(expected_start, expected_end)
        self._start_flag, self._end_flag = self._child.required_flags()
        self._resolved = True

    def interface_flags(self):
        if getattr(self, "_resolved", False):
            return self._start_flag, self._end_flag
        return self.interface.start, self.interface.end

    def required_flags(self):
        return self.interface_flags()

    # -- computation ---------------------------------------------------
    def can_compute(self) -> bool:
        return self._child.can_compute() or \
            self._processed < len(self._child_solutions)

    def compute(self) -> None:
        if self._child.can_compute():
            self._child.run_compute()
        max_solutions = int(self.params.get("max_solutions", 0) or 0)
        while self._processed < len(self._child_solutions):
            if max_solutions and len(self.solutions) >= max_solutions:
                break
            sol = self._child_solutions[self._processed]
            self._processed += 1
            src = sol.start if sol.start is not None else sol.end
            meta = getattr(src, "meta", {}) or {}
            pose = meta.get("target_pose")
            if pose is None:
                continue
            if not isinstance(pose, Pose3):
                pose = Pose3.from_any(pose)
            joint = self.robot.ik(pose, seed=src.joint_state)
            if joint is None:
                self._fail(src, None, "ik failed for candidate pose")
                continue
            state = src.clone(joint_state=joint)
            state.meta = dict(meta)  # target pose travels with the state
            self._emit(Solution(state, state, trajectory=None, cost=0.0,
                                comment=f"ik angle={meta.get('angle', '?')}",
                                response=joint))
            self._push_interface(self._starts, state)  # GENERATE: write both
            self._push_interface(self._ends, state)

    def _push_interface(self, target: list, state) -> None:
        for existing in target:
            if existing.key() == state.key():
                return
        target.append(state)

    def subtree_stages(self) -> list:
        out = [self]
        for child in self.children:
            out.extend(child.subtree_stages())
        return out

    def reset(self) -> None:
        super().reset()
        self._child_solutions = []
        self._processed = 0
        self._resolved = False
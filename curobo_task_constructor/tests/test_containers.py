"""Container tests: Alternatives (incl. plan_batch), Fallbacks,
IndependentComponents — the parallel-container semantics every pick&place
graph relies on.

Validation targets migration step 3 of the plan (the combinatorially
interesting container kinds) against the MockCuroboServer.
"""

from __future__ import annotations

import pytest

import curobo_task_constructor.stages  # noqa: F401  (register builtins)
from curobo_task_constructor.core.registry import register_stage
from curobo_task_constructor.core.robot import GoalsetSpec
from curobo_task_constructor.core.stage import TrajectoryStage
from curobo_task_constructor.executor import TaskExecutor
from curobo_task_constructor.graph.spec import StageSpec
from curobo_task_constructor.stages._util import full_request
from tests.mock_curobo import JOINT_NAMES, MockCuroboServer


def _home_robot(**kw):
    named = {"home": {n: 0.0 for n in JOINT_NAMES}}
    named["home2"] = dict(named["home"])
    named["home2"]["joint_1"] = 0.25
    return MockCuroboServer(named=named, **kw)


def _root_spec(container_type: str, children: list, name="root") -> StageSpec:
    """A serial root driving one parallel container from a current_state."""
    return StageSpec(stage_type="", name=name, container_type="serial",
                     children=[
                         StageSpec(stage_type="current_state", name="cur"),
                         StageSpec(stage_type="", name="par",
                                   container_type=container_type,
                                   children=children),
                     ])


def _move_spec(name, goal_yaml: str) -> StageSpec:
    return StageSpec(stage_type="move_to", name=name,
                     params_yaml=f"goal:\n  {goal_yaml}\n")


# ----------------------------------------------------------------------
# Alternatives
# ----------------------------------------------------------------------
def test_alternatives_all_children_solutions_lift():
    robot = _home_robot()
    spec = _root_spec("alternatives", [
        _move_spec("via_home_joints", "joints: [0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.5]"),
        _move_spec("via_home2_name", "name: home2"),
    ])
    ex = TaskExecutor(spec, robot)
    assert ex.init()
    assert ex.plan()
    assert len(ex.rank()) >= 2  # one root solution per alternative

    par = ex.root.children[1]
    assert par.stage_type() == "alternatives"
    assert len(par.solutions) >= 2


def test_alternatives_same_interface_required():
    """A generator + propagator mix under one Alternatives is invalid."""
    robot = _home_robot()
    spec = _root_spec("alternatives", [
        StageSpec(stage_type="current_state", name="cur_b"),
        _move_spec("via_home_joints", "joints: [0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.5]"),
    ])
    ex = TaskExecutor(spec, robot)
    assert not ex.init()
    assert "same interface" in ex.describe()["comment"]


# ----------------------------------------------------------------------
# Alternatives batching (TrajectoryStage children -> one plan_batch call)
# ----------------------------------------------------------------------
@register_stage("test_batch_motion")
class BatchMotion(TrajectoryStage):
    """A minimal whole-task TrajectoryStage for exercising the batching path."""

    def build_plan_request(self, start):
        goal = self.params.get("goal") or {}
        cfg = self.robot.get_named_joint_config(goal["name"])
        goalset = GoalsetSpec(
            target_joint_positions=list(cfg.positions),
            allowed_collisions=start.scene.all_allowed_links()
            if start.scene else [],
        )
        return full_request(self.robot, start.joint_state, [goalset], self.params)

    def make_end_state(self, start, result, raw=None):
        return start.clone(joint_state=result.last_state)


def test_alternatives_batch_collects_one_plan_batch_call():
    robot = _home_robot()
    spec = _root_spec("alternatives", [
        StageSpec(stage_type="test_batch_motion", name="a",
                  params_yaml="goal:\n  name: home\n"),
        StageSpec(stage_type="test_batch_motion", name="b",
                  params_yaml="goal:\n  name: home2\n"),
    ])
    ex = TaskExecutor(spec, robot)
    assert ex.init()
    assert ex.plan()
    # exactly one batched call, covering both children's requests
    assert robot.batch_calls == 1
    assert len(ex.rank()) >= 2
    par = ex.root.children[1]
    assert len(par.solutions) >= 2


def test_alternatives_no_batch_for_mixed_children():
    """Batching is an optimization that must not change semantics."""
    robot = _home_robot()
    spec = _root_spec("alternatives", [
        StageSpec(stage_type="test_batch_motion", name="a",
                  params_yaml="goal:\n  name: home\n"),
        _move_spec("b", "name: home2"),
    ])
    ex = TaskExecutor(spec, robot)
    assert ex.init()
    assert ex.plan()
    assert robot.batch_calls == 0  # not all children are TrajectoryStage
    assert len(ex.rank()) >= 2


# ----------------------------------------------------------------------
# Fallbacks
# ----------------------------------------------------------------------
def test_fallbacks_first_solution_wins():
    robot = _home_robot(fail_ik=True)  # pose goals fail, joint goals succeed
    spec = _root_spec("fallbacks", [
        _move_spec("bad", "pose:\n  x: 9.0\n  y: 9.0\n  z: 9.0\n"),
        _move_spec("good", "name: home"),
    ])
    ex = TaskExecutor(spec, robot)
    assert ex.init()
    assert ex.plan()
    assert len(ex.rank()) == 1  # first solution wins
    sol = ex.best()
    par_sol = sol.children[1]  # [current_state, fallback-container] chain
    assert par_sol.children[0].stage.name == "good"

    fb = ex.root.children[1]
    assert fb.stage_type() == "fallbacks"
    # the failed attempt is still visible for debugging (Sec. 7 behavior)
    assert fb.children[0].failures
    assert fb.children[0].solutions == []


def test_fallbacks_all_fail_no_solution():
    robot = _home_robot(fail_ik=True)
    spec = _root_spec("fallbacks", [
        _move_spec("bad1", "pose:\n  x: 9.0\n  y: 9.0\n  z: 9.0\n"),
        _move_spec("bad2", "pose:\n  x: 8.0\n  y: 8.0\n  z: 8.0\n"),
    ])
    ex = TaskExecutor(spec, robot)
    assert ex.init()
    assert not ex.plan()
    assert all(c.failures for c in ex.root.children[1].children)


# ----------------------------------------------------------------------
# IndependentComponents
# ----------------------------------------------------------------------
def test_independent_components_lift_every_child():
    robot = _home_robot()
    spec = _root_spec("independent", [
        _move_spec("left", "joints: [0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.5]"),
        _move_spec("right", "name: home"),
    ])
    ex = TaskExecutor(spec, robot)
    assert ex.init()
    assert ex.plan()
    par = ex.root.children[1]
    assert par.stage_type() == "independent"
    assert len(par.solutions) >= 2


# ----------------------------------------------------------------------
# nested containers share the same InterfaceState objects across boundaries
# ----------------------------------------------------------------------
def test_nested_containers_states_thread_by_identity():
    """Alternatives inside a serial container: each alternative's start state
    must be the same object the preceding stage wrote (identity threading)."""
    robot = _home_robot()
    spec = StageSpec(stage_type="", name="root", container_type="serial",
                     children=[
                         StageSpec(stage_type="current_state", name="cur"),
                         StageSpec(stage_type="", name="choose",
                                   container_type="alternatives",
                                   children=[
                                       StageSpec(
                                           stage_type="", name="branch_a",
                                           container_type="serial",
                                           children=[
                                               _move_spec("a1", "name: home"),
                                               _move_spec("a2", "name: home"),
                                           ]),
                                       StageSpec(
                                           stage_type="", name="branch_b",
                                           container_type="serial",
                                           children=[
                                               _move_spec("b1",
                                                          "joints: [0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.4]"),
                                           ]),
                                   ]),
                     ])
    ex = TaskExecutor(spec, robot)
    assert ex.init()
    assert ex.plan()
    for sol in ex.rank():
        chain = ex.flatten_leaves(sol)
        for prev, nxt in zip(chain, chain[1:]):
            # each segment's start is the previous segment's end object
            assert prev.end is nxt.start
            assert id(prev.end) == id(nxt.start)
"""kortex_mtc_test.py as a StageSpec tree — validation target #1.

The legacy MoveIt-pipeline test asked for this stage sequence:

    current_state
    -> move_to (pose goal, base_link)
    -> move_to (named "grasp_home")
    -> move_to (cartesian path variant -> pose goal)
    -> move_to (with IK variant -> pose goal)
    -> move_to (named "open", gripper)
    -> move_to (named "close", gripper)

Reproduced here 1:1 under the declarative format, solved and executed against
the MockCuroboServer — same ordering, same goals.
"""

from __future__ import annotations

import curobo_task_constructor.stages  # noqa: F401  (register builtins)
from curobo_task_constructor.executor import TaskExecutor
from curobo_task_constructor.graph.spec import StageSpec
from tests.mock_curobo import JOINT_NAMES, MockCuroboServer

CLOSE = 0.7


def _named_configs() -> dict:
    zeros = {n: 0.0 for n in JOINT_NAMES}
    cur_arm = {n: 0.0 for n in JOINT_NAMES[:-1]}
    return {
        "home": {**zeros},
        "grasp_home": {**zeros},
        "retract": {**zeros},
        "vertical": {**zeros},
        "perception_pose": {**zeros},
        "open": {**cur_arm, "finger_joint": 0.0},
        "close": {**cur_arm, "finger_joint": CLOSE},
    }


def _pose_move(name: str, x, y, z) -> StageSpec:
    return StageSpec(
        stage_type="move_to", name=name,
        params_yaml=(f"goal:\n  pose:\n    x: {x}\n    y: {y}\n"
                     f"    z: {z}\n    qw: 1.0\n"))


def _named_move(name: str, goal: str) -> StageSpec:
    return StageSpec(stage_type="move_to", name=name,
                     params_yaml=f"goal:\n  name: {goal}\n")


def _kortex_spec() -> StageSpec:
    """The 7-stage kortex_mtc_test sequence as a serial container."""
    return StageSpec(stage_type="", name="kortex task",
                     container_type="serial",
                     children=[
                         StageSpec(stage_type="current_state",
                                   name="current state"),
                         _pose_move("to pre-pose", 0.0, 0.4, 0.0),
                         _named_move("to grasp home", "grasp_home"),
                         _pose_move("cartesian shift", 0.0, 0.32, 0.0),
                         _pose_move("ik approach", 0.0, -0.4, 0.0),
                         _named_move("gripper open", "open"),
                         _named_move("gripper close", "close"),
                     ])


def _executor():
    robot = MockCuroboServer(named=_named_configs())
    return robot, TaskExecutor(_kortex_spec(), robot, task_id="kortex")


def test_kortex_sequence_solves():
    robot, ex = _executor()
    assert ex.init(), ex.describe()["comment"]
    assert ex.plan()
    sol = ex.best()
    assert sol is not None
    assert sol.cost > 0.0


def test_kortex_sequence_ends_gripper_closed():
    robot, ex = _executor()
    assert ex.init() and ex.plan()
    sol = ex.best()
    finger = dict(zip(JOINT_NAMES, sol.end.joint_state.position))[
        "finger_joint"]
    assert abs(finger - CLOSE) < 1e-9


def test_kortex_sequence_leaf_order_matches_stage_order():
    robot, ex = _executor()
    assert ex.init() and ex.plan()
    sol = ex.best()
    chain = ex.flatten_leaves(sol)
    names = [l.stage.name for l in chain]
    assert names == ["current state", "to pre-pose", "to grasp home",
                     "cartesian shift", "ik approach", "gripper open",
                     "gripper close"]
    # one driven motion per move stage (current state has no request)
    assert sum(1 for l in chain if l.plan_request is not None) == 6


def test_kortex_sequence_executes_all_motions_in_order():
    robot, ex = _executor()
    assert ex.init() and ex.plan()
    results = ex.execute(ex.best())
    assert len(results) == 6
    assert len(robot.executed) == 6  # one SendTrajectory drive per move
    # last executed request ends at the close config
    last_req = robot.executed[-1]
    assert last_req.goalsets[0].target_joint_positions[
        JOINT_NAMES.index("finger_joint")] == CLOSE


def test_kortex_sequence_statistics():
    robot, ex = _executor()
    assert ex.init() and ex.plan()
    stats = ex.statistics()
    names = {s["stage_name"] for s in stats["stages"]}
    assert {"current state", "to pre-pose", "to grasp home",
            "cartesian shift", "ik approach", "gripper open",
            "gripper close"} <= names
    assert {a["success"] for a in stats["attempts"]} == {True}
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

The "cartesian path variant" leg is a ``cartesian_path`` stage rather than a
pose ``move_to``. That is the point of the leg: as a plain ``move_to`` it was
a free pose goal, so the planner picked whichever curved joint-space path it
liked and the tool swung round instead of tracking the line. As a
``cartesian_path`` the goalset carries whole-path axis holds and the leg is
solved as a straight line — see ``stages/cartesian_path.py``.
"""

from __future__ import annotations

import math

import pytest

import curobo_task_constructor.stages  # noqa: F401  (register builtins)
from curobo_task_constructor.executor import TaskExecutor
from curobo_task_constructor.graph.spec import StageSpec
from tests.mock_curobo import JOINT_NAMES, MockCuroboServer, fk_positions

CLOSE = 0.7

#: Mid-shell arm configuration (tool at (0.40, 0, 0.10) in the mock's analytic
#: forward kinematics) so the straight-line leg has room to travel in z without
#: running into the 0.30/0.50 m reach limits.
MID_SHELL = {"joint_1": 0.0, "joint_2": math.pi / 2, "joint_3": -math.pi / 2}


def _named_configs() -> dict:
    zeros = {n: 0.0 for n in JOINT_NAMES}
    cur_arm = {n: 0.0 for n in JOINT_NAMES[:-1]}
    return {
        "home": {**zeros},
        "grasp_home": {**MID_SHELL},
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


def _cartesian_shift(name: str, dz: float) -> StageSpec:
    """A straight-line leg: a relative goal (the MoveRelative port) so the hold
    is derived from FK(start) — orientation and x/y pinned, z free."""
    return StageSpec(
        stage_type="cartesian_path", name=name,
        params_yaml=(f"goal:\n  relative:\n    x: 0.0\n    y: 0.0\n"
                     f"    z: {dz}\n    frame: world\n"
                     f"planner: classic\n"))


def _kortex_spec() -> StageSpec:
    """The 7-stage kortex_mtc_test sequence as a serial container."""
    return StageSpec(stage_type="", name="kortex task",
                     container_type="serial",
                     children=[
                         StageSpec(stage_type="current_state",
                                   name="current state"),
                         _pose_move("to pre-pose", 0.0, 0.4, 0.0),
                         _named_move("to grasp home", "grasp_home"),
                         _cartesian_shift("cartesian shift", -0.08),
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


def test_kortex_cartesian_shift_is_a_straight_line():
    """The point of the leg: the tool tracks the line, it does not swing.

    As a pose ``move_to`` the solver is free to bow anywhere; the
    ``cartesian_path`` hold pins x/y and the tool angle for the whole path, so
    every intermediate waypoint's x and y must equal the endpoints' and z must
    move monotonically.
    """
    robot, ex = _executor()
    assert ex.init() and ex.plan()
    chain = ex.flatten_leaves(ex.best())
    leg = next(l for l in chain if l.stage.name == "cartesian shift")
    pts = [fk_positions(dict(zip(JOINT_NAMES, w.position)))
           for w in leg.trajectory]

    assert len(pts) >= 3
    assert {round(p[0], 6) for p in pts} == {round(pts[0][0], 6)}
    assert {round(p[1], 6) for p in pts} == {round(pts[0][1], 6)}
    assert all(b[2] < a[2] for a, b in zip(pts, pts[1:]))  # monotone drop
    assert pts[0][2] - pts[-1][2] == pytest.approx(0.08, abs=1e-6)


def test_kortex_cartesian_shift_records_its_axis_hold():
    """The goalset must actually carry the int8[6] hold, or the leg is a no-op."""
    robot, ex = _executor()
    assert ex.init() and ex.plan()
    chain = ex.flatten_leaves(ex.best())
    leg = next(l for l in chain if l.stage.name == "cartesian shift")
    assert leg.plan_request.goalsets[0].trajectory_constraints == \
        [1, 1, 1, 1, 1, 0]


def test_kortex_sequence_statistics():
    robot, ex = _executor()
    assert ex.init() and ex.plan()
    stats = ex.statistics()
    names = {s["stage_name"] for s in stats["stages"]}
    assert {"current state", "to pre-pose", "to grasp home",
            "cartesian shift", "ik approach", "gripper open",
            "gripper close"} <= names
    assert {a["success"] for a in stats["attempts"]} == {True}
"""MultiToolCartesianPath: one multi-tool goalset with per-tool holds.

Covers the core stage, the MTC-shaped wrapper and the forward-only failure.
"""

from __future__ import annotations

import yaml

import curobo_task_constructor.stages  # noqa: F401  (register builtins)
from curobo_task_constructor.core.state import InterfaceState
from curobo_task_constructor.executor import TaskExecutor
from curobo_task_constructor.graph.spec import StageSpec
from curobo_task_constructor.stages import MultiToolCartesianPath
from tests.mock_curobo import JOINT_NAMES, MockCuroboServer, fk_positions

IDENTITY_Q = (0.0, 0.0, 0.0, 1.0)
HOLD = [1, 1, 1, 0, 1, 0]
ZERO = [0, 0, 0, 0, 0, 0]


def _pose(x, y, z):
    return {"x": x, "y": y, "z": z,
            "qx": IDENTITY_Q[0], "qy": IDENTITY_Q[1],
            "qz": IDENTITY_Q[2], "qw": IDENTITY_Q[3]}


THREE_TOOLS = [
    {"frame": "left_tool", "pose": _pose(0.30, 0.0, 0.20)},
    {"frame": "right_tool", "pose": _pose(0.30, 0.0, 0.10)},
    {"frame": "third_tool", "pose": _pose(0.30, 0.0, 0.15)},
]


def _params(hold=None):
    params = {"goal": {"tools": [
        {"frame": "left_tool", "pose": _pose(0.30, 0.0, 0.20)},
        {"frame": "right_tool", "pose": _pose(0.30, 0.0, 0.10)},
    ]}}
    if hold is not None:
        params["hold"] = list(hold)
    return params


def _plan(robot, params=None):
    root = StageSpec(
        stage_type="", name="root", container_type="serial",
        children=[
            StageSpec(stage_type="current_state", name="current"),
            StageSpec(stage_type="multi_tool_cartesian_path", name="descend",
                      params_yaml=yaml.safe_dump(params or _params())),
        ])
    ex = TaskExecutor(root, robot)
    assert ex.init(), ex.describe()["comment"]
    return ex


def _stage(ex, name="descend"):
    return next(s for s in ex.root.subtree_stages() if s.name == name)


# -----------------------------------------------------------------------------
# core stage
# -----------------------------------------------------------------------------

def test_stage_sends_one_multi_tool_goalset_with_per_tool_holds():
    robot = MockCuroboServer()
    ex = _plan(robot, _params(hold=HOLD))
    assert ex.plan()
    sol = _stage(ex).solutions[0]
    req = sol.plan_request
    gs = req.goalsets[0]

    assert gs.poses == []
    assert gs.tool_frames == ["left_tool", "right_tool"]
    assert [p.position.x for p in gs.tool_poses] == [0.30, 0.30]
    assert [p.position.z for p in gs.tool_poses] == [0.20, 0.10]
    # Legacy setHold -> first tool's group, zeros for every other tool.
    assert gs.tool_trajectory_constraints == HOLD + ZERO
    assert len(gs.tool_trajectory_constraints) == 6 * 2
    # The per-tool field is authoritative; the legacy field stays empty.
    assert gs.trajectory_constraints == []
    assert req.planner == 3  # CARTESIAN, not CLASSIC
    assert "left_tool" in sol.comment
    assert f"holds={[HOLD, ZERO]}" in sol.comment
    assert ex.best() is not None


def test_stage_derives_a_distinct_hold_for_every_tool():
    robot = MockCuroboServer()
    x, y, z = fk_positions(
        dict(zip(JOINT_NAMES, robot.get_current_joint_state().position)))
    params = {"goal": {"tools": [
        {"frame": "left_tool", "pose": _pose(x, y, z - 0.05)},
        {"frame": "right_tool", "pose": _pose(x, y + 0.10, z - 0.05)},
    ]}}
    ex = _plan(robot, params)
    assert ex.plan()

    gs = _stage(ex).solutions[0].plan_request.goalsets[0]
    # First tool: x/y and the orientation agree, z travels.
    # Second tool: y and z travel, so its hold differs from the first.
    left = [1, 1, 1, 1, 1, 0]
    right = [1, 1, 1, 1, 0, 0]
    assert gs.tool_trajectory_constraints == left + right
    assert gs.tool_trajectory_constraints[:6] != \
        gs.tool_trajectory_constraints[6:]
    assert gs.trajectory_constraints == []


def test_stage_accepts_per_tool_holds_as_groups_or_flat():
    groups = [HOLD, [0, 0, 0, 1, 0, 1], [1, 0, 0, 0, 1, 0]]
    flat = [c for group in groups for c in group]
    for holds in (groups, flat):
        robot = MockCuroboServer()
        ex = _plan(robot, {"goal": {"tools": THREE_TOOLS}, "holds": holds})
        assert ex.plan()

        gs = _stage(ex).solutions[0].plan_request.goalsets[0]
        assert gs.tool_trajectory_constraints == flat
        assert len(gs.tool_trajectory_constraints) == 6 * 3
        assert gs.trajectory_constraints == []


def test_stage_accepts_three_tools_in_one_goalset():
    robot = MockCuroboServer()
    ex = _plan(robot, {"goal": {"tools": THREE_TOOLS}, "hold": HOLD})
    assert ex.plan()

    req = _stage(ex).solutions[0].plan_request
    assert len(req.goalsets) == 1
    gs = req.goalsets[0]
    assert gs.poses == []
    assert gs.tool_frames == ["left_tool", "right_tool", "third_tool"]
    assert len(gs.tool_poses) == 3
    assert [p.position.z for p in gs.tool_poses] == [0.20, 0.10, 0.15]
    assert gs.tool_trajectory_constraints == HOLD + ZERO + ZERO
    assert len(gs.tool_trajectory_constraints) == 6 * 3


def test_explicit_hold_must_have_six_ints():
    robot = MockCuroboServer()
    stage = MultiToolCartesianPath(name="descend", params=_params(hold=[1, 0]))
    stage.robot = robot
    stage.compute_forward(
        InterfaceState(joint_state=robot.get_current_joint_state()))

    assert stage.failures
    assert "hold must have 6 ints" in stage.failures[0].message
    assert robot.plan_calls == 0


def test_explicit_per_tool_holds_must_cover_every_tool():
    robot = MockCuroboServer()
    for holds in ([HOLD, HOLD], [1, 0, 0]):
        stage = MultiToolCartesianPath(
            name="descend",
            params={"goal": {"tools": THREE_TOOLS}, "holds": holds})
        stage.robot = robot
        stage.compute_forward(
            InterfaceState(joint_state=robot.get_current_joint_state()))

        assert stage.failures
        assert "holds must be 3 groups" in stage.failures[0].message
    assert robot.plan_calls == 0


def test_backward_fails_with_a_clear_message():
    robot = MockCuroboServer()
    stage = MultiToolCartesianPath(name="descend", params=_params(hold=HOLD))
    stage.robot = robot
    stage.compute_backward(
        InterfaceState(joint_state=robot.get_current_joint_state()))

    assert stage.failures
    assert "forward-only" in stage.failures[0].message
    assert robot.plan_calls == 0


# -----------------------------------------------------------------------------
# MTC-shaped wrapper
# -----------------------------------------------------------------------------

def test_mtc_wrapper_builds_tools_hold_and_cost():
    from curobo_task_constructor.mtc import core, stages

    move = stages.MultiToolCartesianPath(
        "descend", core.PipelinePlanner("classic", "", 1))
    move.setGoals([
        {"frame": "left_tool", "pose": {"x": 0.30, "y": 0.0, "z": 0.20,
                                        "qw": 1.0}},
        ("right_tool", {"x": 0.30, "y": 0.0, "z": 0.10, "qw": 1.0}),
    ])
    move.setHold(HOLD)
    move.setCost("path_length")

    params = yaml.safe_load(move.to_spec().params_yaml)
    assert params["goal"]["tools"] == [
        {"frame": "left_tool",
         "pose": {"x": 0.30, "y": 0.0, "z": 0.20,
                  "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}},
        {"frame": "right_tool",
         "pose": {"x": 0.30, "y": 0.0, "z": 0.10,
                  "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}},
    ]
    assert params["hold"] == HOLD
    assert "holds" not in params
    assert params["cost"] == "path_length"
    assert params["planning_attempts"] == 1


def test_mtc_wrapper_setHolds_builds_one_group_per_tool():
    from curobo_task_constructor.mtc import core, stages

    holds = [HOLD, [0, 0, 0, 1, 0, 1]]
    move = stages.MultiToolCartesianPath(
        "descend", core.PipelinePlanner("classic", "", 1))
    move.setGoals([
        {"frame": "left_tool", "pose": {"x": 0.30, "y": 0.0, "z": 0.20,
                                        "qw": 1.0}},
        {"frame": "right_tool", "pose": {"x": 0.30, "y": 0.0, "z": 0.10,
                                         "qw": 1.0}},
    ])
    move.setHolds(holds)

    params = yaml.safe_load(move.to_spec().params_yaml)
    assert params["holds"] == holds
    assert "hold" not in params


def test_mtc_wrapper_setHold_accepts_groups_and_delegates_to_setHolds():
    from curobo_task_constructor.mtc import core, stages

    holds = [HOLD, ZERO]
    move = stages.MultiToolCartesianPath(
        "descend", core.PipelinePlanner("classic", "", 1))
    move.setGoals([
        {"frame": "left_tool", "pose": {"x": 0.30, "y": 0.0, "z": 0.20,
                                        "qw": 1.0}},
        {"frame": "right_tool", "pose": {"x": 0.30, "y": 0.0, "z": 0.10,
                                         "qw": 1.0}},
    ])
    move.setHold(holds)

    params = yaml.safe_load(move.to_spec().params_yaml)
    assert params["holds"] == holds
    assert "hold" not in params


def test_mtc_wrapper_accepts_three_tools():
    from curobo_task_constructor.mtc import core, stages

    move = stages.MultiToolCartesianPath(
        "descend", core.PipelinePlanner("classic", "", 1))
    move.setGoals([
        {"frame": "left_tool", "pose": {"x": 0.30, "y": 0.0, "z": 0.20,
                                        "qw": 1.0}},
        {"frame": "right_tool", "pose": {"x": 0.30, "y": 0.0, "z": 0.10,
                                         "qw": 1.0}},
        {"frame": "third_tool", "pose": {"x": 0.30, "y": 0.0, "z": 0.15,
                                         "qw": 1.0}},
    ])
    move.setHold(HOLD)

    params = yaml.safe_load(move.to_spec().params_yaml)
    tools = params["goal"]["tools"]
    assert [tool["frame"] for tool in tools] == [
        "left_tool", "right_tool", "third_tool"]
    assert len(tools) == 3
    assert params["hold"] == HOLD


def test_mtc_wrapper_plans_through_the_cartesian_stage():
    from curobo_task_constructor.mtc import core, stages

    robot = MockCuroboServer()
    task = core.Task(robot)
    task.add(stages.CurrentState("current"))
    move = stages.MultiToolCartesianPath(
        "descend", core.PipelinePlanner("classic", "", 1))
    move.setGoals([
        {"frame": "left_tool", "pose": {"x": 0.30, "y": 0.0, "z": 0.20,
                                        "qw": 1.0}},
        {"frame": "right_tool", "pose": {"x": 0.30, "y": 0.0, "z": 0.10,
                                         "qw": 1.0}},
    ])
    move.setHold(HOLD)
    task.add(move)

    assert task.plan()
    leaf = next(s for s in task._executor.root.subtree_stages()
                if s.name == "descend")
    req = leaf.solutions[0].plan_request
    assert req.planner == 3
    assert req.goalsets[0].tool_trajectory_constraints == HOLD + ZERO
    assert req.goalsets[0].trajectory_constraints == []
    assert req.goalsets[0].tool_frames == ["left_tool", "right_tool"]

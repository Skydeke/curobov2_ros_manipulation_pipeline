"""Ordered Cartesian waypoint stages (``MULTIPOINT``).

Covers the single-tool ``cartesian_waypoints`` stage, the multi-tool
``multi_tool_waypoints`` stage (waypoint-major flattening), validation, the
MTC-shaped wrappers, the no-axis-holds contract, and the wire round trips
(Goalset message and executor replay dict).
"""

from __future__ import annotations

import sys

import pytest
import yaml

import curobo_task_constructor.stages  # noqa: F401  (register builtins)
from curobo_task_constructor.core.robot import GoalsetSpec
from curobo_task_constructor.core.state import InterfaceState
from curobo_task_constructor.executor import TaskExecutor
from curobo_task_constructor.graph.spec import StageSpec
from curobo_task_constructor.stages import (
    CartesianWaypoints,
    MultiToolWaypoints,
)
from curobo_task_constructor.stages._util import PLANNER_KEYS
from tests.mock_curobo import JOINT_NAMES, MockCuroboServer, fk_positions

IDENTITY_Q = (0.0, 0.0, 0.0, 1.0)


def _pose(x, y, z):
    return {"x": x, "y": y, "z": z,
            "qx": IDENTITY_Q[0], "qy": IDENTITY_Q[1],
            "qz": IDENTITY_Q[2], "qw": IDENTITY_Q[3]}


def _start_xyz(robot):
    return fk_positions(
        dict(zip(JOINT_NAMES, robot.get_current_joint_state().position)))


def _single_params(robot, count=2, **options):
    x, y, z = _start_xyz(robot)
    params = {"goal": {"waypoints": [
        _pose(x, y, z - 0.03 * (i + 1)) for i in range(count)]}}
    params.update(options)
    return params


def _multi_params(robot, tools=2, count=2, **options):
    x, y, z = _start_xyz(robot)
    entries = [
        {"frame": f"tool_{i}", "waypoints": [
            _pose(x, y + 0.01 * i, z - 0.03 * (k + 1))
            for k in range(count)]}
        for i in range(tools)
    ]
    params = {"goal": {"tools": entries}}
    params.update(options)
    return params


def _plan(robot, stage_type, params):
    root = StageSpec(
        stage_type="", name="root", container_type="serial",
        children=[
            StageSpec(stage_type="current_state", name="current"),
            StageSpec(stage_type=stage_type, name="leg",
                      params_yaml=yaml.safe_dump(params)),
        ])
    ex = TaskExecutor(root, robot)
    assert ex.init(), ex.describe()["comment"]
    return ex


def _stage(ex, name="leg"):
    return next(s for s in ex.root.subtree_stages() if s.name == name)


# -----------------------------------------------------------------------------
# single tool
# -----------------------------------------------------------------------------

def test_single_tool_sends_ordered_waypoints_with_no_holds():
    robot = MockCuroboServer()
    params = _single_params(robot, count=2, link="tool_link")
    ex = _plan(robot, "cartesian_waypoints", params)
    assert ex.plan()
    sol = _stage(ex).solutions[0]
    gs = sol.plan_request.goalsets[0]

    assert sol.plan_request.planner == PLANNER_KEYS["multipoint"] == 4
    assert len(gs.waypoints) == 2
    assert gs.waypoint_count == 2
    assert gs.tool_frames == ["tool_link"]
    assert gs.tool_waypoints == []
    assert gs.poses == []
    assert gs.tool_poses == []
    # Waypoint mode tracks poses; it never sends axis holds.
    assert gs.trajectory_constraints == []
    assert gs.tool_trajectory_constraints == []
    assert "cartesian_waypoints" in sol.comment
    assert "waypoints=2" in sol.comment


def test_single_tool_defaults_to_the_server_tool_frame_and_waypoint_options():
    robot = MockCuroboServer()
    ex = _plan(robot, "cartesian_waypoints", _single_params(robot, count=3))
    assert ex.plan()
    gs = _stage(ex).solutions[0].plan_request.goalsets[0]

    assert gs.tool_frames == []
    assert gs.waypoint_count == 3
    assert gs.path_position_tolerance == pytest.approx(0.05)
    assert gs.path_orientation_tolerance == pytest.approx(0.261799388)
    assert gs.final_position_tolerance == pytest.approx(0.005)
    assert gs.final_orientation_tolerance == pytest.approx(0.05)
    assert gs.track_orientation is True
    assert gs.outer_iterations == 4


def test_single_tool_carries_tolerance_tracking_and_iteration_overrides():
    robot = MockCuroboServer()
    params = _single_params(
        robot, count=1,
        path_position_tolerance=0.02,
        path_orientation_tolerance=0.1,
        final_position_tolerance=0.002,
        final_orientation_tolerance=0.03,
        track_orientation=False,
        outer_iterations=7,
        planner="classic",  # cannot redirect a waypoint goalset
    )
    ex = _plan(robot, "cartesian_waypoints", params)
    assert ex.plan()
    sol = _stage(ex).solutions[0]
    gs = sol.plan_request.goalsets[0]

    assert sol.plan_request.planner == 4
    assert gs.path_position_tolerance == pytest.approx(0.02)
    assert gs.path_orientation_tolerance == pytest.approx(0.1)
    assert gs.final_position_tolerance == pytest.approx(0.002)
    assert gs.final_orientation_tolerance == pytest.approx(0.03)
    assert gs.track_orientation is False
    assert gs.outer_iterations == 7


@pytest.mark.parametrize("goal", [
    {},
    {"waypoints": []},
    {"waypoints": [{"x": 1.0}]},
    {"waypoints": ["not-a-pose"]},
])
def test_single_tool_rejects_a_malformed_path(goal):
    robot = MockCuroboServer()
    ex = _plan(robot, "cartesian_waypoints", {"goal": goal})
    assert not ex.plan()
    failures = _stage(ex).failures
    assert failures
    assert "waypoints" in failures[0].message
    assert robot.plan_calls == 0


@pytest.mark.parametrize("key,value", [
    ("path_position_tolerance", -0.1),
    ("path_position_tolerance", 0.0),
    ("final_orientation_tolerance", float("inf")),
    ("path_orientation_tolerance", 4.0),
    ("outer_iterations", 0),
    ("outer_iterations", "many"),
])
def test_single_tool_rejects_bad_waypoint_options(key, value):
    robot = MockCuroboServer()
    params = _single_params(robot)
    params[key] = value
    ex = _plan(robot, "cartesian_waypoints", params)
    assert not ex.plan()
    assert _stage(ex).failures
    assert robot.plan_calls == 0


def test_single_tool_backward_fails_clearly():
    robot = MockCuroboServer()
    stage = CartesianWaypoints(name="leg", params=_single_params(robot))
    stage.robot = robot
    stage.compute_backward(
        InterfaceState(joint_state=robot.get_current_joint_state()))

    assert stage.failures
    assert "forward-only" in stage.failures[0].message
    assert robot.plan_calls == 0


# -----------------------------------------------------------------------------
# multiple tools
# -----------------------------------------------------------------------------

def test_multi_tool_flattens_waypoint_major():
    robot = MockCuroboServer()
    x, y, z = _start_xyz(robot)
    ex = _plan(robot, "multi_tool_waypoints", _multi_params(robot, tools=2, count=2))
    assert ex.plan()
    sol = _stage(ex).solutions[0]
    gs = sol.plan_request.goalsets[0]

    assert sol.plan_request.planner == 4
    assert gs.tool_frames == ["tool_0", "tool_1"]
    assert gs.waypoint_count == 2
    assert len(gs.tool_waypoints) == 2 * 2
    # Waypoint-major: all tools of wp0 first, then all tools of wp1.
    assert [p.position.y for p in gs.tool_waypoints] == pytest.approx(
        [y, y + 0.01, y, y + 0.01])
    assert [p.position.z for p in gs.tool_waypoints] == pytest.approx(
        [z - 0.03, z - 0.03, z - 0.06, z - 0.06])
    # Version-major would have been [wp0 t0, wp1 t0, wp0 t1, wp1 t1].
    assert gs.tool_waypoints[1].position.z == pytest.approx(
        gs.tool_waypoints[0].position.z)
    assert gs.waypoints == []
    assert gs.poses == []
    assert gs.trajectory_constraints == []
    assert gs.tool_trajectory_constraints == []
    assert "multi_tool_waypoints" in sol.comment
    assert "waypoints=2" in sol.comment


def test_multi_tool_accepts_three_tools_and_keeps_the_order():
    robot = MockCuroboServer()
    x, y, z = _start_xyz(robot)
    ex = _plan(robot, "multi_tool_waypoints",
               _multi_params(robot, tools=3, count=3))
    assert ex.plan()
    gs = _stage(ex).solutions[0].plan_request.goalsets[0]

    assert gs.tool_frames == ["tool_0", "tool_1", "tool_2"]
    assert gs.waypoint_count == 3
    assert len(gs.tool_waypoints) == 9
    for wp in range(3):
        block = gs.tool_waypoints[wp * 3:(wp + 1) * 3]
        assert [p.position.z for p in block] == pytest.approx(
            [z - 0.03 * (wp + 1)] * 3)
        assert [p.position.y for p in block] == pytest.approx(
            [y, y + 0.01, y + 0.02])


def test_multi_tool_accepts_a_frame_to_pose_list_mapping():
    robot = MockCuroboServer()
    x, y, z = _start_xyz(robot)
    ex = _plan(robot, "multi_tool_waypoints", {"goal": {"tools": {
        "left": [_pose(x, y, z - 0.03)],
        "right": [_pose(x, y + 0.01, z - 0.03)],
    }}})
    assert ex.plan()
    gs = _stage(ex).solutions[0].plan_request.goalsets[0]
    assert gs.tool_frames == ["left", "right"]
    assert gs.waypoint_count == 1
    assert len(gs.tool_waypoints) == 2


def test_multi_tool_requires_the_same_waypoint_count_per_tool():
    robot = MockCuroboServer()
    x, y, z = _start_xyz(robot)
    params = {"goal": {"tools": [
        {"frame": "left", "waypoints": [_pose(x, y, z - 0.03)]},
        {"frame": "right", "waypoints": [
            _pose(x, y, z - 0.03), _pose(x, y, z - 0.06)]},
    ]}}
    ex = _plan(robot, "multi_tool_waypoints", params)
    assert not ex.plan()
    failures = _stage(ex).failures
    assert failures
    assert "same number of ordered waypoints" in failures[0].message
    assert robot.plan_calls == 0


def test_multi_tool_rejects_duplicate_frames_and_empty_tools():
    robot = MockCuroboServer()
    x, y, z = _start_xyz(robot)
    pose = _pose(x, y, z - 0.03)
    for params, expected in (
        ({"goal": {"tools": [
            {"frame": "same", "waypoints": [pose]},
            {"frame": "same", "waypoints": [pose]},
        ]}}, "repeats a tool frame"),
        ({"goal": {"tools": []}}, "non-empty list"),
        ({"goal": {"tools": [{"frame": "", "waypoints": [pose]}]}},
         "has no 'frame'"),
    ):
        ex = _plan(robot, "multi_tool_waypoints", params)
        assert not ex.plan()
        assert expected in _stage(ex).failures[0].message
    assert robot.plan_calls == 0


def test_multi_tool_ignores_legacy_hold_params():
    robot = MockCuroboServer()
    params = _multi_params(robot, hold=[1, 1, 1, 0, 1, 0],
                           holds=[[1, 1, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0]])
    ex = _plan(robot, "multi_tool_waypoints", params)
    assert ex.plan()
    gs = _stage(ex).solutions[0].plan_request.goalsets[0]
    assert gs.trajectory_constraints == []
    assert gs.tool_trajectory_constraints == []


# -----------------------------------------------------------------------------
# MTC-shaped wrappers
# -----------------------------------------------------------------------------

def test_mtc_single_tool_wrapper_builds_waypoints_and_options():
    from curobo_task_constructor.mtc import core, stages

    move = stages.CartesianWaypoints(
        "descend", core.PipelinePlanner("multipoint", "", 1))
    move.setWaypoints([
        {"x": 0.3, "y": 0.0, "z": 0.2, "qw": 1.0},
        {"x": 0.3, "y": 0.0, "z": 0.1, "qw": 1.0},
    ])
    move.setLink("tool_link")
    move.setTolerances(0.02, 0.1, 0.002, 0.03)
    move.setTrackOrientation(True)
    move.setOuterIterations(6)
    move.setCost("path_length")

    params = yaml.safe_load(move.to_spec().params_yaml)
    assert params["planner"] == 4
    assert params["link"] == "tool_link"
    assert params["goal"]["waypoints"] == [
        {"x": 0.3, "y": 0.0, "z": 0.2, "qx": 0.0, "qy": 0.0,
         "qz": 0.0, "qw": 1.0},
        {"x": 0.3, "y": 0.0, "z": 0.1, "qx": 0.0, "qy": 0.0,
         "qz": 0.0, "qw": 1.0},
    ]
    assert params["path_position_tolerance"] == pytest.approx(0.02)
    assert params["path_orientation_tolerance"] == pytest.approx(0.1)
    assert params["final_position_tolerance"] == pytest.approx(0.002)
    assert params["final_orientation_tolerance"] == pytest.approx(0.03)
    assert params["track_orientation"] is True
    assert params["outer_iterations"] == 6
    assert "hold" not in params and "holds" not in params


def test_mtc_multi_tool_wrapper_builds_one_waypoint_list_per_tool():
    from curobo_task_constructor.mtc import core, stages

    move = stages.MultiToolWaypoints(
        "descend", core.PipelinePlanner("multipoint", "", 1))
    move.setWaypoints({
        "left_tool": [
            {"x": 0.3, "y": 0.0, "z": 0.2, "qw": 1.0},
            {"x": 0.3, "y": 0.0, "z": 0.1, "qw": 1.0},
        ],
        "right_tool": [
            {"x": 0.3, "y": 0.1, "z": 0.2, "qw": 1.0},
            {"x": 0.3, "y": 0.1, "z": 0.1, "qw": 1.0},
        ],
    })
    move.setCost("path_length")

    params = yaml.safe_load(move.to_spec().params_yaml)
    assert params["planner"] == 4
    assert [tool["frame"] for tool in params["goal"]["tools"]] == [
        "left_tool", "right_tool"]
    assert [len(tool["waypoints"]) for tool in params["goal"]["tools"]] == [2, 2]
    assert "hold" not in params and "holds" not in params


def test_mtc_multi_tool_wrapper_accepts_single_targets_and_plans():
    from curobo_task_constructor.mtc import core, stages

    robot = MockCuroboServer()
    x, y, z = _start_xyz(robot)
    task = core.Task(robot)
    task.add(stages.CurrentState("current"))
    move = stages.MultiToolWaypoints(
        "descend", core.PipelinePlanner("multipoint", "", 1))
    move.setTools([
        {"frame": "left_tool", "pose": _pose(x, y, z - 0.03)},
        ("right_tool", _pose(x, y + 0.01, z - 0.03)),
    ])
    task.add(move)

    assert task.plan()
    leaf = next(s for s in task._executor.root.subtree_stages()
                if s.name == "descend")
    sol = leaf.solutions[0]
    gs = sol.plan_request.goalsets[0]
    assert sol.plan_request.planner == 4
    assert gs.waypoint_count == 1
    assert len(gs.tool_waypoints) == 2
    assert gs.tool_trajectory_constraints == []
    assert gs.trajectory_constraints == []


def test_mtc_multi_tool_wrapper_rejects_mismatched_counts():
    from curobo_task_constructor.mtc import core, stages

    move = stages.MultiToolWaypoints(
        "descend", core.PipelinePlanner("multipoint", "", 1))
    with pytest.raises(ValueError, match="same number of waypoints per tool"):
        move.setWaypoints({
            "left_tool": [{"x": 0.0, "y": 0.0, "z": 0.0}],
            "right_tool": [{"x": 0.0, "y": 0.0, "z": 0.0},
                           {"x": 0.0, "y": 0.0, "z": 0.1}],
        })


# -----------------------------------------------------------------------------
# wire round trips
# -----------------------------------------------------------------------------

def test_executor_io_round_trip_preserves_waypoint_fields():
    from curobo_task_constructor.executor_io import (
        _plan_request_from_dict,
        _plan_request_to_dict,
    )
    from curobo_task_constructor.core.geom import Pose3

    pose_a = Pose3([0.1, 0.2, 0.3], list(IDENTITY_Q))
    pose_b = Pose3([0.1, 0.2, 0.2], list(IDENTITY_Q))
    req = type("Req", (), {})()
    req.goalsets = [GoalsetSpec(
        tool_frames=["left", "right"],
        tool_waypoints=[pose_a, pose_b, pose_a, pose_b],
        waypoint_count=2,
        path_position_tolerance=0.02,
        path_orientation_tolerance=0.1,
        final_position_tolerance=0.002,
        final_orientation_tolerance=0.03,
        track_orientation=False,
        outer_iterations=6,
    )]
    req.start_pose = None
    req.planner = 4

    d = _plan_request_to_dict(req)
    gs = d["goalsets"][0]
    assert gs["waypoint_count"] == 2
    assert len(gs["tool_waypoints"]) == 4
    assert gs["track_orientation"] is False
    assert gs["outer_iterations"] == 6

    back = _plan_request_from_dict(d)
    rebuilt = back.goalsets[0]
    assert rebuilt.tool_frames == ["left", "right"]
    assert len(rebuilt.tool_waypoints) == 4
    assert rebuilt.waypoint_count == 2
    assert rebuilt.path_position_tolerance == pytest.approx(0.02)
    assert rebuilt.final_orientation_tolerance == pytest.approx(0.03)
    assert rebuilt.track_orientation is False
    assert rebuilt.outer_iterations == 6
    assert back.planner == 4


def test_goals_to_msg_carries_waypoint_fields():
    from tests.test_multi_tool_move_to import (
        _install_ros_stubs,
        _remove_ros_stubs,
    )

    mods = _install_ros_stubs()
    try:
        from curobo_task_constructor.core.geom import Pose3
        from curobo_task_constructor.robot.msg import _MsgMixin

        class _Adapter(_MsgMixin):
            pose_cls = sys.modules["geometry_msgs.msg"].Pose

        msg = _Adapter()._to_goalset(GoalsetSpec(
            tool_frames=["left", "right"],
            tool_waypoints=[
                Pose3([0.1, 0.2, 0.3], list(IDENTITY_Q)),
                Pose3([0.4, 0.5, 0.6], list(IDENTITY_Q)),
            ],
            waypoint_count=1,
            path_position_tolerance=0.02,
            path_orientation_tolerance=0.1,
            final_position_tolerance=0.002,
            final_orientation_tolerance=0.03,
            track_orientation=False,
            outer_iterations=6,
        ))
        assert msg.tool_frames == ["left", "right"]
        assert len(msg.tool_waypoints) == 2
        assert msg.tool_waypoints[0].position.x == pytest.approx(0.1)
        assert msg.tool_waypoints[1].position.z == pytest.approx(0.6)
        assert msg.waypoints == []
        assert msg.waypoint_count == 1
        assert msg.path_position_tolerance == pytest.approx(0.02)
        assert msg.path_orientation_tolerance == pytest.approx(0.1)
        assert msg.final_position_tolerance == pytest.approx(0.002)
        assert msg.final_orientation_tolerance == pytest.approx(0.03)
        assert msg.track_orientation is False
        assert msg.outer_iterations == 6
        # Waypoint mode never sends holds.
        assert msg.trajectory_constraints == []
        assert msg.tool_trajectory_constraints == []
    finally:
        _remove_ros_stubs(mods)

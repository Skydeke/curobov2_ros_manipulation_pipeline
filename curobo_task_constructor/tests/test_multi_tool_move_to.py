"""MultiToolMoveTo: one goalset with parallel ``tool_frames``/``tool_poses``.

Covers the core stage, the MTC-shaped wrapper and the wire conversion that
turns a ``GoalsetSpec`` into a ``Goalset`` message.
"""

from __future__ import annotations

import sys
import types

import pytest
import yaml

import curobo_task_constructor.stages  # noqa: F401  (register builtins)
from curobo_task_constructor.core.geom import Pose3
from curobo_task_constructor.core.robot import GoalsetSpec
from curobo_task_constructor.core.state import InterfaceState, SceneDiff
from curobo_task_constructor.executor import TaskExecutor
from curobo_task_constructor.graph.spec import StageSpec
from curobo_task_constructor.stages import MultiToolMoveTo
from tests.mock_curobo import MockCuroboServer

IDENTITY_Q = (0.0, 0.0, 0.0, 1.0)


def _pose(x, y, z):
    return {"x": x, "y": y, "z": z,
            "qx": IDENTITY_Q[0], "qy": IDENTITY_Q[1],
            "qz": IDENTITY_Q[2], "qw": IDENTITY_Q[3]}


class _Vec:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _Quat:
    def __init__(self, x, y, z, w):
        self.x, self.y, self.z, self.w = x, y, z, w


class _PoseLike:
    """Duck-typed geometry_msgs/Pose stand-in (MTC wrapper input)."""

    def __init__(self, x, y, z, q=IDENTITY_Q):
        self.position = _Vec(x, y, z)
        self.orientation = _Quat(*q)


def _params():
    return {"goal": {"tools": [
        {"frame": "left_tool", "pose": _pose(0.30, 0.0, 0.20)},
        {"frame": "right_tool", "pose": _pose(0.30, 0.0, 0.10)},
    ]}}


def _plan(robot, params=None):
    root = StageSpec(
        stage_type="", name="root", container_type="serial",
        children=[
            StageSpec(stage_type="current_state", name="current"),
            StageSpec(stage_type="multi_tool_move_to", name="bimanual",
                      params_yaml=yaml.safe_dump(params or _params())),
        ])
    ex = TaskExecutor(root, robot)
    assert ex.init(), ex.describe()["comment"]
    return ex


def _stage(ex, name="bimanual"):
    return next(s for s in ex.root.subtree_stages() if s.name == name)


# -----------------------------------------------------------------------------
# core stage
# -----------------------------------------------------------------------------

def test_stage_sends_one_goalset_with_parallel_tool_goals():
    robot = MockCuroboServer()
    ex = _plan(robot)
    assert ex.plan()
    stage = _stage(ex)
    sol = stage.solutions[0]
    req = sol.plan_request
    gs = req.goalsets[0]

    assert gs.poses == []
    assert gs.tool_frames == ["left_tool", "right_tool"]
    assert [p.position.x for p in gs.tool_poses] == [0.30, 0.30]
    assert [p.position.z for p in gs.tool_poses] == [0.20, 0.10]
    assert req.planner == 0  # CLASSIC: the server's multi-tool path
    assert gs.allowed_collisions == []
    assert sol.trajectory
    assert "left_tool" in sol.comment and "right_tool" in sol.comment
    assert ex.best() is not None


def test_stage_accepts_three_tool_frames():
    """The interface is N-tool; bimanual is just the two-tool case."""
    params = {"goal": {"tools": [
        {"frame": "arm_a", "pose": _pose(0.30, 0.0, 0.20)},
        {"frame": "arm_b", "pose": _pose(0.30, 0.0, 0.10)},
        {"frame": "arm_c", "pose": _pose(0.30, 0.0, 0.00)},
    ]}}
    robot = MockCuroboServer()
    ex = _plan(robot, params)
    assert ex.plan()
    gs = _stage(ex).solutions[0].plan_request.goalsets[0]
    assert gs.tool_frames == ["arm_a", "arm_b", "arm_c"]
    assert len(gs.tool_poses) == 3


def test_stage_inherits_the_scene_collision_allowance():
    robot = MockCuroboServer()
    stage = MultiToolMoveTo(name="bimanual", params=_params())
    stage.robot = robot
    scene = SceneDiff().with_collisions(
        "object", ["left_finger", "right_finger"], True)
    state = InterfaceState(joint_state=robot.get_current_joint_state(),
                           scene=scene)

    stage.compute_forward(state)

    gs = stage.solutions[0].plan_request.goalsets[0]
    assert gs.allowed_collisions == ["left_finger", "right_finger"]


def test_stage_does_not_fk_for_absolute_tool_poses():
    robot = MockCuroboServer()

    def _boom(*_args, **_kwargs):
        raise AssertionError("multi_tool_move_to must not call FK")

    robot.fk = _boom
    ex = _plan(robot)
    assert ex.plan()


def test_missing_tools_fails_with_a_clear_message():
    robot = MockCuroboServer()
    stage = MultiToolMoveTo(name="bimanual", params={"goal": {}})
    stage.robot = robot
    stage.compute_forward(
        InterfaceState(joint_state=robot.get_current_joint_state()))

    assert stage.failures
    assert "goal.tools" in stage.failures[0].message
    assert robot.plan_calls == 0


def test_backward_fails_with_a_clear_message():
    robot = MockCuroboServer()
    stage = MultiToolMoveTo(name="bimanual", params=_params())
    stage.robot = robot
    stage.compute_backward(
        InterfaceState(joint_state=robot.get_current_joint_state()))

    assert stage.failures
    assert "forward-only" in stage.failures[0].message
    assert robot.plan_calls == 0


# -----------------------------------------------------------------------------
# MTC-shaped wrapper
# -----------------------------------------------------------------------------

def test_mtc_wrapper_builds_tools_and_plans():
    from curobo_task_constructor.mtc import core, stages

    robot = MockCuroboServer()
    task = core.Task(robot)
    task.add(stages.CurrentState("current"))
    move = stages.MultiToolMoveTo(
        "bimanual", core.PipelinePlanner("classic", "", 1))
    move.setGoals([
        {"frame": "left_tool", "pose": {"x": 0.30, "y": 0.0, "z": 0.20,
                                        "qw": 1.0}},
        ("right_tool", _PoseLike(0.30, 0.0, 0.10)),
    ])
    task.add(move)

    params = yaml.safe_load(move.to_spec().params_yaml)
    assert params["goal"]["tools"] == [
        {"frame": "left_tool",
         "pose": {"x": 0.30, "y": 0.0, "z": 0.20,
                  "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}},
        {"frame": "right_tool",
         "pose": {"x": 0.30, "y": 0.0, "z": 0.10,
                  "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}},
    ]
    assert task.plan()
    leaf = next(s for s in task._executor.root.subtree_stages()
                if s.name == "bimanual")
    gs = leaf.solutions[0].plan_request.goalsets[0]
    assert gs.tool_frames == ["left_tool", "right_tool"]


# -----------------------------------------------------------------------------
# GoalsetSpec -> Goalset wire conversion
# -----------------------------------------------------------------------------

def _install_ros_stubs():
    """Minimal ROS module stubs so ``robot.msg`` imports without rclpy."""
    mods = {}

    def _mod(name):
        module = types.ModuleType(name)
        sys.modules[name] = module
        mods[name] = module
        return module

    class _Vec:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x, self.y, self.z = float(x), float(y), float(z)

    class _Quat:
        def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
            self.x, self.y, self.z, self.w = (
                float(x), float(y), float(z), float(w))

    class _Pose:
        def __init__(self):
            self.position = _Vec()
            self.orientation = _Quat()

    geo = _mod("geometry_msgs")
    gmsg = _mod("geometry_msgs.msg")
    gmsg.Point = gmsg.Vector3 = _Vec
    gmsg.Quaternion = _Quat
    gmsg.Pose = _Pose
    geo.msg = gmsg

    class _Goalset:
        def __init__(self):
            self.poses = []
            self.tool_frames = []
            self.tool_poses = []
            self.target_joint_positions = []
            self.trajectory_constraints = []
            self.tool_trajectory_constraints = []
            self.waypoints = []
            self.tool_waypoints = []
            self.waypoint_count = 0
            self.path_position_tolerance = 0.05
            self.path_orientation_tolerance = 0.261799388
            self.final_position_tolerance = 0.005
            self.final_orientation_tolerance = 0.05
            self.track_orientation = True
            self.outer_iterations = 4

    class _TrajectoryGoal:
        def __init__(self):
            self.start_pose = None
            self.goalsets = []
            self.cartesian_step_size = 0.0
            self.cartesian_precision = 0.0

    iface = _mod("curobov2_ros_interfaces")
    imsg = _mod("curobov2_ros_interfaces.msg")
    imsg.Goalset = _Goalset
    imsg.TrajectoryGoal = _TrajectoryGoal
    iface.msg = imsg

    action = _mod("curobov2_ros_interfaces.action")
    action.SendTrajectory = type("SendTrajectory", (), {})
    iface.action = action

    srv = _mod("curobov2_ros_interfaces.srv")
    for name in ("AddObject", "AttachObject", "Fk", "GetSceneObjects", "Ik",
                 "RemoveObject", "SetLinkCollision", "SetPlanner",
                 "TrajectoryGeneration", "TrajectoryGenerationBatch"):
        setattr(srv, name, type(name, (), {}))
    srv.AddObject.Request = types.SimpleNamespace(
        CUBOID=1, BOX=1, SPHERE=2, CYLINDER=3, CAPSULE=4, MESH=5)
    iface.srv = srv

    control = _mod("control_msgs")
    caction = _mod("control_msgs.action")
    for name in ("GripperCommand", "FollowJointTrajectory",
                 "ParallelGripperCommand"):
        setattr(caction, name, type(name, (), {}))
    control.action = caction

    rclpy = _mod("rclpy")
    raction = _mod("rclpy.action")
    raction.ActionClient = type("ActionClient", (), {})
    rclpy.action = raction

    sensor = _mod("sensor_msgs")
    smsg = _mod("sensor_msgs.msg")
    smsg.JointState = type("JointState", (), {})
    sensor.msg = smsg

    builtin = _mod("builtin_interfaces")
    bmsg = _mod("builtin_interfaces.msg")
    bmsg.Duration = type("Duration", (), {})
    builtin.msg = bmsg

    traj = _mod("trajectory_msgs")
    tmsg = _mod("trajectory_msgs.msg")
    tmsg.JointTrajectoryPoint = type("JointTrajectoryPoint", (), {})
    traj.msg = tmsg

    std = _mod("std_srvs")
    ssrv = _mod("std_srvs.srv")
    ssrv.Trigger = type("Trigger", (), {})
    std.srv = ssrv
    return mods


def _remove_ros_stubs(mods):
    for name in mods:
        sys.modules.pop(name, None)


def test_to_goalset_populates_the_multi_tool_fields():
    mods = _install_ros_stubs()
    try:
        from curobo_task_constructor.robot.msg import _MsgMixin

        class _Adapter(_MsgMixin):
            pose_cls = sys.modules["geometry_msgs.msg"].Pose

        holds = [1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1]
        msg = _Adapter()._to_goalset(GoalsetSpec(
            tool_frames=["left_tool", "right_tool"],
            tool_poses=[
                Pose3([0.1, 0.2, 0.3], list(IDENTITY_Q)),
                Pose3([0.4, 0.5, 0.6], list(IDENTITY_Q)),
            ],
            tool_trajectory_constraints=holds))

        assert msg.tool_frames == ["left_tool", "right_tool"]
        assert len(msg.tool_poses) == 2
        assert msg.tool_poses[0].position.x == pytest.approx(0.1)
        assert msg.tool_poses[1].position.z == pytest.approx(0.6)
        assert msg.poses == []
        # Per-tool holds ride the new field; the legacy field stays empty so
        # the bridge applies each group exactly once.
        assert msg.tool_trajectory_constraints == holds
        assert msg.trajectory_constraints == []
    finally:
        _remove_ros_stubs(mods)


def test_to_goalset_keeps_the_legacy_single_tool_hold():
    mods = _install_ros_stubs()
    try:
        from curobo_task_constructor.robot.msg import _MsgMixin

        class _Adapter(_MsgMixin):
            pose_cls = sys.modules["geometry_msgs.msg"].Pose

        hold = [1, 1, 1, 0, 1, 0]
        msg = _Adapter()._to_goalset(GoalsetSpec(
            poses=[Pose3([0.1, 0.2, 0.3], list(IDENTITY_Q))],
            trajectory_constraints=hold))

        assert msg.trajectory_constraints == hold
        assert msg.tool_trajectory_constraints == []
    finally:
        _remove_ros_stubs(mods)


def test_to_goalset_keeps_the_old_wire_form_when_no_tools():
    mods = _install_ros_stubs()
    try:
        from curobo_task_constructor.robot.msg import _MsgMixin

        class _Adapter(_MsgMixin):
            pose_cls = sys.modules["geometry_msgs.msg"].Pose

        msg = _Adapter()._to_goalset(GoalsetSpec(
            poses=[Pose3([0.1, 0.2, 0.3], list(IDENTITY_Q))]))

        assert msg.tool_frames == []
        assert msg.tool_poses == []
        assert len(msg.poses) == 1
        assert msg.tool_trajectory_constraints == []
    finally:
        _remove_ros_stubs(mods)

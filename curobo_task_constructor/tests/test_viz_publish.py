from __future__ import annotations

import pytest

from curobo_task_constructor import viz
from curobo_task_constructor.mtc import core, stages
from tests.mock_curobo import MockCuroboServer


class _Time:
    def __init__(self):
        self.sec = 0
        self.nanosec = 0


class _Point:
    def __init__(self):
        self.positions = []
        self.time_from_start = _Time()


class _Traj:
    def __init__(self):
        self.joint_names = []
        self.points = []


class _JS:
    def __init__(self, names, positions):
        self.name = list(names)
        self.position = list(positions)


def test_to_joint_trajectory_maps_waypoints_to_points():
    names = ["joint_1", "joint_2"]
    wps = [_JS(names, [0.1 * i, 0.2 * i]) for i in range(4)]
    msg = viz.to_joint_trajectory(wps, msg_cls=_Traj, point_cls=_Point, dt=0.05)
    assert msg.joint_names == names
    assert len(msg.points) == 4
    assert msg.points[0].positions == [0.0, 0.0]
    assert msg.points[3].positions[0] == pytest.approx(0.3)
    assert msg.points[3].positions[1] == pytest.approx(0.6)
    assert msg.points[0].time_from_start.sec == 0
    assert msg.points[2].time_from_start.nanosec == int(0.1 * 1e9)


def test_to_joint_trajectory_rejects_empty_or_nameless():
    assert viz.to_joint_trajectory([], msg_cls=_Traj, point_cls=_Point) is None

    class _Bare:
        position = [0.0]

    assert viz.to_joint_trajectory([_Bare()], msg_cls=_Traj,
                                   point_cls=_Point) is None


def test_publish_without_node_only_records():
    robot = MockCuroboServer()
    task = core.Task(robot)
    task.add(stages.CurrentState("current state"))
    assert task.plan()
    task.publish(task.solutions[0])
    assert task.last_published is task.solutions[0]


def test_publish_with_node_publishes_trajectory():
    import sys
    import types
    from unittest.mock import MagicMock

    published = []

    fake_msg = types.ModuleType("trajectory_msgs.msg")

    class _RosPoint:
        def __init__(self):
            self.positions = []
            self.time_from_start = _Time()

    class _RosTraj:
        def __init__(self):
            self.joint_names = []
            self.points = []

    fake_msg.JointTrajectory = _RosTraj
    fake_msg.JointTrajectoryPoint = _RosPoint
    fake_pkg = types.ModuleType("trajectory_msgs")
    fake_pkg.msg = fake_msg
    sys.modules["trajectory_msgs"] = fake_pkg
    sys.modules["trajectory_msgs.msg"] = fake_msg
    try:
        from curobo_task_constructor import viz
        assert viz.SOLUTION_TRAJECTORY_TOPIC == \
            "/curobo_task_constructor/solution_trajectory"

        class _Node:
            def create_publisher(self, cls, topic, qos):
                assert cls is _RosTraj
                assert topic == viz.SOLUTION_TRAJECTORY_TOPIC

                class _Pub:
                    def publish(self, msg):
                        published.append(msg)

                return _Pub()

        robot = MockCuroboServer(
            named={"ready": {"joint_1": 0.1, "joint_2": 0.5}})
        task = core.Task(robot)
        task.add(stages.CurrentState("current state"))
        ready = stages.MoveTo("moveTo ready", core.JointInterpolationPlanner())
        ready.setGoal("ready")
        task.add(ready)
        assert task.plan()
        task.attach_node(_Node())
        task.publish(task.solutions[0])
        assert task.last_published is task.solutions[0]
        assert len(published) == 1
        assert published[0].joint_names
        assert len(published[0].points) > 1
    finally:
        sys.modules.pop("trajectory_msgs", None)
        sys.modules.pop("trajectory_msgs.msg", None)

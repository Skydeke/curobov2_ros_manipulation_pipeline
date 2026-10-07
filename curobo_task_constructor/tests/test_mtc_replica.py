"""MTC-replica API tests: cartesian.py sequence planned locally (no ROS)."""
from math import pi

from curobo_task_constructor.mtc import core, stages

from tests.mock_curobo import MockCuroboServer


class _V3:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _V3S:
    def __init__(self, frame, v):
        self.header = type("H", (), {"frame_id": frame})()
        self.vector = v


class _Twist:
    def __init__(self, lin, ang):
        self.linear, self.angular = lin, ang


class _TS:
    def __init__(self, frame, tw):
        self.header = type("H", (), {"frame_id": frame})()
        self.twist = tw


def _robot():
    cur = {"joint_2": 0.5, "joint_3": 1.2}
    named = {"ready": {"joint_1": 0.1, "joint_2": 0.5, "joint_3": 1.2,
                       "joint_4": 0.0, "joint_5": 0.0, "joint_6": 0.0,
                       "finger_joint": 0.0}}
    return MockCuroboServer(current=cur, named=named)


def _cartesian_task(robot):
    task = core.Task(robot)
    task.name = "cartesian"
    cartesian = core.CartesianPath()
    cartesian.max_cartesian_speed = 0.1
    cartesian.cartesian_speed_limited_link = "tool_link"
    jointspace = core.JointInterpolationPlanner()
    task.add(stages.CurrentState("current state"))
    m = stages.MoveRelative("x +0.05", cartesian)
    m.group = "arm"
    m.setDirection(_V3S("world", _V3(0.05, 0, 0)))
    task.add(m)
    m = stages.MoveRelative("y -0.05", cartesian)
    m.group = "arm"
    m.setDirection(_V3S("world", _V3(0, -0.05, 0)))
    task.add(m)
    m = stages.MoveRelative("rz +45deg", cartesian)
    m.group = "arm"
    m.setDirection(_TS("hand", _Twist(_V3(0, 0, 0), _V3(0, 0, pi / 4))))
    task.add(m)
    m = stages.MoveRelative("joint offset", cartesian)
    m.group = "arm"
    m.setDirection({"joint_1": 0.1, "joint_3": -0.1})
    task.add(m)
    m = stages.MoveTo("moveTo ready", jointspace)
    m.group = "arm"
    m.setGoal("ready")
    task.add(m)
    return task


def test_cartesian_replica_plans_locally():
    task = _cartesian_task(_robot())
    assert task.plan()
    assert len(task.solutions) == 1
    task.publish(task.solutions[0])
    assert task.last_published is task.solutions[0]


def test_spec_round_trips_through_wire_format():
    task = _cartesian_task(_robot())
    spec = task.to_spec()
    assert spec.container_type == "serial"
    assert [c.stage_type for c in spec.children] == [
        "current_state", "move_relative", "move_relative",
        "move_relative", "move_relative", "move_to"]
    msgs = spec.to_msg_list()
    assert msgs[0].parent_id == msgs[0].id == 0
    from curobo_task_constructor.graph.spec import StageSpec
    back = StageSpec.from_msg_list(msgs)
    assert [c.stage_type for c in back.children] == [
        "current_state", "move_relative", "move_relative",
        "move_relative", "move_relative", "move_to"]


def test_setdirection_overloads_hit_expected_params():
    m = stages.MoveRelative("x", None)
    m.setDirection(_V3S("world", _V3(0.05, 0, 0)))
    assert m.params["distance"] == 0.05
    assert m.params["axis"]["frame"] == "world"
    m2 = stages.MoveRelative("rz", None)
    m2.setDirection(_TS("hand", _Twist(_V3(0, 0, 0), _V3(0, 0, pi / 4))))
    assert abs(m2.params["rotation"]["angle"] - pi / 4) < 1e-9
    m3 = stages.MoveRelative("j", None)
    m3.setDirection({"joint_1": 0.5})
    assert m3.params["joint_offsets"] == {"joint_1": 0.5}
    mt = stages.MoveTo("g", None)
    mt.setGoal("ready")
    assert mt.params["goal"] == {"name": "ready"}
    mt.setGoal({"finger_joint": 0.04})
    assert mt.params["goal"] == {"joints": {"finger_joint": 0.04}}

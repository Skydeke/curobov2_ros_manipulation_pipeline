"""A gripper-only action must touch only the jaws.

TIAGo represents an open/close as a ``FollowJointTrajectory`` command to the
pal-gripper controller (two finger joints); the fingers are ``lock_joints`` and
are NOT in the 8-DOF torso+arm cspace. A goal that names only those joints can
therefore not be a cuRobo plan: before this fix it was projected onto the
cspace, leaving an arm-only goal equal to the start pose — a full-arm plan for
an action that should move only the jaws.

These tests pin the two halves of the contract:

* a gripper-only joint goal becomes a ``gripper_command`` segment (no
  ``PlanRequest``), executed on the gripper controller;
* its state starts from what the chain already knows (the previous stage's
  arm pose), filling only missing joints from the live reading, so a command
  after an arm leg stays on that leg's pose while a one-stage open still gets
  the full robot state.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import curobo_task_constructor.stages  # noqa: F401  (register builtins)
from curobo_task_constructor.core.robot import PlanResult
from curobo_task_constructor.core.robot_config import (
    NamedJointConfig,
    canonical_joint_order,
    get_gripper_controllers,
)
from curobo_task_constructor.core.state import InterfaceState, JointStateStub
from curobo_task_constructor.executor import TaskExecutor
from curobo_task_constructor.graph.spec import StageSpec
from curobo_task_constructor.stages.gripper_command import GripperCommand
from tests.mock_curobo import MockCuroboServer

#: TIAGo's 8-DOF planning cspace (torso lift + seven arm joints). The fingers
#: are lock_joints and deliberately absent.
TIAGO_CSPACE = [
    "torso_lift_joint",
    "arm_1_joint",
    "arm_2_joint",
    "arm_3_joint",
    "arm_4_joint",
    "arm_5_joint",
    "arm_6_joint",
    "arm_7_joint",
]
LEFT = "gripper_left_finger_joint"
RIGHT = "gripper_right_finger_joint"
TIAGO_ARM = {
    "torso_lift_joint": 0.149,
    "arm_1_joint": 0.50,
    "arm_2_joint": -1.34,
    "arm_3_joint": -0.48,
    "arm_4_joint": 1.94,
    "arm_5_joint": -1.49,
    "arm_6_joint": 1.37,
    "arm_7_joint": 0.0,
}


class TiagoMock:
    """The TIAGo surface the stages touch, with the descriptor's gripper map."""

    joint_state_cls = None
    pose_cls = None

    def __init__(self):
        self._joint_order = list(TIAGO_CSPACE)
        self._current = JointStateStub(
            names=list(TIAGO_CSPACE) + [LEFT, RIGHT],
            positions=[TIAGO_ARM[n] for n in TIAGO_CSPACE] + [0.0, 0.0],
        )
        #: Same schema the TIAGo descriptor carries: one parallel gripper
        #: commanded through a FollowJointTrajectory controller.
        self._grippers = {
            "gripper": {
                "type": "JointTrajectory",
                "name": "gripper_controller",
                "action_ns": "gripper_cmd",
                "joints": [LEFT, RIGHT],
                "command_joint": LEFT,
                "open": 0.045,
                "close": 0.0,
                "max_effort": 16.0,
            }
        }
        self.named = {
            "open": NamedJointConfig("open", [LEFT, RIGHT], [0.045, 0.045]),
        }
        self.gripper_commands: list = []
        self.executed: list = []
        self.plan_calls = 0

    def get_current_joint_state(self):
        return self._current

    def get_attached_objects(self) -> list:
        return []

    def get_named_joint_config(self, name):
        return self.named[name]

    def gripper_controllers(self) -> dict:
        return dict(self._grippers)

    def gripper_joint(self, gripper: str = "") -> str:
        cfg = self._grippers.get(gripper) or next(
            iter(self._grippers.values()))
        return str(cfg.get("command_joint") or "")

    def gripper_command(self, **cmd):
        self.gripper_commands.append(dict(cmd))
        return PlanResult(True, "gripper")

    def plan(self, request):
        self.plan_calls += 1
        return PlanResult(False, "a gripper-only goal must not plan")

    def execute(self, request, trajectory=None):
        self.executed.append(request)
        return PlanResult(True, "ok")


def _spec(*children) -> StageSpec:
    return StageSpec(stage_type="", name="task", container_type="serial",
                     children=list(children))


def _current_state() -> StageSpec:
    return StageSpec(stage_type="current_state", name="current_state")


def _joint_values(state) -> dict:
    return dict(zip(getattr(state, "name", []),
                    getattr(state, "position", [])))


# ---------------------------------------------------------------------------
# planned gripper goal -> a command, never an arm plan
# ---------------------------------------------------------------------------


def test_planned_tiago_gripper_open_is_a_command_not_an_arm_plan():
    robot = TiagoMock()
    spec = _spec(_current_state(), StageSpec(
        stage_type="move_to", name="open",
        params_yaml=(
            "goal:\n"
            "  joints:\n"
            f"    {LEFT}: 0.045\n"
            f"    {RIGHT}: 0.045\n"
        ),
    ))
    ex = TaskExecutor(spec, robot, task_id="open")
    assert ex.init(), ex.describe()["comment"]
    assert ex.plan(), ex.describe()["comment"]

    leaf = ex.flatten_leaves(ex.best())[-1]
    assert leaf.gripper_command is not None
    assert leaf.plan_request is None, "a gripper goal must not become an arm plan"
    assert robot.plan_calls == 0, "nothing may be solved for a gripper command"

    start = _joint_values(leaf.start.joint_state)
    end = _joint_values(leaf.end.joint_state)
    for name in TIAGO_CSPACE:
        assert start[name] == pytest.approx(TIAGO_ARM[name])
        assert end[name] == pytest.approx(TIAGO_ARM[name]), (
            "the arm must stay exactly where the previous stage left it")
    assert end[LEFT] == pytest.approx(0.045)
    assert end[RIGHT] == pytest.approx(0.045)

    ex.execute(ex.best())
    assert robot.executed == [], "no trajectory may be driven"
    assert robot.gripper_commands[-1]["position"] == pytest.approx(0.045)
    assert robot.gripper_commands[-1]["gripper"] == "gripper"


def test_named_gripper_config_is_also_a_command_not_an_arm_plan():
    robot = TiagoMock()
    spec = _spec(_current_state(), StageSpec(
        stage_type="move_to", name="open",
        params_yaml="goal:\n  name: open\n"))
    ex = TaskExecutor(spec, robot, task_id="open")
    assert ex.init() and ex.plan()
    leaf = ex.flatten_leaves(ex.best())[-1]
    assert leaf.gripper_command is not None
    assert leaf.plan_request is None
    assert robot.plan_calls == 0
    end = _joint_values(leaf.end.joint_state)
    assert end[LEFT] == pytest.approx(0.045)
    assert end[RIGHT] == pytest.approx(0.045)


def test_planned_gripper_open_drives_only_the_gripper_over_the_wire():
    """The serialized solution carries a gripper segment, not a trajectory."""
    from curobo_task_constructor.executor import drive_solution_dict

    robot = TiagoMock()
    spec = _spec(_current_state(), StageSpec(
        stage_type="move_to", name="open",
        params_yaml="goal:\n  joints:\n"
                    f"    {LEFT}: 0.045\n"
                    f"    {RIGHT}: 0.045\n"))
    ex = TaskExecutor(spec, robot, task_id="open")
    assert ex.init() and ex.plan()
    d = ex.solution_to_dict(ex.best())

    segments = d["sub_trajectories"]
    assert len(segments) == 1
    command = segments[0]["replay_goal"]["gripper_command"]
    assert command["position"] == pytest.approx(0.045)
    assert segments[0]["trajectory"]["joint_names"] == []
    assert d["start_state"]["name"], "the task start state must be complete"

    robot2 = TiagoMock()
    results, failed, cancelled = drive_solution_dict(robot2, d)
    assert failed is None and not cancelled
    assert all(r.success for r in results)
    assert robot2.executed == [], "no arm trajectory may be replayed"
    assert robot2.gripper_commands[-1]["position"] == pytest.approx(0.045)
    assert robot2.gripper_commands[-1]["gripper"] == "gripper"


def test_gripper_stage_after_arm_motion_uses_the_previous_stage_state():
    """The chain's arm pose wins over the (older) measured snapshot."""
    robot = MockCuroboServer()
    target = [0.5] + [0.0] * 6
    spec = _spec(
        _current_state(),
        StageSpec(stage_type="move_to", name="arm",
                  params_yaml="{goal: {joints: [" +
                  ", ".join(str(v) for v in target) + "]}}"),
        StageSpec(stage_type="gripper_command", name="open",
                  params_yaml="joint: finger_joint\nposition: 0.7\n"),
    )
    ex = TaskExecutor(spec, robot, task_id="chain")
    assert ex.init(), ex.describe()["comment"]
    assert ex.plan(), ex.describe()["comment"]

    arm, open_leaf = ex.flatten_leaves(ex.best())[-2:]
    assert open_leaf.start is arm.end, "the command must chain off the arm leg"
    assert _joint_values(open_leaf.start.joint_state)["joint_1"] == pytest.approx(0.5)
    assert _joint_values(open_leaf.end.joint_state)["joint_1"] == pytest.approx(0.5)
    assert _joint_values(open_leaf.end.joint_state)["finger_joint"] == pytest.approx(0.7)

    ex.execute(ex.best())
    assert len(robot.executed) == 1, "only the arm leg drives a trajectory"
    assert robot.gripper_commands[-1]["position"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# the display carries every commanded jaw into the following motion
# ---------------------------------------------------------------------------


def test_solution_message_folds_all_gripper_joints_into_motion_points():
    """TIAGo's pal-gripper has two fingers, but the command names only one.

    The trajectory display must show both jaws open during the pre-grasp and
    closed after the close, not leave the second finger at the start value.
    """
    from curobo_task_constructor.msg_convert_to import _carry_gripper_state

    open_cmd = {
        "joint": LEFT,
        "position": 0.045,
        "joints": {LEFT: 0.045, RIGHT: 0.045},
    }
    close_cmd = {
        "joint": LEFT,
        "position": 0.0,
        "joints": {LEFT: 0.0, RIGHT: 0.0},
    }
    segments = [
        {"replay_goal": {"gripper_command": open_cmd},
         "gripper_state": {"name": [LEFT, RIGHT], "position": [0.045, 0.045]},
         "trajectory": {"joint_names": [], "points": [], "velocities": []}},
        {"replay_goal": {"start_pose": {}},
         "trajectory": {"joint_names": ["arm_1_joint"],
                        "points": [[0.1], [0.2]], "velocities": []}},
        {"replay_goal": {"gripper_command": close_cmd},
         "gripper_state": {"name": [LEFT, RIGHT], "position": [0.0, 0.0]},
         "trajectory": {"joint_names": [], "points": [], "velocities": []}},
        {"replay_goal": {"start_pose": {}},
         "trajectory": {"joint_names": ["arm_1_joint"],
                        "points": [[0.2], [0.3]], "velocities": []}},
    ]
    out = _carry_gripper_state(
        segments, ["arm_1_joint", LEFT, RIGHT], [0.0, 0.0, 0.0])
    pre = out[1]["trajectory"]
    retreat = out[3]["trajectory"]
    assert pre["joint_names"][-2:] == [LEFT, RIGHT]
    assert [p[-2:] for p in pre["points"]] == [[0.045, 0.045]] * 2
    assert [p[-2:] for p in retreat["points"]] == [[0.0, 0.0]] * 2
    # The command segment itself updates the display at that chain position.
    assert out[0]["trajectory"]["joint_names"] == [LEFT, RIGHT]
    assert out[0]["trajectory"]["points"] == [[0.045, 0.045]]


# ---------------------------------------------------------------------------
# a bare gripper command fills missing joints from the measured state
# ---------------------------------------------------------------------------


def test_gripper_command_fills_missing_arm_from_the_current_measurement():
    robot = TiagoMock()
    stage = GripperCommand("open")
    stage.robot = robot
    stage.params = {"gripper": "gripper", "joint": LEFT, "position": 0.045}
    # The state a standalone gripper plan may carry: the jaws only.
    state = InterfaceState(joint_state=JointStateStub(
        names=[LEFT], positions=[0.0]))
    stage.compute_forward(state)

    sol = stage.solutions[0]
    # ``start`` is the incoming interface (kept by identity for container
    # chaining); the state the stage WRITES downstream is ``end`` and must
    # carry the full arm pose plus both commanded jaws.
    assert sol.start is state
    end = _joint_values(sol.end.joint_state)
    for name in TIAGO_CSPACE:
        assert end[name] == pytest.approx(TIAGO_ARM[name])
    assert end[LEFT] == pytest.approx(0.045)
    assert end[RIGHT] == pytest.approx(0.045)


# ---------------------------------------------------------------------------
# the real TIAGo descriptor (when this copy carries it)
# ---------------------------------------------------------------------------


def _tiago_descriptor():
    for root in Path(__file__).resolve().parents:
        candidate = (root / "iki_tiago_curobo_config" / "config"
                     / "robots" / "tiago.yaml")
        if candidate.is_file():
            return candidate
    return None


def test_tiago_descriptor_represents_open_as_a_gripper_trajectory():
    path = _tiago_descriptor()
    if path is None:
        pytest.skip("TIAGo descriptor is not present in this robot copy")

    descriptor = yaml.safe_load(path.read_text())
    order = canonical_joint_order(str(path))
    assert order == TIAGO_CSPACE, (
        "the fingers are lock_joints, not planning DOFs")
    assert LEFT not in order and RIGHT not in order

    grippers = get_gripper_controllers(str(path))
    cfg = grippers["gripper"]
    assert cfg["type"] == "JointTrajectory"
    assert cfg["joints"] == [LEFT, RIGHT]
    assert cfg["command_joint"] == LEFT
    assert cfg["open"] > cfg["close"]
    # The descriptor's open value is what the task graph commands.
    assert descriptor["gripper_controllers"]["gripper"]["open"] == 0.045

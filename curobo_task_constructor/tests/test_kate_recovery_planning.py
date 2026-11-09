"""KATE recovery rungs plan their flat ``home`` goal.

KATE is a six-DOF Jaco2. Its jaws are driven by a dedicated gripper
controller and are NOT part of the planner cspace. ``recover_open_detach_home``
therefore chains ``GripperCommand(open) -> MoveTo(home)``, and the command
legitimately leaves the two commanded finger joints in the task state so the
ghost preview is truthful.

The planning request for ``home`` must still start from a cspace-sized vector.
This mock is KATE-shaped: six arm joints in ``_joint_order``, two non-cspace
finger joints in the descriptor's gripper map, and a ``plan()`` that enforces
the joint-space wire contract (start names are the cspace, target length is
the cspace length). Before the regression fix the open rung handed the solver
an eight-joint start for a six-joint goal.
"""

from __future__ import annotations

import pytest

import curobo_task_constructor.stages  # noqa: F401  (register builtins)
from curobo_task_constructor.core.robot import PlanResult
from curobo_task_constructor.core.state import InterfaceState, JointStateStub
from curobo_task_constructor.executor import TaskExecutor
from curobo_task_constructor.graph.spec import StageSpec
from curobo_task_constructor.stages.move_to import MoveTo
from tests.mock_curobo import MockCuroboServer

KATE_ARM = [
    "j2n6s200_joint_1",
    "j2n6s200_joint_2",
    "j2n6s200_joint_3",
    "j2n6s200_joint_4",
    "j2n6s200_joint_5",
    "j2n6s200_joint_6",
]
KATE_FINGERS = [
    "j2n6s200_joint_finger_1",
    "j2n6s200_joint_finger_2",
]
#: The descriptor's perception_pose, i.e. the recovery rungs' flat goal.
HOME = [2.86, 3.92, 5.744, 2.6, 0.55, 0.0]
START = [1.36, 2.66, 5.15, -3.03, 0.53, 2.15]


class KateRecoveryMock(MockCuroboServer):
    """The KATE surface the recovery stages touch, with a strict wire mock."""

    def __init__(self):
        super().__init__()
        self._joint_order = list(KATE_ARM)
        # CurrentState reads the normalized /joint_states cache: arm joints
        # only, exactly like the live KATE adapter.
        self._current = JointStateStub(names=list(KATE_ARM),
                                       positions=list(START))
        self._grippers = {
            "j2n6s200": {
                "type": "GripperCommand",
                "name": "j2n6s200_gripper",
                "action_ns": "gripper_cmd",
                "joints": list(KATE_FINGERS),
                "command_joint": KATE_FINGERS[0],
                "open": 0.0,
                "close": 1.3,
                "max_effort": 0.0,
                "timeout": 5.0,
            }
        }
        #: Every joint-space request this mock was asked to solve.
        self.joint_requests: list = []

    def gripper_controllers(self) -> dict:
        return dict(self._grippers)

    def gripper_joint(self, gripper: str = "") -> str:
        cfg = self._grippers.get(gripper) or next(
            iter(self._grippers.values()))
        return str(cfg.get("command_joint") or "")

    def plan(self, request):
        goalset = (request.goalsets or [None])[0]
        target = list(getattr(goalset, "target_joint_positions", None) or [])
        if target:
            start = getattr(request, "start_pose", None)
            names = list(getattr(start, "name", []) or [])
            positions = list(getattr(start, "position", []) or [])
            # KATE-shaped contract: a joint-space segment starts from the
            # six-joint cspace, in cspace order, with no extra lock joints.
            if names != KATE_ARM:
                return PlanResult(
                    False,
                    "joint goal received a non-cspace start state "
                    f"{names}; expected {KATE_ARM}")
            if len(positions) != len(KATE_ARM):
                return PlanResult(
                    False,
                    f"start has {len(positions)} positions, expected "
                    f"{len(KATE_ARM)}")
            if len(target) != len(KATE_ARM):
                return PlanResult(
                    False,
                    f"goal has {len(target)} positions, expected "
                    f"{len(KATE_ARM)}")
            self.joint_requests.append(request)
        return super().plan(request)


def _task(*children) -> StageSpec:
    return StageSpec(stage_type="", name="recover", container_type="serial",
                     children=list(children))


def _current_state() -> StageSpec:
    return StageSpec(stage_type="current_state", name="current_state")


def _detach() -> StageSpec:
    return StageSpec(stage_type="modify_scene", name="detach",
                     params_yaml="detach: object_0\n")


def _open() -> StageSpec:
    return StageSpec(
        stage_type="gripper_command", name="open",
        params_yaml=(
            "gripper: j2n6s200\n"
            f"joint: {KATE_FINGERS[0]}\n"
            "position: 0.0\n"))


def _home() -> StageSpec:
    return StageSpec(
        stage_type="move_to", name="home",
        params_yaml=(
            "goal:\n  joints:\n"
            + "".join(f"  - {value!r}\n" for value in HOME)
            + "cost: path_length\nplanner: 5\nplanning_attempts: 1\n"))


CASES = {
    "recover_home": [_current_state(), _home()],
    "recover_detach_home": [_current_state(), _detach(), _home()],
    "recover_open_detach_home": [
        _current_state(), _detach(), _open(), _home()],
}


@pytest.mark.parametrize("kind", list(CASES))
def test_kate_recovery_rung_plans_home_from_the_cspace(kind):
    robot = KateRecoveryMock()
    ex = TaskExecutor(_task(*CASES[kind]), robot, task_id=kind)
    assert ex.init(), ex.describe()["comment"]
    assert ex.plan(), ex.describe()["comment"]

    home = next(leaf for leaf in ex.flatten_leaves(ex.best())
                if leaf.stage.name == "home")
    request = home.plan_request
    assert list(request.start_pose.name) == KATE_ARM
    assert list(request.goalsets[0].target_joint_positions) == pytest.approx(HOME)


def test_kate_joint_goal_fills_missing_arm_joints_from_the_measurement():
    """A state carrying only the jaws still plans a full-cspace home goal."""
    robot = KateRecoveryMock()
    stage = MoveTo("home")
    stage.robot = robot
    stage.params = {
        "goal": {"joints": list(HOME)},
        "planner": 5,
        "planning_attempts": 1,
    }
    state = InterfaceState(joint_state=JointStateStub(
        names=list(KATE_FINGERS), positions=[0.0, 0.0]))
    stage.compute_forward(state)

    assert stage.solutions, "home must plan after filling the missing arm"
    request = stage.solutions[0].plan_request
    assert list(request.start_pose.name) == KATE_ARM


def test_kate_open_rung_keeps_the_jaws_in_the_chained_state():
    """Only the planning start is projected; the task state stays complete."""
    robot = KateRecoveryMock()
    ex = TaskExecutor(_task(*CASES["recover_open_detach_home"]), robot,
                      task_id="recover_open_detach_home")
    assert ex.init() and ex.plan()
    open_leaf, home = ex.flatten_leaves(ex.best())[-2:]
    assert open_leaf.stage.name == "open"
    assert home.stage.name == "home"

    # The gripper command wrote both jaws into the chain state, and the motion
    # leg still carries that full state for the next stage / ghost preview.
    end = dict(zip(open_leaf.end.joint_state.name,
                   open_leaf.end.joint_state.position))
    assert end[KATE_FINGERS[0]] == pytest.approx(0.0)
    assert end[KATE_FINGERS[1]] == pytest.approx(0.0)
    chained = set(home.start.joint_state.name)
    assert set(KATE_ARM) <= chained
    assert set(KATE_FINGERS) <= chained

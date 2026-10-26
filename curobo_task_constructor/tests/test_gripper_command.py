"""GripperCommand — the jaws are executed, never planned.

Mirrors MoveIt: a gripper stage carries a joint target only, and execution
sends it to the dedicated GripperCommand controller instead of a trajectory.
"""

from dataclasses import dataclass, field

from curobo_task_constructor.core.robot import PlanResult
from curobo_task_constructor.executor import TaskExecutor, drive_solution_dict
from curobo_task_constructor.mtc import core, stages
from tests.mock_curobo import MockCuroboServer


@dataclass
class FakeRobot:
    commands: list = field(default_factory=list)
    driven: list = field(default_factory=list)

    def gripper_command(self, **cmd):
        self.commands.append(dict(cmd))
        return PlanResult(True, "gripper")

    def execute(self, request, trajectory):
        self.driven.append(request)
        return PlanResult(True, "ok")

    def add_object(self, payload):
        pass

    def remove_object(self, payload):
        pass

    def remove_all_objects(self):
        pass

    def detach_object(self, payload=None):
        pass

    def attach_object(self, payload):
        pass


def _gripper_task() -> core.Task:
    task = core.Task(name="grip")
    task.add(stages.CurrentState("current_state"))
    close = stages.GripperCommand("close")
    close.setGripper("left")
    close.setJoint("finger_joint")
    close.setPosition(1.3)
    task.add(close)
    return task


def test_gripper_stage_executes_without_a_plan_request():
    robot = MockCuroboServer()
    ex = TaskExecutor(_gripper_task().to_spec(), robot)
    assert ex.init()
    assert ex.plan()
    leaves = ex.flatten_leaves(ex.best())
    assert leaves and leaves[-1].gripper_command["position"] == 1.3
    assert leaves[-1].gripper_command["gripper"] == "left"
    assert leaves[-1].plan_request is None
    # The command updates the chain state too: a motion leg planned after it
    # must start from the commanded jaw value, not the planner's lock value.
    end = ex.best().end.joint_state
    assert dict(zip(end.name, end.position))["finger_joint"] == 1.3
    ex.execute(ex.best())
    assert robot.gripper_commands
    assert robot.gripper_commands[-1]["joint"] == "finger_joint"
    assert robot.gripper_commands[-1]["gripper"] == "left"


def test_drive_solution_dict_routes_gripper_segment_without_execute():
    robot = FakeRobot()
    d = {"sub_trajectories": [{
        "info": {"id": 1, "cost": 0.0, "comment": "", "stage_id": 1,
                 "planner_id": ""},
        "execution_info": {"controller_names": ["j2n6s200_gripper"]},
        "trajectory": {"joint_names": [], "points": []},
        "scene_diff": {"before": {"added": [], "removed": [], "attached": None,
                                  "detached": None},
                       "after": {"added": [], "removed": [], "attached": None,
                                 "detached": None}},
        "replay_goal": {"gripper_command": {"joint": "finger_joint",
                                            "position": 0.0}},
    }]}
    results, failed, cancelled = drive_solution_dict(robot, d)
    assert failed is None and not cancelled
    assert robot.driven == []
    assert robot.commands == [{"joint": "finger_joint", "position": 0.0}]

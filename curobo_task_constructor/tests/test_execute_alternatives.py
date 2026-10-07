"""Executing different alternatives (MTC ExecuteTaskSolution mechanic).

Any ranked complete solution drives with no replanning: the executor replays
the chain's own segment requests. The panel lists the ranked chains and sends
whichever one is selected to ExecuteSolution.
"""

import threading

import curobo_task_constructor.stages  # noqa: F401  (register builtins)
from curobo_task_constructor.executor import TaskExecutor
from curobo_task_constructor.graph.spec import StageSpec
from tests.mock_curobo import JOINT_NAMES, MockCuroboServer


def _spec():
    def _move(name, joints):
        return {"stage_type": "move_to", "name": name, "container_type": "",
                "params_yaml": f"{{goal: {{joints: [{joints}]}}}}",
                "children": []}

    return StageSpec.from_dict({
        "stage_type": "serial", "name": "t", "container_type": "serial",
        "params_yaml": "",
        "children": [
            {"stage_type": "current_state", "name": "s", "container_type": "",
             "params_yaml": "{}", "children": []},
            {"stage_type": "", "name": "choice", "container_type": "alternatives",
             "params_yaml": "",
             "children": [
                 _move("left",
                       ", ".join(["0.5"] + ["0.0"] * 6)),
                 _move("right",
                       ", ".join(["-0.5"] + ["0.0"] * 6)),
             ]},
        ],
    })


def _executor(robot=None):
    robot = robot or MockCuroboServer()
    ex = TaskExecutor(_spec(), robot, task_id="t")
    assert ex.init()
    assert ex.plan()
    return robot, ex


def test_two_alternatives_rank():
    _, ex = _executor()
    assert len(ex.rank()) == 2


def test_executing_the_second_alternative_drives_its_plan():
    robot, ex = _executor()
    sols = ex.rank()
    results = ex.execute(sols[1])
    assert results and all(r.success for r in results)
    driven = robot.executed[-1]
    goal = driven.goalsets[0].target_joint_positions
    assert goal[0] == sols[1].end.joint_state.position[0]


def test_executing_the_first_alternative_is_independent():
    robot, ex = _executor()
    sols = ex.rank()
    ex.execute(sols[0])
    goal = robot.executed[-1].goalsets[0].target_joint_positions
    assert goal[0] == sols[0].end.joint_state.position[0]


def test_cancel_stops_the_chain_between_segments():
    def _two_moves():
        def _move(name, joints):
            return {"stage_type": "move_to", "name": name,
                    "container_type": "",
                    "params_yaml": f"{{goal: {{joints: [{joints}]}}}}",
                    "children": []}

        return StageSpec.from_dict({
            "stage_type": "serial", "name": "t", "container_type": "serial",
            "params_yaml": "",
            "children": [
                {"stage_type": "current_state", "name": "s",
                 "container_type": "", "params_yaml": "{}", "children": []},
                _move("first", ", ".join(["0.5"] + ["0.0"] * 6)),
                _move("second", ", ".join(["-0.5"] + ["0.0"] * 6)),
            ],
        })

    robot = MockCuroboServer()
    ex = TaskExecutor(_two_moves(), robot, task_id="t")
    assert ex.init()
    assert ex.plan()
    sol = ex.best()
    assert sol is not None
    event = threading.Event()
    seen = []
    results = ex.execute(sol, progress_callback=lambda name: (
        seen.append(name), event.set()), cancel_event=event)
    assert seen == ["first"]  # second segment never drove
    assert len(results) == 1
    assert len(robot.executed) == 1

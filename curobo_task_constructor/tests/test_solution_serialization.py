"""Solution serialization (MTC Solution.toMsg mechanic).

A planned solution converts to plain dicts with MTC Solution field names
(``solution_to_dict``), survives a YAML round-trip (the
``replay_goal_yaml`` wire form), and drives on a FRESH executor — the
execute-server situation in another process — via ``execute_dict``,
replaying the validated plan instead of re-solving.
"""

import yaml

import curobo_task_constructor.stages  # noqa: F401  (register builtins)
from curobo_task_constructor.executor import TaskExecutor
from curobo_task_constructor.graph.spec import StageSpec
from tests.mock_curobo import MockCuroboServer


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
            _move("m1", ", ".join(["0.5"] + ["0.0"] * 6)),
            _move("m2", ", ".join(["-0.5"] + ["0.0"] * 6)),
        ],
    })


def _planned():
    robot = MockCuroboServer()
    ex = TaskExecutor(_spec(), robot, task_id="t")
    assert ex.init()
    assert ex.plan()
    return robot, ex


def test_solution_to_dict_has_mtc_fields():
    _, ex = _planned()
    d = ex.solution_to_dict(ex.best())
    assert d["task_id"] == "t"
    assert isinstance(d["sub_solution"], list) and d["sub_solution"]
    assert isinstance(d["sub_trajectories"], list) and d["sub_trajectories"]
    for sub in d["sub_solution"]:
        assert set(sub) == {"info", "sub_solution_id"}
        assert set(sub["info"]) == {"id", "cost", "comment", "stage_id",
                                    "planner_id"}
    for traj in d["sub_trajectories"]:
        assert set(traj) == {"info", "execution_info", "trajectory",
                             "scene_diff", "replay_goal"}
        assert traj["trajectory"]["points"]
        assert traj["replay_goal"]["goalsets"]


def test_sub_solution_tree_mirrors_children():
    _, ex = _planned()
    d = ex.solution_to_dict(ex.best())
    # root entry first, with the whole chain below it
    assert d["sub_solution"][0]["sub_solution_id"]
    leaves = ex.flatten_leaves(ex.best())
    assert len(d["sub_trajectories"]) == sum(
        1 for leaf in leaves if leaf.plan_request is not None)


def test_yaml_round_trip_and_execute_on_fresh_executor():
    _, ex = _planned()
    d = ex.solution_to_dict(ex.best())
    wire = yaml.safe_dump(d, default_flow_style=False)
    back = yaml.safe_load(wire)

    robot2 = MockCuroboServer()
    ex2 = TaskExecutor(_spec(), robot2, task_id="t")
    assert ex2.init()
    names = []
    results, failed = ex2.execute_dict(
        back, progress_callback=names.append)
    assert failed == ""
    assert results and all(r.success for r in results)
    # same motion leaves drove, in chain order
    assert names == ["m1", "m2"]
    goals = [r.goalsets[0].target_joint_positions[0]
             for r in robot2.executed]
    assert goals == [0.5, -0.5]


def test_execute_dict_applies_scene_ops():
    robot = MockCuroboServer()
    spec = StageSpec.from_dict({
        "stage_type": "serial", "name": "t", "container_type": "serial",
        "params_yaml": "",
        "children": [
            {"stage_type": "current_state", "name": "s", "container_type": "",
             "params_yaml": "{}", "children": []},
            {"stage_type": "modify_scene", "name": "add",
             "container_type": "",
             "params_yaml": ("{add: {name: box, shape: cuboid, "
                             "pose: {x: 0.5, y: 0.0, z: 0.25}, "
                             "dimensions: [0.1, 0.1, 0.1]}}"),
             "children": []},
            {"stage_type": "move_to", "name": "m", "container_type": "",
             "params_yaml": "{goal: {joints: [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}}",
             "children": []},
        ],
    })
    ex = TaskExecutor(spec, robot, task_id="t")
    assert ex.init()
    assert ex.plan()
    d = ex.solution_to_dict(ex.best())

    robot2 = MockCuroboServer()
    ex2 = TaskExecutor(spec, robot2, task_id="t")
    assert ex2.init()
    results, failed = ex2.execute_dict(d)
    assert failed == "" and all(r.success for r in results)
    assert ("add", "box") in robot2.world_ops

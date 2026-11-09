"""Executing a selected stage solution must drive only the chain PREFIX.

The RViz panel caches each broadcast Solution under every sub-solution id;
that payload is the whole task, so sending it to ExecuteTaskSolution always
replayed every segment regardless of the selected row. GetSolution now
returns the prefix rooted at the requested id (server-side truncation), and
the panel executes that payload. These tests cover the converter and the
service callback without ROS.
"""

import curobo_task_constructor.stages  # noqa: F401  (register builtins)
from curobo_task_constructor import msg_convert
from curobo_task_constructor.executor import TaskExecutor, drive_solution_dict
from curobo_task_constructor.graph.spec import StageSpec
from curobo_task_constructor.mtc import core, stages
from tests.mock_curobo import MockCuroboServer


def _task(robot=None):
    robot = robot or MockCuroboServer()
    task = core.Task(robot, name="prefix", task_id="prefix")
    task.add(stages.CurrentState("current state"))
    for name, joint_2 in (("m1", 0.4), ("m2", -0.3), ("m3", 0.2)):
        move = stages.MoveTo(name, core.JointInterpolationPlanner())
        move.setGoal({"joint_2": joint_2})
        task.add(move)
    assert task.plan()
    return task


def _find(node, name):
    if getattr(getattr(node, "stage", None), "name", None) == name:
        return node
    for child in (getattr(node, "children", None) or []):
        hit = _find(child, name)
        if hit is not None:
            return hit
    return None


def _segment_ids(d):
    return [int((seg.get("info", None) or {}).get("id", 0) or 0)
            for seg in (d.get("sub_trajectories", None) or [])]


def test_converter_truncates_chain_to_selected_stage():
    task = _task()
    ex = task.executor
    root = ex.best()
    d = ex.solution_to_dict(root)
    m1 = _find(root, "m1")
    m2 = _find(root, "m2")
    m3 = _find(root, "m3")
    assert m1 is not None and m2 is not None and m3 is not None

    m2_id = ex.global_solution_id(m2)
    prefix = msg_convert.solution_prefix_dict(d, m2_id)
    assert _segment_ids(prefix) == [
        ex.global_solution_id(m1), ex.global_solution_id(m2)]
    # The solution tree is pruned too: m3 is not announced.
    announced = {int((sub.get("info", None) or {}).get("id", 0) or 0)
                 for sub in prefix["sub_solution"]}
    assert ex.global_solution_id(m2) in announced
    assert ex.global_solution_id(m3) not in announced

    # Selecting the root keeps the complete task (same object, no copy).
    assert msg_convert.solution_prefix_dict(
        d, ex.global_solution_id(root)) is d
    # An id from a different plan/tree is not silently truncated.
    assert msg_convert.solution_prefix_dict(d, 999999) is d


def test_getsolution_returns_prefix_for_intermediate_row():
    task = _task()
    ex = task.executor
    root = ex.best()
    m2 = _find(root, "m2")
    m2_id = ex.global_solution_id(m2)

    class _Req:
        solution_id = m2_id

    class _Resp:
        solution = None

    original = msg_convert.solution_to_msg
    msg_convert.solution_to_msg = lambda d, msg_cls=None: d
    try:
        response = task._on_get_solution(_Req(), _Resp())
    finally:
        msg_convert.solution_to_msg = original
    assert response.solution is not None
    assert len(response.solution["sub_trajectories"]) == 2


def _driven_joint_2(robot):
    """Final joint_2 of every replayed trajectory, in drive order."""
    out = []
    for traj in robot.replayed:
        names = list(traj.get("joint_names", None) or [])
        points = list(traj.get("points", None) or [])
        if not points or "joint_2" not in names:
            continue
        out.append(round(points[-1][names.index("joint_2")], 6))
    return out


def test_drive_prefix_executes_only_prefix_segments():
    task = _task()
    ex = task.executor
    root = ex.best()
    d = ex.solution_to_dict(root)
    m2_id = ex.global_solution_id(_find(root, "m2"))

    prefix = msg_convert.solution_prefix_dict(d, m2_id)
    robot = MockCuroboServer()
    results, failed, cancelled = drive_solution_dict(robot, prefix)
    assert failed is None and cancelled is False
    assert _driven_joint_2(robot) == [0.4, -0.3]

    # Root selection still drives every segment.
    robot = MockCuroboServer()
    results, failed, cancelled = drive_solution_dict(robot, d)
    assert failed is None and cancelled is False
    assert _driven_joint_2(robot) == [0.4, -0.3, 0.2]


def test_prefix_keeps_selected_scene_leaf_between_motions():
    """m1 -> ModifyScene(add box) -> m2, select the ModifyScene solution.

    Its ops are folded into m2's ``scene_diff.before`` by the serializer.
    The prefix must keep that add (as m1's ``after``) while dropping m2.
    """
    spec = StageSpec.from_dict({
        "stage_type": "serial", "name": "t", "container_type": "serial",
        "params_yaml": "",
        "children": [
            {"stage_type": "current_state", "name": "s",
             "container_type": "", "params_yaml": "{}", "children": []},
            {"stage_type": "move_to", "name": "m1", "container_type": "",
             "params_yaml": "{goal: {joints: [0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}}",
             "children": []},
            {"stage_type": "modify_scene", "name": "add_box",
             "container_type": "",
             "params_yaml": ("{add: {name: box, shape: cuboid, "
                             "pose: {x: 0.5, y: 0.0, z: 0.25}, "
                             "dimensions: [0.1, 0.1, 0.1]}}"),
             "children": []},
            {"stage_type": "move_to", "name": "m2", "container_type": "",
             "params_yaml": "{goal: {joints: [-0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}}",
             "children": []},
        ],
    })
    ex = TaskExecutor(spec, MockCuroboServer(), task_id="t")
    assert ex.init()
    assert ex.plan()
    d = ex.solution_to_dict(ex.best())
    add_stage = next(st for st in ex.root.subtree_stages()
                     if st.name == "add_box")
    assert add_stage.solutions
    prefix = msg_convert.solution_prefix_dict(
        d, ex.global_solution_id(add_stage.solutions[0]))
    # m1 still drives; the selected scene leaf's add rides on its "after".
    assert len(prefix["sub_trajectories"]) == 1

    robot = MockCuroboServer()
    results, failed, cancelled = drive_solution_dict(robot, prefix)
    assert failed is None and cancelled is False
    assert ("add", "box") in robot.world_ops
    # Only m1 was driven: the prefix stops before m2.
    assert len(robot.replayed) == 1
    assert [int(seg["info"]["id"]) for seg in prefix["sub_trajectories"]] == [
        int(d["sub_trajectories"][0]["info"]["id"])]


def test_prefix_drops_scene_ops_belonging_to_later_stages():
    spec = StageSpec.from_dict({
        "stage_type": "serial", "name": "t", "container_type": "serial",
        "params_yaml": "",
        "children": [
            {"stage_type": "current_state", "name": "s",
             "container_type": "", "params_yaml": "{}", "children": []},
            {"stage_type": "move_to", "name": "m1", "container_type": "",
             "params_yaml": "{goal: {joints: [0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}}",
             "children": []},
            {"stage_type": "modify_scene", "name": "add", "container_type": "",
             "params_yaml": ("{add: {name: box, shape: cuboid, "
                             "pose: {x: 0.5, y: 0.0, z: 0.25}, "
                             "dimensions: [0.1, 0.1, 0.1]}}"),
             "children": []},
        ],
    })
    ex = TaskExecutor(spec, MockCuroboServer(), task_id="t")
    assert ex.init()
    assert ex.plan()
    d = ex.solution_to_dict(ex.best())
    m1 = next(st for st in ex.root.subtree_stages() if st.name == "m1")
    prefix = msg_convert.solution_prefix_dict(
        d, ex.global_solution_id(m1.solutions[0]))
    assert len(prefix["sub_trajectories"]) == 1
    assert not (prefix["sub_trajectories"][0]["scene_diff"].get(
        "after", {}).get("added"))

    robot = MockCuroboServer()
    results, failed, cancelled = drive_solution_dict(robot, prefix)
    assert failed is None and cancelled is False
    assert ("add", "box") not in robot.world_ops

    robot = MockCuroboServer()
    results, failed, cancelled = drive_solution_dict(robot, d)
    assert failed is None and cancelled is False
    assert ("add", "box") in robot.world_ops

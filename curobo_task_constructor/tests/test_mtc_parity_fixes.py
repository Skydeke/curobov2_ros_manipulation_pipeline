"""Parity fixes: MoveItErrorCode returns, ns topics, per-group planner loader."""
from curobo_task_constructor.mtc import core, stages
from tests.mock_curobo import MockCuroboServer


def _task():
    t = core.Task(MockCuroboServer())
    t.add(stages.CurrentState("c"))
    m = stages.MoveTo("m", core.JointInterpolationPlanner())
    m.setGoal({"joint_1": 0.1})
    t.add(m)
    return t


def test_plan_returns_moveit_error_code_truthy():
    r = _task().plan()
    assert isinstance(r, core.MoveItErrorCode)
    assert r.val == core.MoveItErrorCode.SUCCESS
    assert bool(r)


def test_execute_offline_returns_moveit_error_code():
    t = _task()
    assert t.plan()
    r = t.execute()
    assert isinstance(r, core.MoveItErrorCode)
    assert bool(r) and r.val == core.MoveItErrorCode.SUCCESS


def test_ns_topics_and_service():
    t = core.Task(MockCuroboServer(), ns="my_ns")
    assert t.description_topic() == "/my_ns/task_description"
    assert t.statistics_topic() == "/my_ns/task_statistics"
    assert t.solution_topic() == "/my_ns/solution"
    assert t.execute_action_name() == "/my_ns/execute_task_solution"
    assert t.introspection().get_solution_service_name() == "/my_ns/get_solution_task"


def test_load_planner_configs_jaco_shape(tmp_path):
    p = tmp_path / "groups.yaml"
    p.write_text(
        "planning_groups: ['arm', 'gripper']\n"
        "arm:\n  planning_pipeline_id: ompl\n  planner_id: RRTConnectkConfigDefault\n"
        "  planning_time: 15.0\n  planning_attempts: 2\n"
        "  max_velocity_scaling_factor: 0.5\n"
        "gripper:\n  planning_pipeline_id: ompl\n  planner_id: ''\n")
    cfgs = core.load_planner_configs(str(p))
    assert cfgs["arm"].pipeline == "ompl"
    assert cfgs["arm"].planner_id == "RRTConnectkConfigDefault"
    assert cfgs["arm"].num_planning_attempts == 2
    assert cfgs["gripper"].planner_id == ""
    assert cfgs["arm"].planner_key() == core.PLANNER_JOINT_SPACE

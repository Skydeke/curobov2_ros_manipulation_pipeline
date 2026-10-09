"""demo_ws-style configuration parity (MTCManager::createTask shape).

The old pipeline configures planners and stages with the MTC call shape
(PipelinePlanner(node, pipeline, id), setProperty/setTimeout/
setMaxVelocityScalingFactor, stage setGroup/setGoal/setTimeout/
setPathConstraints/setCostTerm, Connect planners list, GeneratePose +
ComputeIK wrappers). These tests pin that every one of those calls exists
and lands where the backend reads it.
"""

import types

from curobo_task_constructor.mtc import core, stages

from tests.mock_curobo import MockCuroboServer


def _pose_stamped(x=0.4, y=0.0, z=0.3):
    header = types.SimpleNamespace(frame_id="base_link")
    pose = types.SimpleNamespace(
        position=types.SimpleNamespace(x=x, y=y, z=z),
        orientation=types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0))
    return types.SimpleNamespace(header=header, pose=pose)


def _node():
    return types.SimpleNamespace()


def test_pipeline_planner_mtc_call_shape():
    ompl = core.PipelinePlanner(_node(), "ompl", "RRTConnectkConfigDefault")
    assert (ompl.pipeline, ompl.planner_id) == ("ompl", "RRTConnectkConfigDefault")
    ompl.setProperty("num_planning_attempts", 2)
    assert ompl.num_planning_attempts == 2
    ompl.setMaxVelocityScalingFactor(0.5)
    ompl.setMaxAccelerationScalingFactor(0.5)
    assert ompl.max_velocity_scaling_factor == 0.5
    assert ompl.max_acceleration_scaling_factor == 0.5
    ompl.init(object())
    assert ompl.getPlannerId() == "RRTConnectkConfigDefault"


def test_cartesian_planner_mtc_call_shape():
    cart = core.CartesianPath()
    cart.setMaxVelocityScalingFactor(0.3)
    cart.setMaxAccelerationScalingFactor(0.3)
    cart.setPrecision(core.CartesianPrecision(0.01, 0.05, 1e-3))
    cart.setProperty("jump_threshold", 0.2)
    cart.setMinFraction(1.0)
    cart.setIKFrame("grasping_frame")
    assert cart.min_fraction == 1.0
    assert cart.getPlannerId() == "CartesianPath"


def test_move_to_mtc_call_shape():
    ompl = core.PipelinePlanner(_node(), "ompl", "RRTConnectkConfigDefault")
    move = stages.MoveTo("move_to_home", ompl)
    move.setGroup("manipulator")
    move.setGoal("home")
    move.setProperty("goal_position_tolerance", 0.025)
    move.setProperty("goal_orientation_tolerance", 0.01)
    move.setProperty("goal_joint_tolerance", 0.025)
    move.setTimeout(15.0)
    move.setPathConstraints({"position_constraints": []})
    move.setCostTerm("path_length")
    assert move.group == "manipulator"
    assert move.timeout == 15.0
    assert move.properties["goal_position_tolerance"] == 0.025
    assert move.path_constraints == {"position_constraints": []}


def test_connect_mtc_call_shape():
    ompl = core.PipelinePlanner(_node(), "ompl", "RRTConnectkConfigDefault")
    connect = stages.Connect("plan to IK", [("manipulator", ompl)])
    connect.setTimeout(10.0)
    connect.setProperty("goal_joint_tolerance", 0.05)
    connect.setProperty("goal_position_tolerance", 0.02)
    connect.setProperty("goal_orientation_tolerance", 0.1)
    connect.setPathConstraints({"position_constraints": []})
    connect.setCostTerm("path_length")
    connect.properties().configureInitFrom(1, ["group", "eef"])
    assert connect.timeout == 10.0
    assert connect.params["groups"] == ["manipulator"]


def test_generate_pose_compute_ik_mtc_call_shape():
    gen = stages.GeneratePose("target pose")
    gen.setMonitoredStage(object())
    gen.setPose(_pose_stamped())
    assert gen.params["pose"]["x"] == 0.4
    ik = stages.ComputeIK("IK", gen)
    ik.setGroup("manipulator")
    ik.setIKFrame("grasping_frame")
    ik.setTargetPose(_pose_stamped())
    ik.setMaxIKSolutions(20)
    ik.setTimeout(3.0)
    ik.setIgnoreCollisions(False)
    spec = ik.to_spec()
    assert spec.stage_type == "compute_ik"
    assert len(spec.children) == 1
    assert spec.children[0].stage_type == "generate_pose"


def test_generate_pose_compute_ik_plans():
    task = core.Task(MockCuroboServer())
    gen = stages.GeneratePose("target pose")
    gen.params["pose"] = {"x": 0.3, "y": 0.0, "z": 0.3,
                          "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
    task.add(stages.ComputeIK("IK", gen))
    assert task.plan()
    assert len(task.solutions) == 1


def test_move_gripper_mtc_call_shape():
    interp = core.JointInterpolationPlanner()
    move = stages.MoveTo("open_gripper", interp)
    move.setGroup("gripper")
    move.setGoal("open")
    move.setTimeout(5.0)
    assert move.group == "gripper"
    assert move.params["goal"] == {"name": "open"}


def test_modify_scene_mtc_call_shape():
    mod = stages.ModifyPlanningScene()
    group = types.SimpleNamespace(
        get_joint_model_names=lambda: ["finger_1", "finger_2"])
    mod.allowCollisions("object_0", group, True)
    mod.attachObject("object_0", "grasping_frame")
    assert mod.params["allow_collisions"]["links"] == ["finger_1", "finger_2"]
    mod2 = stages.ModifyPlanningScene()
    mod2.allowCollisions("object_0", group, False)
    mod2.detachObject("object_0", "grasping_frame")
    mod2.removeObject("object_0")
    assert mod2.params["detach"] == "object_0"
    assert mod2.params["remove"] == "object_0"

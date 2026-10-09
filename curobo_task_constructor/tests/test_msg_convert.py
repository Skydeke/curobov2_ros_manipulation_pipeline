"""msg_convert round-trips with duck-typed stand-in messages.

The real ROS messages are unavailable here; these tests install stub
modules (the test_viz_publish pattern) shaped like the MTC-identical
interface messages and verify dict -> msg -> dict preserves everything
the execute server needs (waypoints, replay goal, scene ops).
"""

import sys
import types

import curobo_task_constructor.stages  # noqa: F401  (register builtins)
from curobo_task_constructor.executor import TaskExecutor
from curobo_task_constructor.graph.spec import StageSpec
from tests.mock_curobo import MockCuroboServer


def _install_stubs():
    mods = {}

    def _mod(name):
        m = types.ModuleType(name)
        sys.modules[name] = m
        mods[name] = m
        return m

    class _List(list):
        def append(self, v):
            super().append(v)

    def _msg(**fields):
        obj = types.SimpleNamespace()
        for key, value in fields.items():
            setattr(obj, key, value)
        return obj

    # -- curobo_task_constructor_interfaces.msg --
    ifaces = _mod("curobo_task_constructor_interfaces")
    imsg = _mod("curobo_task_constructor_interfaces.msg")

    class _SolutionInfo:
        def __init__(self):
            self.id = 0
            self.cost = float("inf")
            self.comment = ""
            self.stage_id = 0
            self.planner_id = ""
            self.markers = _List()

    class _ExecInfo:
        def __init__(self):
            self.controller_names = _List()

    class _SubTraj:
        def __init__(self):
            self.info = _SolutionInfo()
            self.execution_info = _ExecInfo()
            self.trajectory = None
            self.scene_diff = None
            self.replay_goal_yaml = ""

    class _SubSol:
        def __init__(self):
            self.info = _SolutionInfo()
            self.sub_solution_id = _List()

    class _Solution:
        def __init__(self):
            self.task_id = ""
            self.start_scene = None
            self.sub_solution = _List()
            self.sub_trajectory = _List()

    imsg.SolutionInfo = _SolutionInfo
    imsg.SubTrajectory = _SubTraj
    imsg.SubSolution = _SubSol
    imsg.Solution = _Solution
    ifaces.msg = imsg

    # -- trajectory_msgs --
    traj = _mod("trajectory_msgs")
    tmsg = _mod("trajectory_msgs.msg")

    class _Time:
        def __init__(self):
            self.sec = 0
            self.nanosec = 0

    class _Point:
        def __init__(self):
            self.positions = []
            self.time_from_start = _Time()

    class _JointTraj:
        def __init__(self):
            self.joint_names = []
            self.points = _List()

    tmsg.JointTrajectory = _JointTraj
    tmsg.JointTrajectoryPoint = _Point
    traj.msg = tmsg

    # -- moveit_msgs --
    moveit = _mod("moveit_msgs")
    mmsg = _mod("moveit_msgs.msg")

    class _RobotTraj:
        def __init__(self):
            self.joint_trajectory = _JointTraj()
            self.multi_dof_joint_trajectory = _msg(joint_names=[])

    class _CollisionObject:
        ADD = 0
        REMOVE = 2

        def __init__(self):
            self.id = ""
            self.operation = 0
            self.primitives = _List()
            self.primitive_poses = _List()

    class _Attached:
        def __init__(self):
            self.object = _CollisionObject()
            self.link_name = ""

    class _PlanningScene:
        def __init__(self):
            self.world = _msg(collision_objects=_List())
            self.robot_state = _msg(attached_collision_objects=_List())
            self.is_diff = False

    mmsg.RobotTrajectory = _RobotTraj
    mmsg.CollisionObject = _CollisionObject
    mmsg.AttachedCollisionObject = _Attached
    mmsg.PlanningScene = _PlanningScene
    moveit.msg = mmsg

    # -- geometry_msgs / shape_msgs --
    geo = _mod("geometry_msgs")
    gmsg = _mod("geometry_msgs.msg")

    class _Pose:
        def __init__(self):
            self.position = _msg(x=0.0, y=0.0, z=0.0)
            self.orientation = _msg(x=0.0, y=0.0, z=0.0, w=1.0)

    gmsg.Pose = _Pose
    geo.msg = gmsg

    shape = _mod("shape_msgs")
    smsg = _mod("shape_msgs.msg")

    class _Prim:
        BOX = 1
        SPHERE = 2
        CYLINDER = 3

        def __init__(self):
            self.type = 1
            self.dimensions = []

    smsg.SolidPrimitive = _Prim
    shape.msg = smsg
    return mods


def _remove_stubs(mods):
    for name in mods:
        sys.modules.pop(name, None)


def _spec():
    return StageSpec.from_dict({
        "stage_type": "serial", "name": "t", "container_type": "serial",
        "params_yaml": "",
        "children": [
            {"stage_type": "current_state", "name": "s", "container_type": "",
             "params_yaml": "{}", "children": []},
            {"stage_type": "move_to", "name": "m", "container_type": "",
             "params_yaml": "{goal: {joints: [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}}",
             "children": []},
        ],
    })


def test_solution_round_trip_through_messages():
    mods = _install_stubs()
    try:
        from curobo_task_constructor import msg_convert
        robot = MockCuroboServer()
        ex = TaskExecutor(_spec(), robot, task_id="t")
        assert ex.init()
        assert ex.plan()
        d = ex.solution_to_dict(ex.best())

        msg = msg_convert.solution_to_msg(d)
        assert msg.task_id == "t"
        assert len(msg.sub_trajectory) == 1
        sub = msg.sub_trajectory[0]
        assert len(sub.trajectory.joint_trajectory.points) > 0
        assert sub.replay_goal_yaml.strip()

        back = msg_convert.msg_to_solution_dict(msg)
        assert back["task_id"] == "t"
        assert len(back["sub_trajectories"]) == 1
        seg = back["sub_trajectories"][0]
        assert seg["trajectory"]["points"] == \
            d["sub_trajectories"][0]["trajectory"]["points"]
        assert seg["replay_goal"]["goalsets"]
    finally:
        _remove_stubs(mods)


def test_scene_diff_round_trip():
    mods = _install_stubs()
    try:
        from curobo_task_constructor import msg_convert
        diff = {"added": [{"name": "box", "shape": "cuboid",
                           "pose": {"x": 0.5, "y": 0.0, "z": 0.25,
                                    "qx": 0.0, "qy": 0.0, "qz": 0.0,
                                    "qw": 1.0},
                           "dimensions": [0.1, 0.1, 0.1],
                           "mesh_path": None, "vertices": None,
                           "triangles": None}],
                "removed": ["old"], "attached": "box", "detached": None}
        msg = msg_convert.scene_diff_to_msg(diff)
        assert len(msg.world.collision_objects) == 2
        assert len(msg.robot_state.attached_collision_objects) == 1
        back = msg_convert.msg_to_scene_diff(msg)
        assert back["removed"] == ["old"]
        assert back["attached"] == "box"
        assert back["added"][0]["name"] == "box"
        assert back["added"][0]["pose"]["x"] == 0.5
    finally:
        _remove_stubs(mods)

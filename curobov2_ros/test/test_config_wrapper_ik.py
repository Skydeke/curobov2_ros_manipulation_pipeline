# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the standalone IK config wrapper (``ConfigWrapperIK``).

curobo, torch and the ROS message packages are stubbed, so this pins the
wrapper's *control flow* — shared (not rebuilt) motion context, ROS-param
resolution of the seed budget and CUDA-graph mode (including the
``CUROBO_USE_CUDA_GRAPH`` env override), lazy rebuild detection, and the
``update_ik_config`` rebuild path — on any machine, with no GPU and no robot.

The stubbed boundary is deliberate: the one thing these tests cannot check is
whether curobo actually solves IK, which is what the franka ``launch_test``
kinematics suite covers.
"""

import importlib
import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _ParamValue:
    """Mimics a ROS parameter value (integer/bool/double slots)."""

    def __init__(self, integer_value=0, bool_value=False, double_value=0.0):
        self.integer_value = integer_value
        self.bool_value = bool_value
        self.double_value = double_value


class _Param:
    def __init__(self, value):
        self._value = value

    def get_parameter_value(self):
        return self._value


class _Logger:
    def __init__(self):
        self.infos = []
        self.warns = []
        self.errors = []

    def info(self, msg):
        self.infos.append(str(msg))

    def warn(self, msg):
        self.warns.append(str(msg))

    def error(self, msg):
        self.errors.append(str(msg))


class FakeNode:
    """Minimal stand-in for the planner node (params + services + logger)."""

    def __init__(self, params=None):
        # name -> _ParamValue
        self._params = dict(params or {})
        self.logger = _Logger()
        self.services = []
        self._sync = False

    def has_parameter(self, name):
        return name in self._params

    def get_parameter(self, name):
        return _Param(self._params[name])

    def set_param(self, name, value):
        self._params[name] = value

    def get_logger(self):
        return self.logger

    def get_name(self):
        return "unified_planner"

    def create_service(self, srv_type, name, callback):
        self.services.append((srv_type, name, callback))
        return ("service", name)

    def torch_sync_enabled(self):
        return self._sync


def _package(name):
    """A namespace-style stub module, so submodules can be imported under it."""
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _fresh_import(monkeypatch, name):
    """Import ``name`` bound to the just-installed stubs.

    ``importlib.import_module`` returns the cached module when it ran before,
    which would still reference a previous test's stubs — evict it (and its
    dependents) first so this test's stubs take effect. ``monkeypatch``
    restores ``sys.modules`` afterwards.
    """
    for mod in (name,
                "curobov2_ros.core.ik_services",
                "curobov2_ros.core.config_wrapper_ik",
                "curobov2_ros.core.config_wrapper"):
        monkeypatch.delitem(sys.modules, mod, raising=False)
    return importlib.import_module(name)


def _install_stubs(monkeypatch):
    """Stub the module-level imports of ``config_wrapper_ik``/``ik_services``."""
    torch = types.ModuleType("torch")
    torch.float32 = "float32"
    torch.device = lambda *a, **k: ("device", a, k)
    cuda = types.SimpleNamespace(synchronize=lambda: None)
    torch.cuda = cuda
    monkeypatch.setitem(sys.modules, "torch", torch)

    for name in ("curobo", "curobo._src", "curobo._src.types",
                 "curobo._src.util", "curobov2_ros_interfaces",
                 "sensor_msgs", "std_msgs"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, _package(name))

    scene_mod = types.ModuleType("curobo.scene")

    class Scene:
        def __init__(self):
            self.empty = True

    scene_mod.Scene = Scene
    monkeypatch.setitem(sys.modules, "curobo.scene", scene_mod)

    ik_mod = types.ModuleType("curobo.inverse_kinematics")
    created_cfgs = []

    class InverseKinematicsCfg:
        @staticmethod
        def create(**kwargs):
            created_cfgs.append(dict(kwargs))
            return {"cfg": dict(kwargs)}

    class _ToolPoses:
        def as_goal(self):
            return "goal"

    class _KinState:
        tool_poses = _ToolPoses()

    class _Kinematics:
        joint_names = ["panda_joint1", "panda_joint2"]
        tool_frames = ["panda_hand"]

    class InverseKinematics:
        instances = []

        def __init__(self, cfg):
            self.cfg = cfg
            self.kinematics = _Kinematics()
            self.worlds = []
            InverseKinematics.instances.append(self)

        def sample_configs(self, batch):
            return [batch]

        def compute_kinematics(self, js):
            return _KinState()

        def solve_pose(self, goal_tool_poses=None):
            return "result"

        def update_world(self, scene):
            self.worlds.append(scene)

    ik_mod.InverseKinematics = InverseKinematics
    ik_mod.InverseKinematicsCfg = InverseKinematicsCfg
    monkeypatch.setitem(sys.modules, "curobo.inverse_kinematics", ik_mod)

    types_mod = types.ModuleType("curobo.types")

    class JointState:
        @staticmethod
        def from_position(position, joint_names=None):
            return ("js", position, joint_names)

    class Pose:
        pass

    class GoalToolPose:
        @staticmethod
        def from_poses(mapping):
            return ("goal_pose", mapping)

    types_mod.JointState = JointState
    types_mod.Pose = Pose
    types_mod.GoalToolPose = GoalToolPose
    monkeypatch.setitem(sys.modules, "curobo.types", types_mod)

    # Manager modules imported by the real config_wrapper base (stubbed: the
    # wrapper under test never constructs them, it adopts them by reference).
    for mod_name, cls_name in (
        ("curobov2_ros.core.config_manager", "ConfigManager"),
        ("curobov2_ros.core.robot_model_manager", "RobotModelManager"),
        ("curobov2_ros.core.obstacle_manager", "ObstacleManager"),
        ("curobov2_ros.core.camera_system_manager", "CameraSystemManager"),
        ("curobov2_ros.core.laser_system_manager", "LaserSystemManager"),
        ("curobov2_ros.core.ros_service_manager", "RosServiceManager"),
    ):
        if mod_name not in sys.modules:
            mod = types.ModuleType(mod_name)
            setattr(mod, cls_name, type(cls_name, (), {}))
            monkeypatch.setitem(sys.modules, mod_name, mod)

    distances = []

    collision_mod = types.ModuleType("curobov2_ros.core.collision_distance")

    def _compute_sphere_distance(wrapper, node, response):
        distances.append((wrapper, node))
        response.nb_sphere = 0
        response.data = []
        return response

    collision_mod._compute_sphere_distance = _compute_sphere_distance
    monkeypatch.setitem(
        sys.modules, "curobov2_ros.core.collision_distance", collision_mod)

    lanes_mod = types.ModuleType("curobov2_ros.core.gpu_lanes")
    lanes_mod.PRIO_KIN = 1
    monkeypatch.setitem(sys.modules, "curobov2_ros.core.gpu_lanes", lanes_mod)

    std_srvs = types.ModuleType("std_srvs")

    class Trigger:
        class Request:
            pass

        class Response:
            def __init__(self):
                self.success = False
                self.message = ""

    std_srvs.srv = types.SimpleNamespace(Trigger=Trigger)
    monkeypatch.setitem(sys.modules, "std_srvs", std_srvs)
    monkeypatch.setitem(sys.modules, "std_srvs.srv", std_srvs.srv)

    srv = types.ModuleType("curobov2_ros_interfaces.srv")

    def _make_srv_type():
        class Request:
            pass

        class Response:
            def __init__(self):
                self.success = False
                self.message = ""

        return type("SrvType", (), {"Request": Request, "Response": Response})

    for cls_name in ("Ik", "IkBatch", "WarmupIK", "GetCollisionDistance"):
        setattr(srv, cls_name, _make_srv_type())
    monkeypatch.setitem(sys.modules, "curobov2_ros_interfaces.srv", srv)

    sensor_msgs = sys.modules["sensor_msgs"]
    sensor_msgs.msg = types.SimpleNamespace(JointState=type("JointState", (), {}))
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", sensor_msgs.msg)
    std_msgs = sys.modules["std_msgs"]
    std_msgs.msg = types.SimpleNamespace(Bool=type("Bool", (), {}))
    monkeypatch.setitem(sys.modules, "std_msgs.msg", std_msgs.msg)

    ik_module = _fresh_import(monkeypatch, "curobov2_ros.core.config_wrapper_ik")
    return ik_module, ik_mod, created_cfgs, distances


class FakeObstacleManager:
    def __init__(self):
        self.collision_cache = {"cuboid": 1, "mesh": 0, "voxel": None}
        self.pushed = []

    def primitives_only_scene(self):
        return "primitives-scene"

    def collision_world_scene(self):
        return "collision-scene"


def _motion_context():
    """Shared motion-wrapper context the IK wrapper adopts (by reference)."""
    return types.SimpleNamespace(
        robot="robot",
        config_manager="config-manager",
        robot_model_manager=types.SimpleNamespace(robot_cfg="robot-cfg"),
        obstacle_manager=FakeObstacleManager(),
        camera_system_manager="cameras",
        laser_system_manager="lasers",
        ros_service_manager="ros-services",
    )


def _node_with_ik_params(num_seeds=32, use_graph=True):
    return FakeNode(params={
        "num_ik_seeds": _ParamValue(integer_value=num_seeds),
        "use_cuda_graph": _ParamValue(bool_value=use_graph),
    })


@pytest.fixture
def ctx(monkeypatch):
    ik_module, ik_mod, created_cfgs, distances = _install_stubs(monkeypatch)
    ik_mod.InverseKinematics.instances.clear()
    node = _node_with_ik_params()
    motion = _motion_context()
    wrapper = ik_module.ConfigWrapperIK(node, motion)
    return types.SimpleNamespace(
        module=ik_module, ik_mod=ik_mod, created=created_cfgs,
        distances=distances, node=node, motion=motion, wrapper=wrapper)


# ---------------------------------------------------------------------------
# Shared context
# ---------------------------------------------------------------------------


class TestSharedContext:
    def test_adopts_motion_managers_by_reference(self, ctx):
        """No second URDF parse, voxel grid, or service manager is built."""
        for attr in ("config_manager", "robot_model_manager",
                     "obstacle_manager", "camera_system_manager",
                     "laser_system_manager", "ros_service_manager"):
            assert getattr(ctx.wrapper, attr) is getattr(ctx.motion, attr)
        assert ctx.wrapper.robot is ctx.motion.robot
        assert ctx.wrapper.node is ctx.node

    def test_registers_update_ik_config_service(self, ctx):
        names = [name for _, name, _ in ctx.node.services]
        assert "unified_planner/update_ik_config" in names

    def test_solver_starts_uninitialized(self, ctx):
        assert ctx.wrapper.ik_solver is None
        assert ctx.wrapper.ik_batch_size == 0


# ---------------------------------------------------------------------------
# Build configuration (ROS params)
# ---------------------------------------------------------------------------


class TestBuildConfig:
    def test_build_passes_ros_params_into_cfg(self, ctx):
        ctx.wrapper.build(4)
        assert len(ctx.created) == 1
        cfg = ctx.created[0]
        assert cfg["num_seeds"] == 32
        assert cfg["use_cuda_graph"] is True
        assert cfg["max_batch_size"] == 4
        assert cfg["robot"] == "robot-cfg"
        assert cfg["position_tolerance"] == 0.005
        assert cfg["orientation_tolerance"] == 0.05
        assert cfg["self_collision_check"] is True
        assert ctx.wrapper.ik_batch_size == 4
        assert ctx.wrapper.ik_num_seeds == 32
        assert ctx.wrapper.ik_use_cuda_graph is True

    def test_build_honors_explicit_seeds(self, ctx):
        ctx.wrapper.build(2, num_seeds=8)
        assert ctx.created[0]["num_seeds"] == 8
        assert ctx.wrapper.ik_num_seeds == 8

    def test_env_overrides_ros_param(self, ctx, monkeypatch):
        monkeypatch.setenv("CUROBO_USE_CUDA_GRAPH", "0")
        ctx.wrapper.build(2)
        assert ctx.created[0]["use_cuda_graph"] is False
        assert ctx.wrapper.ik_use_cuda_graph is False

    def test_param_off_disables_graphs(self, ctx):
        ctx.node.set_param("use_cuda_graph", _ParamValue(bool_value=False))
        ctx.wrapper.build(2)
        assert ctx.created[0]["use_cuda_graph"] is False

    def test_build_pushes_real_scene_after_empty_build(self, ctx):
        ctx.wrapper.build(2)
        solver = ctx.wrapper.ik_solver
        assert solver.worlds == ["primitives-scene"]

    def test_missing_num_seeds_param_falls_back_to_default(self, ctx):
        del ctx.node._params["num_ik_seeds"]
        ctx.wrapper.build(2)
        assert ctx.created[0]["num_seeds"] == 32


# ---------------------------------------------------------------------------
# Rebuild detection + lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_needs_rebuild_detects_batch_seed_and_graph_changes(self, ctx):
        ctx.wrapper.build(4)
        assert ctx.wrapper.needs_rebuild(4, 32) is False
        assert ctx.wrapper.needs_rebuild(8, 32) is True
        assert ctx.wrapper.needs_rebuild(4, 16) is True
        ctx.node.set_param("use_cuda_graph", _ParamValue(bool_value=False))
        assert ctx.wrapper.needs_rebuild(4, 32) is True

    def test_rebuild_is_noop_before_warmup(self, ctx):
        ctx.wrapper.rebuild()
        assert ctx.created == []
        assert ctx.wrapper.ik_solver is None

    def test_rebuild_reresolves_params(self, ctx):
        ctx.wrapper.build(4)
        ctx.node.set_param("num_ik_seeds", _ParamValue(integer_value=16))
        ctx.wrapper.rebuild()
        assert len(ctx.created) == 2
        assert ctx.created[1]["num_seeds"] == 16
        assert ctx.wrapper.ik_batch_size == 4

    def test_set_ik_config_rebuilds_and_reports(self, ctx):
        from std_srvs.srv import Trigger

        ctx.wrapper.build(4)
        ctx.node.set_param("num_ik_seeds", _ParamValue(integer_value=8))
        response = Trigger.Response()
        out = ctx.wrapper.set_ik_config(ctx.node, Trigger.Request(), response)
        assert out.success is True
        assert out.message == "IK solver config set"
        assert len(ctx.created) == 2
        assert ctx.created[1]["num_seeds"] == 8
        assert ctx.wrapper.ik_batch_size == 4

    def test_invalidate_drops_solver(self, ctx):
        ctx.wrapper.build(4)
        ctx.wrapper.invalidate()
        assert ctx.wrapper.ik_solver is None
        assert ctx.wrapper.ik_batch_size == 0
        assert ctx.wrapper.ik_use_cuda_graph is None
        assert ctx.wrapper.needs_rebuild(4, 32) is True

    def test_update_world_pushes_given_scene(self, ctx):
        ctx.wrapper.build(2)
        ctx.wrapper.update_world("resolved-scene")
        assert ctx.wrapper.ik_solver.worlds[-1] == "resolved-scene"

    def test_update_world_derives_scene_when_missing(self, ctx):
        ctx.wrapper.build(2)
        ctx.wrapper.update_world()
        assert ctx.wrapper.ik_solver.worlds[-1] == "collision-scene"

    def test_update_world_noop_before_warmup(self, ctx):
        ctx.wrapper.update_world("resolved-scene")

    def test_update_world_config_pushes_collision_scene(self, ctx):
        ctx.wrapper.build(2)
        ctx.wrapper.update_world_config(ctx.node)
        assert ctx.wrapper.ik_solver.worlds[-1] == "collision-scene"

    def test_collision_distance_delegates_to_shared_harness(self, ctx):
        response = types.SimpleNamespace(nb_sphere=-1, data=None)
        out = ctx.wrapper.callback_get_collision_distance(
            ctx.node, object(), response)
        assert out.nb_sphere == 0
        assert out.data == []
        assert ctx.distances == [(ctx.wrapper, ctx.node)]


# ---------------------------------------------------------------------------
# IKServices delegation
# ---------------------------------------------------------------------------


class TestServicesDelegation:
    def test_services_delegate_lifecycle_to_wrapper(self, ctx, monkeypatch):
        services_mod = _fresh_import(monkeypatch, "curobov2_ros.core.ik_services")
        services = services_mod.IKServices(ctx.node, ctx.wrapper)
        assert services.warmup(4) == "IK solver ready (batch_size=4)"
        assert services._ik_solver is ctx.wrapper.ik_solver
        services.update_world("resolved-scene")
        assert ctx.wrapper.ik_solver.worlds[-1] == "resolved-scene"
        services.rebuild()
        assert len(ctx.created) == 2
        assert services._resolve_num_seeds() == 32

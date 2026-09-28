# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the joint-lock runtime switch (``RobotModelManager``).

curobo, torch and the generated ``isaac_ros_cumotion_interfaces`` package are
stubbed, so this pins the switch's *control flow* — idempotency, which joints
may be locked at all, merging a partial change onto the locks in force,
override bookkeeping, and not leaving a half-applied model behind when the
rebuild fails — on any machine, with no GPU and no robot.

The stubbed boundary is deliberate: the one thing these tests cannot check is
whether curobo actually honors the override, which is what the franka
``launch_test`` suite and the benchmark's own A/B (``--no-mpinets-locks``)
cover. What they can check is that the switch never silently no-ops, never
widens the lockable joint set past the robot's own joints, and never commits a
broken model.
"""

import importlib
import json
import sys
import types
import unittest.mock as mock

import pytest


FRANKA_LOCKS = {"panda_finger_joint1": 0.04, "panda_finger_joint2": 0.04}
# The robot's joints in kinematic-tree order, as the franka reference config
# models them: the 7 cspace joints, then the two fingers the config locks, then
# a hand joint it leaves free (so a test can lock a joint the YAML does not).
FRANKA_TREE_JOINTS = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
    "panda_hand_joint",
]


def _package(name):
    """A namespace-style stub module, so submodules can be imported under it."""
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _install_stubs(monkeypatch):
    """Stub the module-level imports of ``robot_model_manager``.

    Returns the stub module handle for ``KinematicsCfg``/``Kinematics`` so a
    test can assert how many times (and with what) the model was rebuilt.
    """
    torch = types.ModuleType("torch")
    torch.float32 = "float32"
    torch.device = lambda *a, **k: "cuda"
    monkeypatch.setitem(sys.modules, "torch", torch)

    for name in ("curobo", "curobo._src", "curobo._src.types",
                 "curobo._src.util", "isaac_ros_cumotion_interfaces"):
        monkeypatch.setitem(sys.modules, name, _package(name))

    kinematics_mod = types.ModuleType("curobo.kinematics")

    class KinematicsCfg:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.kinematics_config = kwargs

        @staticmethod
        def from_robot_yaml_file(path, **kwargs):
            return KinematicsCfg(path=path, **kwargs)

    class Kinematics:
        """Stand-in for the GPU model.

        Mirrors what curobo's loader does with ``lock_joints``: the locked
        joints leave ``joint_names`` (they are out of the cspace) and reappear
        in ``lock_jointstate``. Order is approximated — only the *set* and the
        resulting model order matter to the switch, not the exact sequence
        curobo's loader produces.
        """

        def __init__(self, kinematics):
            self.kinematics = kinematics
            locks = (getattr(kinematics, "kwargs", {}) or {}).get("lock_joints") or {}
            self.lock_jointstate = types.SimpleNamespace(
                joint_names=[j for j in FRANKA_TREE_JOINTS if j in locks]
            )
            self.joint_names = [j for j in FRANKA_TREE_JOINTS if j not in locks]

    kinematics_mod.KinematicsCfg = KinematicsCfg
    kinematics_mod.Kinematics = Kinematics
    monkeypatch.setitem(sys.modules, "curobo.kinematics", kinematics_mod)

    device_cfg = types.ModuleType("curobo._src.types.device_cfg")

    class DeviceCfg:
        pass

    device_cfg.DeviceCfg = DeviceCfg
    monkeypatch.setitem(sys.modules, "curobo._src.types.device_cfg", device_cfg)

    src_robot = types.ModuleType("curobo._src.types.robot")

    class RobotCfg:
        def __init__(self, kinematics=None, dynamics=None):
            self.kinematics = kinematics
            self.dynamics = dynamics

        @staticmethod
        def _create_dynamics_config(kinematics_config=None, device_cfg=None):
            return {"dynamics": True}

    src_robot.RobotCfg = RobotCfg
    monkeypatch.setitem(sys.modules, "curobo._src.types.robot", src_robot)

    curobo_types = types.ModuleType("curobo.types")
    curobo_types.JointState = object
    monkeypatch.setitem(sys.modules, "curobo.types", curobo_types)

    srv = types.ModuleType("isaac_ros_cumotion_interfaces.srv")
    srv.SetLinkCollision = object
    monkeypatch.setitem(sys.modules, "isaac_ros_cumotion_interfaces.srv", srv)

    config_io = types.ModuleType("curobo._src.util.config_io")
    # Mutable so a test can simulate the robot YAML being edited on disk
    # between calls (the config is re-read on every switch).
    yaml_state = {"content": {"robot_cfg": {"kinematics": {
        "lock_joints": dict(FRANKA_LOCKS),
    }}}}
    config_io.load_yaml = lambda path: json.loads(json.dumps(yaml_state["content"]))
    monkeypatch.setitem(sys.modules, "curobo._src.util.config_io", config_io)

    module = importlib.import_module("isaac_ros_cumotion.core.robot_model_manager")
    return module, kinematics_mod, yaml_state


@pytest.fixture
def manager(monkeypatch):
    """A ``RobotModelManager`` built without ``__init__`` (no GPU, no robot).

    Yields ``(instance, module, kinematics_stub, yaml_state)``. Patch
    ``module.Kinematics`` (not the stub's attribute) to make the model build
    fail: the module under test did ``from curobo.kinematics import
    Kinematics``, so it holds its own reference.
    """
    module, kinematics_mod, yaml_state = _install_stubs(monkeypatch)
    instance = module.RobotModelManager.__new__(module.RobotModelManager)
    instance.robot_config_file = "config/franka.curobo.reference.yml"
    instance.robot = None
    instance.base_link = "base_link"
    instance.node = None
    instance._load_dynamics_active = False
    instance._lock_joints_override = None
    # Seed the model __init__ would have parsed, so the torque switch has a
    # kin_cfg to re-wrap from the very first call. The franka locks its two
    # fingers at startup, so the fake model starts with the same two joints out
    # of its cspace.
    instance.kin_cfg = kinematics_mod.KinematicsCfg(
        path="startup", lock_joints=dict(FRANKA_LOCKS)
    )
    instance.robot_cfg = instance._wrap_robot_cfg(instance.kin_cfg)
    instance.kin_model = kinematics_mod.Kinematics(instance.robot_cfg.kinematics)
    return instance, module, kinematics_mod, yaml_state


class TestConfigLockJoints:
    def test_reads_the_robots_own_values(self, manager):
        instance, _, _, _ = manager
        assert instance.config_lock_joints() == FRANKA_LOCKS

    def test_no_override_reports_the_config_values(self, manager):
        instance, _, _, _ = manager
        assert instance.lock_joints() == FRANKA_LOCKS

    def test_override_wins_until_restored(self, manager):
        instance, _, _, _ = manager
        assert instance.set_lock_joints(
            {"panda_finger_joint1": 0.025, "panda_finger_joint2": 0.025}
        ) is True
        assert instance.lock_joints() == {
            "panda_finger_joint1": 0.025,
            "panda_finger_joint2": 0.025,
        }
        # the YAML keeps its own values; only the effective set moved
        assert instance.config_lock_joints() == FRANKA_LOCKS
        assert instance.set_lock_joints(None) is True
        assert instance.lock_joints() == FRANKA_LOCKS


class TestSetLockJointsValidation:
    """The lockable joint set comes from the robot model, never the caller."""

    def test_partial_joint_set_is_the_way_to_release(self, manager):
        """A mapping that names only some joints releases the rest.

        Locking and releasing are the same operation on one dict: a joint left
        out is no longer locked, so a caller releases the fingers by sending
        ``{}`` rather than needing a separate verb.
        """
        instance, _, _, _ = manager
        assert instance.set_lock_joints({}) is True
        assert instance.lock_joints() == {}
        assert instance.lockable_joints() == FRANKA_TREE_JOINTS

    def test_active_joint_can_be_locked(self, manager):
        """Not just the joints the YAML already locks: any joint of the model
        can be pinned (and therefore dropped from the cspace)."""
        instance, _, _, _ = manager
        assert instance.set_lock_joints(
            {**FRANKA_LOCKS, "panda_joint7": 0.1}
        ) is True
        assert instance.lock_joints()["panda_joint7"] == 0.1

    def test_unknown_joint_rejected(self, manager):
        instance, _, _, _ = manager
        with pytest.raises(ValueError, match="not part of the robot model"):
            instance.set_lock_joints(
                {
                    "panda_finger_joint1": 0.025,
                    "panda_finger_joint2": 0.025,
                    "panda_elbow_joint": 0.1,
                }
            )

    def test_non_finite_value_rejected(self, manager):
        """A NaN would be baked straight into a fixed joint transform, so it
        must fail before the model is re-parsed."""
        instance, _, _, _ = manager
        with pytest.raises(ValueError, match="finite"):
            instance.set_lock_joints(
                {**FRANKA_LOCKS, "panda_joint7": float("nan")}
            )

    def test_rejected_request_leaves_state_untouched(self, manager):
        instance, module, kinematics_mod, _ = manager
        before_kin, before_cfg = instance.kin_cfg, instance.robot_cfg
        with pytest.raises(ValueError):
            instance.set_lock_joints({"panda_elbow_joint": 0.1})
        assert instance._lock_joints_override is None
        assert instance.lock_joints() == FRANKA_LOCKS
        assert instance.kin_cfg is before_kin
        assert instance.robot_cfg is before_cfg

    def test_lockable_set_is_the_model_not_the_active_locks(self, manager):
        """Whatever the YAML or the live override says, the set of joints that
        may be named is the model's — so an override can never widen it, and a
        lock joint the config gains is lockable the moment the model has it.

        The model was parsed from that same config, so this still cannot drift
        from it: a joint the model does not have is not a joint of the robot.
        """
        instance, _, _, yaml_state = manager
        assert instance.set_lock_joints(
            {"panda_finger_joint1": 0.025, "panda_finger_joint2": 0.025}
        ) is True
        # A joint the model does not have stays un-lockable even though the
        # config on disk would happily name it.
        yaml_state["content"]["robot_cfg"]["kinematics"]["lock_joints"][
            "panda_elbow_joint"
        ] = 0.0
        with pytest.raises(ValueError, match="not part of the robot model"):
            instance.set_lock_joints({"panda_elbow_joint": 0.0})
        # …and a joint the config locks but the current override dropped is
        # still lockable (the model has it free).
        assert instance.update_lock_joints(
            lock={"panda_hand_joint": 0.0}
        ) is True
        assert instance.lock_joints()["panda_hand_joint"] == 0.0


class TestUpdateLockJoints:
    """The partial form: change some joints, leave the rest in force."""

    def test_lock_touches_only_the_named_joint(self, manager):
        instance, _, _, _ = manager
        assert instance.update_lock_joints(
            lock={"panda_finger_joint1": 0.025}
        ) is True
        assert instance.lock_joints() == {
            "panda_finger_joint1": 0.025,
            "panda_finger_joint2": 0.04,
        }

    def test_unlock_touches_only_the_named_joint(self, manager):
        instance, _, _, _ = manager
        assert instance.update_lock_joints(
            unlock=["panda_finger_joint1"]
        ) is True
        assert instance.lock_joints() == {"panda_finger_joint2": 0.04}
        # the released joint is back in the model's cspace
        assert "panda_finger_joint1" in instance.kin_model.joint_names

    def test_successive_calls_compose(self, manager):
        """Each call applies to the locks in force, so a run can pin the
        gripper and then the arm without restating the gripper."""
        instance, _, _, _ = manager
        instance.update_lock_joints(lock={"panda_finger_joint1": 0.025})
        instance.update_lock_joints(lock={"panda_joint7": 0.1})
        instance.update_lock_joints(unlock=["panda_finger_joint2"])
        assert instance.lock_joints() == {
            "panda_joint7": 0.1,
            "panda_finger_joint1": 0.025,
        }

    def test_repeated_request_is_a_noop(self, manager):
        instance, _, _, _ = manager
        assert instance.update_lock_joints(
            lock={"panda_finger_joint1": 0.025}
        ) is True
        before = (instance.kin_cfg, instance.robot_cfg, instance.kin_model)
        assert instance.update_lock_joints(
            lock={"panda_finger_joint1": 0.025}
        ) is False
        assert (instance.kin_cfg, instance.robot_cfg, instance.kin_model) == before

    def test_unlocking_a_free_joint_is_a_noop(self, manager):
        """Releases are idempotent, so a caller can assert a state without
        tracking whether it already holds it."""
        instance, _, _, _ = manager
        before = (instance.kin_cfg, instance.robot_cfg, instance.kin_model)
        assert instance.update_lock_joints(unlock=["panda_joint3"]) is False
        assert instance.lock_joints() == FRANKA_LOCKS
        assert (instance.kin_cfg, instance.robot_cfg, instance.kin_model) == before

    def test_joint_cannot_be_locked_and_unlocked_at_once(self, manager):
        instance, _, _, _ = manager
        with pytest.raises(ValueError, match="same call"):
            instance.update_lock_joints(
                lock={"panda_finger_joint1": 0.025},
                unlock=["panda_finger_joint1"],
            )

    def test_empty_request_is_a_noop(self, manager):
        instance, _, _, _ = manager
        before = (instance.kin_cfg, instance.robot_cfg, instance.kin_model)
        assert instance.update_lock_joints() is False
        assert (instance.kin_cfg, instance.robot_cfg, instance.kin_model) == before

    def test_reported_in_model_order_not_request_order(self, manager):
        """The response (and the loader) see the model's joint order, whatever
        order the caller named the joints in."""
        instance, _, _, _ = manager
        instance.update_lock_joints(
            lock={"panda_hand_joint": 0.0, "panda_finger_joint2": 0.025},
            unlock=["panda_finger_joint1"],
        )
        assert list(instance.lock_joints()) == [
            "panda_finger_joint2",
            "panda_hand_joint",
        ]


class TestSetLockJointsIdempotency:
    """A rebuild triggered for another reason must not pay for this switch."""

    def test_same_values_do_not_rebuild(self, manager):
        instance, _, _, _ = manager
        requested = {"panda_finger_joint1": 0.025, "panda_finger_joint2": 0.025}
        assert instance.set_lock_joints(requested) is True
        kin_cfg, robot_cfg, kin_model = (
            instance.kin_cfg,
            instance.robot_cfg,
            instance.kin_model,
        )
        # Same values again, and again: no re-parse, no new model objects.
        assert instance.set_lock_joints(dict(requested)) is False
        assert instance.set_lock_joints(requested) is False
        assert (instance.kin_cfg, instance.robot_cfg, instance.kin_model) == (
            kin_cfg,
            robot_cfg,
            kin_model,
        )

    def test_repeating_the_config_values_is_a_noop(self, manager):
        """The restore the runner issues to learn the joint names must not
        re-parse the model when nothing is overridden yet."""
        instance, _, _, _ = manager
        before = (instance.kin_cfg, instance.robot_cfg, instance.kin_model)
        assert instance.set_lock_joints(None) is False
        assert (instance.kin_cfg, instance.robot_cfg, instance.kin_model) == before

    def test_changing_one_value_rebuilds(self, manager):
        instance, _, _, _ = manager
        assert instance.set_lock_joints(
            {"panda_finger_joint1": 0.025, "panda_finger_joint2": 0.025}
        ) is True
        assert instance.set_lock_joints(
            {"panda_finger_joint1": 0.04, "panda_finger_joint2": 0.04}
        ) is True

    def test_restore_rebuilds_only_after_an_override(self, manager):
        instance, _, _, _ = manager
        assert instance.set_lock_joints(
            {"panda_finger_joint1": 0.025, "panda_finger_joint2": 0.025}
        ) is True
        assert instance.set_lock_joints(None) is True
        # and it is idempotent again on the second restore
        assert instance.set_lock_joints(None) is False

    def test_restore_after_the_config_gained_a_lock_joint(self, manager):
        """A restore reads the config fresh. If that config has grown a lock
        joint since the override was set, the two key sets differ and the
        switch must rebuild — not compare values positionally and trip over a
        joint the override does not have.

        The config is edited to *agree* with the override on the shared joints,
        so nothing short-circuits the comparison before the new joint: the key
        sets themselves have to be what decides.
        """
        instance, _, _, yaml_state = manager
        locks = yaml_state["content"]["robot_cfg"]["kinematics"]["lock_joints"]
        assert instance.set_lock_joints(
            {"panda_finger_joint1": 0.025, "panda_finger_joint2": 0.025}
        ) is True
        # config now shares the override's values AND has an extra lock joint
        locks["panda_finger_joint1"] = 0.025
        locks["panda_finger_joint2"] = 0.025
        locks["panda_hand_joint"] = 0.0

        assert instance.set_lock_joints(None) is True
        assert instance.lock_joints() == {
            "panda_finger_joint1": 0.025,
            "panda_finger_joint2": 0.025,
            "panda_hand_joint": 0.0,
        }
        assert instance._lock_joints_override is None


class TestSetLockJointsRebuild:
    """The override must reach curobo's loader, and only commit once complete."""

    def test_loader_receives_the_override_as_a_build_kwarg(self, manager):
        instance, module, kinematics_mod, _ = manager
        instance.set_lock_joints(
            {"panda_finger_joint1": 0.025, "panda_finger_joint2": 0.025}
        )
        # kwargs land on the *unwrapped* kinematics dict (what
        # KinematicsCfg.from_content_path does), and the PATH is passed so
        # relative urdf_path / collision_sphere entries still resolve.
        assert instance.kin_cfg.kwargs == {
            "path": "config/franka.curobo.reference.yml",
            "lock_joints": {
                "panda_finger_joint1": 0.025,
                "panda_finger_joint2": 0.025,
            },
        }

    def test_kin_model_is_rebuilt_from_the_new_cfg(self, manager):
        instance, _, _, _ = manager
        instance.set_lock_joints(
            {"panda_finger_joint1": 0.025, "panda_finger_joint2": 0.025}
        )
        assert instance.kin_model.kinematics is instance.kin_cfg
        assert instance.robot_cfg.kinematics is instance.kin_cfg

    def test_torque_mode_survives_the_switch(self, manager):
        """The dynamics wrapper is re-derived from the current mode, so a lock
        change on a torque-limited server does not silently drop the payload."""
        instance, _, _, _ = manager
        instance._load_dynamics_active = True
        instance.set_lock_joints(
            {"panda_finger_joint1": 0.025, "panda_finger_joint2": 0.025}
        )
        assert instance.robot_cfg.dynamics == {"dynamics": True}
        instance._load_dynamics_active = False
        instance.set_lock_joints(None)
        assert instance.robot_cfg.dynamics is None

    def test_failed_rebuild_commits_nothing(self, manager, monkeypatch):
        """A Kinematics() failure must leave the node on the model it had, not
        on a re-parsed cfg with no GPU model behind it."""
        instance, module, kinematics_mod, _ = manager
        instance.set_lock_joints(
            {"panda_finger_joint1": 0.025, "panda_finger_joint2": 0.025}
        )
        good_kin_cfg, good_robot_cfg, good_model = (
            instance.kin_cfg,
            instance.robot_cfg,
            instance.kin_model,
        )
        good_override = dict(instance._lock_joints_override)

        monkeypatch.setattr(
            module, "Kinematics", mock.Mock(side_effect=RuntimeError("cuda oom"))
        )
        with pytest.raises(RuntimeError, match="cuda oom"):
            instance.set_lock_joints(
                {"panda_finger_joint1": 0.04, "panda_finger_joint2": 0.04}
            )
        assert instance.kin_cfg is good_kin_cfg
        assert instance.robot_cfg is good_robot_cfg
        assert instance.kin_model is good_model
        assert instance._lock_joints_override == good_override
        assert instance.lock_joints() == {
            "panda_finger_joint1": 0.025,
            "panda_finger_joint2": 0.025,
        }


class TestSetTorqueMode:
    """The torque switch re-wraps robot_cfg but must not touch the kinematics."""

    def test_switch_is_idempotent(self, manager):
        instance, _, _, _ = manager
        assert instance.set_torque_mode(False) is False  # already off
        assert instance.set_torque_mode(True) is True
        assert instance.set_torque_mode(True) is False

    def test_kin_cfg_is_reused_across_the_switch(self, manager):
        """Only `dynamics` varies, so solvers already holding a GPU model of
        this kin_cfg stay valid and no re-parse is needed."""
        instance, module, kinematics_mod, _ = manager
        instance.kin_cfg = kinematics_mod.KinematicsCfg(path="pinned")
        kin_cfg = instance.kin_cfg
        instance.set_torque_mode(True)
        assert instance.robot_cfg.kinematics is kin_cfg
        assert instance.robot_cfg.dynamics == {"dynamics": True}
        instance.set_torque_mode(False)
        assert instance.robot_cfg.kinematics is kin_cfg
        assert instance.robot_cfg.dynamics is None

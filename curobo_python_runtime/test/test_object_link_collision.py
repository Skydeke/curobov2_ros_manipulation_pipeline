"""Per-object/per-link collision allowance semantics.

The task constructor's ``allowCollisions(object, links)`` must not turn a link
blind to the world. These tests exercise the layer bookkeeping directly with a
fake kinematics model and scene checker, so no CUDA/cuRobo install is needed.
"""

from __future__ import annotations

import curobo_python_runtime.scene as scene
from curobo_python_runtime import runtime

ALL_BITS = 0xFFFFFFFF


class _FakeKinematics:
    def __init__(self):
        self.link_name_to_idx_map = {
            "finger_1": 0,
            "finger_2": 1,
            "wrist": 2,
        }
        self.masks = {name: ALL_BITS for name in self.link_name_to_idx_map}

    def set_link_collision_mask(self, link_name, layers):
        assert link_name in self.masks
        self.masks[link_name] = int(layers)


class _FakeSceneData:
    def __init__(self, names):
        self.names = set(names)
        self.layers = {}

    def check_obstacle_exists(self, name):
        return name in self.names

    def set_obstacle_layers(self, name, layers):
        assert name in self.names
        self.layers[name] = int(layers)


class _FakeChecker:
    def __init__(self, names):
        self.data = _FakeSceneData(names)


class _FakeAttachmentManager:
    def __init__(self):
        self.kinematics_params = _FakeKinematics()


def _install_fakes(handle, monkeypatch):
    entry = {}
    runtime._REGISTRY[handle] = entry
    am = _FakeAttachmentManager()
    checker = _FakeChecker({"object_0", "object_1"})
    monkeypatch.setattr(scene, "_attachment_manager", lambda _entry: am)
    monkeypatch.setattr(scene, "_solver_scene_checkers", lambda _entry: [checker])
    return entry, am, checker


def _cleanup(handle):
    runtime._REGISTRY.pop(handle, None)


def test_disabling_one_link_leaves_every_other_link_colliding(monkeypatch):
    handle = 9101
    try:
        _entry, am, checker = _install_fakes(handle, monkeypatch)
        result = scene.set_object_link_collision(
            handle, "object_0", ["finger_1"], enabled=False)

        assert result["applied"] == ["finger_1"]
        bit = checker.data.layers["object_0"]
        assert bit > 1
        assert am.kinematics_params.masks["finger_1"] & bit == 0
        # finger_2 was not named: it still sees the object.
        assert am.kinematics_params.masks["finger_2"] == ALL_BITS
        assert am.kinematics_params.masks["finger_2"] & bit
        # The wrist is also untouched and still collides with the object.
        assert am.kinematics_params.masks["wrist"] == ALL_BITS
        assert am.kinematics_params.masks["wrist"] & bit

        scene.set_object_link_collision(
            handle, "object_0", ["finger_1"], enabled=True)
        assert checker.data.layers["object_0"] == 1
        assert am.kinematics_params.masks["finger_1"] == ALL_BITS
        assert am.kinematics_params.masks["finger_2"] == ALL_BITS
        assert am.kinematics_params.masks["wrist"] == ALL_BITS
    finally:
        _cleanup(handle)


def test_two_objects_use_independent_layers(monkeypatch):
    handle = 9102
    try:
        _entry, am, checker = _install_fakes(handle, monkeypatch)
        scene.set_object_link_collision(
            handle, "object_0", ["finger_1"], enabled=False)
        scene.set_object_link_collision(
            handle, "object_1", ["finger_2"], enabled=False)

        bit0 = checker.data.layers["object_0"]
        bit1 = checker.data.layers["object_1"]
        assert bit0 != bit1
        # finger_1 ignores object_0 but still collides with object_1.
        assert am.kinematics_params.masks["finger_1"] & bit0 == 0
        assert am.kinematics_params.masks["finger_1"] & bit1
        # finger_2 ignores object_1 but still collides with object_0.
        assert am.kinematics_params.masks["finger_2"] & bit1 == 0
        assert am.kinematics_params.masks["finger_2"] & bit0
        # The wrist still collides with both.
        assert am.kinematics_params.masks["wrist"] & bit0
        assert am.kinematics_params.masks["wrist"] & bit1
    finally:
        _cleanup(handle)

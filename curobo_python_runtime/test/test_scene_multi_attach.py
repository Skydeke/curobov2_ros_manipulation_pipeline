"""Bridge attach bookkeeping for one payload per arm.

These tests stay ROS-free and torch-free: they exercise the pure bookkeeping
helpers plus ``detach_object`` with a fake AttachmentManager, so no CUDA
context is needed.
"""

from __future__ import annotations

import curobo_python_runtime.scene as scene


def _two_attached_entry() -> dict:
    entry: dict = {}
    scene._record_attached(
        entry, "left", "attached_object_left", "left_grasp_frame")
    scene._record_attached(
        entry, "right", "attached_object_right", "right_grasp_frame")
    return entry


def test_record_attached_keeps_both_objects_and_primary_compat_fields():
    entry = _two_attached_entry()
    assert entry["attached_name"] == "right"
    assert entry["attach_link_used"] == "attached_object_right"
    assert entry["attach_parent_link_used"] == "right_grasp_frame"
    assert entry["attached_disabled"] == ["left", "right"]
    assert scene._attached_names(entry) == ["left", "right"]
    assert scene._attached_links(entry) == {
        "left": "attached_object_left",
        "right": "attached_object_right",
    }
    assert scene._attached_parent_links(entry) == {
        "left": "left_grasp_frame",
        "right": "right_grasp_frame",
    }


def test_drop_attached_removes_only_the_named_object():
    entry = _two_attached_entry()
    remaining = scene._drop_attached(entry, "left")
    assert remaining == ["right"]
    assert entry["attached_name"] == "right"
    assert entry["attach_link_used"] == "attached_object_right"
    assert entry["attached_disabled"] == ["right"]
    assert scene._attached_links(entry) == {
        "right": "attached_object_right",
    }


def test_clear_attached_releases_every_name():
    entry = _two_attached_entry()
    released = scene._clear_attached(entry)
    assert released == ["left", "right"]
    assert entry["attached_name"] == ""
    assert entry["attach_link_used"] == ""
    assert entry["attached_disabled"] == []
    assert scene._attached_names(entry) == []


def test_legacy_single_attachment_entry_is_upgraded():
    entry = {
        "attached_name": "object",
        "attach_link_used": "attached_object",
        "attach_parent_link_used": "flange",
        "attached_disabled": ["object"],
    }
    assert scene._attached_names(entry) == ["object"]
    assert scene._attached_links(entry) == {"object": "attached_object"}
    assert scene._attached_parent_links(entry) == {"object": "flange"}

    scene._record_attached(entry, "object_2", "attached_object_left", "left")
    assert scene._attached_names(entry) == ["object", "object_2"]
    assert entry["attached_disabled"] == ["object", "object_2"]


class _FakeChecker:
    def __init__(self):
        self.disabled: set = set()

    def check_obstacle_exists(self, name):
        return True

    def enable_obstacle(self, name, enable=True, env_idx=None):
        if enable:
            self.disabled.discard(name)
        else:
            self.disabled.add(name)


def test_reapply_disables_every_attached_name():
    entry = _two_attached_entry()
    checker = _FakeChecker()
    entry["checker"] = checker
    scene._reapply_attached_disables(entry)
    assert checker.disabled == {"left", "right"}


class _FakeAttachmentManager:
    def __init__(self):
        self.detached: list = []
        self._attached_link_name = None
        self._disabled_obstacle_names = []
        self._disabled_num_envs = 0

    def detach(self, link_name=None, enable_obstacle_names=None):
        self.detached.append((link_name, list(enable_obstacle_names or [])))
        self._attached_link_name = None
        self._disabled_obstacle_names = []
        self._disabled_num_envs = 0


def test_detach_by_name_releases_one_and_detach_all_releases_rest(
        monkeypatch):
    handle = 424242
    entry = _two_attached_entry()
    entry["attached_num_envs"] = 1
    fake_am = _FakeAttachmentManager()
    monkeypatch.setitem(scene._REGISTRY, handle, entry)
    monkeypatch.setattr(scene, "_attachment_manager", lambda e: fake_am)
    monkeypatch.setattr(scene, "world_push", lambda h: True)

    released = scene.detach_object(handle, "left")
    assert released == "left"
    assert fake_am.detached == [
        ("attached_object_left", ["left"]),
    ]
    assert entry["attached_name"] == "right"
    assert entry["attached_disabled"] == ["right"]
    assert fake_am._attached_link_name == "attached_object_right"
    assert fake_am._disabled_obstacle_names == ["right"]

    released = scene.detach_object(handle)
    assert released == "right"
    assert fake_am.detached[-1] == (
        "attached_object_right", ["right"])
    assert entry["attached_name"] == ""
    assert entry["attached_disabled"] == []
    assert fake_am._disabled_obstacle_names == []


def test_detach_all_returns_comma_joined_names(monkeypatch):
    handle = 424243
    entry = _two_attached_entry()
    fake_am = _FakeAttachmentManager()
    monkeypatch.setitem(scene._REGISTRY, handle, entry)
    monkeypatch.setattr(scene, "_attachment_manager", lambda e: fake_am)
    monkeypatch.setattr(scene, "world_push", lambda h: True)

    assert scene.detach_object(handle) == "left, right"
    assert entry["attached_disabled"] == []

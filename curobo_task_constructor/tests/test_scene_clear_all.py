"""The clear-all scene ops survive the execute wire.

A scene-setup task clears the server world with ``remove_all`` and releases any
stale attach with ``detach_all``. Both are boolean ops with no MTC counterpart:
a ``PlanningScene`` diff is name-keyed and has no wildcard REMOVE. Dropping them
is not harmless — the task's re-adds then land on the previous cycle's world,
where ``add_object`` refuses the duplicate names, so every pick cycle plans
against stale boxes and the setup never lands.
"""

from curobo_task_constructor.executor import (
    _object_spec_from_dict, _scene_ops_from_dict, _scene_ops_to_dict,
)
from curobo_task_constructor.msg_convert import _merge_scene_diff


def _spec(name):
    """An ObjectSpec whose dict form comes back byte-for-byte."""
    return _object_spec_from_dict(
        {"name": name, "shape": "cuboid",
         "pose": {"x": 0.0, "y": 0.0, "z": 0.0,
                  "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0},
         "dimensions": [0.1, 0.1, 0.1],
         "mesh_path": None, "vertices": None, "triangles": None})


def _empty():
    return {"added": [], "removed": [], "attached": None, "detached": None,
            "remove_all": False, "detach_all": False}


def test_scene_ops_dict_carries_the_clear_all_flags():
    d = _scene_ops_to_dict([("detach_all", None), ("remove_all", None),
                            ("add", _spec("object_0"))])
    assert d["remove_all"] is True and d["detach_all"] is True
    assert [s["name"] for s in d["added"]] == ["object_0"]


def test_scene_ops_dict_defaults_the_flags_off():
    d = _scene_ops_to_dict([("add", _spec("object_0"))])
    assert d["remove_all"] is False and d["detach_all"] is False


def test_clear_all_is_reconstructed_before_the_adds():
    d = _scene_ops_to_dict([("detach_all", None), ("remove_all", None),
                            ("add", _spec("object_0"))])
    kinds = [kind for kind, _ in _scene_ops_from_dict(d)]
    assert kinds == ["detach_all", "remove_all", "add"]


def test_merge_scene_diff_keeps_a_clear_all_from_either_half():
    merged = _merge_scene_diff(dict(_empty(), remove_all=True),
                               dict(_empty(), detach_all=True))
    assert merged["remove_all"] is True and merged["detach_all"] is True


def test_clear_all_survives_a_merged_segment_diff():
    before = _scene_ops_to_dict([("detach_all", None), ("remove_all", None)])
    after = _scene_ops_to_dict([("add", _spec("object_0"))])
    merged = _merge_scene_diff(before, after)
    kinds = [kind for kind, _ in _scene_ops_from_dict(merged)]
    assert kinds == ["detach_all", "remove_all", "add"]

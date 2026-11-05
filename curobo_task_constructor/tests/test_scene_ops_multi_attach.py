"""Two attach/detach ops survive the execute wire without overwriting."""

from __future__ import annotations

from curobo_task_constructor.executor import (
    _scene_ops_from_dict, _scene_ops_to_dict,
)
from curobo_task_constructor.msg_convert import _merge_scene_diff


def test_scene_ops_dict_keeps_both_attaches():
    d = _scene_ops_to_dict([("attach", "left"), ("attach", "right")])
    assert d["attached"] == "right"  # last/primary compatibility field
    assert d["attached_list"] == ["left", "right"]
    assert _scene_ops_from_dict(d) == [
        ("attach", "left"),
        ("attach", "right"),
    ]


def test_scene_ops_dict_keeps_both_detaches():
    d = _scene_ops_to_dict([("detach", "left"), ("detach", "right")])
    assert d["detached"] == "right"
    assert d["detached_list"] == ["left", "right"]
    assert _scene_ops_from_dict(d) == [
        ("detach", "left"),
        ("detach", "right"),
    ]


def test_merge_scene_diff_keeps_two_attaches():
    before = _scene_ops_to_dict([("attach", "left")])
    after = _scene_ops_to_dict([("attach", "right")])
    merged = _merge_scene_diff(before, after)
    assert merged["attached_list"] == ["left", "right"]
    assert merged["attached"] == "right"
    assert _scene_ops_from_dict(merged) == [
        ("attach", "left"),
        ("attach", "right"),
    ]


def test_merge_scene_diff_last_operation_wins_per_name():
    before = _scene_ops_to_dict([("attach", "left"), ("attach", "right")])
    after = _scene_ops_to_dict([("detach", "left")])
    merged = _merge_scene_diff(before, after)
    assert merged["attached_list"] == ["right"]
    assert merged["detached_list"] == ["left"]

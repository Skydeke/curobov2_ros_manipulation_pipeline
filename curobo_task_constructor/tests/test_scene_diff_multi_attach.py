"""SceneDiff with one payload per arm (bimanual attach)."""

from __future__ import annotations

from curobo_task_constructor.core.state import SceneDiff


def _two_attached() -> SceneDiff:
    return (SceneDiff()
            .with_attached("left", link="attached_object_left")
            .with_attached("right", link="attached_object_right"))


def test_two_attachments_are_kept_and_primary_is_last():
    scene = _two_attached()
    assert scene.attached_names() == ["left", "right"]
    assert scene.attached_object == "right"
    assert scene.attached_objects == {
        "left": "attached_object_left",
        "right": "attached_object_right",
    }
    assert scene.detached_object is None
    assert scene.detached_objects == []


def test_detach_one_keeps_the_other():
    scene = _two_attached().with_detached("left")
    assert scene.attached_names() == ["right"]
    assert scene.attached_object == "right"
    assert scene.attached_objects == {"right": "attached_object_right"}
    assert scene.detached_objects == ["left"]
    assert scene.detached_object == "left"


def test_detach_all_clears_every_attachment():
    scene = _two_attached().with_detached(None)
    assert scene.attached_names() == []
    assert scene.attached_object is None
    assert scene.attached_objects == {}
    assert scene.detached_objects == ["left", "right"]
    assert scene.detached_object == "right"


def test_ops_to_materialize_emits_each_attach_and_detach():
    scene = _two_attached().with_detached("left")
    assert scene.ops_to_materialize() == [
        ("detach", "left"),
        ("attach", "right"),
    ]


def test_ops_to_materialize_keeps_both_attach_ops():
    scene = _two_attached()
    assert scene.ops_to_materialize() == [
        ("attach", "left"),
        ("attach", "right"),
    ]


def test_copy_and_key_track_all_attachments_without_aliasing():
    scene = _two_attached()
    copied = scene.copy()
    assert copied.key() == scene.key()
    assert copied.attached_objects == scene.attached_objects

    changed = copied.with_detached("left")
    assert changed.key() != scene.key()
    # The original cumulative diff is unchanged by the derived copy.
    assert scene.attached_names() == ["left", "right"]


def test_is_empty_accounts_for_multi_attachment():
    assert SceneDiff().is_empty()
    assert not _two_attached().is_empty()
    assert not _two_attached().with_detached("left").is_empty()
    assert not _two_attached().with_detached(None).is_empty()


def test_single_attachment_constructor_remains_compatible():
    scene = SceneDiff(attached_object="object")
    assert scene.attached_names() == ["object"]
    assert scene.attached_objects == {"object": None}
    assert scene.attached_object == "object"

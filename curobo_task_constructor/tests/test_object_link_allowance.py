"""The adapter must keep the object half of allowCollisions(object, links).

Before this, the SceneDiff pairs were flattened to a bare link list, so the
server disabled those links against every obstacle. That is what made a
descend ignore the table. These tests pin the object-scoped call shape and the
legacy fallback separately.
"""

from __future__ import annotations

from types import SimpleNamespace

from curobo_task_constructor.robot.plan_client import _PlanClientMixin


class _Dummy(_PlanClientMixin):
    def __init__(self):
        self.calls = []
        self._allowance_tls = SimpleNamespace(depth=0, links=None)
        self._session = SimpleNamespace(
            collision_transition_needed=lambda *a, **k: True,
            note_collision_sent=lambda *a, **k: None,
        )

    def _send_object_collision_update(self, object_name, links, enabled):
        self.calls.append((object_name, list(links), bool(enabled)))

    def _send_collision_update(self, links, enabled):
        self.calls.append(("__legacy__", list(links), bool(enabled)))


def _request(pairs=(), links=()):
    return SimpleNamespace(
        scene=SimpleNamespace(disabled_collisions=set(pairs)),
        goalsets=[SimpleNamespace(allowed_collisions=list(links))],
    )


def test_scene_pairs_are_grouped_by_object_and_keep_the_links():
    dummy = _Dummy()
    request = _request(
        pairs={
            ("object_0", "finger_1"),
            ("object_0", "finger_2"),
            ("object_1", "wrist"),
        }
    )
    _PlanClientMixin._set_links_collision(dummy, [request], False)
    assert dummy.calls == [
        ("object_0", ["finger_1", "finger_2"], False),
        ("object_1", ["wrist"], False),
    ]


def test_an_object_allowance_is_restored_with_the_same_object():
    dummy = _Dummy()
    request = _request(pairs={("object_0", "finger_1")})
    _PlanClientMixin._set_links_collision(dummy, [request], True)
    assert dummy.calls == [("object_0", ["finger_1"], True)]


def test_callers_without_scene_pairs_keep_the_legacy_link_switch():
    dummy = _Dummy()
    request = SimpleNamespace(
        scene=None,
        goalsets=[SimpleNamespace(allowed_collisions=["finger_1", "finger_2"])],
    )
    _PlanClientMixin._set_links_collision(dummy, [request], False)
    assert dummy.calls == [
        ("__legacy__", ["finger_1", "finger_2"], False),
    ]

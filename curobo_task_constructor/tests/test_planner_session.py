"""Sticky planner/collision sessions: skip identical setup calls, never skip
across state changes.

The server switches planners and collision allowances on its own during
execution replay, so the session must forget everything on execute() and on
any failed call — a stale skip would plan under the wrong planner or with
the wrong spheres off. Within one planning burst (single thread, no executes
interleaved) identical repeats are pure waste: three best-of-3 attempts of
one request, consecutive same-planner legs.
"""

import pytest

from curobo_task_constructor.robot.curobo import PlannerSessionState


def test_first_planner_request_always_sends():
    s = PlannerSessionState()
    assert s.planner_request_needed(5) is True


def test_identical_planner_repeats_skip():
    s = PlannerSessionState()
    assert s.planner_request_needed(5) is True
    s.note_planner_sent(5)
    assert s.planner_request_needed(5) is False
    assert s.planner_request_needed(5) is False


def test_planner_change_sends():
    s = PlannerSessionState()
    s.note_planner_sent(5)
    assert s.planner_request_needed(0) is True
    s.note_planner_sent(0)
    assert s.planner_request_needed(0) is False
    assert s.planner_request_needed(5) is True


def test_execute_invalidates_planner():
    s = PlannerSessionState()
    s.note_planner_sent(5)
    assert s.planner_request_needed(5) is False
    s.invalidate()
    assert s.planner_request_needed(5) is True


def test_error_invalidates_planner():
    s = PlannerSessionState()
    s.note_planner_sent(5)
    s.invalidate()
    assert s.planner_request_needed(5) is True


def test_empty_links_never_need_a_call():
    s = PlannerSessionState()
    assert s.collision_transition_needed([], False) is False
    assert s.collision_transition_needed([], True) is False
    s.note_collision_sent(["a"], False)
    assert s.collision_transition_needed([], False) is False


def test_identical_collision_repeats_skip():
    s = PlannerSessionState()
    links = ["left_outer_finger", "right_outer_finger"]
    assert s.collision_transition_needed(links, False) is True
    s.note_collision_sent(links, False)
    assert s.collision_transition_needed(links, False) is False
    # Order-insensitive: same set in another order is the same state.
    assert s.collision_transition_needed(
        ["right_outer_finger", "left_outer_finger"], False) is False


def test_collision_enable_flap_sends_each_transition():
    s = PlannerSessionState()
    links = ["a"]
    s.note_collision_sent(links, False)
    assert s.collision_transition_needed(links, True) is True
    s.note_collision_sent(links, True)
    assert s.collision_transition_needed(links, True) is False
    assert s.collision_transition_needed(links, False) is True


def test_collision_link_change_sends():
    s = PlannerSessionState()
    s.note_collision_sent(["a"], False)
    assert s.collision_transition_needed(["a", "b"], False) is True


def test_execute_invalidates_collision():
    s = PlannerSessionState()
    s.note_collision_sent(["a"], False)
    assert s.collision_transition_needed(["a"], False) is False
    s.invalidate()
    assert s.collision_transition_needed(["a"], False) is True


def test_planner_and_collision_track_independently():
    s = PlannerSessionState()
    s.note_planner_sent(5)
    s.note_collision_sent(["a"], False)
    s.invalidate()
    assert s.planner_request_needed(5) is True
    assert s.collision_transition_needed(["a"], False) is True
    # Re-arming one must not re-arm the other.
    s.note_planner_sent(5)
    assert s.planner_request_needed(5) is False
    assert s.collision_transition_needed(["a"], False) is True


class _FakeFuture:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error
        self._callbacks = []
        self._done = response is not None or error is not None
        self.cancelled = False

    def done(self):
        return self._done

    def result(self):
        if self._error is not None:
            raise self._error
        return self._response

    def cancel(self):
        self.cancelled = True
        self._done = True
        return True

    def add_done_callback(self, callback):
        if self._done:
            callback(self)
        else:
            self._callbacks.append(callback)


class _FakeClient:
    """Recording stand-in for an rclpy service client."""

    def __init__(self, response=None, error=None):
        self.srv_name = "fake"
        self._response = response
        self._error = error
        self.requests = []

    def wait_for_service(self, timeout_sec=None):
        return True

    def call_async(self, request):
        self.requests.append(request)
        return _FakeFuture(response=self._response, error=self._error)


def _make_iface():
    """CuroboServerInterface on a real node with fake service clients.

    Needs rclpy (init here); the interface itself never spins in these
    tests — pre-resolved futures complete inline.
    """
    rclpy = pytest.importorskip("rclpy")
    from curobov2_ros_interfaces.srv import SetLinkCollision, SetPlanner

    from curobo_task_constructor.robot.curobo import CuroboServerInterface

    mine = not rclpy.ok()
    if mine:
        rclpy.init()
    try:
        node = rclpy.create_node("planner_session_probe")
        try:
            iface = CuroboServerInterface(node, service_timeout=5.0)
        finally:
            node.destroy_node()
        iface._planner_client = _FakeClient(
            response=SetPlanner.Response(success=True))
        iface._link_collision_client = _FakeClient(
            response=SetLinkCollision.Response(success=True))
        return iface
    finally:
        if mine and rclpy.ok():
            rclpy.shutdown()


def test_wire_skips_identical_planner_repeats():
    from curobov2_ros_interfaces.srv import SetPlanner  # noqa: F401 (type check)

    iface = _make_iface()
    iface._ensure_planner(5)
    iface._ensure_planner(5)
    iface._ensure_planner(5)
    assert [r.planner_type for r in iface._planner_client.requests] == [5]
    iface._ensure_planner(0)
    assert [r.planner_type for r in iface._planner_client.requests] == [5, 0]


def test_wire_skips_identical_collision_repeats():
    from curobo_task_constructor.core.robot import GoalsetSpec, PlanRequest

    def req():
        return PlanRequest(
            goalsets=[GoalsetSpec(allowed_collisions=["b", "a"])])

    iface = _make_iface()
    iface._set_links_collision([req()], False)
    iface._set_links_collision([req()], False)
    iface._set_links_collision([req()], True)
    iface._set_links_collision([req()], True)
    iface._set_links_collision([req()], False)
    sent = [(sorted(r.link_names), r.enabled)
            for r in iface._link_collision_client.requests]
    assert sent == [(["a", "b"], False), (["a", "b"], True),
                    (["a", "b"], False)]


def test_wire_failure_forgets_session():
    from curobov2_ros_interfaces.srv import SetPlanner
    from curobo_task_constructor.core.robot import ServiceError

    iface = _make_iface()
    iface._ensure_planner(5)
    assert len(iface._planner_client.requests) == 1
    iface._planner_client = _FakeClient(error=RuntimeError("down"))
    with pytest.raises(ServiceError):
        iface._ensure_planner(0)
    iface._planner_client = _FakeClient(
        response=SetPlanner.Response(success=True))
    # The failure forgot the session: the same key sends again...
    iface._ensure_planner(0)
    assert len(iface._planner_client.requests) == 1
    # ...and identical repeats skip again.
    iface._ensure_planner(0)
    assert len(iface._planner_client.requests) == 1


def _links_req(*links):
    from curobo_task_constructor.core.robot import GoalsetSpec, PlanRequest

    return PlanRequest(goalsets=[GoalsetSpec(allowed_collisions=list(links))])


def test_scope_holds_allowance_across_nested_calls():
    # One stage attempt loop: outer scope + 3 nested plan() OFF/ON pairs
    # (as robot.plan brackets each call) collapse to a single OFF...ON.
    iface = _make_iface()
    req = _links_req("a", "b")
    with iface.allowance_scope([req]):
        for _ in range(3):
            iface._set_links_collision([req], False)
            iface._set_links_collision([req], True)
    sent = [(sorted(r.link_names), r.enabled)
            for r in iface._link_collision_client.requests]
    assert sent == [(["a", "b"], False), (["a", "b"], True)]


def test_standalone_calls_keep_today_behavior():
    # No scope: each balanced pair sends, exactly like before the change.
    iface = _make_iface()
    req = _links_req("a")
    iface._set_links_collision([req], False)
    iface._set_links_collision([req], True)
    iface._set_links_collision([req], False)
    iface._set_links_collision([req], True)
    sent = [(sorted(r.link_names), r.enabled)
            for r in iface._link_collision_client.requests]
    assert sent == [(["a"], False), (["a"], True),
                    (["a"], False), (["a"], True)]


def test_scopes_are_thread_independent():
    import threading

    iface = _make_iface()
    errors = []

    def worker():
        try:
            req = _links_req("a")
            with iface.allowance_scope([req]):
                for _ in range(2):
                    iface._set_links_collision([req], False)
                    iface._set_links_collision([req], True)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    sent = [(sorted(r.link_names), r.enabled)
            for r in iface._link_collision_client.requests]
    # Two threads x one OFF...ON each; order between threads is free.
    assert sorted(sent) == [(["a"], False), (["a"], False),
                            (["a"], True), (["a"], True)]


def test_scope_exception_still_restores():
    iface = _make_iface()
    req = _links_req("a")
    with pytest.raises(RuntimeError):
        with iface.allowance_scope([req]):
            raise RuntimeError("stage blew up")
    sent = [(sorted(r.link_names), r.enabled)
            for r in iface._link_collision_client.requests]
    assert sent == [(["a"], False), (["a"], True)]

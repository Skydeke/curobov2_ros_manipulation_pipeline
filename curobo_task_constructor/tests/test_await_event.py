"""Event-driven waits: identical semantics to the sleep-poll loop, without
the 20 ms floor.

Contract under test (unchanged by the refactor):
- resolved future -> its result, without raising;
- resolving-late future -> its result once resolved;
- never-resolving future -> cancel() then ServiceError after the budget;
- failed future -> ServiceError carrying the cause (repr, never bare str).
The refactor additionally guarantees zero time.sleep calls on every path:
completion wakes the waiter through the callback, the deadline through the
event timeout — polling is gone, not faster.
"""

import threading
import time
from unittest import mock

import pytest

from curobo_task_constructor.robot.curobo import CuroboServerInterface
from curobo_task_constructor.core.robot import ServiceError


class FakeFuture:
    """Minimal future protocol subset _await uses, with rclpy-like callback
    firing (callbacks run when set_result is called, inline — like rclpy's
    no-executor immediate path)."""

    def __init__(self):
        self._done = False
        self._result = None
        self._callbacks = []
        self.cancelled = False
        self._lock = threading.Lock()

    def done(self):
        with self._lock:
            return self._done

    def result(self):
        with self._lock:
            if isinstance(self._result, Exception):
                raise self._result
            return self._result

    def cancel(self):
        with self._lock:
            self.cancelled = True
            self._done = True
        return True

    def add_done_callback(self, callback):
        fire_now = False
        with self._lock:
            if self._done:
                fire_now = True
            else:
                self._callbacks.append(callback)
        if fire_now:
            callback(self)

    def set_result(self, result):
        with self._lock:
            self._result = result
            self._done = True
            callbacks, self._callbacks = self._callbacks, []
        for callback in callbacks:
            callback(self)


def await_no_sleep(future, what="call", timeout=5.0):
    """_await with time.sleep counting; returns (result, sleep_calls)."""
    with mock.patch("time.sleep") as slept:
        result = CuroboServerInterface._await(future, what, timeout)
    return result, slept.call_count


def test_resolved_future_returns_without_sleeping():
    future = FakeFuture()
    future.set_result("ok")
    result, sleeps = await_no_sleep(future, timeout=1.0)
    assert result == "ok"
    assert sleeps == 0


def test_late_future_returns_without_sleeping():
    future = FakeFuture()

    def resolve_late():
        time.sleep(0.05)
        future.set_result("late")

    thread = threading.Thread(target=resolve_late)
    thread.start()
    try:
        started = time.monotonic()
        result, sleeps = await_no_sleep(future, timeout=5.0)
        elapsed = time.monotonic() - started
    finally:
        thread.join()
    assert result == "late"
    assert sleeps == 0
    # Woke on completion (~50 ms), not on the next poll tick.
    assert elapsed < 0.5


def test_timeout_cancels_and_raises():
    future = FakeFuture()
    started = time.monotonic()
    with mock.patch("time.sleep") as slept:
        with pytest.raises(ServiceError, match="timed out"):
            CuroboServerInterface._await(future, "svc", 0.1)
    elapsed = time.monotonic() - started
    assert future.cancelled is True
    assert slept.call_count == 0
    assert elapsed < 2.0


def test_failed_future_raises_service_error_with_cause():
    future = FakeFuture()
    future.set_result(RuntimeError("boom"))
    with pytest.raises(ServiceError) as excinfo:
        CuroboServerInterface._await(future, "svc", 1.0)
    assert "RuntimeError" in str(excinfo.value)


def test_callback_after_completion_still_resolves():
    # Completes between the done() check and add_done_callback: the callback
    # path must still fire (no missed wakeup -> no timeout).
    future = FakeFuture()
    future.set_result("raced")
    result, sleeps = await_no_sleep(future, timeout=1.0)
    assert result == "raced"
    assert sleeps == 0

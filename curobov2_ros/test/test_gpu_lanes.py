import threading, time
import pytest
from curobov2_ros.core.gpu_lanes import (
    GpuScheduler, GpuLane, CaptureBarrier, LanePoisoned, JobCancelled, NotLaneOwner,
    PRIO_REACTIVE, PRIO_PLAN, PRIO_BULK,
)


@pytest.fixture
def sch():
    s = GpuScheduler()
    yield s
    s.shutdown()


def test_result_and_exception_roundtrip(sch):
    ln = sch.lane("a", use_cuda_stream=False)
    assert ln.call(lambda x: x + 1, 41) == 42
    with pytest.raises(ValueError):
        ln.call(lambda: (_ for _ in ()).throw(ValueError("boom")))
    assert not ln.poisoned  # ordinary errors do not poison


def test_single_thread_ownership(sch):
    ln = sch.lane("a", use_cuda_stream=False)
    names = {ln.call(lambda: threading.current_thread().name) for _ in range(20)}
    assert names == {"gpu-lane-a"}
    with pytest.raises(NotLaneOwner):
        ln.assert_owner("solver")
    ln.call(lambda: ln.assert_owner("solver"))  # inside lane: ok


def test_submit_from_inside_lane_runs_inline_no_deadlock(sch):
    ln = sch.lane("a", use_cuda_stream=False)
    assert ln.call(lambda: ln.call(lambda: 7), timeout=2) == 7


def test_priority_order(sch):
    ln = sch.lane("a", use_cuda_stream=False)
    gate = threading.Event()
    order = []
    ln.submit(lambda: gate.wait(2))           # occupy the lane
    time.sleep(0.05)
    fs = [ln.submit(order.append, "bulk", prio=PRIO_BULK),
          ln.submit(order.append, "plan", prio=PRIO_PLAN),
          ln.submit(order.append, "react", prio=PRIO_REACTIVE)]
    gate.set()
    for f in fs:
        f.result(2)
    assert order == ["react", "plan", "bulk"]


def test_fifo_within_priority(sch):
    ln = sch.lane("a", use_cuda_stream=False)
    gate = threading.Event(); out = []
    ln.submit(lambda: gate.wait(2)); time.sleep(0.05)
    fs = [ln.submit(out.append, i) for i in range(10)]
    gate.set(); [f.result(2) for f in fs]
    assert out == list(range(10))


def test_cancel_before_start(sch):
    ln = sch.lane("a", use_cuda_stream=False)
    gate = threading.Event(); ev = threading.Event()
    ln.submit(lambda: gate.wait(2)); time.sleep(0.05)
    f = ln.submit(lambda: 1, cancel=ev)
    ev.set(); gate.set()
    with pytest.raises(JobCancelled):
        f.result(2)


def test_poison_fails_fast_and_callback(sch):
    seen = []
    s = GpuScheduler(on_poisoned=lambda l, e: seen.append(l.name))
    try:
        ln = s.lane("p", use_cuda_stream=False)
        with pytest.raises(RuntimeError):
            ln.call(lambda: (_ for _ in ()).throw(RuntimeError("CUDA error: an illegal memory access was encountered")))
        assert ln.poisoned and seen == ["p"] and s.any_poisoned()
        with pytest.raises(LanePoisoned):
            ln.call(lambda: 1)
    finally:
        s.shutdown()


def test_exclusive_quiesces_other_lanes(sch):
    a = sch.lane("a", use_cuda_stream=False)
    b = sch.lane("b", use_cuda_stream=False)
    running_b = threading.Event(); stop_b = threading.Event()
    violations = []
    state = {"exclusive": False}

    def b_job():
        if state["exclusive"]:
            violations.append("b ran during exclusive")
        running_b.set()
        time.sleep(0.02)

    def a_capture():
        with sch.exclusive():
            state["exclusive"] = True
            time.sleep(0.2)               # "capture" window
            state["exclusive"] = False

    def hammer():
        while not stop_b.is_set():
            b.call(b_job)

    t = threading.Thread(target=hammer); t.start()
    running_b.wait(2)
    a.call(a_capture, timeout=5)
    stop_b.set(); t.join(5)
    assert violations == []


def test_two_lanes_both_want_exclusive_no_deadlock(sch):
    a = sch.lane("a", use_cuda_stream=False)
    b = sch.lane("b", use_cuda_stream=False)
    done = []

    def cap(tag):
        with sch.exclusive():
            time.sleep(0.05)
            done.append(tag)

    fa = a.submit(cap, "a"); fb = b.submit(cap, "b")
    fa.result(3); fb.result(3)
    assert sorted(done) == ["a", "b"]


def test_lanes_actually_overlap(sch):
    a = sch.lane("a", use_cuda_stream=False)
    b = sch.lane("b", use_cuda_stream=False)
    t0 = time.monotonic()
    fa = a.submit(time.sleep, 0.3); fb = b.submit(time.sleep, 0.3)
    fa.result(2); fb.result(2)
    assert time.monotonic() - t0 < 0.5   # parallel, not 0.6


def test_writer_not_starved(sch):
    a = sch.lane("a", use_cuda_stream=False)
    b = sch.lane("b", use_cuda_stream=False)
    stop = threading.Event()

    def hammer():
        while not stop.is_set():
            b.call(time.sleep, 0.005)

    t = threading.Thread(target=hammer); t.start()
    t0 = time.monotonic()
    def cap():
        with sch.exclusive():
            pass
    a.call(cap, timeout=2)
    assert time.monotonic() - t0 < 1.0
    stop.set(); t.join(2)


def test_submit_if_idle_drops_when_busy(sch):
    ln = sch.lane("v", use_cuda_stream=False)
    gate = threading.Event()
    ln.submit(lambda: gate.wait(2)); time.sleep(0.05)
    assert ln.submit_if_idle(lambda: 1) is None
    gate.set(); time.sleep(0.05)
    f = ln.submit_if_idle(lambda: 5)
    assert f is not None and f.result(1) == 5


def test_shutdown_fails_pending(sch):
    s = GpuScheduler()
    ln = s.lane("x", use_cuda_stream=False)
    gate = threading.Event()
    ln.submit(lambda: gate.wait(1))
    time.sleep(0.05)
    f = ln.submit(lambda: 1)
    gate.set()
    s.shutdown()
    assert f.done()


# ---- barrier / legacy lock -------------------------------------------------
from curobov2_ros.core.gpu_lanes import LegacyGpuLock, DeadlockRisk


def test_try_exclusive_fails_while_reader_and_state_is_restored():
    b = CaptureBarrier()
    got = {}
    ready = threading.Event(); release = threading.Event()

    def reader():
        with b.shared():
            ready.set(); release.wait(2)

    t = threading.Thread(target=reader); t.start(); ready.wait(2)
    assert b.acquire_exclusive(blocking=False) is False
    # a thread that holds shared and fails a try-upgrade must keep its shared hold
    with b.shared():
        assert b.acquire_exclusive(blocking=False) is False
        assert b.holds_shared()
    release.set(); t.join(2)
    assert b.acquire_exclusive(blocking=False) is True
    b.release_exclusive()


def test_upgrade_restores_shared_hold():
    b = CaptureBarrier()
    with b.shared():
        with b.exclusive():
            assert b.holds_exclusive()
        assert b.holds_shared() and not b.holds_exclusive()
    assert not b.holds_shared()
    assert b.acquire_exclusive(blocking=False)
    b.release_exclusive()


def test_legacy_lock_semantics():
    b = CaptureBarrier(); lk = LegacyGpuLock(b)
    with lk:
        with lk:  # re-entrant like RLock
            pass
        res = {}
        t = threading.Thread(target=lambda: res.setdefault("ok", lk.acquire(blocking=False)))
        t.start(); t.join(2)
        assert res["ok"] is False  # other thread cannot take it
    t = threading.Thread(target=lambda: res.update(ok2=lk.acquire(blocking=False)))
    t.start(); t.join(2)
    assert res["ok2"] is True


def test_waiting_on_lane_while_holding_legacy_lock_raises_instead_of_deadlocking(sch):
    ln = sch.lane("a", use_cuda_stream=False)
    lk = sch.legacy_lock()
    with lk:
        with pytest.raises(DeadlockRisk):
            ln.submit(lambda: 1)


def test_legacy_lock_blocks_lane_jobs_until_released(sch):
    ln = sch.lane("a", use_cuda_stream=False)
    lk = sch.legacy_lock()
    out = []
    futs = []
    lk.acquire()
    # submit from ANOTHER thread (the holder itself would trip the deadlock guard)
    t = threading.Thread(target=lambda: futs.append(ln.submit(out.append, 1)))
    t.start(); t.join(2)
    time.sleep(0.15)
    assert out == []          # lane job waits for the legacy holder
    lk.release()
    futs[0].result(2)
    assert out == [1]


def test_sync_hook_runs_once_per_outermost_exclusive_enter_and_exit():
    b = CaptureBarrier(); calls = []
    b.sync_hook = lambda: calls.append(1)
    with b.exclusive():
        with b.exclusive():          # nested: no extra calls
            pass
        assert len(calls) == 1
    assert len(calls) == 2
    with b.shared():                 # shared never syncs
        pass
    assert len(calls) == 2


def test_sync_hook_runs_on_upgrade_from_shared():
    b = CaptureBarrier(); calls = []
    b.sync_hook = lambda: calls.append(1)
    with b.shared():
        with b.exclusive():
            pass
    assert len(calls) == 2


def test_scheduler_installs_a_sync_hook_that_is_safe_without_cuda(sch):
    assert sch.barrier.sync_hook is not None
    with sch.exclusive():            # must not raise on a CPU-only box
        pass

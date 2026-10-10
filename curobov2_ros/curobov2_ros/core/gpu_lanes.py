# SPDX-License-Identifier: Apache-2.0
"""GPU lane scheduler for the curobov2_ros server.

WHY THIS EXISTS
---------------
The node used ONE exclusive ``gpu_lock`` for two different jobs:
  (a) keep CUDA-graph *capture* (process-global) away from all other CUDA work;
  (b) serialise *all* GPU use (plans, IK, FK, depth integrate, viz).
Job (b) makes every GPU user wait for every other one, and the lock is still
not taken by every CUDA user (reactive MPC replays, result ``.cpu()`` calls),
which is how capture/replay races -> illegal memory access happen.

THE MODEL
---------
* A **lane** is ONE Python thread that is the only thread allowed to touch a
  given group of solvers / graphs / device tensors. Everyone else submits
  closures and gets a ``Future`` back (results must be plain CPU data).
* Each lane runs on its **own CUDA stream** (when torch+CUDA exist), so lanes
  can overlap on the GPU.
* Every job runs inside ``barrier.shared()``. Anything that may CAPTURE, reset
  or rebuild a CUDA graph (or mutate state shared between lanes) runs inside
  ``barrier.exclusive()``: other lanes finish their current job and pause until
  it ends. Replays / eager work never take the exclusive side.
* Priorities: lower number = served first (REACTIVE < PLAN < KIN < BULK).
* A lane that sees a CUDA error is **poisoned**: later jobs fail fast with
  ``LanePoisoned`` instead of hammering a dead context.

This module imports torch lazily and works without it (unit tests use fakes).
"""
from __future__ import annotations

import itertools
import logging
import queue
import threading
import time
from concurrent.futures import Future
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

log = logging.getLogger(__name__)

# Priorities (lower = more urgent).
PRIO_REACTIVE = 0
PRIO_PLAN = 10
PRIO_KIN = 20
PRIO_PERCEPTION = 30
PRIO_BULK = 40

_CUDA_FATAL_MARKERS = (
    "illegal memory access",
    "cudaerrorstreamcapture",
    "cuda error",
    "cuda_error",
    "device-side assert",
    "cugraphlaunch",
)


class LanePoisoned(RuntimeError):
    """The lane hit a fatal CUDA error; the CUDA context is unusable."""


class LaneStopped(RuntimeError):
    """The lane was shut down before the job could run."""


class JobCancelled(RuntimeError):
    """The job's cancel event was set before it started."""


class NotLaneOwner(RuntimeError):
    """Owned GPU state was touched from a thread that is not its lane."""


# --------------------------------------------------------------------------
# Capture barrier (readers = normal jobs, writer = capture / reset / rebuild)
# --------------------------------------------------------------------------
class DeadlockRisk(RuntimeError):
    """A thread holding the exclusive side tried to wait on a lane."""


class CaptureBarrier:
    """Writer-preferring read/write lock, re-entrant, with same-thread upgrade.

    shared    : normal lane jobs (replay / eager work). Many at once.
    exclusive : graph capture / reset / rebuild / edits of state shared between
                lanes. Waits for every other thread's shared section to end and
                keeps new ones out until released.
    A thread already inside shared() may call exclusive(): its read hold is
    dropped while waiting and restored afterwards, so two lanes that both want
    exclusive cannot deadlock each other.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writer: Optional[int] = None
        self._writers_waiting = 0
        self._tl = threading.local()  # shared_depth, excl_depth, saved_shared
        # Called (with no lock held by the hook itself) right after the OUTERMOST
        # exclusive acquire and right before the OUTERMOST exclusive release.
        # The scheduler points it at a device-wide cuda synchronize so work
        # queued on one lane's stream is finished before another lane captures
        # or edits shared state, and vice versa.
        self.sync_hook: Optional[Callable[[], None]] = None

    def _sd(self) -> int:
        return getattr(self._tl, "shared_depth", 0)

    def _ed(self) -> int:
        return getattr(self._tl, "excl_depth", 0)

    def holds_exclusive(self) -> bool:
        return self._ed() > 0

    def holds_shared(self) -> bool:
        return self._sd() > 0

    def _run_sync_hook(self) -> None:
        hook = self.sync_hook
        if hook is not None:
            hook()

    # ---- shared
    def acquire_shared(self) -> None:
        if self._sd() > 0 or self._ed() > 0:  # nested
            self._tl.shared_depth = self._sd() + 1
            return
        with self._cond:
            while self._writer is not None or self._writers_waiting > 0:
                self._cond.wait()
            self._readers += 1
        self._tl.shared_depth = 1

    def release_shared(self) -> None:
        d = self._sd()
        if d <= 0:
            raise RuntimeError("release_shared without acquire_shared")
        self._tl.shared_depth = d - 1
        if d == 1 and self._ed() == 0:
            with self._cond:
                self._readers -= 1
                self._cond.notify_all()

    # ---- exclusive
    def acquire_exclusive(self, blocking: bool = True) -> bool:
        if self._ed() > 0:  # re-entrant
            self._tl.excl_depth = self._ed() + 1
            return True
        me = threading.get_ident()
        held_shared = self._sd() > 0
        with self._cond:
            if held_shared:
                self._readers -= 1  # drop our own read hold while we wait
            if not blocking and (self._writer is not None or self._readers > 0):
                if held_shared:
                    self._readers += 1
                return False
            self._writers_waiting += 1
            try:
                while self._writer is not None or self._readers > 0:
                    self._cond.wait()
            finally:
                self._writers_waiting -= 1
            self._writer = me
        self._tl.saved_shared = self._sd() if held_shared else 0
        self._tl.shared_depth = 0
        self._tl.excl_depth = 1
        self._run_sync_hook()
        return True

    def release_exclusive(self) -> None:
        d = self._ed()
        if d <= 0:
            raise RuntimeError("release_exclusive without acquire_exclusive")
        if d == 1:
            self._run_sync_hook()
        self._tl.excl_depth = d - 1
        if d > 1:
            return
        saved = getattr(self._tl, "saved_shared", 0)
        self._tl.saved_shared = 0
        with self._cond:
            self._writer = None
            if saved > 0:
                self._readers += 1  # restore the read hold we dropped
            self._cond.notify_all()
        self._tl.shared_depth = saved

    @contextmanager
    def shared(self):
        self.acquire_shared()
        try:
            yield
        finally:
            self.release_shared()

    @contextmanager
    def exclusive(self):
        self.acquire_exclusive(True)
        try:
            yield
        finally:
            self.release_exclusive()


class LegacyGpuLock:
    """Drop-in for the old ``threading.RLock`` named ``node.gpu_lock``.

    Maps the old semantics ("nobody else touches CUDA while I hold it") onto
    the barrier's exclusive side. Supports ``with``, ``acquire(blocking=...)``
    and ``release()``. Migrated code should stop using it; it exists so
    not-yet-migrated call sites stay correct next to lanes.
    """

    def __init__(self, barrier: CaptureBarrier) -> None:
        self._b = barrier

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        return self._b.acquire_exclusive(blocking)

    def release(self) -> None:
        self._b.release_exclusive()

    def __enter__(self):
        self._b.acquire_exclusive(True)
        return self

    def __exit__(self, *exc):
        self._b.release_exclusive()
        return False


# --------------------------------------------------------------------------
# Lane
# --------------------------------------------------------------------------
_seq = itertools.count()


class _Job:
    __slots__ = ("prio", "seq", "fn", "args", "kwargs", "future", "cancel", "name", "t_submit")

    def __init__(self, prio, fn, args, kwargs, future, cancel, name):
        self.prio = prio
        self.seq = next(_seq)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.future = future
        self.cancel = cancel
        self.name = name
        self.t_submit = time.monotonic()

    def __lt__(self, other: "_Job"):
        return (self.prio, self.seq) < (other.prio, other.seq)


class GpuLane:
    """One thread = sole owner of a set of solvers/graphs/tensors."""

    def __init__(
        self,
        name: str,
        barrier: CaptureBarrier,
        *,
        use_cuda_stream: bool = True,
        device: Optional[str] = None,
        on_poisoned: Optional[Callable[["GpuLane", BaseException], None]] = None,
    ) -> None:
        self.name = name
        self._barrier = barrier
        self._q: "queue.PriorityQueue[_Job]" = queue.PriorityQueue()
        self._stop = threading.Event()
        self._poisoned: Optional[BaseException] = None
        self._use_stream = use_cuda_stream
        self._device = device
        self._on_poisoned = on_poisoned
        self._busy = False
        self._cuda_stream = None
        self.stats = {"jobs": 0, "failed": 0, "max_wait_s": 0.0, "max_run_s": 0.0}
        self._thread = threading.Thread(target=self._run, name=f"gpu-lane-{name}", daemon=True)
        self._started = threading.Event()
        self._thread.start()
        self._started.wait(10.0)

    # -- public API ---------------------------------------------------
    @property
    def poisoned(self) -> bool:
        return self._poisoned is not None

    def owns_current_thread(self) -> bool:
        return threading.get_ident() == self._thread.ident

    def assert_owner(self, what: str = "") -> None:
        if not self.owns_current_thread():
            raise NotLaneOwner(
                f"{what or 'GPU state'} owned by lane '{self.name}' was touched from "
                f"thread '{threading.current_thread().name}'. Submit a job instead."
            )

    def idle(self) -> bool:
        return (not self._busy) and self._q.empty()

    def submit(
        self,
        fn: Callable[..., Any],
        *args: Any,
        prio: int = PRIO_PLAN,
        cancel: Optional[threading.Event] = None,
        name: str = "",
        **kwargs: Any,
    ) -> Future:
        fut: Future = Future()
        if self._stop.is_set():
            fut.set_exception(LaneStopped(self.name))
            return fut
        if self._poisoned is not None:
            fut.set_exception(LanePoisoned(f"lane '{self.name}': {self._poisoned!r}"))
            return fut
        if self.owns_current_thread():
            # Called from inside the lane itself: run inline (no deadlock).
            self._execute_inline(fut, fn, args, kwargs)
            return fut
        if self._barrier.holds_exclusive():
            # The lane's job needs the shared side, which we are blocking.
            raise DeadlockRisk(
                f"thread '{threading.current_thread().name}' holds the exclusive GPU "
                f"barrier (legacy gpu_lock?) and tried to wait on lane '{self.name}'. "
                "Release it first or move this code into the lane."
            )
        self._q.put(_Job(prio, fn, args, kwargs, fut, cancel, name or getattr(fn, "__name__", "job")))
        return fut

    def call(self, fn: Callable[..., Any], *args: Any, timeout: Optional[float] = None, **kw: Any) -> Any:
        """submit() + wait. Re-raises the job's exception."""
        return self.submit(fn, *args, **kw).result(timeout=timeout)

    def submit_if_idle(self, fn: Callable[..., Any], *args: Any, **kw: Any) -> Optional[Future]:
        """Drop-if-busy submit for rate-limited publishers (viz, stats)."""
        if not self.idle() or self._poisoned is not None or self._stop.is_set():
            return None
        return self.submit(fn, *args, prio=kw.pop("prio", PRIO_BULK), **kw)

    def shutdown(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._q.put(_Job(10**9, lambda: None, (), {}, Future(), None, "stop"))
        self._thread.join(timeout)
        # fail anything left
        while True:
            try:
                j = self._q.get_nowait()
            except queue.Empty:
                break
            if not j.future.done():
                j.future.set_exception(LaneStopped(self.name))

    # -- internals ----------------------------------------------------
    def _is_fatal(self, exc: BaseException) -> bool:
        msg = str(exc).lower()
        return isinstance(exc, RuntimeError) and any(m in msg for m in _CUDA_FATAL_MARKERS)

    def _execute_inline(self, fut: Future, fn, args, kwargs) -> None:
        try:
            fut.set_result(fn(*args, **kwargs))
        except BaseException as e:  # noqa: BLE001
            fut.set_exception(e)

    def _run(self) -> None:
        stream_ctx = None
        try:
            try:
                import torch  # type: ignore

                if self._use_stream and torch.cuda.is_available():
                    dev = torch.device(self._device or "cuda")
                    torch.cuda.set_device(dev)
                    stream = torch.cuda.Stream(device=dev)
                    stream_ctx = torch.cuda.stream(stream)
                    stream_ctx.__enter__()
                    self._cuda_stream = stream
            except Exception as e:  # noqa: BLE001
                log.debug("lane %s: no CUDA stream (%s)", self.name, e)
                stream_ctx = None
            self._started.set()
            while True:
                job = self._q.get()
                if self._stop.is_set() and job.prio >= 10**9:
                    return
                if self._stop.is_set():
                    if not job.future.done():
                        job.future.set_exception(LaneStopped(self.name))
                    continue
                self._run_job(job)
        finally:
            self._started.set()
            if stream_ctx is not None:
                try:
                    stream_ctx.__exit__(None, None, None)
                except Exception:  # noqa: BLE001
                    pass

    def _run_job(self, job: _Job) -> None:
        fut = job.future
        if not fut.set_running_or_notify_cancel():
            return
        if self._poisoned is not None:
            fut.set_exception(LanePoisoned(f"lane '{self.name}': {self._poisoned!r}"))
            return
        if job.cancel is not None and job.cancel.is_set():
            fut.set_exception(JobCancelled(job.name))
            return
        t0 = time.monotonic()
        self.stats["max_wait_s"] = max(self.stats["max_wait_s"], t0 - job.t_submit)
        self._busy = True
        try:
            with self._barrier.shared():
                res = job.fn(*job.args, **job.kwargs)
                if self._cuda_stream is not None:
                    # Finish this lane's GPU work BEFORE leaving the shared side: an
                    # exclusive section (capture / edit of shared tensors) or another
                    # lane's reader of our results must never overlap in-flight kernels.
                    # Also surfaces async CUDA errors here, where they poison the lane.
                    self._cuda_stream.synchronize()
            fut.set_result(res)
        except BaseException as e:  # noqa: BLE001
            self.stats["failed"] += 1
            if self._is_fatal(e):
                self._poisoned = e
                log.error("lane %s POISONED by %r", self.name, e)
                if self._on_poisoned:
                    try:
                        self._on_poisoned(self, e)
                    except Exception:  # noqa: BLE001
                        log.exception("on_poisoned callback failed")
            fut.set_exception(e)
        finally:
            self._busy = False
            self.stats["jobs"] += 1
            self.stats["max_run_s"] = max(self.stats["max_run_s"], time.monotonic() - t0)


# --------------------------------------------------------------------------
# Scheduler
# --------------------------------------------------------------------------
def _cuda_device_sync() -> None:
    """Device-wide synchronize if torch+CUDA exist; no-op otherwise."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
    except ImportError:
        pass


class GpuScheduler:
    """Owns the barrier and the named lanes."""

    def __init__(self, on_poisoned: Optional[Callable[[GpuLane, BaseException], None]] = None) -> None:
        self.barrier = CaptureBarrier()
        self.barrier.sync_hook = _cuda_device_sync
        self._lanes: Dict[str, GpuLane] = {}
        self._on_poisoned = on_poisoned
        self._lock = threading.Lock()

    def lane(self, name: str, **kw: Any) -> GpuLane:
        with self._lock:
            ln = self._lanes.get(name)
            if ln is None:
                ln = GpuLane(name, self.barrier, on_poisoned=self._on_poisoned, **kw)
                self._lanes[name] = ln
            return ln

    def exclusive(self):
        """Use around capture / graph reset / solver rebuild / shared-state edits."""
        return self.barrier.exclusive()

    def legacy_lock(self) -> LegacyGpuLock:
        """Object to assign to ``node.gpu_lock`` during the migration."""
        return LegacyGpuLock(self.barrier)

    def any_poisoned(self) -> bool:
        return any(l.poisoned for l in self._lanes.values())

    def stats(self) -> Dict[str, Dict[str, float]]:
        return {n: dict(l.stats) for n, l in self._lanes.items()}

    def shutdown(self) -> None:
        for l in list(self._lanes.values()):
            l.shutdown()

"""Opt-in per-call GPU time accounting for the bridge.

Enable with  CUROBO_BRIDGE_PROFILE=1  (period: CUROBO_BRIDGE_PROFILE_PERIOD, s).
Every 5 s prints, per bridge function: calls/s, mean ms, and busy % of wall
clock (CUDA-synchronized, so it reflects real GPU time; adds a little overhead).
Sum of busy% ~ the nvidia-smi utilisation this node causes.
"""
from __future__ import annotations

import functools
import os
import threading
import time

WATCH = (
    "depth_integrate", "robot_segment", "lidar_integrate", "esdf_refresh",
    "world_push", "voxel_grid", "collision_spheres_with_attached",
    "collision_distance", "plan_cspace", "plan_pose", "ik", "fk", "validate",
    "mpc_step", "attach_object", "detach_object", "scene_set",
)
_STATS: dict = {}
_LOCK = threading.Lock()


def _wrap(name, fn):
    @functools.wraps(fn)
    def inner(*a, **k):
        import torch
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            with _LOCK:
                s = _STATS.setdefault(name, [0, 0.0, 0.0])
                s[0] += 1
                s[1] += dt
                s[2] = max(s[2], dt)
    return inner


def _report(period):
    last = time.perf_counter()
    while True:
        time.sleep(period)
        now = time.perf_counter()
        wall = now - last
        last = now
        with _LOCK:
            snap = {k: list(v) for k, v in _STATS.items()}
            _STATS.clear()
        if not snap:
            print("[bridge-profile] idle: no bridge calls in window", flush=True)
            continue
        rows, total = [], 0.0
        for k, (n, tot, mx) in sorted(snap.items(), key=lambda kv: -kv[1][1]):
            total += tot
            rows.append(f"  {k:34s} {n / wall:6.1f}/s  mean {1e3 * tot / n:8.2f} ms"
                        f"  max {1e3 * mx:8.2f} ms  busy {100 * tot / wall:5.1f}%")
        print(f"[bridge-profile] window {wall:.1f}s, total busy {100 * total / wall:.1f}%\n"
              + "\n".join(rows), flush=True)


def install(module):
    if os.environ.get("CUROBO_BRIDGE_PROFILE", "") not in ("1", "true", "True"):
        return
    for name in WATCH:
        fn = getattr(module, name, None)
        if callable(fn):
            setattr(module, name, _wrap(name, fn))
    period = float(os.environ.get("CUROBO_BRIDGE_PROFILE_PERIOD", "5"))
    threading.Thread(target=_report, args=(period,), daemon=True).start()
    print("[bridge-profile] enabled", flush=True)

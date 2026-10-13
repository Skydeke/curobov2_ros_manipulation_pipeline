"""Refresh-policy decision matrix: the world must stay current, and a static
scene must not pay a rebuild per request.

``should_refresh_world`` is the entire policy; ``refresh_perception_world``
only executes it. Every row below is a collision-safety invariant — a wrong
``False`` plans blind, a wrong ``True`` only costs time.
"""

from curobov2_ros.core.unified_planner_node import should_refresh_world


def test_force_with_unchanged_epoch_still_skips():
    # Force only bypasses the throttle, not physics: with nothing
    # integrated since the last push there is nothing to rebuild — same as
    # today's early return.
    assert should_refresh_world(
        force=True, epoch=5, last_pushed_epoch=5,
        now=100.0, last_push_time=99.9, min_interval_s=2.0) is False


def test_force_with_new_epoch_refreshes():
    assert should_refresh_world(
        force=True, epoch=6, last_pushed_epoch=5,
        now=100.0, last_push_time=99.9, min_interval_s=2.0) is True


def test_unchanged_epoch_never_refreshes():
    # Nothing integrated since the last push: identical world, skip even
    # when the throttle bound is long past.
    assert should_refresh_world(
        force=False, epoch=5, last_pushed_epoch=5,
        now=1000.0, last_push_time=0.0, min_interval_s=2.0) is False


def test_new_epoch_past_throttle_refreshes():
    assert should_refresh_world(
        force=False, epoch=6, last_pushed_epoch=5,
        now=100.0, last_push_time=97.0, min_interval_s=2.0) is True


def test_new_epoch_within_throttle_skips():
    # Depth arrived 200 ms after the last push in a static cell: the ESDF
    # is for all practical purposes identical; planning proceeds on it.
    assert should_refresh_world(
        force=False, epoch=6, last_pushed_epoch=5,
        now=100.0, last_push_time=99.9, min_interval_s=2.0) is False


def test_never_pushed_always_refreshes():
    # Fresh process (or servers that never pushed): no baseline exists, so
    # even force=False must build one.
    assert should_refresh_world(
        force=False, epoch=0, last_pushed_epoch=-1,
        now=100.0, last_push_time=0.0, min_interval_s=2.0) is True


def test_throttle_boundary_refreshes():
    assert should_refresh_world(
        force=False, epoch=6, last_pushed_epoch=5,
        now=102.0, last_push_time=100.0, min_interval_s=2.0) is True


def test_zero_interval_recovers_today_behavior():
    # min_interval_s=0 disables the throttle: any new epoch refreshes, as
    # before this change.
    assert should_refresh_world(
        force=False, epoch=6, last_pushed_epoch=5,
        now=100.0, last_push_time=100.0, min_interval_s=0.0) is True


def test_reactive_forces_through_throttle():
    # The servo loop must never run on throttled perception: force bypasses
    # both the epoch and the throttle gates.
    assert should_refresh_world(
        force=True, epoch=6, last_pushed_epoch=5,
        now=100.0, last_push_time=100.0, min_interval_s=60.0) is True

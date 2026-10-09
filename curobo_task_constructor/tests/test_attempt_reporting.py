"""Per-try live reporting (MTC attempt mechanic).

MTC records every planning attempt's outcome the moment it happens — a
failed plan becomes a failed solution immediately, published live — and
publishes task statistics after every compute pass. The client's
best-of-N loop must not hide tries until the Nth one finishes: each failed
try is failed immediately (one row per try, like MTC's failed solutions),
each successful try ticks the count, and only the cheapest result emits.
"""

import curobo_task_constructor.stages  # noqa: F401  (register builtins)
from curobo_task_constructor.core.robot import PlanResult
from curobo_task_constructor.executor import TaskExecutor
from curobo_task_constructor.graph.spec import StageSpec
from tests.mock_curobo import MockCuroboServer


class _FlakyRobot(MockCuroboServer):
    """Fails the first ``failures`` plan calls, then plans normally."""

    def __init__(self, failures=2, **kw):
        super().__init__(**kw)
        self._failures_left = failures

    def plan(self, request):
        if self._failures_left > 0:
            self._failures_left -= 1
            self.plan_calls += 1
            return PlanResult(False, "flaky try failed")
        return super().plan(request)


def _spec():
    return StageSpec.from_dict({
        "stage_type": "serial", "name": "t", "container_type": "serial",
        "params_yaml": "",
        "children": [
            {"stage_type": "current_state", "name": "s", "container_type": "",
             "params_yaml": "{}", "children": []},
            {"stage_type": "move_to", "name": "m", "container_type": "",
             "params_yaml": "{goal: {joints: [0.1, 0.5, 1.2, 0, 0, 0, 0]}, "
                            "planning_attempts: 3}",
             "children": []},
        ],
    })


def test_each_failed_try_is_recorded_immediately():
    """Two failed tries then a success: two failure rows, one solution."""
    robot = _FlakyRobot(failures=2)
    ex = TaskExecutor(_spec(), robot, task_id="t")
    assert ex.init()
    assert ex.plan()
    move = next(s for s in ex.root.subtree_stages() if s.name == "m")
    assert len(move.failures) == 2  # one row per failed try, not one total
    assert len(move.solutions) == 1
    # 3 planner calls; the winning emit spends its try instead of adding one
    assert move.attempt_count == 3


def test_all_failed_tries_are_recorded_without_duplicate_tail():
    robot = _FlakyRobot(failures=99)
    ex = TaskExecutor(_spec(), robot, task_id="t")
    assert ex.init()
    assert not ex.plan()
    move = next(s for s in ex.root.subtree_stages() if s.name == "m")
    assert len(move.failures) == 3  # each try once; no extra end-fail
    assert move.attempt_count == 3


def test_progress_callback_fires_per_compute_pass():
    """MTC publishTaskState: statistics stream after every compute pass."""
    passes = []
    robot = MockCuroboServer()
    ex = TaskExecutor(_spec(), robot, task_id="t")
    assert ex.init()
    assert ex.plan(progress_callback=lambda: passes.append(1))
    assert len(passes) >= 1


def test_on_progress_fires_per_planner_try():
    """Successful tries tick the count live, before the winner emits."""
    seen = []
    robot = MockCuroboServer()
    ex = TaskExecutor(_spec(), robot, task_id="t")
    assert ex.init()
    move = next(s for s in ex.root.subtree_stages() if s.name == "m")
    move.on_progress = lambda stg: seen.append(stg.attempt_count)
    assert ex.plan()
    assert seen == [1, 2, 3]  # one tick per try, ahead of the emit


def test_rows_sum_to_attempts_when_seeds_lose():
    """Three succeeding tries: winner plus two stored loser rows.

    attempts(3) == successful rows(3) + failed rows(0): every planner call
    is stored (MTC stores every computed solution) and visible exactly
    once; ranking still returns only the cheapest via best().
    """
    considered = []
    robot = MockCuroboServer()
    ex = TaskExecutor(_spec(), robot, task_id="t")
    assert ex.init()
    move = next(s for s in ex.root.subtree_stages() if s.name == "m")
    move.on_considered = considered.append
    assert ex.plan()
    assert len(move.solutions) == 3
    assert ex.best() is not None
    assert len(move.failures) == 0
    assert len(considered) == 3  # every try streams live, winner included
    assert move.attempt_count == 3


class _SlowRobot(MockCuroboServer):
    """Sleeps per plan call so compute-time ticking is observable."""

    def __init__(self, delay=0.03, **kw):
        super().__init__(**kw)
        self._delay = float(delay)

    def plan(self, request):
        import time as _time

        _time.sleep(self._delay)
        return super().plan(request)


def test_compute_time_ticks_during_tries_not_just_after():
    """The time column must move while attempts run: each planner call ticks
    the stage clock live (run_compute only adds the unaccounted remainder,
    so the total stays exactly the wall duration)."""
    robot = _SlowRobot(delay=0.03)
    ex = TaskExecutor(_spec(), robot, task_id="t")
    assert ex.init()
    move = next(s for s in ex.root.subtree_stages() if s.name == "m")
    seen = []
    orig_progress = move.on_progress
    move.on_progress = lambda stg: (seen.append(stg.compute_time),
                                    orig_progress(stg) if orig_progress else None)
    assert ex.plan()
    assert seen == sorted(seen) and seen[-1] > seen[0], (
        f"time must tick live across tries, got {seen}")
    total = move.compute_time
    assert 0.06 <= total <= 0.30, (
        f"total must equal wall time once, not double-counted: {total}")


def test_statistics_publish_per_try():
    """Task streams statistics per planner call (MTC formula, finer
    granularity than per-pass): a 3-attempt leg emits at least 3 snapshots."""
    from curobo_task_constructor.mtc import core as mtc_core
    from curobo_task_constructor.mtc import stages as mtc_stages

    calls = []
    orig = mtc_core.Introspection.publishTaskState
    mtc_core.Introspection.publishTaskState = (
        lambda self: calls.append(1))
    try:
        robot = MockCuroboServer()
        task = mtc_core.Task(robot)
        task.add(mtc_stages.CurrentState("c"))
        mv = mtc_stages.MoveTo(
            "m", mtc_core.JointInterpolationPlanner(), planning_attempts=3)
        mv.setGoal({"joint_2": 0.4})
        task.add(mv)
        assert task.plan()
    finally:
        mtc_core.Introspection.publishTaskState = orig
    assert len(calls) >= 3, (
        f"statistics must stream per try, got {len(calls)} snapshots")

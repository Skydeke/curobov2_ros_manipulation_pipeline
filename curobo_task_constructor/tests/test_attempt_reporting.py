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

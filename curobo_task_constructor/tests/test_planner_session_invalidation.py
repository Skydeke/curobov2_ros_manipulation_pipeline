"""A plan burst must forget a planner belief formed before a remote execute.

The adapter skips a redundant ``SetPlanner`` when it believes the requested
planner is already active. That belief is only sound WITHIN one planning
burst: ``task.execute()`` hands the solution to the execute server in ANOTHER
process, which drives every segment through its OWN interface and mutates the
shared ``/curobo_server`` planner. If the belief survives, the next task's
first leg can skip its ``SetPlanner`` and be planned under the wrong planner —
a joint-space ``home`` leg landing on Classic, say (Classic rejects joint
goals), and because that comes back as a graceful failure rather than a
service error the session never self-heals and the failure spams.

So both burst boundaries drop it: ``Task.plan`` at the start, ``Task.execute``
at the end. These tests observe the drop on the robot interface, ROS-free.
"""
from curobo_task_constructor.mtc import core, stages

from tests.mock_curobo import MockCuroboServer


class _RecordingRobot(MockCuroboServer):
    """A mock that counts ``invalidate_planner_session`` calls."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.invalidations = 0

    def invalidate_planner_session(self):
        self.invalidations += 1


def _robot():
    return _RecordingRobot(
        current={"joint_2": 0.5, "joint_3": 1.2},
        named={"ready": {"joint_1": 0.1, "joint_2": 0.5, "joint_3": 1.2,
                         "joint_4": 0.0, "joint_5": 0.0, "joint_6": 0.0,
                         "finger_joint": 0.0}},
    )


def _task(robot):
    task = core.Task(robot)
    planner = core.PipelinePlanner("joint_space", "", 1)
    task.add(stages.CurrentState("current"))
    move = stages.MoveTo("ready", planner)
    move.group = "arm"
    move.setGoal("ready")
    task.add(move)
    return task


def test_plan_forgets_a_stale_planner_belief():
    # A burst must not trust a belief formed before a remote execute mutated
    # the shared planner: the first motion leg has to re-assert its planner.
    robot = _robot()
    task = _task(robot)
    assert task.plan()
    assert robot.invalidations >= 1


def test_execute_forgets_the_belief_it_just_mutated():
    # The drive runs segments under their own planners; the next burst must
    # start from a clean slate, so execute() drops the belief as well.
    robot = _robot()
    task = _task(robot)
    assert task.plan()
    before = robot.invalidations
    task.execute(task.solutions[0])
    assert robot.invalidations > before


def test_default_invalidation_is_a_noop():
    # Interfaces that never cache a planner belief inherit the no-op.
    robot = MockCuroboServer()
    assert robot.invalidate_planner_session() is None

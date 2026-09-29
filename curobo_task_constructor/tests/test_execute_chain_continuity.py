"""Chain-continuity: a task must not drive a chain that has already drifted.

A Task is planned as a chain of segments and then executed one at a time.
Each segment's request carries a ``start_pose`` baked in at PLAN time - the
planned end state of the segment before it. The server re-solves from scratch
whenever its single-slot trajectory cache misses, and ``_resolve_start_state``
prefers the request's ``start_pose`` over the arm's live pose.

So once one segment fails to land where the plan said, every remaining request
in the chain is anchored to a pose the arm was never asked to be at. Each
re-solve then compounds it: the measured failure walked joint_2 from 0.600 to
1.4529 to 2.240 across the `return` and `open` drives and ended in a genuine
self-collision with the goal state clear, so nothing could plan out of it.

These are behaviour tests, not AST guards: ``executor`` imports no torch or
curobo, so the real code runs here. What is pinned is that the chain stops and
reports, rather than driving on.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from curobo_task_constructor.core.robot import PlanResult  # noqa: E402
from curobo_task_constructor.core.stage import Solution  # noqa: E402
from curobo_task_constructor.executor import (  # noqa: E402
    EXECUTE_CONTINUITY_TOLERANCE,
    TaskExecutor,
)


@dataclass
class FakeJointState:
    position: list = field(default_factory=list)


@dataclass
class FakeStage:
    name: str = "stage"


@dataclass
class FakeRobot:
    """Replays a scripted trajectory per segment, recording the calls."""
    driven: list = field(default_factory=list)
    #: index -> trajectory to return (list of waypoints, each a list of joints)
    script: dict = field(default_factory=dict)
    #: index -> PlanResult to return instead of a success
    fail_at: dict = field(default_factory=dict)

    def execute(self, request):
        i = len(self.driven)
        self.driven.append(request)
        if i in self.fail_at:
            return self.fail_at[i]
        traj = self.script.get(i, [])
        return PlanResult(True, "ok",
                          trajectory=[FakeJointState(list(w)) for w in traj])


def _leaf(name, traj, *, request=True):
    return Solution(
        start=None, end=None,
        trajectory=[FakeJointState(list(w)) for w in traj] if traj else None,
        plan_request=object() if request else None,
        stage=FakeStage(name),
    )


def _chain(*leaves):
    """Wrap leaves as a single root solution."""
    return Solution(start=None, end=None, children=list(leaves))


def _executor(robot) -> TaskExecutor:
    """A TaskExecutor holding only what ``execute()`` touches.

    ``__init__`` runs ``build_tree`` over a StageSpec, which is a whole graph
    to stand up for a test about replaying an already-planned solution. This
    bypasses it deliberately: ``execute`` reads exactly two attributes -
    ``self.robot`` and ``self._applied_ops`` - and the subject of these tests
    is the replay loop, not the build.
    """
    ex = TaskExecutor.__new__(TaskExecutor)
    ex.robot = robot
    ex._applied_ops = []
    return ex


A = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
B = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0]
C = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]


# ---------------------------------------------------------------- chain stops

def test_a_failed_drive_stops_the_chain():
    """The pre-existing behaviour this replaces drove every segment anyway.

    So a failed `close_0` was still followed by `attach_0` - fitting the
    collision geometry of an object the gripper had not grasped - and then by
    the retreat, the return and the release.
    """
    robot = FakeRobot(fail_at={0: PlanResult(False, "close failed")})
    sol = _chain(_leaf("close_0", [A, B]), _leaf("attach_0", [B, C]))
    results = _executor(robot).execute(sol)

    assert len(robot.driven) == 1, "must not drive attach_0 after close_0 failed"
    assert len(results) == 1
    assert not results[0].success
    assert "close failed" in results[0].message


def test_a_diverged_segment_stops_the_chain():
    """The case this was added for.

    Segment 0 planned to end at B, and the re-solve ended it somewhere else.
    Segment 1's baked start_pose is B, so driving it now would ask the arm to
    plan from a pose it is not in - and that mismatch compounds.
    """
    robot = FakeRobot(script={0: [A, C], 1: [C, A]})
    sol = _chain(_leaf("return", [A, B]), _leaf("open", [B, C]))
    results = _executor(robot).execute(sol)

    assert len(robot.driven) == 1, "must not drive 'open' after 'return' diverged"
    assert len(results) == 1
    assert not results[0].success
    assert "cache miss" in results[0].message
    assert "return" in results[0].message


def test_a_chain_that_lands_where_planned_runs_to_the_end():
    """The check must not fire on the good path, or nothing ever executes."""
    robot = FakeRobot(script={0: [A, B], 1: [B, C]})
    sol = _chain(_leaf("return", [A, B]), _leaf("open", [B, C]))
    results = _executor(robot).execute(sol)

    assert len(robot.driven) == 2
    assert [r.success for r in results] == [True, True]


# ------------------------------------------------------- what counts as drift

def test_only_the_endpoints_are_compared():
    """A different path through free space is not a defect.

    trajopt is stochastic, so a re-solve may wiggle between the same two
    points. What the chain depends on is only where each segment begins and
    ends, so a wholly different MIDDLE must not trip the check.
    """
    wiggly = [A, [0.9] * 8, B]
    robot = FakeRobot(script={0: wiggly})
    sol = _chain(_leaf("return", [A, B]))
    results = _executor(robot).execute(sol)

    assert len(robot.driven) == 1
    assert results[0].success, results[0].message


def test_a_moved_start_is_caught_too():
    """The original bug was a stale START, not a stale end.

    The re-solve was anchored to the plan-time start pose while the arm was
    elsewhere, and cuRobo answered with a different IK branch - same tool pose,
    completely different joints. A check that only looked at the end state
    would have missed every instance of it.
    """
    robot = FakeRobot(script={0: [B, B]})
    sol = _chain(_leaf("return", [A, B]))
    results = _executor(robot).execute(sol)

    assert not results[0].success
    assert "start" in results[0].message


def test_drift_just_under_the_tolerance_is_tolerated():
    """The tolerance is a real boundary, not decoration.

    Without this, any change to the constant would silently turn every
    execution into a failure, and the test suite would still pass.
    """
    just_under = [x + EXECUTE_CONTINUITY_TOLERANCE * 0.9 for x in A]
    robot = FakeRobot(script={0: [just_under, B]})
    sol = _chain(_leaf("return", [A, B]))
    assert _executor(robot).execute(sol)[0].success


def test_drift_just_over_the_tolerance_is_caught():
    just_over = [x + EXECUTE_CONTINUITY_TOLERANCE * 1.1 for x in A]
    robot = FakeRobot(script={0: [just_over, B]})
    sol = _chain(_leaf("return", [A, B]))
    assert not _executor(robot).execute(sol)[0].success


def test_a_short_waypoint_is_compared_over_the_joints_it_has():
    """A 7-DOF arm group against an 8-joint arm is a real request shape.

    ``_resolve_start_state`` pads a short start pose from the live pose, so
    the comparison has to tolerate a short waypoint the same way rather than
    raising or reporting a spurious delta against a missing joint.
    """
    robot = FakeRobot(script={0: [A[:7], B[:7]]})
    sol = _chain(_leaf("return", [A[:7], B[:7]]))
    assert _executor(robot).execute(sol)[0].success


def test_a_result_with_no_trajectory_is_not_treated_as_drift():
    """Absent evidence is not evidence of divergence.

    A trajectory-less result is how a non-motion leaf reports; treating it as a
    mismatch would fail every scene-only chain.
    """
    robot = FakeRobot()
    robot.execute = lambda request: PlanResult(True, "ok", trajectory=[])
    sol = _chain(_leaf("return", [A, B]))
    assert _executor(robot).execute(sol)[0].success


def test_plain_lists_are_accepted_as_waypoints():
    """Waypoints arrive as JointState messages, but tests and the raw
    responses hand back plain lists; both must compare."""
    from curobo_task_constructor.executor import _max_joint_delta
    assert _max_joint_delta([0.0, 0.0], [0.0, 0.1]) == pytest.approx(0.1)
    assert _max_joint_delta(FakeJointState([0.0]), FakeJointState([0.2])) == \
        pytest.approx(0.2)


# ------------------------------------------------------------------ reporting

def test_the_failure_names_the_stage_that_broke_the_chain():
    """The caller reports `failed_stage_name` from the failing result, and the
    orchestrator's recovery ladder keys off it."""
    robot = FakeRobot(script={0: [A, C]})
    sol = _chain(_leaf("return_to_perception", [A, B]))
    results = _executor(robot).execute(sol)
    assert "return_to_perception" in results[0].message


def test_scene_ops_before_the_broken_segment_still_applied():
    """Everything up to the failure is real and must not be rolled back.

    An `allow_collisions` or `attach` that ran before the break has already
    changed the server's world; the ladder downstream depends on that state.
    """
    applied = []

    class Robot(FakeRobot):
        def add_object(self, spec):
            applied.append(spec)

    sol = _chain(_leaf("allow_0", [A, B]),
                 _leaf("return", [A, B]),
                 _leaf("open", [B, C]))
    ex = _executor(Robot(script={1: [A, C]}))
    # The "add" payload is a spec object, not a bare name - _apply_op keys on
    # payload.name, so a string would not be a faithful stand-in.
    spec = SimpleNamespace(name="object_0")
    sol.children[0].scene_ops = [("add", spec)]
    ex.execute(sol)
    assert applied == [spec]


def test_the_continuity_tolerance_is_moveits_allowed_start_tolerance():
    """One start-agreement number for the whole pipeline: 0.01 rad.

    ``iki_kortex_moveit_config/config/moveit_controllers.yaml`` sets
    ``allowed_start_tolerance: 0.01``. A curobo chain must refuse to drive a
    segment from a configuration that differs more than that from where the
    previous segment anchored it - the same wall the MoveIt stack enforces
    before it plans. A larger value here would let a segment land where the
    plan was not and have the next one drive anyway, which is the measured
    failure this whole module exists to pin.
    """
    assert EXECUTE_CONTINUITY_TOLERANCE == pytest.approx(0.01)

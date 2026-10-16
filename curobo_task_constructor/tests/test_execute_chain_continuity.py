"""Execution replays each segment's own validated trajectory.

A Task is planned as a chain of segments and then executed one segment at a
time. MTC ``Task::execute`` parity: each segment is driven from the trajectory
the ``Solution`` ALREADY carries (see ``executor._replay_trajectory_dict``), so
the server streams exactly those waypoints — there is no goal re-match against
the server's trajectory cache. A multi-solution pose goal therefore CANNOT land
on a different IK branch than the one that was planned, collision-checked and
displayed, which is the failure the old client-side chain-continuity guard
existed to catch.

The old guard compared the driven trajectory against the plan and aborted the
chain on any endpoint mismatch. That guard was the wrong fix: it fired on a
legitimate re-solve (whose endpoints a stochastic trajopt is free to move), not
on a real defect — and MTC's execute never second-guesses the controller this
way. Start-state agreement (the arm actually being where a replay assumes) is
the controller's ``allowed_start_tolerance`` check, exactly where MoveIt puts
it, so the cuRobo executor no longer duplicates it.

These are behaviour tests, not AST guards: ``executor`` imports no torch or
curobo, so the real code runs here. What is pinned is (a) that each segment's
OWN waypoints are handed to the drive, and (b) that the chain still STOPS at the
first segment whose drive failed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from curobo_task_constructor.core.robot import PlanResult  # noqa: E402
from curobo_task_constructor.core.stage import Solution  # noqa: E402
from curobo_task_constructor.executor import TaskExecutor  # noqa: E402


@dataclass
class FakeJointState:
    position: list = field(default_factory=list)
    velocity: list = field(default_factory=list)
    name: list = field(default_factory=list)


@dataclass
class FakeStage:
    name: str = "stage"


@dataclass
class FakeRobot:
    """Records each drive and returns a scripted result per segment."""
    driven: list = field(default_factory=list)
    #: trajectory dict handed to each execute() call, in drive order.
    replayed: list = field(default_factory=list)
    #: index -> PlanResult to return instead of a success.
    fail_at: dict = field(default_factory=dict)

    def execute(self, request, trajectory):
        i = len(self.driven)
        self.driven.append(request)
        self.replayed.append(trajectory)
        if i in self.fail_at:
            return self.fail_at[i]
        return PlanResult(True, "ok", trajectory=[])


def _leaf(name, traj, *, request=True, joint_names=("j0", "j1")):
    wps = [FakeJointState(list(w), [], list(joint_names))
           for w in traj] if traj else None
    return Solution(
        start=None, end=None,
        trajectory=wps,
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


A = [0.0] * 8
B = [0.5] * 8
C = [1.0] * 8


# ------------------------------------------------------------- verbatim replay

def test_each_segment_replays_its_own_waypoints():
    """The core of the MTC model: the segment's OWN trajectory is driven."""
    robot = FakeRobot()
    sol = _chain(_leaf("return", [A, B]), _leaf("open", [B, C]))
    results = _executor(robot).execute(sol)

    assert [r.success for r in results] == [True, True]
    assert robot.replayed[0]["points"] == [A, B]
    assert robot.replayed[1]["points"] == [B, C]


def test_replay_carries_the_joint_names():
    robot = FakeRobot()
    sol = _chain(_leaf("return", [A, B], joint_names=("j0", "j1", "j2")))
    _executor(robot).execute(sol)
    assert robot.replayed[0]["joint_names"] == ["j0", "j1", "j2"]


def test_replay_carries_velocities_when_the_solver_produced_them():
    """A RobotTrajectory carries speeds; they must reach the streamer (it
    clamps/taper them) rather than being silently dropped to positions-only."""
    robot = FakeRobot()
    sol = _chain(_leaf("return", [A, B]))
    sol.children[0].trajectory[0].velocity = [0.1] * 8
    _executor(robot).execute(sol)
    assert robot.replayed[0]["velocities"][0] == [0.1] * 8
    assert robot.replayed[0]["velocities"][1] == []


def test_a_server_that_reports_other_waypoints_is_not_a_client_failure():
    """The old guard aborted here; the replay model does not compare.

    The caller drives what it SENT. If the arm was not where the trajectory
    starts, the controller's ``allowed_start_tolerance`` refuses it — MTC does
    not have the executor re-derive and compare the plan.
    """
    robot = FakeRobot()
    calls = []

    def _other(request, trajectory=None):
        calls.append(trajectory)
        return PlanResult(True, "ok", trajectory=[FakeJointState(C)])

    robot.execute = _other
    sol = _chain(_leaf("return", [A, B]), _leaf("open", [B, C]))
    results = _executor(robot).execute(sol)

    assert [r.success for r in results] == [True, True]
    assert len(calls) == 2


# ---------------------------------------------------------------- chain stops

def test_a_failed_drive_stops_the_chain():
    """The pre-existing behaviour this keeps: a failed `close_0` must not be
    followed by `attach_0` - fitting collision geometry for an object the
    gripper never grasped - nor by the retreat, return and release."""
    robot = FakeRobot(fail_at={0: PlanResult(False, "close failed")})
    sol = _chain(_leaf("close_0", [A, B]), _leaf("attach_0", [B, C]))
    results = _executor(robot).execute(sol)

    assert len(robot.driven) == 1, "must not drive attach_0 after close_0 failed"
    assert len(results) == 1
    assert not results[0].success
    assert "close failed" in results[0].message


def test_a_chain_that_all_succeed_runs_to_the_end():
    robot = FakeRobot()
    sol = _chain(_leaf("return", [A, B]), _leaf("open", [B, C]))
    results = _executor(robot).execute(sol)

    assert len(robot.driven) == 2
    assert [r.success for r in results] == [True, True]


def test_a_mutation_leaf_without_a_request_is_skipped():
    """A scene-only (mutation) leaf carries no plan_request and no drive."""
    robot = FakeRobot()
    sol = _chain(_leaf("allow_0", None, request=False),
                 _leaf("return", [A, B]))
    results = _executor(robot).execute(sol)

    assert len(robot.driven) == 1
    assert results[0].success


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
    ex = _executor(Robot(fail_at={1: PlanResult(False, "boom")}))
    # The "add" payload is a spec object, not a bare name - _apply_op keys on
    # payload.name, so a string would not be a faithful stand-in.
    spec = SimpleNamespace(name="object_0")
    sol.children[0].scene_ops = [("add", spec)]
    ex.execute(sol)
    assert applied == [spec]

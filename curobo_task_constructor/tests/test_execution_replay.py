"""MTC-literal execution: the solution's trajectory is replayed verbatim.

``Task::execute`` in MoveIt/MTC hands each sub-trajectory's stored
``RobotTrajectory`` to the controller and never re-derives it — so there is no
server-side plan cache to miss and no IK branch to jump. This suite pins that
model end to end:

* ``TaskExecutor.execute`` / ``drive_solution_dict`` forward each segment's OWN
  waypoints to the robot,
* the rclpy adapter carries them on the wire (``SendTrajectory.replay``),
* the cuRobo node streams that trajectory instead of matching/re-solving a goal,
* the ``SendTrajectory`` action declares the fields all of that rides on, and
* the solution wire form carries positions AND velocities (MTC's
  ``RobotTrajectory`` does; a replayed trajectory must not lose its speeds).

``executor``/``msg_convert`` import no torch or curobo, so their half runs for
real; the adapter and node import ROS/cuRobo at module scope, so their half is
an AST guard.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

#: parents[0] = tests/, [1] = curobo_task_constructor/, [2] = the submodule root.
ROOT = Path(__file__).resolve().parents[2]
CLIENT = ROOT / "curobo_task_constructor" / "curobo_task_constructor"
NODE = ROOT / "curobov2_ros" / "curobov2_ros" / "core" / "unified_planner_node.py"
ACTION = (ROOT / "curobov2_ros_interfaces" / "action" / "SendTrajectory.action")

for _p in (NODE, ACTION, CLIENT / "robot" / "curobo.py"):
    assert _p.is_file(), f"guard points at a missing file: {_p}"

from curobo_task_constructor.executor import (  # noqa: E402
    _replay_trajectory_dict,
)


def _func(path: Path, name: str) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no function {name!r} in {path.name}")


# ------------------------------------------------------------------ executor

def test_replay_dict_packs_names_positions_and_velocities():
    wps = [SimpleNamespace(name=["a", "b"], position=[0.1, 0.2],
                           velocity=[1.0, 2.0]),
           SimpleNamespace(name=["a", "b"], position=[0.3, 0.4],
                           velocity=[])]
    packed = _replay_trajectory_dict(wps)
    assert packed["joint_names"] == ["a", "b"]
    assert packed["points"] == [[0.1, 0.2], [0.3, 0.4]]
    assert packed["velocities"] == [[1.0, 2.0], []]


def test_replay_dict_is_none_without_waypoints():
    assert _replay_trajectory_dict([]) is None
    assert _replay_trajectory_dict(None) is None


def test_task_executor_forwards_the_leaf_trajectory():
    body = _func(CLIENT / "executor.py", "_replay_trajectory_dict")
    assert "position" in ast.unparse(body) and "velocity" in ast.unparse(body)

    ex = ast.parse((CLIENT / "executor.py").read_text())
    execute = next(n for n in ast.walk(ex)
                   if isinstance(n, ast.FunctionDef) and n.name == "execute"
                   and "sol" in [a.arg for a in n.args.args])
    text = ast.unparse(execute)
    assert "_replay_trajectory_dict" in text, \
        "Task.execute must hand each segment's own waypoints to the drive"
    assert "trajectory=" in text


def test_drive_solution_dict_forwards_the_segment_trajectory():
    """The execute server (another process) hands the segment's trajectory on."""
    ex = ast.parse((CLIENT / "executor.py").read_text())
    drive = next(n for n in ast.walk(ex)
                 if isinstance(n, ast.FunctionDef)
                 and n.name == "drive_solution_dict")
    text = ast.unparse(drive)
    assert "trajectory=traj" in text, \
        "the segment's OWN waypoints must be replayed, not re-matched"
    assert "_diverged_from_plan" not in text, \
        "the client no longer re-derives and compares the plan (MTC parity)"


def test_client_execute_sets_the_replay_fields():
    body = _func(CLIENT / "robot" / "curobo.py", "execute")
    text = ast.unparse(body)
    assert "replay" in text and "replay_trajectory" in text, \
        "the adapter must carry the caller's trajectory on the wire"
    assert "_to_joint_trajectory_msg" in text


def test_client_builds_a_joint_trajectory_message():
    body = _func(CLIENT / "robot" / "curobo.py", "_to_joint_trajectory_msg")
    text = ast.unparse(body)
    assert "JointTrajectory" in text
    assert "positions" in text and "velocities" in text, \
        "positions and the solver's speeds both ride the wire"


# ----------------------------------------------------------------- wire form

def test_send_trajectory_action_declares_the_replay_fields():
    text = ACTION.read_text()
    assert "bool replay" in text, \
        "the execute action needs a replay flag (MTC ExecuteTaskSolution)"
    assert "trajectory_msgs/JointTrajectory replay_trajectory" in text, \
        "and a field to carry the validated waypoints"


# --------------------------------------------------------------------- server

def test_server_dispatches_replay_before_planning():
    """A replay needs no goalset, so it must skip validation AND the cache."""
    body = _func(NODE, "_execute_goal")
    text = ast.unparse(body)
    assert "_execute_replay" in text, "the replay path must be dispatched"
    assert text.index("replay") < text.index("_check_goal_request"), \
        "replay must run before goal validation (it carries its own plan)"


def test_server_replay_streams_without_the_cache():
    body = _func(NODE, "_execute_replay")
    text = ast.unparse(body)
    assert "replay(" in text, \
        "the server must stage+stream the supplied trajectory"
    for banned in ("_pending_plan_matches", "_refresh_and_plan",
                   "_store_pending_plan"):
        assert banned not in text, \
            f"a replay must not touch the plan cache / re-solve ({banned})"


def test_server_replay_reports_the_driven_trajectory():
    body = _func(NODE, "_execute_replay")
    assert "_fill_replayed_trajectory" in ast.unparse(body), \
        "the replayed trajectory must reach the caller (viz + inspection)"


def test_server_rejects_replay_on_a_reactive_planner():
    """Only open-loop planners stream a waypoint list; a reactive one servos."""
    body = _func(NODE, "_execute_replay")
    assert "is_open_loop" in ast.unparse(body)


# ------------------------------------------------- no cache / no re-solve left

def test_the_plan_cache_is_gone():
    """The old caching execution is removed, not merely unused.

    A reopen of the multi-entry preview cache / execute-time re-solve would
    resurrect the IK-branch jump this pipeline exists to avoid, so the guard is
    on the SOURCE, not on behaviour.
    """
    text = NODE.read_text()
    for banned in (
        "_pending_plan",
        "_store_pending_plan",
        "_pending_plan_matches",
        "_drop_pending_plan",
        "_prune_plan_cache",
        "_check_execution_start_drift",
        "trajectory_cache_ttl",
        "trajectory_cache_size",
        "execution_start_tolerance",
        "execution_start_warn_drift",
        "clear_trajectory",
        "allow_cached",
        "force_cached",
    ):
        assert banned not in text, f"cache machinery is back: {banned}"
    # `_refresh_and_plan_reactive` stays (the reactive servo path uses it); the
    # open-loop re-solve helper `_refresh_and_plan` must not.
    assert "_refresh_and_plan(" not in text, \
        "the open-loop execute re-solve helper is back"


def test_action_has_no_cache_flags():
    text = ACTION.read_text()
    assert "allow_cached" not in text and "force_cached" not in text, \
        "the execute action no longer exposes cache-reuse flags"


def test_server_refuses_open_loop_execute_without_a_trajectory():
    """No cache to miss, so an open-loop execute with no plan is an error."""
    body = _func(NODE, "_execute_goal")
    text = ast.unparse(body)
    assert "is_open_loop" in text, \
        "the open-loop branch must require a supplied trajectory"
    assert "_execute_replay" in text


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))

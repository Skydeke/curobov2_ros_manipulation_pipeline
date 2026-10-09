"""Introspection completeness: every solution/failure the executor records
reaches the panel topics (ROS-free audit at the dict level).

Mirrors the panel's join logic: TaskStatistics carries per-stage GLOBAL
solution ids; every id must name a Solution payload the panel can fetch
(GetSolution -> solutionFromId), and every payload id must resolve.
A pick-shaped tree (fallbacks with a losing line strategy + a free one,
3 attempts per leg) exercises successes, live failures and loser rows.
"""

import curobo_task_constructor.stages  # noqa: F401  (register builtins)
from curobo_task_constructor import msg_convert
from curobo_task_constructor.mtc import core, stages
from curobo_task_constructor.mtc.core import _DISPLAY_PROPERTIES

from tests.mock_curobo import MockCuroboServer


def _task(robot):
    jointspace = core.JointInterpolationPlanner()
    cartesian = core.CartesianPath()
    task = core.Task(robot)
    task.name = "audit"
    task.add(stages.CurrentState("current state"))
    fb = core.Fallbacks("strategies")

    line = core.SerialContainer("strategy_0")
    reach = stages.MoveTo("reach_0", jointspace, planning_attempts=3)
    reach.setGoal({"joint_2": 0.4})
    line.add(reach)
    descend = stages.CartesianPath("descend_0", cartesian, planning_attempts=3)
    descend.setRelative(0.0, 0.0, -0.05, frame="world")
    line.add(descend)
    fb.add(line)

    free = core.SerialContainer("strategy_1")
    reach1 = stages.MoveTo("reach_1", jointspace, planning_attempts=3)
    reach1.setGoal({"joint_2": 0.4})
    free.add(reach1)
    descend1 = stages.CartesianPath("descend_1", cartesian, planning_attempts=3)
    descend1.setRelative(0.0, 0.0, -0.05, frame="world")
    descend1.setHold([0, 0, 0, 0, 0, 0])
    descend1.setCheckStraightness(False)
    free.add(descend1)
    fb.add(free)

    task.add(fb)
    tail = stages.MoveTo("park", jointspace, planning_attempts=3)
    tail.setGoal({"joint_2": 0.2})
    task.add(tail)
    return task


def _planned(task):
    """Plan with publishSolution recorded (no ROS needed)."""
    calls = []
    orig = core.Introspection.publishSolution
    core.Introspection.publishSolution = (
        lambda self, solution: calls.append(solution))
    try:
        assert task.plan(), "audit tree must solve via the free strategy"
    finally:
        core.Introspection.publishSolution = orig
    return calls


def _payload_ids(task, calls):
    ids = set()
    for obj in calls:
        d = task._solution_dict(obj)
        assert d is not None
        for sub in d["sub_solution"]:
            ids.add(int(sub["info"]["id"]))
    return ids


def test_no_live_solution_publish_only_explicit():
    """Planning emits no Solution messages on its own (MTC: solutions go
    out solely through explicit publish calls); a single explicit publish
    then carries the best solution."""
    robot = MockCuroboServer(bow=0.04)
    task = _task(robot)
    calls = _planned(task)
    assert calls == [], (
        f"planning must not auto-publish, got {len(calls)} publishes")
    best = task.best()
    assert best is not None
    task.publish(best)
    # publish() with introspection disabled (no ROS here) records locally.
    assert task.last_published is best


def test_every_recorded_id_resolves_via_getsolution():
    """Everything the executor records (solutions and failures) resolves
    through the GetSolution path, so panel rows — announced by statistics
    — always have fetchable payloads even though nothing auto-publishes."""
    robot = MockCuroboServer(bow=0.04)
    task = _task(robot)
    assert task.plan()
    ex = task.executor
    n = 0
    for stage in ex.root.subtree_stages():
        for sol in list(stage.solutions) + list(stage.failures):
            gid = ex.global_solution_id(sol)
            assert ex.solutionFromId(gid) is not None
            d = task._solution_dict(sol)
            assert d is not None
            assert any(int(sub["info"]["id"]) == gid
                       for sub in d["sub_solution"])
            n += 1
    assert n > 0, "audit tree must record solutions and failures"


def test_losing_strategy_failures_are_recorded():
    robot = MockCuroboServer(bow=0.04)
    task = _task(robot)
    assert task.plan()
    ex = task.executor
    line_descend = next(
        s for s in ex.root.subtree_stages() if s.name == "descend_0")
    assert line_descend.failures, "bowed line descend must fail its attempts"
    for failure in line_descend.failures:
        gid = ex.global_solution_id(failure)
        assert ex.solutionFromId(gid) is not None, (
            "failed attempts must stay fetchable (panel red rows)")


def test_root_solution_published_explicitly():
    robot = MockCuroboServer(bow=0.04)
    task = _task(robot)
    assert task.plan()
    calls = []
    orig = core.Introspection.publishSolution
    core.Introspection.publishSolution = (
        lambda self, solution: calls.append(solution))
    try:
        task.publish(task.best())
    finally:
        core.Introspection.publishSolution = orig
    assert len(calls) == 1 and calls[0] is task.best(), (
        "explicit publish sends exactly the best solution")
    payload_ids = _payload_ids(task, calls)
    best = task.best()
    assert task._executor.global_solution_id(best) in payload_ids, (
        "top-level solution must publish (panel task row)")


def test_best_of_n_attempts_all_recorded():
    """Best-of-3 legs store all 3 attempts (winner promoted downstream)
    without auto-publishing any of them; every stored try stays fetchable
    for the panel, and explicit publish sends the best."""
    robot = MockCuroboServer()
    jointspace = core.JointInterpolationPlanner()
    task = core.Task(robot)
    task.name = "losers"
    task.add(stages.CurrentState("current state"))
    move = stages.MoveTo("reach", jointspace, planning_attempts=3)
    move.setGoal({"joint_2": 0.4})
    task.add(move)

    payloads = []
    orig = core.Introspection.publishSolution
    core.Introspection.publishSolution = (
        lambda self, solution: payloads.append(task._solution_dict(solution)))
    try:
        assert task.plan()
        # No auto-publish during planning (MTC parity)...
        assert payloads == []
        # ...then one explicit publish carries the best.
        task.publish(task.best())
    finally:
        core.Introspection.publishSolution = orig
    assert len(payloads) == 1, "explicit publish sends exactly one payload"

    ex = task.executor
    reach = next(s for s in ex.root.subtree_stages() if s.name == "reach")
    assert len(reach.solutions) == 3, "winner + 2 stored rows"
    assert task.best() is not None, "ranking still yields a best solution"
    reach_ids = set()
    for d in payloads:
        assert d is not None
        for sub in d["sub_solution"]:
            if int(sub["info"]["stage_id"]) == reach.stage_id:
                reach_ids.add(int(sub["info"]["id"]))
    for sol in reach.solutions:
        gid = ex.global_solution_id(sol)
        assert ex.solutionFromId(gid) is not None, (
            f"stored id {gid} must resolve (panel GetSolution path)")
        d = task._solution_dict(sol)
        assert any(int(sub["info"]["id"]) == gid for sub in d["sub_solution"])
    assert len(reach_ids) >= 1, "explicit payload names stage solutions"


def test_stage_properties_match_mtc_declared_set():
    """Published StageDescription properties stay at the MTC-declared set:
    planner wire keys, attempt budgets, cost terms and solver extras ride
    params_yaml to the executor, never the properties pane."""
    recorded = {}
    orig = msg_convert.property_to_msg
    msg_convert.property_to_msg = (
        lambda key, value: recorded.setdefault(key, value) or (key, value))
    try:
        robot = MockCuroboServer()
        task = _task(robot)
        assert task.plan()
        ex = task.executor
        for stage in ex.root.subtree_stages():
            recorded.clear()
            try:
                allowed = _DISPLAY_PROPERTIES.get(
                    stage.stage_type(), ("group", "timeout"))
            except Exception:
                allowed = ("group", "timeout")
            params = getattr(stage, "params", None) or {}
            for key in sorted(params):
                if key in allowed:
                    msg_convert.property_to_msg(key, params[key])
            for key in recorded:
                assert key in allowed, (
                    f"stage '{stage.name}': property {key!r} not in "
                    f"MTC-declared set for {stage.stage_type()}")
        move = next(s for s in ex.root.subtree_stages() if s.name == "reach_0")
        assert "goal" in [k for k in (getattr(move, "params", None) or {}) if k in
                         _DISPLAY_PROPERTIES.get("move_to", ())]
        assert "planning_attempts" not in recorded
        assert "planner" not in recorded
        assert "cost" not in recorded
    finally:
        msg_convert.property_to_msg = orig


def test_identical_sibling_failures_attribute_to_own_stage():
    """Value-equal failures from sibling stages sharing one input (every
    strategy's pre_grasp failing the same goal the same way) must attribute
    to the stage that recorded them — not, via dataclass __eq__, to the
    first sibling in tree order (3 attempts then show as 9 rows under one
    stage while the others show none).
    """
    robot = MockCuroboServer(fail_ik=True)
    jointspace = core.JointInterpolationPlanner()
    task = core.Task(robot)
    task.name = "siblings"
    task.add(stages.CurrentState("current state"))
    fb = core.Fallbacks("strategies")
    moves = []
    for i in range(3):
        s = core.SerialContainer(f"strategy_{i}")
        move = stages.MoveTo(f"pre_grasp_{i}", jointspace, planning_attempts=1)
        move.setGoals([{"x": 9.0, "y": 9.0, "z": 9.0,
                        "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}])
        s.add(move)
        fb.add(s)
        moves.append(f"pre_grasp_{i}")
    task.add(fb)
    assert not task.plan()
    ex = task.executor
    by_name = {s.name: s for s in ex.root.subtree_stages()}
    for name in moves:
        stage = by_name[name]
        assert len(stage.failures) == 1
        assert task._failure_stage(stage.failures[0]) is stage, (
            f"{name}: failure attributed to another stage")
    # Panel join emulation: exactly one row per stage.
    from collections import Counter
    counts = Counter()
    for stage in ex.root.subtree_stages():
        for failure in stage.failures:
            d = task._solution_dict(failure)
            for sub in d["sub_solution"]:
                counts[sub["info"]["stage_id"]] += 1
    for name in moves:
        assert counts[by_name[name].stage_id] == 1, (
            f"{name}: expected 1 row, got {counts[by_name[name].stage_id]}")


def test_pose_to_dict_covers_all_pose_shapes():
    """Marker plumbing converts every pose shape the pipeline carries."""
    from curobo_task_constructor import msg_convert
    from curobo_task_constructor.core.geom import Pose3

    assert msg_convert._pose_to_dict(None) is None
    assert msg_convert._pose_to_dict({"a": 1}) is None
    flat = {"x": 1.0, "y": 2.0, "z": 3.0}
    assert msg_convert._pose_to_dict(flat)["x"] == 1.0
    assert msg_convert._pose_to_dict(flat)["qw"] == 1.0
    assert msg_convert._pose_to_dict(
        Pose3([0.1, 0.2, 0.3], [0.0, 0.0, 0.0, 1.0])) == {
            "x": 0.1, "y": 0.2, "z": 0.3,
            "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}

    class _RosLike:
        class _P:
            x, y, z = 0.4, 0.5, 0.6
        class _Q:
            x, y, z, w = 0.0, 0.0, 0.0, 1.0
        position, orientation = _P(), _Q()

    assert msg_convert._pose_to_dict(_RosLike())["z"] == 0.6


def test_attempt_markers_reach_the_solution_dict():
    """Markers generated for a motion leaf land in its solution payload
    (the panel republishes them to selected_solution_markers).

    Regression test: msg_convert lacked _pose_to_dict, so every marker
    helper raised AttributeError into a bare except and all payloads
    carried zero markers while the topic kept publishing empties.
    """
    from curobo_task_constructor import msg_convert

    made = []
    orig_sphere = msg_convert.sphere_marker
    orig_list = msg_convert.sphere_list_marker
    msg_convert.sphere_marker = (
        lambda ns, i, pose, size, color: made.append((ns, i)) or ("M", ns, i))
    msg_convert.sphere_list_marker = (
        lambda ns, i, poses, size, color: made.append((ns, i)) or ("L", ns, i))
    try:
        robot = MockCuroboServer()
        task = core.Task(robot)
        task.name = "markers"
        task.add(stages.CurrentState("current state"))
        move = stages.MoveTo("reach", core.JointInterpolationPlanner(),
                             planning_attempts=1)
        move.setGoal({"joint_2": 0.4})
        task.add(move)
        assert task.plan()
        out = task._markers_for_object(task.best())
        assert made, "motion leaf must yield start/end markers"
        assert any(v for v in out.values()), "marker groups must be non-empty"
    finally:
        msg_convert.sphere_marker = orig_sphere
        msg_convert.sphere_list_marker = orig_list

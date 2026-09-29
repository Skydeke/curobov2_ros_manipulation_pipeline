"""Guard on the execution-start-drift check.

The check lives in ``unified_planner_node.py``, which imports curobo and torch
at module scope, so it cannot be imported in this environment. These are
therefore AST guards, like ``test_allowed_collisions_bracket.py`` beside it:
they assert on the module's STRUCTURE rather than calling it. That is enough
for the properties that matter here, all of which are about WHERE the check
sits relative to the plan and the cache - the one thing that is easy to get
subtly wrong and hard to notice, because the code still works.

The behaviour being guarded
---------------------------
A Task goal is planned as a chain of segments, then executed one segment at a
time. Each segment's request carries a ``start_pose`` baked in at PLAN time.
``TaskExecutor.execute`` replays those requests, and the server re-solves from
scratch on a cache miss. ``_resolve_start_state`` prefers the request's
``start_pose`` over the arm's live pose. So if the arm is not actually there
when the segment is reached, the re-solve is anchored to a configuration the
arm is not in, cuRobo picks a different IK branch, and the trajectory that
gets driven is not the trajectory that was planned and displayed - while the
goal reports success and the controller reports success.

Two things must be true, and they pull in opposite directions:

  * the check must be on the RE-SOLVE path, or it fires on legitimate cache
    hits (a hit already pins the start state, so the arm cannot have moved);
  * the check must be BEFORE the solve, or the divergent plan is computed and
    then thrown away instead of refused.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: parents[0] = tests/, [1] = curobo_task_constructor/, [2] = the submodule
#: root. One level too shallow reads a path that does not exist, which fails
#: every test with a FileNotFoundError that looks like a code problem.
NODE = (Path(__file__).resolve().parents[2] / "isaac_ros_cumotion"
        / "isaac_ros_cumotion" / "core" / "unified_planner_node.py")

assert NODE.is_file(), f"guard points at a missing file: {NODE}"


def _module() -> ast.Module:
    return ast.parse(NODE.read_text())


def _func(name: str) -> ast.FunctionDef:
    for node in ast.walk(_module()):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no function {name!r} in {NODE.name}")


def _calls(node: ast.AST) -> set:
    return {c.func.attr if isinstance(c.func, ast.Attribute) else c.func.id
            for c in ast.walk(node) if isinstance(c, ast.Call)
            and isinstance(c.func, (ast.Attribute, ast.Name))}


def _find_if_within(body: ast.AST, needle: str):
    """The innermost-ish ``if`` inside ``body`` whose test mentions ``needle``."""
    for node in ast.walk(body):
        if isinstance(node, ast.If) and needle in ast.unparse(node.test):
            return node
    return None


def test_the_check_exists_and_compares_both_poses():
    """It has to read the request's start pose AND the arm's live pose.

    Reading only the request would be a tautology; reading only the live pose
    would not catch the stale-anchor bug at all.
    """
    body = _func("_check_execution_start_drift")
    text = ast.unparse(body)
    assert "start_pose" in text, "must read the goal's start_pose"
    assert "get_joint_pose" in text, "must read the arm's actual pose"


def test_the_check_compares_joint_by_joint_and_reports_the_worst_one():
    """A single max-delta summary, naming the joint.

    An aggregate norm would hide the failure this exists to catch: cuRobo
    wrapping one joint through several radians while the rest stay put is
    exactly the observed self-collision spin, and it is invisible in an L2 norm
    spread across seven joints.
    """
    body = _func("_check_execution_start_drift")
    text = ast.unparse(body)
    assert "max(" in text
    assert "joint" in text, "the message must name the offending joint"
    # Both poses must reach the message, not just the delta, so the log is
    # diagnosable without re-running.
    assert text.count("requested") >= 1 and "actual" in text


def test_small_drift_warns_and_large_drift_refuses():
    """The two-tier behaviour, and that the tiers are ordered.

    Refusing all drift would fail the pipeline on ordinary servo tracking
    error, where re-solving from the arm's real pose is the correct response.
    Reporting all drift as a warning would let the divergent trajectory be
    driven, which is the bug.
    """
    body = _func("_check_execution_start_drift")
    text = ast.unparse(body)
    assert "'execution_start_warn_drift'" in text, "a warn tier must exist"
    assert "'execution_start_tolerance'" in text, "a refuse tier must exist"
    # The refusal is tested FIRST. If the warn were tested first it would
    # swallow every drift above the tolerance, and the check would degrade into
    # a log line - which is the pre-existing behaviour.
    refuse = text.index("if drift > tol:")
    warn = text.index("if drift > warn:")
    assert refuse < warn
    # And the comparison is strict, so exactly-at-tolerance is allowed.
    assert "if drift > tol:" in text
    # The refusal must actually return, and return False.
    refusal_returns = [n for n in ast.walk(body)
                       if isinstance(n, ast.Return)
                       and isinstance(n.value, ast.Tuple)
                       and n.value.elts
                       and getattr(n.value.elts[0], "value", None) is False]
    assert refusal_returns, "the over-tolerance branch must return False"


def test_both_tiers_are_parameters_with_sane_defaults():
    """Not hard-coded, and the refuse threshold is above the warn threshold.

    A default that fails normal operation would take the whole stack out; a
    default that never fires is the same as no check.
    """
    defaults = {}
    for n in ast.walk(_module()):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "declare_parameter" and len(n.args) > 1
                and isinstance(n.args[0], ast.Constant)):
            defaults[n.args[0].value] = n.args[1]
    tol = defaults.get("execution_start_tolerance")
    warn = defaults.get("execution_start_warn_drift")
    assert tol is not None and warn is not None
    assert tol.value > warn.value > 0.0, (
        f"tolerance {tol.value} must sit above the warn threshold {warn.value}")


def test_the_check_is_on_the_resolve_path_not_the_cache_hit_path():
    """A cache hit must not be gated by this.

    ``_target_signature`` already includes the start state, so a hit proves the
    arm has not moved since the plan. Gating hits would reject correct
    executions for no reason.
    """
    execute = _func("_execute_goal")
    reuse_branch = _find_if_within(execute, "reuse")
    assert reuse_branch is not None, "the cache-hit branch was not found"
    resolve_branch = reuse_branch.orelse
    assert resolve_branch, "the re-solve branch was not found"
    re_solve_calls = set()
    for stmt in resolve_branch:
        re_solve_calls |= _calls(stmt)
    assert "_check_execution_start_drift" in re_solve_calls
    # ... and the cache-hit branch must not call it.
    hit_calls = set()
    for stmt in reuse_branch.body:
        hit_calls |= _calls(stmt)
    assert "_check_execution_start_drift" not in hit_calls, \
        "a cache hit already pins the start state; do not gate it"


def test_the_refusal_happens_before_the_solve():
    """Otherwise the divergent plan is computed and then discarded.

    Cheaper not to solve, but the real reason is that a solve that starts from
    a state the arm is not in can report success and leave the caller with a
    trajectory it never validated - which is the thing being prevented.
    """
    execute = _func("_execute_goal")
    reuse_branch = _find_if_within(execute, "reuse")
    text = ast.unparse(reuse_branch.orelse)
    check = text.index("_check_execution_start_drift")
    plan = text.index("planner.plan(")
    assert check < plan, "check must precede the solve"


def test_a_refusal_aborts_the_goal_and_reports_failure():
    """A refusal that still reports success is worse than no check.

    The caller has no way to tell a refused execution from a completed one
    except through ``success``, so it must be false and the goal must be
    aborted - not merely logged and then planned anyway.
    """
    execute = _func("_execute_goal")
    branch = _find_if_within(execute, "not drift_ok")
    assert branch is not None, "the refusal branch was not found"
    calls = _calls(branch)
    assert "abort" in calls, "a refused execution must abort the goal"
    text = ast.unparse(branch)
    assert "success = False" in text
    assert "drift_msg" in text, "the reason must reach the caller"
    # `return` is a statement, not a call, so it is not in _calls.
    assert any(isinstance(n, ast.Return) for n in ast.walk(branch)), \
        "and it must not fall through to the solve"


def test_a_goal_with_no_start_pose_is_never_gated():
    """With no start_pose the planner is already anchored to the live pose.

    ``_resolve_start_state`` falls back to the arm's current position in that
    case, so there is no stale anchor to catch and the check must not invent
    one.
    """
    body = _func("_check_execution_start_drift")
    text = ast.unparse(body)
    guard = text.index("start_pose is None")
    compare = text.index("get_joint_pose")
    assert guard < compare, "the empty-start_pose early return must come first"


def test_a_pose_that_will_not_convert_is_not_treated_as_drift():
    """A type problem must not become a refused motion.

    The comparison reads curobo tensors; if that conversion throws, the honest
    answer is "cannot tell", not "the arm is in the wrong place".
    """
    body = _func("_check_execution_start_drift")
    text = ast.unparse(body)
    assert "except" in text, "the float() conversion must be guarded"
    handlers = [n for n in ast.walk(body) if isinstance(n, ast.Try)]
    assert handlers, "there must be a try around the conversion"
    for handler in handlers[0].handlers:
        types = {ast.unparse(t) for t in handler.type.elts} \
            if isinstance(handler.type, ast.Tuple) else {ast.unparse(handler.type)}
        assert types, "the except clause must name what it catches"
        # Every handler must return the ALLOW verdict. A bare True would be a
        # different arity from the (ok, message) contract.
        allowed = [n for n in ast.walk(handler)
                   if isinstance(n, ast.Return) and isinstance(n.value, ast.Tuple)
                   and n.value.elts
                   and getattr(n.value.elts[0], "value", None) is True]
        assert allowed, \
            "an unconvertible pose must be allowed, not refused"


def test_the_check_is_not_applied_to_the_reactive_path():
    """A reactive solver re-aims every cycle, so a start pose is advisory.

    The open-loop path is the one that caches a plan and replays it; that is
    where a stale anchor silently changes the trajectory. Gating the reactive
    branch would refuse motions that are correct by construction.
    """
    execute = _func("_execute_goal")
    text = ast.unparse(execute)
    # The reactive branch is the `else` of `if planner.is_open_loop():`.
    open_loop = _find_if_within(execute, "is_open_loop")
    assert open_loop is not None
    reactive = open_loop.orelse
    assert reactive, "the reactive branch was not found"
    reactive_calls = set()
    for stmt in reactive:
        reactive_calls |= _calls(stmt)
    assert "_check_execution_start_drift" not in reactive_calls


def test_both_documented_thresholds_are_read_from_parameters_at_call_time():
    """Not captured once at init.

    Reading a parameter per call is what lets the threshold be retuned on a
    running node; capturing it into an attribute at construction time would
    silently ignore every later change.
    """
    body = _func("_check_execution_start_drift")
    gets = [n for n in ast.walk(body)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "get_parameter"]
    assert len(gets) >= 2, "both thresholds must be fetched here"
    names = {getattr(g.args[0], "value", None) for g in gets}
    assert names == {"execution_start_warn_drift", "execution_start_tolerance"}


def test_start_tolerance_default_matches_moveit_allowed_start_tolerance():
    """One start-agreement number for the whole pipeline: 0.01 rad.

    ``iki_kortex_moveit_config/config/moveit_controllers.yaml`` sets
    ``allowed_start_tolerance: 0.01``. The curobo chain must not accept a
    start state MoveIt would have refused to plan from - if the two stacks
    disagree on the wall, a handoff between MoveIt-planned and curobo-driven
    segments silently plans from configurations the other would reject. The
    warn tier must stay below it.
    """
    defaults = {}
    for n in ast.walk(_module()):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "declare_parameter" and len(n.args) > 1
                and isinstance(n.args[0], ast.Constant)):
            defaults[n.args[0].value] = n.args[1]
    tol = defaults.get("execution_start_tolerance")
    warn = defaults.get("execution_start_warn_drift")
    assert tol is not None and warn is not None
    assert tol.value == pytest.approx(0.01), (
        f"execution_start_tolerance is {tol.value}, must match MoveIt's "
        f"allowed_start_tolerance of 0.01")
    assert 0.0 < warn.value < tol.value, \
        "the warn tier must sit below the refuse tier"


def test_the_execute_action_reports_the_driven_trajectory():
    """An execute result with no waypoints is how a diverged chain drove on.

    ``TaskExecutor._diverged_from_plan`` compares the plan-time endpoints
    against the DRIVEN trajectory; when the execute action's TrajectoryResult
    carries no trajectory it returns "nothing to compare" and the chain
    continues into segments that are anchored to the PLANNED endpoint - which
    is the execution/display mismatch this suite exists to prevent. Both
    open-loop paths must therefore report the trajectory they are about to
    drive: the freshly re-solved plan on the re-solve path, the replayed
    cached plan on the cache-hit path.
    """
    execute = _func("_execute_goal")
    text = ast.unparse(execute)
    assert "fill_arrays=True" in text, \
        "the re-solve path must return the executed trajectory"
    assert "_fill_reused_trajectory" in text, \
        "the cache-hit path must return the replayed trajectory"

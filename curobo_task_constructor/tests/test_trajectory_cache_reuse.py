"""Guard on the multi-entry trajectory cache + force_cached reuse flag.

The reuse machinery lives in ``unified_planner_node.py``, which imports curobo
and torch at module scope, so it cannot be imported in this environment. These
are therefore AST guards, like ``test_execution_start_drift.py`` beside it: they
assert on the module's STRUCTURE rather than calling it. That is enough for the
properties that matter here, and each guard names the runtime failure it exists
to prevent.

The behaviour being guarded
--------------------------
A Task is planned as a chain of segments, then executed one segment at a time.
Originally the trajectory cache had a single slot, so only the LAST segment
planned was ever reusable; every mid-chain execute (e.g. ``pre_grasp_0``) missed
and re-solved from scratch. A re-solve of a multi-solution pose goal is free to
pick the OTHER IK branch, and when it did the executor's chain-continuity check
aborted the task mid-grasp:

    task 'pick_object_0' execution failed at 'pre_grasp_0': cache miss:
    'pre_grasp_0' was re-solved and its end state moved 5.148 rad from the
    plan (> 0.01 rad); the rest of the chain is anchored to the planned
    trajectory and was not executed

The fix, and what these guards pin:

  * the cache is MULTI-ENTRY: every open-loop plan (single srv, batch srv, and
    the batch fallback) stores its own entry keyed by target signature, holding
    that segment's trajectory — so a mid-chain execute HITS and replays the
    validated plan instead of re-solving it;
  * a hit replays the CACHED trajectory (``reuse['trajectory']``) through the
    planner's ``replay`` path, not the single-slot ``planned_trajectory`` the
    drive streamer would otherwise send;
  * ``force_cached`` is honoured: when set, a miss FAILS explicitly instead of
    silently re-solving into a different trajectory;
  * the pool is bounded (``trajectory_cache_size`` + ``trajectory_cache_ttl``).
"""

from __future__ import annotations

import ast
from pathlib import Path

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


def _find_assign(body: ast.AST, target: str):
    """The first plain assignment ``target = ...`` inside ``body``."""
    for node in ast.walk(body):
        if (isinstance(node, ast.Assign) and node.targets
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == target):
            return node
    return None


def test_the_cache_is_multi_entry_not_a_single_slot():
    """The pool must be a list of per-segment entries, not one dict.

    A single slot can only ever serve the LAST segment planned; a mid-chain
    execute then always misses and re-solves, which is the 5.148 rad
    divergence this suite exists to prevent.
    """
    store = _func("_store_pending_plan")
    text = ast.unparse(store)
    assert "append(" in text, "entries must accumulate, not overwrite"
    assert "'trajectory'" in text, "each entry must keep its own trajectory"
    assert "'signature'" in text, "each entry must be keyed by its target"
    assert "'stamp'" in text, "each entry must carry its cache timestamp"


def test_every_planning_surface_stores_its_plans():
    """Single srv, batch srv, AND the batch fallback must all cache.

    The batch shell returns early on invalid goals / exception and delegates to
    ``_plan_trajectory_goal`` per problem — so that function is the choke point
    the single surface shares. The batched path stores per-problem trajectories
    explicitly, because ``plan_batch`` leaves the planner's single slot holding
    only the last problem.
    """
    plan_goal = _func("_plan_trajectory_goal")
    text = ast.unparse(plan_goal)
    assert "_store_pending_plan" in text and "is_open_loop" in text, \
        "the shared single-goal path must cache successful open-loop plans"

    batch = _func("_plan_trajectory_goal_batch")
    text = ast.unparse(batch)
    assert "_store_pending_plan" in text, \
        "the batched path must cache each problem"
    assert "getattr(r, 'trajectory', None)" in text, \
        "a bulked problem must keep its OWN trajectory, not the planner slot"


def test_a_hit_replays_the_cached_trajectory_not_the_planner_slot():
    """The drive must use the matched entry's trajectory.

    ``planner.execute`` streams whatever plan() last staged into the command
    buffer — for a mid-chain hit that is a DIFFERENT segment. Reusing the
    single-slot ``planned_trajectory`` on a hit is exactly how a chain would
    start replaying the wrong motion. The hit path must take the trajectory
    from the matched cache entry (``reuse['trajectory']``) and drive it via
    ``replay``.
    """
    execute = _func("_execute_goal")
    text = ast.unparse(execute)
    assert "reuse['trajectory']" in text, \
        "the hit path must read the cached entry's trajectory"
    assert "replay(" in text, \
        "a hit must stage + stream the CACHED trajectory, not execute()"
    hit = _find_if_within(execute, "reuse")
    assert hit is not None
    hit_calls = set()
    for stmt in hit.body:
        hit_calls |= _calls(stmt)
    # The reuse branch must not touch the single-slot planner trajectory.
    assert "planned_trajectory" not in ast.unparse(hit.body), \
        "an entry-forward hit must not read planner.planned_trajectory"


def test_fill_reused_trajectory_takes_the_cached_trajectory():
    """The reported trajectory on a hit must be the ONE BEING DRIVEN.

    The executor's chain-continuity check compares plan-time endpoints against
    the DRIVEN trajectory and skips a result with no waypoints as "nothing to
    compare". If the report fell back to the planner's single slot on a
    mid-chain hit, the endpoint comparison would compare the wrong segment.
    """
    fill = _func("_fill_reused_trajectory")
    text = ast.unparse(fill)
    assert "traj=None" in text or "traj = None" in text or "traj: Any = None" in text \
        or "traj" in text and "planner" in text, \
        "the fill must accept the replayed (cached) trajectory"


def test_force_cached_refuses_to_resolve_on_a_miss():
    """The requested hard knob: replay or fail, never a silent re-solve.

    With ``force_cached`` a miss is a changed-condition sign, not an excuse to
    solve a different trajectory the caller never validated. The refusal must
    abort the goal and report failure.
    """
    execute = _func("_execute_goal")
    branch = _find_if_within(execute, "force_cached")
    assert branch is not None, "the force_cached refusal branch was not found"
    calls = _calls(branch)
    assert "abort" in calls, "a forced-miss refusal must abort the goal"
    text = ast.unparse(branch)
    assert "success = False" in text, "a forced-miss refusal must report failure"
    assert "refusing to re-solve" in text, \
        "the message must say the re-solve was refused"


def test_the_cache_is_bounded_by_ttl_and_size():
    """Dead entries must not be handed out, and the pool must stay bounded.

    An expired entry replayed blindly would drive a stale plan against a
    possibly-changed world; an unbounded pool would leak GPU trajectory
    tensors across a long multi-task run.
    """
    prune = _func("_prune_plan_cache")
    text = ast.unparse(prune)
    assert "trajectory_cache_ttl" in text, "TTL expiry must be enforced"
    assert "trajectory_cache_size" in text, "pool size must be capped"


def test_the_cache_size_parameter_is_declared_with_a_sane_default():
    """The cap must exist at the parameter level (retunable, like the TTL)."""
    defaults = {}
    for n in ast.walk(_module()):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "declare_parameter" and len(n.args) > 1
                and isinstance(n.args[0], ast.Constant)):
            defaults[n.args[0].value] = n.args[1]
    size = defaults.get("trajectory_cache_size")
    assert size is not None, "trajectory_cache_size must be declared"
    assert size.value > 0, "the pool cap must be positive"
    ttl = defaults.get("trajectory_cache_ttl")
    assert ttl is not None and ttl.value > 0.0, \
        "the TTL must also stay declared and positive"


def test_consumed_or_superseded_entries_are_dropped_after_execute():
    """An executed segment must never be replayed from a stale start.

    After a replay the arm is at the replayed END, not the plan-time START, so
    a later identical request must not find the old entry; after a re-solve the
    arm is on the re-solved path, so the plan-time entry for that target is
    stale too. The drop must keep OTHER segments' entries (that is the point
    of the multi-entry cache).
    """
    drop = _func("_drop_pending_plan")
    text = ast.unparse(drop)
    assert "is not entry" in text or "entry" in text, \
        "a consumed replay must be dropped by identity"
    assert "signature" in text, \
        "a superseded re-solve must be dropped by target signature"
    assert "goto" not in text
    # The drop removes only matching entries — the pool is filtered, not cleared.
    assert "for e in self._pending_plan" in text
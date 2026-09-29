"""Every goalset-sending method must bracket its call with the collision allowance.

``GoalsetSpec.allowed_collisions`` is the task constructor's ONLY channel for
contact allowance: the wire ``Goalset`` carries no collision concept, so
``CuroboServerInterface._set_links_collision`` -> the server's
``set_link_collision`` service is the only way the solver ever learns that e.g.
the finger collision spheres may be turned off to let them touch the object
being grasped. A method that sends a goalset without that bracket is therefore
not merely inconsistent — it silently plans with one world and solves with
another.

The failure this exists to catch, observed on a real pick: ``plan`` and
``plan_batch`` both bracketed, ``execute`` did not. The grasp close planned
cleanly with the finger spheres off, then the server's execute path re-solved
that same goalset (the trajectory cache is a single slot, so a mid-chain
segment misses it) against spheres that the batch's ``finally`` had already
restored — and the goal state came back rejected:

    goal: 5 contacts
      right_inner_finger_pad  -> plane_0   5.0 mm past surface
      right_inner_finger      -> object_0  5.0 mm past surface
      left_inner_finger_pad   -> plane_0   1.8 mm past surface
      right_inner_knuckle     -> object_0  1.2 mm past surface
      left_inner_finger       -> object_0  0.4 mm past surface

which surfaced as "pick task failed at 'close_0'". No fallbacks container can
absorb that: the container commits to a child at PLAN time, and by the time a
segment is driven the plan is long since chosen.

``CuroboServerInterface`` cannot be imported without a ROS 2 install
(``geometry_msgs``), so this asserts the invariant over the source AST instead
of instantiating the class. That is the point: the guard has to run in the
ROS-free environment where every other test in this package runs.
"""

import ast
import pathlib

ADAPTER = (pathlib.Path(__file__).resolve().parents[1]
           / "curobo_task_constructor" / "robot" / "curobo.py")

#: Methods that hand a goalset to the server and therefore must bracket.
#: Keep in sync by ADDING here, never by relaxing the assertions — the
#: "no sender escaped the list" test fails when a new one appears unlisted.
GOALSET_SENDING_METHODS = ("plan", "plan_batch", "execute")

#: Pure wire conversions. They build a Goalset message but never send it, so
#: they legitimately have no bracket.
CONVERTERS = ("_to_goal", "_to_goalset", "_to_joint_msg", "_to_options",
              "_from_result")


def _source():
    return ADAPTER.read_text(encoding="utf-8")


def _methods():
    """name -> FunctionDef, for every method of every class in the adapter."""
    tree = ast.parse(_source())
    out = {}
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for item in cls.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[item.name] = item
    return out


def _self_calls(node, attr):
    """Every ``self.<attr>(...)`` call inside ``node``, nested ones included."""
    return [n for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == attr
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "self"]


def _literal_flag(calls, value):
    """The calls whose second positional argument is the literal ``value``."""
    return [c for c in calls
            if len(c.args) >= 2 and isinstance(c.args[1], ast.Constant)
            and c.args[1].value is value]


def test_adapter_source_is_found():
    assert ADAPTER.is_file(), f"adapter source not at {ADAPTER}"


def test_named_methods_exist():
    methods = _methods()
    missing = [n for n in GOALSET_SENDING_METHODS if n not in methods]
    assert not missing, (
        f"CuroboServerInterface.{missing} is gone — re-point "
        "GOALSET_SENDING_METHODS at its replacement, do not just drop it")


def test_every_goalset_sending_method_brackets_the_collision_allowance():
    """Each of plan / plan_batch / execute turns the spheres off before its
    send and back on afterwards."""
    methods = _methods()
    problems = []
    for name in GOALSET_SENDING_METHODS:
        calls = _self_calls(methods[name], "_set_links_collision")
        if not _literal_flag(calls, False):
            problems.append(f"{name}(): never disables the allowed links "
                            "(no _set_links_collision(..., False))")
        if not _literal_flag(calls, True):
            problems.append(f"{name}(): never restores the allowed links "
                            "(no _set_links_collision(..., True))")
        offs = _literal_flag(calls, False)
        ons = _literal_flag(calls, True)
        if offs and ons and ons[0].lineno < offs[0].lineno:
            problems.append(f"{name}(): restores the links before disabling them")
    assert not problems, (
        "contact allowance is not bracketed around the server call:\n  "
        + "\n  ".join(problems))


def test_every_goalset_sending_method_restores_in_a_finally():
    """A rejected goal, a timeout, or an exception mid-solve must not leave the
    finger spheres off for whatever runs next — the restore belongs in a
    ``finally``, not on the success path."""
    methods = _methods()
    problems = []
    for name in GOALSET_SENDING_METHODS:
        tries = [n for n in ast.walk(methods[name])
                 if isinstance(n, ast.Try) and n.finalbody]
        if not tries:
            problems.append(f"{name}(): no try/finally — an early return or a "
                            "raised ServiceError would leak the allowance")
        elif not any(_literal_flag(_self_calls(t, "_set_links_collision"), True)
                     for t in tries):
            problems.append(f"{name}(): the finally block does not restore the "
                            "allowed links")
    assert not problems, "\n  ".join(problems)


def test_no_goalset_sending_method_escaped_the_list():
    """Guard the guard: a NEW method that builds a goalset must be classified,
    so the list above cannot quietly rot and stop covering anything."""
    senders = set()
    for fn in _methods().values():
        if fn.name in CONVERTERS:
            continue
        if _self_calls(fn, "_to_goal") or _self_calls(fn, "_to_goalset"):
            senders.add(fn.name)
    unclassified = senders - set(GOALSET_SENDING_METHODS)
    assert not unclassified, (
        "these methods build a goalset but are not in GOALSET_SENDING_METHODS: "
        f"{sorted(unclassified)} — either they send a goalset to the server and "
        "need the bracket, or they are a pure conversion and belong in CONVERTERS")

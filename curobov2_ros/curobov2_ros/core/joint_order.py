"""Resolve name-keyed joint lists into the robot's active cspace order.

The planning wire is POSITIONAL: ``TrajectoryGoal.start_pose.position`` and
``Goalset.target_joint_positions`` are float lists the node resolves verbatim
in the model's active cspace order
(``kinematics.cspace.joint_names``). Clients therefore had to know that order
themselves, which is why a robot descriptor and a client-side reorder
existed in curobo_task_constructor at all.

Both fields now optionally carry their names
(``TrajectoryGoal.start_pose.name`` for the start state, added
``Goalset.target_joint_names`` for goals), and this module is the ONE place
that maps names onto the model order. An empty or length-mismatched name list
is treated as "the caller sent already-ordered positions", which is exactly
what a bare list means, so every existing name-less client behaves as before.

Deliberately strict about what it rejects: an unknown or duplicated name is
an ERROR, never a silent drop. A request that names the wrong joint must
fail loudly, because the failure mode it replaces (a start state shifted by
one joint) looks like a limit violation or a pile of phantom self-contacts
and costs hours to diagnose.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence


def _names_of(joint_state: Any) -> List[str]:
    """Names of a JointState-like, [] when it carries none."""
    if joint_state is None:
        return []
    return [str(n) for n in (getattr(joint_state, "name", None) or [])]


def _positions_of(joint_state: Any) -> List[float]:
    if joint_state is None:
        return []
    return [float(v) for v in (getattr(joint_state, "position", None) or [])]


def named_order(names: Sequence[str], values: Sequence[float],
                model_names: Sequence[str]) -> Optional[List[float]]:
    """``values`` re-ordered into ``model_names``, or None if not name-keyed.

    ``names`` parallel to ``values``. Returns None when the request is NOT
    name-keyed — empty names, or a length that disagrees with ``values`` —
    in which case the caller keeps the legacy positional reading.

    Raises ValueError when the names ARE the key (length matches) but name a
    joint the model does not have, or name one twice.
    """
    name_list = [str(n) for n in (names or [])]
    if not name_list or len(name_list) != len(values or []):
        return None
    model = [str(n) for n in (model_names or [])]
    model_set = set(model)
    unknown = [n for n in name_list if n not in model_set]
    if unknown:
        raise ValueError(
            f"unknown joint name(s) {unknown}: the robot's active cspace is "
            f"{model}. Fix the name, or send a nameless position list to keep "
            f"the legacy positional reading."
        )
    if len(set(name_list)) != len(name_list):
        seen, dup = set(), []
        for name in name_list:
            if name in seen:
                dup.append(name)
            seen.add(name)
        raise ValueError(f"duplicate joint name(s) {dup}; each joint may be named once")
    # Map REQUEST name -> its position in the request's value list, then walk
    # the model's order and pick the value each model joint was given.
    request_index = {name: i for i, name in enumerate(name_list)}
    out: List[float] = []
    for name in model:
        pos = request_index.get(name)
        if pos is not None:
            out.append(float(values[pos]))
    return out


def resolve_joint_targets(names: Sequence[str], values: Sequence[float],
                          model_names: Sequence[str]) -> List[float]:
    """Positions for a joint-space goal, in the model's active order.

    Name-keyed requests are reordered here; nameless ones pass through
    untouched (legacy: already in active-DOF order).
    """
    ordered = named_order(names, values, model_names)
    return list(values) if ordered is None else ordered


def resolve_start_pose(start_pose: Any, model_names: Sequence[str],
                       current: Sequence[float]) -> List[float]:
    """Start joint list, re-ordered when ``start_pose`` carries names.

    A short start pose (the MoveIt arm group sends only its own DOF) keeps its
    historical meaning: it is a PREFIX of the cspace, and the trailing DOF are
    padded from ``current``. Padding happens AFTER the reorder, so the length
    the reorder sees is the length the caller actually named.

    A named pose that is neither full-length nor a prefix is rejected: padding
    it silently would put its values into the wrong joints, which is the exact
    failure this whole mechanism exists to prevent.
    """
    names = _names_of(start_pose)
    positions = _positions_of(start_pose)
    if not positions or not names:
        return list(positions) if positions else list(current)
    ordered = named_order(names, positions, model_names)
    if ordered is None:  # name/position length disagree -> legacy positional
        return list(positions)
    model = [str(n) for n in (model_names or [])]
    if len(ordered) < len(model):
        prefix = model[:len(ordered)]
        named_set = {str(n) for n in names}
        if named_set != set(prefix):
            raise ValueError(
                f"named start pose covers {sorted(named_set)}, which is not "
                f"the cspace prefix {prefix}. A partial start pose is read as "
                f"a prefix and padded from the robot's current pose, so name "
                f"either the full cspace or that prefix."
            )
    return ordered

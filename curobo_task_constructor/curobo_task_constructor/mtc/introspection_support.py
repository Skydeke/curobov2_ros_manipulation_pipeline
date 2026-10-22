"""MTC-shaped local task API over cuRobo (``core`` namespace). Mirrors ``moveit.task_constructor.core`` so ``cartesian.py``-style scripts run
with only the import changed::

    from curobo_task_constructor.mtc import core, stages

    task = core.Task(robot)
    task.name = "cartesian"
    task.add(stages.CurrentState("current state"))
    ... ``Task.plan()`` runs **locally in-process** (MTC C++ semantics): it converts
the added wrapper stages to a ``StageSpec`` tree and drives ``TaskExecutor``
directly against the injected ``RobotInterface``."""

from __future__ import annotations

from typing import Any, Optional

#: Introspection namespace default (MTC task ns; absolute default kept for compat).
#: MTC uses RELATIVE topics ``description/statistics/solution`` in the task's
#: namespace plus service ``get_solution_<task_id>``. This package keeps the
#: absolute default ``/curobo_task_constructor/...`` but honours ``Task(ns)``:
#: ``Task(ns="my_ns")`` publishes ``my_ns/description`` etc. and serves
#: ``my_ns/get_solution_<task_id>`` — see ``Task._ns_base``.


_DISPLAY_PROPERTIES = {
    "current_state": ("require_not_attached", "require_attached", "timeout"),
    "fixed_state": ("goal", "timeout"),
    "move_to": ("group", "goal", "ik_frame", "path_constraints", "timeout"),
    "move_relative": ("group", "ik_frame", "axis", "distance", "rotation",
                      "joint_offsets", "min_distance", "max_distance",
                      "path_constraints", "timeout"),
    "connect": ("merge_mode", "max_distance", "path_constraints", "timeout"),
    "compute_ik": ("group", "eef", "max_solutions", "target_pose", "ik_frame",
                   "ignore_collisions", "timeout"),
    "generate_pose": ("pose", "timeout"),
    "cartesian_path": ("goal", "hold", "step_size", "precision",
                       "ik_frame", "timeout"),
    "modify_scene": ("add", "remove", "remove_all", "attach", "attach_link",
                     "detach", "detach_all", "allow_collisions", "timeout"),
}


#: Process-global GetSolution services by rendezvous name, served on the
#: shared introspection node. MTC never collides (its task_id embeds
#: hostname_pid_this-pointer), but this pipeline reuses labels across cycles
#: ("setup_scene" every cycle, "pick" every pick), so a new Task must
#: deterministically TAKE OVER the name: depending on garbage-collection
#: timing, a bare create_service would otherwise raise (failing
#: _ensure_publishers closed, silencing the whole task — no description,
#: no statistics, no solutions, panel never updates) or leave the STALE
#: task answering ids from a reset registry (wrong/empty payloads).
_SERVICE_HANDLES: dict = {}
_SERVICE_LOCK: Any = None


def _service_lock():
    """Process lock guarding _SERVICE_HANDLES."""
    global _SERVICE_LOCK
    import threading

    if _SERVICE_LOCK is None:
        _SERVICE_LOCK = threading.Lock()
    return _SERVICE_LOCK


def _replace_service(node: Any, srv_type: Any, name: str, callback: Any):
    """Create a named service, destroying any live predecessor first."""
    with _service_lock():
        old = _SERVICE_HANDLES.get(name)
        if old is not None:
            try:
                node.destroy_service(old)
            except Exception:
                pass
            _SERVICE_HANDLES.pop(name, None)
        handle = node.create_service(srv_type, name, callback)
        _SERVICE_HANDLES[name] = handle
        return handle


#: Process-shared introspection publishers by topic (see ``_shared_publisher``).
_SHARED_PUBLISHERS: dict = {}


def _shared_publisher(node: Any, msg_type: Any, topic: str, depth: int):
    """One publisher per topic for the whole process (the ``viz.py`` pattern). Every motion builds a fresh ``Task``, so a publisher per task is a
    publisher leak per motion: the node retains every created publisher in
    ``node._publishers`` until ``destroy_publisher`` runs, and dropping the
    Python handle does NOT destroy it — the process accumulates 3
    TRANSIENT_LOCAL publishers (+ their latched samples) per motion,
    unbounded. The topics are task-namespace rendezvous, not per-task state,
    so all tasks with the same namespace publish the same three topics and
    share one handle each."""
    with _service_lock():
        pub = _SHARED_PUBLISHERS.get(topic)
        if pub is None:
            from rclpy.qos import (
                DurabilityPolicy,
                HistoryPolicy,
                QoSProfile,
                ReliabilityPolicy,
            )

            # MTC introspection.cpp:118-126 rendezvous, verbatim (depths 2/1/1,
            # transient_local).
            pub = node.create_publisher(
                msg_type, topic,
                QoSProfile(
                    depth=depth, history=HistoryPolicy.KEEP_LAST,
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL))
            _SHARED_PUBLISHERS[topic] = pub
        return pub

__all__ = [
    '_DISPLAY_PROPERTIES',
    '_SERVICE_HANDLES',
    '_SERVICE_LOCK',
    '_service_lock',
    '_replace_service',
    '_SHARED_PUBLISHERS',
    '_shared_publisher',
]

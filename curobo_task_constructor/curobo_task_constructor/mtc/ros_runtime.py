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


class MoveItErrorCode:
    """Minimal ``moveit_msgs/MoveItErrorCodes`` equivalent (MTC ``plan/execute`` return).

    ``Task.plan/execute`` return this (not bool) so callers can check
    ``.val == MoveItErrorCode.SUCCESS`` exactly like MoveIt. Truthiness
    follows MTC success (``bool(code)`` is ``val == SUCCESS``) so existing
    ``assert task.plan()`` / ``if task.execute():`` call sites keep working.
    """

    SUCCESS = 1
    FAILURE = 99999
    PLANNING_FAILED = -1
    INVALID_MOTION_PLAN = -3
    CONTROL_FAILED = -6
    PREEMPTED = -5

    def __init__(self, val: int = SUCCESS):
        self.val = int(val)

    def __bool__(self) -> bool:
        return self.val == MoveItErrorCode.SUCCESS

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, MoveItErrorCode):
            return self.val == other.val
        if isinstance(other, int):
            return self.val == other
        return NotImplemented

    def __repr__(self) -> str:
        return f"MoveItErrorCode(val={self.val})"


def _ros_ok() -> bool:
    """MTC's ``rclcpp::ok()`` — is a ROS context initialized?

    MTC enables introspection at Task construction only when it is
    (task.cpp:102-104). Same gate here: without rclpy up, a Task plans and
    executes locally and publishes nothing, rather than half-creating an
    introspection node that cannot spin.
    """
    try:
        import rclpy

        return bool(rclpy.ok())
    except Exception:
        return False


class _IntrospectionExecutor:
    """Shared, ref-counted spinning executor (MTC ``IntrospectionExecutor``). MTC gives each ``Introspection`` ITS OWN node
    (``introspection_<task_id>`` in the task's namespace) on one static
    executor that spins a thread while any of those nodes is alive and stops
    when the last one goes away (introspection.cpp:82-110). That is why the
    MTC API has no ``attach_node``: the caller never supplies a node, and the
    panel is fed regardless."""

    _executor: Any = None
    _thread: Any = None
    _count: int = 0
    _lock: Any = None

    @classmethod
    def _lock_(cls):
        import threading

        if cls._lock is None:
            cls._lock = threading.Lock()
        return cls._lock

    @classmethod
    def add_node(cls, node: Any) -> None:
        with cls._lock_():
            if cls._executor is None:
                from rclpy.executors import SingleThreadedExecutor

                cls._executor = SingleThreadedExecutor()
            cls._executor.add_node(node)
            if cls._count == 0:
                import threading

                # The ONE and ONLY spin() in this package, and it is on an
                # Executor: start the shared executor's spin loop exactly
                # once, at the 0 -> 1 transition. Everything else that needs
                # serviced uses spin_once() on an executor that ONLY
                # the calling thread drives - sharing an already-spinning
                # executor across threads is what raises "Executor is
                # already spinning" and, downstream, took the planner down.
                # NEVER call spin() on a node, and never add a second spin()
                # to this executor.
                cls._thread = threading.Thread(
                    target=cls._executor.spin, daemon=True)
                cls._thread.start()
            cls._count += 1

    @classmethod
    def remove_node(cls, node: Any) -> None:
        with cls._lock_():
            if cls._executor is None:
                return
            cls._executor.remove_node(node)
            cls._count -= 1
            if cls._count > 0:
                return
            # Last node gone: cancel, join, drop (MTC does exactly this).
            executor, thread = cls._executor, cls._thread
            cls._executor = None
            cls._thread = None
            cls._count = 0
        executor.shutdown()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)


#: The one introspection node in this process (MTC's per-Introspection
#: ``node_``, made process-shared because a Task per motion would otherwise
#: leak a node per motion — see ``Introspection._ensure_publishers``).
_INTROSPECTION_NODE: Any = None
_INTROSPECTION_NODE_LOCK: Any = None


def _private_action_executor() -> Any:
    """A PRIVATE (node, executor) pair for action round trips. The executor has no ``spin()`` running on it anywhere, and is to be
    serviced by the CALLING thread with ``spin_once()`` only. It is
    structurally impossible to get the shared introspection executor from
    here: that one has a background ``spin()``, and driving it from two
    threads is what raised "Executor is already spinning"."""
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    node = rclpy.create_node("curobo_task_constructor_executor")
    executor = SingleThreadedExecutor()
    assert executor is not _IntrospectionExecutor._executor, (
        "_private_action_executor must never return the shared executor")
    executor.add_node(node)
    return node, executor


#: Process-shared ExecuteTaskSolution endpoints by action name:
#: ``[node, executor, client, lock]``. See ``_shared_execute_endpoint``.
_EXEC_ENDPOINTS: dict = {}


def _shared_execute_endpoint(action_name: str, action_type: Any):
    """One (node, executor, ActionClient, lock) per action, process lifetime. ``Task.execute`` runs once per motion, so the MTC-shaped "private node
    per execute" is a node per motion — and node create/destroy cycles leak
    (~350KB each at the rcl layer, measured; entity counts stay flat while
    RSS climbs) even with ``destroy_node()`` in a ``finally``. The endpoint
    is therefore created once per action rendezvous and reused: one node,
    one client, zero churn."""
    import threading

    from rclpy.action.client import ActionClient

    with _service_lock():
        ep = _EXEC_ENDPOINTS.get(action_name)
        if ep is None:
            node, executor = _private_action_executor()
            client = ActionClient(node, action_type, action_name)
            ep = [node, executor, client, threading.Lock()]
            _EXEC_ENDPOINTS[action_name] = ep
        return ep[0], ep[1], ep[2], ep[3]


def _shared_introspection_node() -> Any:
    """Create (once) and return the shared introspection node.

    ``None`` when rclpy is not up (plain unit tests): publication then fails
    closed, exactly as MTC's introspection stays off without ``rclcpp::ok()``.
    """
    global _INTROSPECTION_NODE, _INTROSPECTION_NODE_LOCK
    import threading

    if _INTROSPECTION_NODE_LOCK is None:
        _INTROSPECTION_NODE_LOCK = threading.Lock()
    with _INTROSPECTION_NODE_LOCK:
        if _INTROSPECTION_NODE is not None:
            return _INTROSPECTION_NODE
        try:
            import rclpy

            if not rclpy.ok():
                return None
            node = rclpy.create_node("curobo_task_constructor_introspection")
            _IntrospectionExecutor.add_node(node)
        except Exception:
            return None
        _INTROSPECTION_NODE = node
        return node


def shutdown_shared_introspection() -> None:
    """Tear down the shared introspection node, execute endpoints, and spin thread. MTC stops its introspection executor when the last introspection goes
    away; the shared node pins the refcount at 1 forever, so without this the
    background ``spin()`` outlives every task and ``rclpy.shutdown()`` aborts
    the process ("terminate called without an active exception"). Process-exit
    cleanup only: the next ``_ensure_publishers`` / ``_execute_remote`` after
    this recreates everything, so a task that outlives the call keeps
    publishing onto dead handles (caught per-call) until it is released."""
    global _INTROSPECTION_NODE
    lock = _INTROSPECTION_NODE_LOCK
    if lock is None:
        node = _INTROSPECTION_NODE
        _INTROSPECTION_NODE = None
    else:
        with lock:
            node, _INTROSPECTION_NODE = _INTROSPECTION_NODE, None
    with _service_lock():
        # Handles die with the nodes below (destroy_node destroys every
        # tracked entity); the registries must not keep naming them.
        _SERVICE_HANDLES.clear()
        _SHARED_PUBLISHERS.clear()
        endpoints = list(_EXEC_ENDPOINTS.values())
        _EXEC_ENDPOINTS.clear()
    for ep_node, ep_executor, ep_client, _ep_lock in endpoints:
        try:
            ep_client.destroy()
        except Exception:
            pass
        try:
            ep_executor.remove_node(ep_node)
        except Exception:
            pass
        try:
            ep_node.destroy_node()
        except Exception:
            pass
    if node is None:
        return
    try:
        _IntrospectionExecutor.remove_node(node)
    except Exception:
        pass
    try:
        node.destroy_node()
    except Exception:
        pass


class _DoneFuture:

    def __init__(self, result: Any = None):
        self._result = result
        self._callbacks: list = []
        if result is not None:
            self._fire()

    def set_result(self, result: Any) -> "_DoneFuture":
        self._result = result
        self._fire()
        return self

    def _fire(self) -> None:
        callbacks, self._callbacks = self._callbacks, []
        for callback in callbacks:
            try:
                callback(self)
            except Exception:
                pass

    def add_done_callback(self, callback) -> None:
        if self._result is not None:
            try:
                callback(self)
            except Exception:
                pass
        else:
            self._callbacks.append(callback)

    def result(self) -> Any:
        result = self._result
        while isinstance(result, _DoneFuture):
            result = result._result
        return result


def _result_ok(res: Any) -> bool:
    """ExecuteTaskSolution result -> success bool (MTC error_code check).

    MTC reports ``moveit_msgs/MoveItErrorCodes`` (SUCCESS == 1); accept
    that as well as a plain ``success`` bool for duck-typed results.
    """
    code = getattr(res, "error_code", None)
    if code is not None:
        val = getattr(code, "val", code)
        try:
            return int(val) == 1
        except (TypeError, ValueError):
            pass
    res = getattr(res, "result", res)
    code = getattr(res, "error_code", None)
    if code is not None:
        val = getattr(code, "val", code)
        try:
            return int(val) == 1
        except (TypeError, ValueError):
            pass
        return False
    return bool(getattr(res, "success", False))


def roscpp_init(name: str, *args, **kwargs) -> None:
    """MTC ``py_binding_tools.roscpp_init`` equivalent: init rclpy if present."""
    try:
        import rclpy

        if not rclpy.ok():
            rclpy.init(args=None)
    except Exception:
        pass

__all__ = [
    'MoveItErrorCode',
    '_ros_ok',
    '_IntrospectionExecutor',
    '_INTROSPECTION_NODE',
    '_INTROSPECTION_NODE_LOCK',
    '_private_action_executor',
    '_EXEC_ENDPOINTS',
    '_shared_execute_endpoint',
    '_shared_introspection_node',
    'shutdown_shared_introspection',
    '_DoneFuture',
    '_result_ok',
    'roscpp_init',
]

from .introspection_support import _SERVICE_HANDLES, _SHARED_PUBLISHERS, _service_lock

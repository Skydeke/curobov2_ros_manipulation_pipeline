
from __future__ import annotations

from typing import Any, Optional

#: Introspection namespace default (MTC task ns; absolute default kept for compat).
#: MTC uses RELATIVE topics ``description/statistics/solution`` in the task's
#: namespace plus service ``get_solution_<task_id>``. This package keeps the
#: absolute default ``/curobo_task_constructor/...`` but honours ``Task(ns)``:
#: ``Task(ns="my_ns")`` publishes ``my_ns/description`` etc. and serves
#: ``my_ns/get_solution_<task_id>`` — see ``Task._ns_base``.

from curobo_task_constructor.mtc.constants import *
from curobo_task_constructor.mtc.ros_runtime import *
from curobo_task_constructor.mtc.planners import *
from curobo_task_constructor.mtc.containers import *
from curobo_task_constructor.mtc.introspection_support import *
from curobo_task_constructor.mtc.introspection import *

class _TaskRemoteMixin:
    def _invalidate_planner_session(self) -> None:
        """Forget any cached "planner already active" belief on the robot.

        The adapter may skip a redundant ``SetPlanner`` round trip within one
        burst, but a remote ``execute()`` (or a sibling node) mutates the
        shared ``/curobo_server`` planner behind our back. Dropping the belief
        is a no-op for interfaces that never cache (see
        ``RobotInterface.invalidate_planner_session``).
        """
        robot = getattr(self, "robot", None)
        forget = getattr(robot, "invalidate_planner_session", None)
        if callable(forget):
            forget()

    def _execute_remote(self, solution: Any):
        """Send a full Solution message through the ExecuteTaskSolution action. MTC's ``Task::execute`` (task.cpp:282-330) drives the action on a
        PRIVATE node and a LOCAL single-threaded executor that the CALLING
        thread services with ``spin_once()``, waits 0.5 s for the action
        server and then blocks for the result, honouring ``preempt()`` as a
        goal cancel. Same shape here."""
        import rclpy
        from curobo_task_constructor_interfaces.action import (
            ExecuteTaskSolution,
        )
        from curobo_task_constructor import msg_convert

        if isinstance(solution, dict):
            d = solution
        elif self._executor is not None:
            d = self._executor.solution_to_dict(solution)
        else:
            raise RuntimeError(
                "Task.execute(): cannot serialize solution without executor")
        goal = ExecuteTaskSolution.Goal()
        goal.solution = msg_convert.solution_to_msg(d)

        # Shared endpoint (node + LOCAL executor + client), driven by THIS
        # thread only while holding the endpoint lock (see the spin invariant
        # in _IntrospectionExecutor.add_node and _shared_execute_endpoint).
        # Never the shared, already-spinning introspection executor.
        _, executor, client, lock = _shared_execute_endpoint(
            self.execute_action_name(), ExecuteTaskSolution)
        with lock:
            # Bounded wait (MTC: 0.5s) - an execution backend that is not up
            # must fail the call, never hang the caller's thread.
            if not client.wait_for_server(
                    timeout_sec=rclpy.duration.Duration(
                        seconds=EXECUTE_WAIT_SEC)):
                return MoveItErrorCode(MoveItErrorCode.FAILURE)

            goal_future = client.send_goal_async(goal)
            if not self._spin_until(executor, goal_future):
                return MoveItErrorCode(MoveItErrorCode.FAILURE)
            goal_handle = goal_future.result()
            if goal_handle is None or not goal_handle.accepted:
                return MoveItErrorCode(MoveItErrorCode.FAILURE)

            result_future = goal_handle.get_result_async()
            while not result_future.done():
                if self._preempted:
                    cancel_future = client.async_cancel_goal(goal_handle)
                    self._preempted = False
                    self._spin_until(executor, cancel_future)
                    return MoveItErrorCode(MoveItErrorCode.PREEMPTED)
                # rclpy has no spin_some() - that is a C++ rclcpp method
                # (MTC's Task::execute calls executor.spin_some()). The
                # rclpy equivalent for "service this once, then check" is
                # spin_once(), which is safe here because this executor has
                # no background spin() and the endpoint lock holds off every
                # other driver.
                executor.spin_once(timeout_sec=0.01)
            raw = result_future.result()
            # Mirror MTC: surface the server's MoveItErrorCodes value, default
            # CONTROL_FAILED when the wrapper only yields a bool.
            code = getattr(getattr(raw, "result", raw), "error_code", None)
            val = getattr(code, "val", None)
            if val is not None:
                try:
                    return MoveItErrorCode(int(val))
                except (TypeError, ValueError):
                    pass
            ok = _result_ok(raw)
            return MoveItErrorCode(
                MoveItErrorCode.SUCCESS if ok else MoveItErrorCode.CONTROL_FAILED)

    @staticmethod
    def _spin_until(executor, future) -> bool:
        """Drive ``executor`` until ``future`` completes; False on timeout.

        The executor is private to the caller (never the shared, spinning
        one), so ``spin_until_future_complete`` is safe here — it wakes on
        completion instead of polling at a fixed granularity.
        """
        executor.spin_until_future_complete(
            future, timeout_sec=EXECUTE_WAIT_SEC)
        return future.done()

__all__ = [
    '_TaskRemoteMixin',
]

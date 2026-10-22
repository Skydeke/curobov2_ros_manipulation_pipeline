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


DEFAULT_TASK_NS = "/curobo_task_constructor"
DESCRIPTION_TOPIC = f"{DEFAULT_TASK_NS}/task_description"
STATISTICS_TOPIC = f"{DEFAULT_TASK_NS}/task_statistics"
SOLUTION_TOPIC = f"{DEFAULT_TASK_NS}/solution"
GET_SOLUTION_SERVICE = f"{DEFAULT_TASK_NS}/get_solution"
#: ExecuteTaskSolution rendezvous. MTC uses the RELATIVE action name
#: ``execute_task_solution`` on its own private node
#: (``moveit_task_constructor_executor_<this>``, ``task.cpp:282-288``), and
#: ``Task::execute`` waits **0.5 s** for that server before returning
#: ``MoveItErrorCode::FAILURE`` — an execution backend that is not up must
#: fail the call, never hang the caller.
EXECUTE_ACTION = "/curobo_task_constructor/execute_task_solution"
#: MTC ``wait_for_action_server(0.5s)``.
EXECUTE_WAIT_SEC = 0.5

PLANNER_CLASSIC = 0
#: SetPlanner CARTESIAN (=3): the MTC ``solvers::CartesianPath`` backend. It
#: plans a straight line through the axis-hold cost.
PLANNER_CARTESIAN = 3
PLANNER_JOINT_SPACE = 5

__all__ = [
    'DEFAULT_TASK_NS',
    'DESCRIPTION_TOPIC',
    'STATISTICS_TOPIC',
    'SOLUTION_TOPIC',
    'GET_SOLUTION_SERVICE',
    'EXECUTE_ACTION',
    'EXECUTE_WAIT_SEC',
    'PLANNER_CLASSIC',
    'PLANNER_CARTESIAN',
    'PLANNER_JOINT_SPACE',
]

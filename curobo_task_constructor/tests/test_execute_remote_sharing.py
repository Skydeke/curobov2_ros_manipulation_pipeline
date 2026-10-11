"""Remote execute reuses one shared action endpoint across tasks.

``Task.execute`` runs once per motion, so a private action node per execute
is a node per motion — and node create/destroy cycles leak (~350KB each at
the rcl layer, measured with flat entity counts) even with
``destroy_node()``. The endpoint (node + executor + client) is therefore
shared per action rendezvous. This drives the full ``_execute_remote`` path
(plan against the mock, serialize, send, spin-drive, result) against a stub
action server and pins both the result and the single endpoint.
"""

import threading
import time

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip(
    "curobo_task_constructor_interfaces.action",
    reason="curobo_task_constructor_interfaces actions are not built",
)

from rclpy.action import ActionServer  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402

from curobo_task_constructor.mtc import core  # noqa: E402
from curobo_task_constructor.mtc import stages  # noqa: E402
from curobo_task_constructor_interfaces.action import (  # noqa: E402
    ExecuteTaskSolution,
)
from tests.mock_curobo import MockCuroboServer  # noqa: E402

ACTION = "/curobo_task_constructor/execute_task_solution"


def _execute_one(robot, label, joint_1):
    """Plan against the mock and drive the solution through the action."""
    task = core.Task(robot, name="exec", task_id=label)
    task.add(stages.CurrentState("current state"))
    move = stages.MoveTo("move", core.JointInterpolationPlanner())
    move.setGoal({"joint_1": joint_1, "finger_joint": 0.0})
    task.add(move)
    assert task.plan(), f"task {label} must plan against the mock"
    code = task.execute(task.best())
    task.enableIntrospection(False)
    return code


def test_remote_execute_reuses_one_endpoint():
    mine = not rclpy.ok()
    if mine:
        rclpy.init()
    succeeded = {"n": 0}
    try:
        srv_node = rclpy.create_node("stub_execute_server")

        def run_goal(goal_handle):
            assert len(
                goal_handle.request.solution.sub_trajectory or []) >= 1
            succeeded["n"] += 1
            goal_handle.succeed()
            result = ExecuteTaskSolution.Result()
            result.error_code.val = 1
            return result

        server = ActionServer(
            srv_node, ExecuteTaskSolution, ACTION, run_goal)
        executor = MultiThreadedExecutor()
        executor.add_node(srv_node)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()
        # Let discovery settle so wait_for_server (0.5 s budget) is not a
        # race on a loaded machine.
        time.sleep(1.0)

        robot = MockCuroboServer()
        codes = [_execute_one(robot, f"exec_{i}", 0.05 * i) for i in range(3)]
        assert [int(c.val) for c in codes] == [1, 1, 1]
        assert succeeded["n"] == 3
        assert len(core._EXEC_ENDPOINTS) == 1, (
            f"expected one shared execute endpoint, "
            f"found {len(core._EXEC_ENDPOINTS)}"
        )
    finally:
        core.shutdown_shared_introspection()
        try:
            executor.shutdown()
        except Exception:
            pass
        try:
            srv_node.destroy_node()
        except Exception:
            pass
        if mine and rclpy.ok():
            rclpy.shutdown()

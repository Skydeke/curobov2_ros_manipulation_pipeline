"""Introspection must not accumulate DDS entities across tasks.

The pipeline builds a fresh ``Task`` for every motion of every pick cycle.
The introspection node is process-shared, so anything created per task on it
and never destroyed is a per-motion leak in long-running nodes (the grasp
orchestrator in continuous mode): 3 TRANSIENT_LOCAL publishers per task made
process RSS ratchet once per motion. The publishers are therefore shared —
one handle per topic for the whole process — while the
``get_solution_<task_id>`` service stays per task with deterministic
takeover. This pins both halves: publisher count stays flat across tasks,
and no service handle survives its task's release.
"""

import gc
import time

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip(
    "curobo_task_constructor_interfaces.msg",
    reason="curobo_task_constructor_interfaces (with Solution) is not built",
)

from curobo_task_constructor.mtc import core  # noqa: E402
from curobo_task_constructor.mtc import stages  # noqa: E402
from tests.mock_curobo import MockCuroboServer  # noqa: E402


def _plan_publish_release(robot, label, joint_1):
    """One orchestrator-like motion: build, plan, publish, tear down.

    Its own frame, so no caller local keeps a stage (whose progress hook
    closes over the task) alive past the return.
    """
    task = core.Task(robot, name="churn", task_id=label)
    task.add(stages.CurrentState("current state"))
    move = stages.MoveTo("move", core.JointInterpolationPlanner())
    move.setGoal({"joint_1": joint_1, "finger_joint": 0.0})
    task.add(move)
    assert task.plan(), f"task {label} must plan against the mock"
    assert task.best() is not None
    task.publish(task.best())
    task.enableIntrospection(False)


def _introspection_topics(node):
    pubs = getattr(node, "_publishers", None)
    if pubs is None:  # pragma: no cover - rclpy internals
        pytest.skip("rclpy Node exposes no publisher list here")
    return sorted(p.topic_name for p in pubs)


def _plan_publish_release_dark(robot, label, joint_1):
    """One dark motion: disable, plan, serialize (the execute path).

    Its own frame, so no caller local keeps anything alive past return.
    """
    task = core.Task(robot, name="churn", task_id=label)
    task.enableIntrospection(False)
    task.add(stages.CurrentState("current state"))
    move = stages.MoveTo("move", core.JointInterpolationPlanner())
    move.setGoal({"joint_1": joint_1, "finger_joint": 0.0})
    task.add(move)
    assert task.plan(), f"task {label} must plan while dark"
    assert task.best() is not None
    d = task._executor.solution_to_dict(task.best())
    assert d["sub_trajectories"], "dark serialize must still work"


def test_dark_task_creates_no_endpoints():
    """A task with introspection disabled stays dark through plan+serialize.

    MTC gates init on the flag: no hooks, no description, no statistics, no
    get_solution service — and solution ids come from the executor registry,
    so even the execute-path serialization creates nothing. This is what
    makes the orchestrator's publish_solutions:=false bisection valid.
    """
    mine = not rclpy.ok()
    if mine:
        rclpy.init()
    try:
        robot = MockCuroboServer()
        for i in range(4):
            _plan_publish_release_dark(robot, f"dark_{i}", 0.05 * i)
        gc.collect()
        assert core._INTROSPECTION_NODE is None, (
            "dark tasks must not create the shared node")
        assert dict(core._SERVICE_HANDLES) == {}
        assert dict(core._SHARED_PUBLISHERS) == {}
        assert [o for o in gc.get_objects()
                if type(o) is core.Introspection] == []
    finally:
        core.shutdown_shared_introspection()
        if mine and rclpy.ok():
            rclpy.shutdown()


def test_task_churn_leaves_no_publishers_or_services_behind():
    mine = not rclpy.ok()
    if mine:
        rclpy.init()
    try:
        robot = MockCuroboServer()
        labels = [f"churn_{i}" for i in range(8)]
        for i, label in enumerate(labels):
            _plan_publish_release(robot, label, 0.05 * i)
        # Let the shared node's spin thread cycle past the last service
        # teardown (its waitable list briefly retains the destroyed
        # service's callback), then collect repeatedly: anything still
        # alive after that is genuinely retained, not in flight.
        time.sleep(0.5)
        gc.collect()
        gc.collect()
        gc.collect()

        node = core._INTROSPECTION_NODE
        assert node is not None
        topics = _introspection_topics(node)
        for topic in ("/curobo_task_constructor/task_description",
                      "/curobo_task_constructor/task_statistics",
                      "/curobo_task_constructor/solution"):
            assert topics.count(topic) == 1, (
                f"{topic} has {topics.count(topic)} publishers after "
                f"{len(labels)} tasks; the handle must be shared, not per task"
            )
        assert dict(core._SERVICE_HANDLES) == {}, (
            "released tasks must not keep services: "
            f"{sorted(core._SERVICE_HANDLES)}"
        )
        assert [o for o in gc.get_objects() if type(o) is core.Task] == []

        from curobo_task_constructor_interfaces.action import (
            ExecuteTaskSolution,
        )
        ep1 = core._shared_execute_endpoint(
            "/curobo_task_constructor/execute_task_solution",
            ExecuteTaskSolution)
        ep2 = core._shared_execute_endpoint(
            "/curobo_task_constructor/execute_task_solution",
            ExecuteTaskSolution)
        assert ep1[0] is ep2[0] and ep1[2] is ep2[2], (
            "the execute endpoint (node + action client) must be shared, "
            "not rebuilt per motion: a private node per execute leaks at "
            "the rcl layer even with destroy_node()"
        )
        assert len(core._EXEC_ENDPOINTS) == 1
    finally:
        core.shutdown_shared_introspection()
        if mine and rclpy.ok():
            rclpy.shutdown()

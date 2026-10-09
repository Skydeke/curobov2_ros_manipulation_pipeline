"""Task.publish publishes ONE Solution message - MTC's exact contract.

MTC binds ``Task.publish`` to *only*
``self.introspection().publishSolution(*solution)``
(``moveit_task_constructor/core/python/bindings/src/core.cpp:481``), documented
as "Publish the given solution to the ROS topic ``solution``". Nothing else:
rendering is the consumer's job (MTC's RViz Display subscribes to ``solution``;
our panel fetches via GetSolution and republishes the trajectory for the
CuroboTrajectoryDisplay).

These tests pin that contract so a second publish on the trajectory topic
cannot creep back in: the trajectory publisher is counted and must stay 0.
"""

from curobo_task_constructor import viz
from curobo_task_constructor.mtc import core, stages
from tests.mock_curobo import MockCuroboServer


def test_publish_without_node_only_records():
    """Without a node (MTC: without ``rclcpp::ok()``) publish only records."""
    robot = MockCuroboServer()
    task = core.Task(robot)
    task.add(stages.CurrentState("current state"))
    assert task.plan()
    task.publish(task.solutions[0])
    assert task.last_published is task.solutions[0]


def test_publish_calls_publishSolution_and_nothing_else():
    """``Task.publish`` == ``introspection().publishSolution(solution)``.

    Spy on the introspection object rather than faking the generated message
    types: what is being pinned is *which* publish call happens, and the wire
    encoding is covered by test_msg_convert / test_solution_serialization.
    """
    published_topics = []
    trajectory_publishes = []

    class _Node:
        def create_publisher(self, cls, topic, qos):
            published_topics.append(topic)

            class _Pub:
                def publish(self, msg):
                    if topic == viz.SOLUTION_TRAJECTORY_TOPIC:
                        trajectory_publishes.append(topic)

            return _Pub()

        def create_service(self, srv_cls, name, cb):
            return object()

    robot = MockCuroboServer(named={"ready": {"joint_1": 0.1, "joint_2": 0.5}})
    task = core.Task(robot, task_id="pick")
    task.add(stages.CurrentState("current state"))
    ready = stages.MoveTo("moveTo ready", core.JointInterpolationPlanner())
    ready.setGoal("ready")
    task.add(ready)
    assert task.plan()

    introspect = task.introspection()
    real_publish_solution = introspect.publishSolution
    calls = {"solution": 0}

    def spy_publish_solution(sol):
        calls["solution"] += 1
        return real_publish_solution(sol)

    introspect.publishSolution = spy_publish_solution

    task.publish(task.solutions[0])

    assert task.last_published is task.solutions[0]
    assert calls["solution"] == 1, \
        "publish() must call publishSolution exactly once"
    assert not trajectory_publishes, (
        "Task.publish must not publish a trajectory topic - that is the "
        "consumer's (rviz display/panel) job, per MTC's API contract")


def test_introspection_owns_only_mtc_publish_calls():
    """The Introspection surface is MTC's - no trajectory publisher in it."""
    robot = MockCuroboServer()
    task = core.Task(robot)
    task.add(stages.CurrentState("current state"))
    assert task.plan()
    introspect = task.introspection()
    for name in ("publishTaskDescription", "publishTaskState",
                 "publishSolution", "publishAllSolutions", "reset",
                 "registerSolution", "solutionId", "solutionFromId",
                 "getSolution"):
        assert callable(getattr(introspect, name)), name
    # MTC's Introspection has no such method: rendering is the consumer's
    # responsibility, and there is no "setup" entry point either - MTC's
    # IntrospectionPrivate constructor does that work at construction time.
    assert not hasattr(introspect, "displaySolution"), (
        "MTC's Introspection has no displaySolution: rendering is the "
        "consumer's responsibility")
    assert not hasattr(introspect, "setup"), (
        "MTC's Introspection has no setup(): the publishers are created by "
        "its constructor, not by a call the user makes")

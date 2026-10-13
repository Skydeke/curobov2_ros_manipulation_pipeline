"""Identity-goal fast path: joint requests already at the goal succeed
without a server round trip (ROS-free)."""

from curobo_task_constructor.core.robot import GoalsetSpec, PlanRequest
from curobo_task_constructor.core.state import JointStateStub
from curobo_task_constructor.robot.curobo import CuroboServerInterface


def _req(joints, start):
    return PlanRequest(
        goalsets=[GoalsetSpec(target_joint_positions=list(joints))],
        start_pose=JointStateStub(names=["a", "b"], positions=list(start)))


def test_identity_joint_goal_succeeds_without_server():
    res = CuroboServerInterface._trivial_goal(_req([1.0, 2.0], [1.0, 2.0]))
    assert res is not None and res.success
    assert len(res.trajectory) == 1
    assert res.cost == 0.0


def test_near_identity_within_tolerance_succeeds():
    res = CuroboServerInterface._trivial_goal(
        _req([1.0, 2.0], [1.0 + 5e-7, 2.0 - 5e-7]))
    assert res is not None and res.success


def test_encoder_noise_level_still_trivial():
    # Live joint readings sit ~1e-4..1e-3 off a constant goal (e.g. the
    # finger at 0.001 vs requested 0.000): no-op legs must take the fast
    # path instead of burning a full trajopt solve on a milliradian.
    res = CuroboServerInterface._trivial_goal(
        _req([3.14, 0.6, 0.0, -1.5, 0.0, -1.57, 0.0, 0.0],
             [3.14, 0.6, 1e-4, -1.5, 0.0, -1.57, 0.0, 0.001]))
    assert res is not None and res.success
    assert len(res.trajectory) == 1


def test_above_tolerance_falls_through():
    assert CuroboServerInterface._trivial_goal(
        _req([1.0, 2.0], [1.0 + 5e-3, 2.0])) is None


def test_real_motion_falls_through_to_server():
    assert CuroboServerInterface._trivial_goal(
        _req([1.0, 2.5], [1.0, 2.0])) is None


def test_pose_goals_and_empty_requests_fall_through():
    start = JointStateStub(names=["a", "b"], positions=[1.0, 2.0])
    assert CuroboServerInterface._trivial_goal(
        PlanRequest(goalsets=[GoalsetSpec(poses=[{"x": 1.0}])],
                    start_pose=start)) is None
    assert CuroboServerInterface._trivial_goal(
        PlanRequest(goalsets=[], start_pose=start)) is None
    multi = PlanRequest(
        goalsets=[GoalsetSpec(target_joint_positions=[1.0, 2.0]),
                  GoalsetSpec(target_joint_positions=[1.0, 2.0])],
        start_pose=start)
    assert CuroboServerInterface._trivial_goal(multi) is None

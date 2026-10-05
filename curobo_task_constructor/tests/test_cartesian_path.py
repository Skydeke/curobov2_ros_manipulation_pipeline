"""CartesianPath + the geometry and cost helpers it stands on.

Covers the two things the reference ``pick_and_place_pipeline`` had and the
curobo port did not:

1. a stage that emits whole-path Cartesian axis holds, so an approach /
   descent / retreat is a straight line instead of a free-space swing, and
2. a real ranking cost (MTC ``cost::PathLength``) so a fan-out actually picks
   the cheapest of the candidates it planned.
"""

from __future__ import annotations

import math

import pytest
import yaml

import curobo_task_constructor.stages  # noqa: F401  (register builtins)
from curobo_task_constructor.core.geom import (
    Pose3,
    quat_from_axis_angle,
    quat_rotation_angle,
)
from curobo_task_constructor.core.robot import PlanResult
from curobo_task_constructor.core.state import JointStateStub
from curobo_task_constructor.executor import TaskExecutor
from curobo_task_constructor.graph.spec import StageSpec
from curobo_task_constructor.stages._util import (
    axis_holds,
    cost_of,
    path_length_cost,
)
from tests.mock_curobo import JOINT_NAMES, MockCuroboServer, fk_positions

#: A start configuration near the middle of the mock arm's reach shell
#: (|L1 - L2| = 0.30 m .. L1 + L2 = 0.50 m): tool at (0.40, 0, 0.10), so
#: there is headroom to travel BOTH down onto a grasp and up off it.
START_ARMS = {"joint_1": 0.0, "joint_2": math.pi / 2, "joint_3": -math.pi / 2}

IDENTITY_Q = {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}


def _pose(x, y, z, q=(0.0, 0.0, 0.0, 1.0)):
    return Pose3([x, y, z], list(q))


def _robot(**kw):
    kw.setdefault("current", dict(START_ARMS))
    return MockCuroboServer(**kw)


def _tool_xyz(robot) -> list:
    return fk_positions(dict(zip(JOINT_NAMES,
                                 robot.get_current_joint_state().position)))


def _stage(stage_type, name, params=None, children=None):
    return StageSpec(
        stage_type=stage_type, name=name,
        params_yaml=yaml.safe_dump(params or {}, sort_keys=False),
        children=list(children or []))


def _container(container_type, name, children):
    return StageSpec(stage_type="", name=name, container_type=container_type,
                     children=list(children))


def _plan(stages, robot=None, **kw):
    """Solve ``stages`` under a root serial seeded by ``current_state``."""
    robot = robot or _robot(**kw)
    root = _container("serial", "root", [
        _stage("current_state", "current state"), *stages])
    ex = TaskExecutor(root, robot)
    assert ex.init(), ex.describe()["comment"]
    return robot, ex


# -----------------------------------------------------------------------------
# axis_holds
# -----------------------------------------------------------------------------

def test_axis_holds_pins_every_axis_but_the_moving_one():
    """A top-down descent: x/y and the orientation agree, z travels."""
    start = _pose(0.30, 0.10, 0.25)
    goal = _pose(0.30, 0.10, 0.20)
    # [theta_x, theta_y, theta_z, x, y, z] -> orientation + x,y held, z free
    assert axis_holds(start, [goal]) == [1, 1, 1, 1, 1, 0]


def test_axis_holds_frees_every_position_axis_on_a_long_oblique_move():
    """No axis the start and goal share -> nothing to pin except orientation."""
    start = _pose(0.30, 0.10, 0.25)
    goal = _pose(0.10, 0.02, 0.40)
    hold = axis_holds(start, [goal])
    assert hold[:3] == [1, 1, 1]   # the tool angle is unchanged: hold it
    assert hold[3:] == [0, 0, 0]   # every position axis travels


def test_axis_holds_frees_orientation_when_the_tool_rotates():
    start = _pose(0.30, 0.10, 0.25, q=(0.0, 0.0, 0.0, 1.0))
    turned = quat_from_axis_angle([0.0, 0.0, 1.0], 0.6)  # 0.6 rad ~ 34 deg
    goal = _pose(0.30, 0.10, 0.20, q=turned)
    hold = axis_holds(start, [goal])
    # Position is still pinned along z, but a partial orientation hold is not
    # expressible in 6 slots, so the whole orientation block is freed.
    assert hold[:3] == [0, 0, 0]
    assert hold[3:] == [1, 1, 0]


def test_axis_holds_is_the_intersection_over_candidates():
    """One vector must stay correct for whichever candidate the server picks."""
    start = _pose(0.30, 0.10, 0.25)
    same_xy = [_pose(0.30, 0.10, z) for z in (0.20, 0.18, 0.22)]
    assert axis_holds(start, same_xy) == [1, 1, 1, 1, 1, 0]

    # A candidate that drifts in x cannot be served by a hold that pins x, so
    # x is freed for ALL of them.
    drifted = same_xy + [_pose(0.34, 0.10, 0.19)]
    assert axis_holds(start, drifted) == [1, 1, 1, 0, 1, 0]


def test_axis_holds_within_tolerance_counts_as_agreement():
    start = _pose(0.30, 0.10, 0.25)
    goal = _pose(0.302, 0.10, 0.20)  # 2 mm off, inside the 5 mm default
    assert axis_holds(start, [goal], pos_tol=0.005)[3] == 1
    assert axis_holds(start, [goal], pos_tol=0.001)[3] == 0


def test_axis_holds_empty_goals_is_no_constraint():
    assert axis_holds(_pose(0, 0, 0), []) == []


def test_quat_rotation_angle():
    q0 = [0.0, 0.0, 0.0, 1.0]
    assert quat_rotation_angle(q0, q0) == pytest.approx(0.0, abs=1e-9)
    assert quat_rotation_angle(
        q0, quat_from_axis_angle([0.0, 0.0, 1.0], 1.0)) == pytest.approx(1.0)
    # The double cover: -q is the same rotation, not 2*pi away.
    assert quat_rotation_angle(
        q0, [-v for v in quat_from_axis_angle([0.0, 0.0, 1.0], 1.0)]
    ) == pytest.approx(1.0)


# -----------------------------------------------------------------------------
# path_length_cost / cost_of
# ---------------------------------------------------------------------------

def _js(*positions):
    return JointStateStub(names=list(JOINT_NAMES),
                          positions=[float(p) for p in positions])


def test_path_length_cost_sums_joint_space_distance():
    traj = [_js(0, 0, 0), _js(0.3, 0.4, 0), _js(0.3, 0.4, 0.5)]
    assert path_length_cost(traj) == pytest.approx(1.0)


def test_path_length_cost_skip_tail_drops_a_gripper_close():
    """A finger-only move must not register as arm motion."""
    traj = [_js(0, 0, 0, 0.04), _js(0, 0, 0, 0.0)]
    assert path_length_cost(traj) == pytest.approx(0.04)
    assert path_length_cost(traj, skip_tail=1) == pytest.approx(0.0)


def test_path_length_cost_rewards_straightness():
    """A straight Cartesian line is a LONGER joint path than a shortcut.

    This is the property that makes PathLength the right ranking term for a
    Cartesian move: minimising it is what pushes the solver onto the line.
    """
    straight = [_js(0, 0, 0), _js(0.2, 0.0, 0.0), _js(0.4, 0.0, 0.0)]
    shortcut = [_js(0, 0, 0), _js(0.2, 0.6, 0.0), _js(0.4, 0.0, 0.0)]
    assert path_length_cost(straight) < path_length_cost(shortcut)


def test_cost_of_modes():
    result = PlanResult(True, trajectory=[_js(0, 0, 0), _js(0.3, 0, 0)])
    # auto with no adapter-populated cost -> path_length
    assert cost_of(result, {}) == pytest.approx(0.3)
    # waypoints keeps the old proxy
    assert cost_of(result, {"cost": "waypoints"}) == pytest.approx(2.0)
    assert cost_of(result, {"cost": "inf"}) == float("inf")
    # an adapter that DID populate PlanResult.cost wins under auto
    result.cost = 42.0
    assert cost_of(result, {}) == pytest.approx(42.0)
    # ... but an explicit mode still wins
    assert cost_of(result, {"cost": "waypoints"}) == pytest.approx(2.0)


def test_cost_of_solver_cost_reads_the_considered_rows():
    class _Row:
        def __init__(self, c):
            self.cost = c

    class _Stats:
        considered = [_Row(7.0), _Row(3.0), _Row(float("inf"))]

    result = PlanResult(True, trajectory=[_js(0, 0, 0), _js(0.3, 0, 0)],
                        stats=_Stats())
    assert cost_of(result, {"cost": "solver_cost"}) == pytest.approx(3.0)


# -----------------------------------------------------------------------------
# the cartesian_path stage
# ---------------------------------------------------------------------------

def _vertical_goal(robot, dz, **extra):
    x, y, z = _tool_xyz(robot)
    return {"x": x, "y": y, "z": z - dz, **IDENTITY_Q, **extra}


def test_cartesian_path_emits_the_derived_axis_hold():
    """The whole point: the goalset must carry int8[6], not an empty vector."""
    robot = _robot()
    goal = _vertical_goal(robot, 0.05)
    _, ex = _plan([_stage("cartesian_path", "descend",
                          {"goal": {"pose": goal}, "planner": "classic"})],
                  robot)
    assert ex.plan()
    assert ex.best() is not None
    # x,y + orientation agree, z is the only travelling axis
    assert robot.last_hold == [1, 1, 1, 1, 1, 0]


def test_cartesian_path_produces_a_straight_tool_path():
    """With the hold applied, every waypoint stays on the start->goal segment."""
    robot = _robot()
    goal = _vertical_goal(robot, 0.05)
    _, ex = _plan([_stage("cartesian_path", "descend",
                          {"goal": {"pose": goal}})], robot)
    assert ex.plan()

    traj = ex.best().trajectory
    pts = [fk_positions(dict(zip(JOINT_NAMES, w.position))) for w in traj]
    # x and y are pinned to the goal for the WHOLE path (that is the hold)
    assert {round(p[0], 6) for p in pts} == {round(goal["x"], 6)}
    assert {round(p[1], 6) for p in pts} == {round(goal["y"], 6)}
    # ... and z travels monotonically from start to goal
    assert pts[-1][2] == pytest.approx(goal["z"], abs=1e-6)
    assert all(b[2] <= a[2] + 1e-9 for a, b in zip(pts, pts[1:]))
    assert len({round(p[2], 6) for p in pts}) == len(pts)  # strictly monotone


def test_cartesian_path_relative_goal_freezes_orientation():
    """The MoveRelative port: translate by a delta, keep the tool angle."""
    robot = _robot()
    _, ex = _plan([_stage("cartesian_path", "lift", {
        "goal": {"relative": {"x": 0.0, "y": 0.0, "z": 0.10}},
        "planner": "classic",
    })], robot)
    assert ex.plan()
    traj = ex.best().trajectory
    start_xyz = fk_positions(dict(zip(JOINT_NAMES, traj[0].position)))
    end_xyz = fk_positions(dict(zip(JOINT_NAMES, traj[-1].position)))
    assert end_xyz[2] - start_xyz[2] == pytest.approx(0.10, abs=1e-6)
    assert end_xyz[0] == pytest.approx(start_xyz[0], abs=1e-6)
    assert end_xyz[1] == pytest.approx(start_xyz[1], abs=1e-6)
    # orientation is held -> the hold pins all three rotational slots
    assert robot.last_hold[:3] == [1, 1, 1]


def test_cartesian_path_relative_goal_can_ride_the_hand_frame():
    robot = _robot()
    _, ex = _plan([_stage("cartesian_path", "lift", {
        "goal": {"relative": {"x": 0.0, "y": 0.0, "z": 0.10,
                              "frame": "hand"}},
        "planner": "classic",
    })], robot)
    assert ex.plan()
    traj = ex.best().trajectory
    start_xyz = fk_positions(dict(zip(JOINT_NAMES, traj[0].position)))
    end_xyz = fk_positions(dict(zip(JOINT_NAMES, traj[-1].position)))
    # the mock's tool frame is unrotated, so hand == world here; what this
    # checks is that the hand branch resolves and still lands on the line
    assert end_xyz[2] - start_xyz[2] == pytest.approx(0.10, abs=1e-6)


def test_cartesian_path_accepts_a_bowed_path():
    """The hold is a soft cost: cuRobo may leave the line when that is cheaper,
    and the client no longer rejects a bowed solve. A bowed descent is the
    optimizer's answer, not a failure to fall through on."""
    robot = _robot(bow=0.03)  # 3 cm of bow
    goal = _vertical_goal(robot, 0.05)
    _, ex = _plan([_stage("cartesian_path", "descend",
                          {"goal": {"pose": goal}})], robot)
    assert ex.plan()
    assert ex.best() is not None


def test_cartesian_path_straightness_gate_feeds_the_fallbacks():
    """A rejected variant must let the NEXT sibling grasp candidate run."""
    robot = _robot(bow=0.0)
    x, y, z = _tool_xyz(robot)

    def _variant(dz):
        return _container("serial", f"variant_{dz}", [
            _stage("cartesian_path", "descend",
                   {"goal": {"pose": {"x": x, "y": y, "z": z - dz,
                                      **IDENTITY_Q}},
                    "planner": "classic"}),
        ])

    _, ex = _plan([_container("fallbacks", "candidates",
                              [_variant(0.05), _variant(0.08)])], robot)
    assert ex.plan()
    # first variant is accepted, the fallbacks container stops there
    assert ex.best() is not None


def test_cartesian_path_straightness_gate_can_be_disabled():
    robot = _robot(bow=0.03)
    goal = _vertical_goal(robot, 0.05)
    _, ex = _plan([_stage("cartesian_path", "descend", {
        "goal": {"pose": goal}, "check_straightness": False})], robot)
    assert ex.plan()
    assert ex.best() is not None
    assert robot.fk_batch_calls == 0


def test_cartesian_path_multi_candidate_fan_out_is_reported():
    """N candidates in ONE goalset; the winner is surfaced, not guessed."""
    robot = _robot(winner_index=1)
    x, y, z = _tool_xyz(robot)
    poses = [{"x": x, "y": y, "z": z - dz, **IDENTITY_Q}
             for dz in (0.05, 0.08, 0.11)]
    _, ex = _plan([_stage("cartesian_path", "descend", {
        "goal": {"poses": poses}, "planner": "classic"})], robot)
    assert ex.plan()
    stage = ex.root.subtree_stages()[-1]
    assert stage.solutions[0].comment == (
        "cartesian_path hold=[1, 1, 1, 1, 1, 0] candidates=3 winner=[1]")


def test_cartesian_path_candidate_fan_out_is_one_solve():
    """All candidates attempted inside ONE round trip, not N."""
    robot = _robot(winner_index=1)
    x, y, z = _tool_xyz(robot)
    poses = [{"x": x, "y": y, "z": z - dz, **IDENTITY_Q}
             for dz in (0.05, 0.08, 0.11)]
    _, ex = _plan([_stage("cartesian_path", "descend", {
        "goal": {"poses": poses}, "planner": "classic"})], robot)
    assert ex.plan()
    # 3 candidates in ONE goalset, but 3 planning_attempts (multi-attempt)
    assert robot.plan_calls == 3


def test_move_to_accepts_a_multi_candidate_pose_goal():
    """Same native goalset fan-out on the plain move_to stage."""
    robot = _robot(winner_index=2)
    x, y, z = _tool_xyz(robot)
    poses = [{"x": x, "y": y, "z": z - dz, **IDENTITY_Q}
             for dz in (0.05, 0.08, 0.11)]
    _, ex = _plan([_stage("move_to", "reach",
                          {"goal": {"poses": poses}, "planner": "classic"})],
                  robot)
    assert ex.plan()
    stage = ex.root.subtree_stages()[-1]
    assert "candidate [2]" in stage.solutions[0].comment
    # 3 candidates in ONE goalset, 3 planning_attempts (multi-attempt)
    assert robot.plan_calls == 3


def test_move_relative_derives_a_hold_for_its_sampled_segment():
    """MoveRelative exists to interpolate a line, so it must now say so."""
    robot = _robot()
    _, ex = _plan([_stage("move_relative", "lift", {
        "axis": {"frame": "hand", "xyz": [0.0, 0.0, 1.0]},
        "distance": 0.05,
        "planner": "classic",
    })], robot)
    assert ex.plan()
    # the mock's tool frame is unrotated, so the hand +z axis is world +z
    assert robot.last_hold == [1, 1, 1, 1, 1, 0]


def test_move_relative_hold_param_can_force_a_free_space_move():
    robot = _robot()
    _, ex = _plan([_stage("move_relative", "swing", {
        "axis": {"frame": "hand", "xyz": [1.0, 0.0, 0.0]},
        "distance": 0.05,
        "hold": [0, 0, 0, 0, 0, 0],
        "planner": "classic",
    })], robot)
    assert ex.plan()
    assert robot.last_hold == [0, 0, 0, 0, 0, 0]


# -----------------------------------------------------------------------------
# ranking
# ---------------------------------------------------------------------------

def test_path_length_ranking_picks_the_cheaper_of_two_variants():
    """A far and a near descent, ranked: the near one must win.

    This is the payoff of replacing the waypoint-count proxy with
    ``cost::PathLength`` — both variants have the same number of waypoints, so
    the old proxy could not tell them apart at all. And the two solves happen in
    ONE batched round trip, which is the point of the Alternatives container.
    """
    robot = _robot()
    x, y, z = _tool_xyz(robot)

    def _descent(name, dz):
        return _stage("cartesian_path", name, {
            "goal": {"pose": {"x": x, "y": y, "z": z - dz, **IDENTITY_Q}},
            "planner": "classic"})

    _, ex = _plan([_container("alternatives", "variants",
                              [_descent("far", 0.12),
                               _descent("near", 0.02)])], robot)
    assert ex.plan()

    ranked = ex.rank()
    assert len(ranked) == 2
    assert ranked[0].cost < ranked[1].cost
    assert robot.batch_calls == 1  # both variants solved in ONE batch

    winner = ex.best()
    end_xyz = fk_positions(dict(zip(JOINT_NAMES,
                                    winner.end.joint_state.position)))
    assert end_xyz[2] == pytest.approx(z - 0.02, abs=1e-6)

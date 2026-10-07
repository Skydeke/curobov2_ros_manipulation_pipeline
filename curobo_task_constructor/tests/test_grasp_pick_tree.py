"""The REAL pick graph from ``kortex_curobo_grasping``, replayed on the mock.

``grasp_orchestrator.py`` is rclpy-only, so the graph it builds was moved into
``kortex_curobo_grasping/ctc.py`` — pure data, no ROS. That makes the
deliverable itself testable: this file imports the *production* builder (not a
re-declaration that could silently drift), hands it a candidate window, and
solves the resulting tree against ``MockCuroboServer``.

What is asserted is the behaviour the rewrite was for:

* the tree plans at all, and plans it in one round trip per leg however many
  candidates are offered;
* the descent and the retreat are straight lines, not joint-space swings;
* the close does not need an IK result for the arm;
* a solve that bows off the line is rejected and falls through to the next
  approach strategy.

The mock arm is 3-DOF with reach ``|L1 - L2| .. L1 + L2`` = 0.30 .. 0.50 m, so
the starts below sit mid-shell with room to travel up and down.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

import curobo_task_constructor.stages  # noqa: F401  (register builtins)
from curobo_task_constructor.core.state import ObjectSpec, SceneDiff
from curobo_task_constructor.executor import TaskExecutor
from tests.mock_curobo import JOINT_NAMES, MockCuroboServer, fk_positions

#: The orchestrator package lives in a sibling workspace package, not in this
#: repo. Add it to the path only if it is actually there, so the suite still
#: runs (minus this file) in a bare checkout of curobo_task_constructor.
for _root in (Path(__file__).resolve().parents[i] for i in range(1, 6)):
    _pkg = _root / "iki_kortex_curobo_grasping"
    if (_pkg / "kortex_curobo_grasping" / "ctc_manager.py").is_file():
        sys.path.insert(0, str(_pkg))
        break

ctc = pytest.importorskip(
    "kortex_curobo_grasping.ctc_manager",
    reason="kortex_curobo_grasping (the grasp orchestrator) is not present",
)

OBJECT = "object_0"
CLOSE = ctc.GRIPPER_CLOSE
LIFT = 0.15

#: The orchestrator's own default: how far back along the approach axis the
#: pre-grasp sits.
APPROACH_OFFSET = 0.12
STRATEGIES = ctc.STRATEGY_NAMES

#: Top-down grasp orientation: tool +z pointing DOWN at the object, so backing
#: off along the approach axis moves the hand UP. (The mock's FK reports an
#: identity tool orientation for every configuration, so the descent's hold
#: frees the rotational block — the real arm's start pose would already match
#: and the hold would pin all three rotation axes too.)
TOP_DOWN_Q = ctc._Quat(1.0, 0.0, 0.0, 0.0)

#: Mock-arm start: tool at (0.40, 0, 0.10), inside the reach shell with room
#: to travel both down onto a grasp and back up off it.
START_ARMS = {"joint_1": 0.0, "joint_2": math.pi / 2, "joint_3": -math.pi / 2}


# -----------------------------------------------------------------------------
# candidate window
# -----------------------------------------------------------------------------

def _grasp_poses(robot, depths=(0.06, 0.09, 0.12)):
    """N top-down grasp candidates at increasing depth, best score first."""
    x, y, z = _start_xyz(robot)
    out = []
    for d in depths:
        g = ctc.PoseLike(
            position=ctc._Vec(x, y, z - d),
            orientation=ctc._Quat(TOP_DOWN_Q.x, TOP_DOWN_Q.y,
                                        TOP_DOWN_Q.z, TOP_DOWN_Q.w))
        pre = ctc.offset_along_approach(g, APPROACH_OFFSET)
        out.append(ctc.GraspCandidate(pre_grasp=pre, grasp=g))
    return out


def _start_xyz(robot):
    return fk_positions(dict(zip(JOINT_NAMES,
                                 robot.get_current_joint_state().position)))


def _robot(**kw):
    kw.setdefault("current", dict(START_ARMS))
    return MockCuroboServer(**kw)


def _scene():
    return SceneDiff().with_object_added(
        ObjectSpec(name=OBJECT, shape="cuboid", dimensions=[0.05] * 3))


def _executor(robot=None, strategies=STRATEGIES, depths=(0.06, 0.09, 0.12),
              **kw):
    robot = robot or _robot(**kw)
    tree = ctc.pick_root(
        _grasp_poses(robot, depths),
        object_name=OBJECT,
        strategies=list(strategies),
        approach_offset=APPROACH_OFFSET,
        lift_offset=LIFT,
    )
    ex = TaskExecutor(tree, robot,
                      base_scene=_scene(), task_id="pick")
    assert ex.init(), ex.describe()["comment"]
    return robot, ex


def _chain(ex):
    return ex.flatten_leaves(ex.best())


# -----------------------------------------------------------------------------
# the graph the orchestrator builds
# -----------------------------------------------------------------------------

def test_pick_tree_solves():
    _, ex = _executor()
    assert ex.plan()
    assert ex.best() is not None


def test_pick_tree_shape():
    """current_state -> ready -> fallbacks(strategies) -> allow -> return.

    The pick task stops with the finger still CLOSED: the orchestrator has to
    be able to read whether the grasp actually took hold, and a task that
    reopened the gripper on its way out would erase that evidence. The reopen
    and the detach are ``release_root``, sent as a separate task afterwards.
    """
    root = ctc.pick_root(
        _grasp_poses(_robot()), object_name=OBJECT)
    assert root.container_type == "serial"
    top = [(c.name, c.stage_type, c.container_type)
           for c in root.children]
    assert top == [
        ("current_state", "current_state", ""),
        ("ready", "move_to", ""),
        ("strategies", "", "fallbacks"),
        ("allow", "modify_scene", ""),
        ("return", "move_to", ""),
    ]
    strat = root.children[2].children[0]
    assert strat.name == "strategy_0"
    # attach BEFORE close, as the reference does: the attach fits the object's
    # collision geometry to the gripper, and it also puts the one stage that
    # must never run on a failed grasp ahead of the stage that fails it.
    assert [c.name for c in strat.children] == [
        "pre_grasp_0", "allow_0", "descend_0", "attach_0", "close_0",
        "retreat_0"]


def test_pick_tree_fans_each_candidate_set_out_in_one_round_trip():
    """The whole point: N candidates are ONE goalset in ONE solve, not N solves.

    The old graph solved N joint goals from the orchestrator's own single-seed
    IK call and then ran N fallback children — N round trips, one IK solution
    each, planned one at a time.
    """
    robot, ex = _executor()
    assert ex.plan()
    # 3 candidates x 3 strategies, but only the first strategy is planned
    # (fallbacks stop at the first child that solves):
    #   pre_grasp, descend, close, retreat, return, open  = 6 stages
    # Each stage plans 3 times (planning_attempts=3, multi-attempt):
    assert robot.plan_calls == 18
    assert robot.batch_calls == 0  # no Alternatives: the fan-out is in-band


def test_pick_tree_sends_all_candidates_in_one_goalset():
    robot, ex = _executor()
    assert ex.plan()
    legs = {l.stage.name: l for l in _chain(ex)}
    assert len(legs["pre_grasp_0"].plan_request.goalsets[0].poses) == 3
    assert len(legs["descend_0"].plan_request.goalsets[0].poses) == 3
    # ... and the winner is reported, not guessed
    assert "winner=" in legs["descend_0"].stage.solutions[0].comment


def test_pick_tree_takes_the_cheapest_strategy_first():
    """Fallbacks priority = best strategy first; the rest are not even planned."""
    _, ex = _executor()
    assert ex.plan()
    names = [l.stage.name for l in _chain(ex)]
    assert names[:7] == ["current_state", "ready", "pre_grasp_0", "allow_0",
                         "descend_0", "attach_0", "close_0"]
    assert not any(n.endswith(("_1", "_2")) for n in names)


def test_pick_tree_refuses_a_stale_attach():
    """The pick must not be built on an object still held by a previous try."""
    robot = _robot()
    root = ctc.pick_root(_grasp_poses(robot), object_name=OBJECT)
    params = root.children[0]
    assert "require_not_attached" in params.params_yaml
    assert OBJECT in params.params_yaml


# -----------------------------------------------------------------------------
# the two legs that used to be joint-space swings
# -----------------------------------------------------------------------------

def test_descent_is_a_straight_line_with_a_derived_hold():
    _, ex = _executor()
    assert ex.plan()
    descend = next(l for l in _chain(ex) if l.stage.name == "descend_0")
    # x/y pinned, z free. The rotational block is freed because the mock's FK
    # reports an identity tool orientation while the goal asks for a top-down
    # one, and a PARTIAL orientation hold is not expressible in six slots.
    assert descend.plan_request.goalsets[0].trajectory_constraints == \
        [0, 0, 0, 1, 1, 0]

    pts = [fk_positions(dict(zip(JOINT_NAMES, w.position)))
           for w in descend.trajectory]
    assert {round(p[0], 6) for p in pts} == {round(pts[0][0], 6)}
    assert {round(p[1], 6) for p in pts} == {round(pts[0][1], 6)}
    assert all(b[2] < a[2] for a, b in zip(pts, pts[1:]))
    assert len({round(p[2], 6) for p in pts}) == len(pts)


def test_retreat_is_a_straight_line_up_from_the_pose_actually_reached():
    """Relative to the achieved grasp, not the requested one.

    A retreat anchored to the *requested* grasp pose would start in mid-air
    whenever the straight-line solve stops short of it.
    """
    _, ex = _executor()
    assert ex.plan()
    legs = {l.stage.name: l for l in _chain(ex)}
    retreat = legs["retreat_0"]
    # The retreat's relative goal keeps the (frozen) start orientation, which
    # here is the mock's identity — so the whole rotational block IS pinned.
    assert retreat.plan_request.goalsets[0].trajectory_constraints == \
        [1, 1, 1, 1, 1, 0]

    grasp_end = fk_positions(dict(zip(JOINT_NAMES,
                                       legs["descend_0"].trajectory[-1].position)))
    pts = [fk_positions(dict(zip(JOINT_NAMES, w.position)))
           for w in retreat.trajectory]
    assert pts[0] == pytest.approx(grasp_end, abs=1e-6)
    assert pts[-1][2] - pts[0][2] == pytest.approx(LIFT, abs=1e-6)
    assert {round(p[0], 6) for p in pts} == {round(grasp_end[0], 6)}
    assert {round(p[1], 6) for p in pts} == {round(grasp_end[1], 6)}
    assert all(b[2] > a[2] for a, b in zip(pts, pts[1:]))


def test_free_space_strategy_sends_no_hold_at_all():
    """The last-resort strategy keeps the old behaviour (and the old risk).

    Only reached once the two line-constrained strategies have been rejected
    for bowing off the line, which is what the mock's ``bow`` simulates.
    """
    _, ex = _executor(bow=0.04)
    assert ex.plan()
    names = [l.stage.name for l in _chain(ex)]
    assert "descend_2" in names and "descend_0" not in names
    descend = next(l for l in _chain(ex) if l.stage.name == "descend_2")
    # all-zero hold == "no axis is pinned" == the old free pose goal
    assert descend.plan_request.goalsets[0].trajectory_constraints == \
        [0, 0, 0, 0, 0, 0]
    # the gate is off for it, or it would reject its own bowed solve
    assert descend.stage.params["check_straightness"] is False


def test_a_bowed_descent_falls_through_to_the_next_strategy():
    """cuRobo's hold is soft, so the straightness gate is what actually
    enforces the line — and a rejection must reach the fallbacks."""
    # the only strategy is line-constrained, so the pick cannot be solved
    robot, ex = _executor(strategies=STRATEGIES[:1], bow=0.04)
    assert not ex.plan()

    # with free_space available it succeeds, on the unconstrained leg
    robot, ex = _executor(bow=0.04)
    assert ex.plan()
    assert "descend_2" in [l.stage.name for l in _chain(ex)]


def test_high_approach_stands_off_further():
    """The second strategy backs off one extra ``approach_offset`` before
    descending, so the line it must hold is longer and starts further clear of
    whatever sits beside the object."""
    _, base = _executor(strategies=("cartesian",))
    assert base.plan()
    _, hi = _executor(strategies=("high_approach",))
    assert hi.plan()

    def _standoff(ex):
        leg = next(l for l in _chain(ex) if l.stage.name == "pre_grasp_0")
        tool_z = fk_positions(
            dict(zip(JOINT_NAMES, leg.trajectory[-1].position)))[2]
        return tool_z - _start_xyz(_robot())[2]

    assert _standoff(hi) == pytest.approx(_standoff(base) + APPROACH_OFFSET,
                                          abs=1e-6)


# -----------------------------------------------------------------------------
# the close needs no IK result
# -----------------------------------------------------------------------------

def test_close_parks_the_arm_and_drives_only_the_finger():
    """A sparse joint goal: the arm stays exactly where the descent ended."""
    _, ex = _executor()
    assert ex.plan()
    legs = {l.stage.name: l for l in _chain(ex)}
    before = legs["descend_0"].trajectory[-1].position
    after = legs["close_0"].trajectory[-1].position
    for i, joint in enumerate(JOINT_NAMES):
        if joint != "finger_joint":
            assert after[i] == pytest.approx(before[i], abs=1e-9)
    assert after[JOINT_NAMES.index("finger_joint")] == pytest.approx(CLOSE)


def test_open_after_the_pick_reopens_the_finger():
    _, ex = _executor()
    assert ex.plan()
    end = dict(zip(JOINT_NAMES, ex.best().end.joint_state.position))
    assert end["finger_joint"] == pytest.approx(ctc.GRIPPER_OPEN)


# -----------------------------------------------------------------------------
# scene ops
# -----------------------------------------------------------------------------

def test_contact_is_allowed_before_the_descent_not_after():
    """The allow stage has to sit ABOVE the descent so the fingers may touch
    the object along the straight line, not only at its endpoint."""
    _, ex = _executor()
    assert ex.plan()
    names = [l.stage.name for l in _chain(ex)]
    assert names.index("allow_0") < names.index("descend_0")
    # ... and the descendent goalset inherits the allowance
    descend = next(l for l in _chain(ex) if l.stage.name == "descend_0")
    assert sorted(descend.end.scene.all_allowed_links()) == \
        sorted(ctc.GRIPPER_CONTACT_LINKS)


def test_attach_happens_after_the_descent_and_before_the_close():
    """The reference's order: descend, attach, close, retreat.

    The attach fits the object's collision geometry to the gripper, so doing
    it first means the close is planned against an object already accounted
    for - and it means the stage that must never run on a failed grasp comes
    before the stage that fails the grasp. The detach is no longer here: the
    pick task ends holding the finger closed so the caller can read whether it
    holds anything, and the release is a separate task.
    """
    robot, ex = _executor()
    assert ex.plan()
    names = [l.stage.name for l in _chain(ex)]
    assert names.index("descend_0") < names.index("attach_0")
    assert names.index("attach_0") < names.index("close_0")
    assert names.index("close_0") < names.index("retreat_0")
    assert ex.execute(ex.best())
    # The pick task attaches and stops there.
    assert [kind for kind, _ in robot.world_ops] == ["attach"]


def test_pick_chain_is_continuous():
    _, ex = _executor()
    assert ex.plan()
    chain = _chain(ex)
    for prev, nxt in zip(chain, chain[1:]):
        assert prev.end is nxt.start


# -----------------------------------------------------------------------------
# the scene-setup task (scene_root)
# -----------------------------------------------------------------------------

def _scene_specs():
    """Two perceived cuboids, the shape ``_scene_object_specs`` produces."""
    return [
        ctc.scene_object(
            "object_0", ctc.PoseLike(ctc._Vec(0.40, 0.0, 0.24),
                                           TOP_DOWN_Q),
            ctc._Vec(0.05, 0.05, 0.03)),
        ctc.scene_object(
            "plane_0", ctc.PoseLike(ctc._Vec(0.0, 0.0, 0.17),
                                          TOP_DOWN_Q),
            ctc._Vec(0.40, 0.40, 0.01)),
    ]


def test_scene_setup_task_clears_a_stale_attach_and_world_then_re_adds():
    """``scene_root`` must return the server to EXACTLY the perceived world —
    objects only, nothing attached — even when a previous cycle left an object
    attached and boxes behind. That is what stops a stale attach leaking into
    the next pick (the ``current_state`` 'already attached' refusal, or the
    re-added box left collision-inert by reapply_attached_disables)."""
    robot = _robot()
    # A leftover world: last cycle's box AND an object still in the hand.
    robot.add_object(ObjectSpec(name="old_box", shape="cuboid",
                                dimensions=[0.1] * 3))
    robot.add_object(ObjectSpec(name=OBJECT, shape="cuboid",
                                dimensions=[0.05] * 3))
    assert robot.attach_object(OBJECT)
    seed_ops = len(robot.world_ops)  # the two seed adds above

    specs = _scene_specs()
    root = ctc.scene_root(specs, remove_all=True)
    ex = TaskExecutor(root, robot, task_id="scene")
    ex.base_scene = ex.build_base_scene()  # the node's own construction
    assert ex.init(), ex.describe()["comment"]
    assert ex.plan(), "a mutation-only scene task must always solve"
    sol = ex.best()
    assert sol is not None
    # Mutation-only leaves drive nothing, so execute() yields [] (the task
    # node's own motion-leaves filter treats that as success); the record is
    # the ops it applied on the server.
    ex.execute(sol)

    # Chain order (the task's own ops only): clear the hand, clear the world,
    # then the adds. The detach_all must precede the adds, or the re-added
    # object stays disabled.
    kinds = [kind for kind, _ in robot.world_ops[seed_ops:]]
    assert kinds == ["detach_all", "remove_all", "add", "add"]

    # The world is exactly the new specs and nothing is in the hand.
    assert sorted(robot.world) == [s["name"] for s in specs]
    assert robot.get_attached_objects() == []


def test_scene_add_flat_pose_dict_parses_through_the_stage():
    """``scene_object`` rides as YAML, so the pose is a flat
    ``{x,y,z,qx,qy,qz,qw}`` dict — message types cannot ride YAML — and the
    ``add`` stage must rebuild a Pose-like from it, not pass the dict on."""
    robot = _robot()
    spec = ctc.scene_object(
        OBJECT, ctc.PoseLike(ctc._Vec(0.30, 0.0, 0.24),
                                   TOP_DOWN_Q),
        ctc._Vec(0.05, 0.05, 0.03))
    root = ctc.scene_root([spec], remove_all=False)
    ex = TaskExecutor(root, robot, task_id="scene")
    assert ex.init(), ex.describe()["comment"]
    assert ex.plan()
    ex.execute(ex.best())  # mutation-only: [] results == success (see above)
    assert sorted(robot.world) == [OBJECT]
    assert robot.world[OBJECT] == pytest.approx([0.30, 0.0, 0.24])

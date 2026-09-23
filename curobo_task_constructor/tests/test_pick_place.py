"""pick_place_task.cpp as a StageSpec tree — validation target #2.

Reproduces the full MTC demo graph structure (Sec. 1.1 of the investigation):

    current_state
    -> move_to open (gripper)
    -> connect (move to pick)
    -> pick serial [approach bwd, ComputeIK(GenerateGraspPose),
                    allow collisions, close, attach,
                    allow collisions (surface), lift, forbid]
    -> connect (move to place)
    -> place serial [lower bwd, ComputeIK(GeneratePlacePose),
                     open, forbid, detach, retreat]
    -> move_to home (manipulator) FORWARD

The graph exercises every interface type (generator / forward+backward
propagator / connector) and both backward approach edges — the trickiest part
of the framework — against the analytic MockCuroboServer.
"""

from __future__ import annotations

import pytest

import curobo_task_constructor.stages  # noqa: F401  (register builtins)
from curobo_task_constructor.core.state import ObjectSpec, SceneDiff
from curobo_task_constructor.executor import TaskExecutor
from curobo_task_constructor.graph.spec import StageSpec
from tests.mock_curobo import JOINT_NAMES, MockCuroboServer, make_pick_world

OBJECT = "object"
TOOL_LINK = "tool_link"
SURFACE_LINK = "table_link"


def _pick_place_spec() -> StageSpec:
    """The full pick&place StageSpec tree (MTC demo port)."""
    open_gripper = StageSpec(
        stage_type="move_to", name="open gripper",
        params_yaml="goal:\n  name: open\n")
    close_gripper = StageSpec(
        stage_type="move_to", name="close gripper",
        params_yaml="goal:\n  name: close\n")
    home = StageSpec(
        stage_type="move_to", name="move home",
        params_yaml="goal:\n  name: home\n")

    allow_tool = StageSpec(
        stage_type="modify_scene", name="allow object-tool",
        params_yaml=f"allow_collisions:\n  object: {OBJECT}\n"
                    f"  links: [{TOOL_LINK}]\n  enabled: true\n")
    allow_surface = StageSpec(
        stage_type="modify_scene", name="allow object-surface",
        params_yaml=f"allow_collisions:\n  object: {OBJECT}\n"
                    f"  links: [{SURFACE_LINK}]\n  enabled: true\n")
    forbid = StageSpec(
        stage_type="modify_scene", name="forbid object-surface",
        params_yaml=f"allow_collisions:\n  object: {OBJECT}\n"
                    f"  links: [{TOOL_LINK}, {SURFACE_LINK}]\n"
                    f"  enabled: false\n")
    attach = StageSpec(
        stage_type="modify_scene", name="attach object",
        params_yaml=f"attach: {OBJECT}\n")
    detach = StageSpec(
        stage_type="modify_scene", name="detach object",
        params_yaml=f"detach: {OBJECT}\n")

    approach = StageSpec(
        stage_type="move_relative", name="approach",
        params_yaml="axis:\n  frame: hand\n  xyz: [0.0, 0.0, 1.0]\n"
                    f"link: {TOOL_LINK}\ndistance: 0.05\n")
    lift = StageSpec(
        stage_type="move_relative", name="lift",
        params_yaml="axis:\n  frame: hand\n  xyz: [0.0, 0.0, 1.0]\n"
                    f"link: {TOOL_LINK}\ndistance: 0.05\n")
    lower = StageSpec(
        stage_type="move_relative", name="lower",
        params_yaml="axis:\n  frame: hand\n  xyz: [0.0, 0.0, 1.0]\n"
                    f"link: {TOOL_LINK}\ndistance: 0.05\n")
    retreat = StageSpec(
        stage_type="move_relative", name="retreat",
        params_yaml="axis:\n  frame: hand\n  xyz: [0.0, 0.0, 1.0]\n"
                    f"link: {TOOL_LINK}\ndistance: 0.05\n")

    grasp_ik = StageSpec(
        stage_type="compute_ik", name="compute pick pose",
        children=[StageSpec(
            stage_type="generate_grasp_pose", name="sample grasp",
            params_yaml=f"object: {OBJECT}\nangle_delta: 0.5236\n"
                        "pre_grasp:\n  name: home\n")]
    )
    place_ik = StageSpec(
        stage_type="compute_ik", name="compute place pose",
        children=[StageSpec(
            stage_type="generate_grasp_pose", name="sample place",
            params_yaml=f"object: {OBJECT}\nmode: place\nangle_delta: 0.5236\n"
                        "pre_grasp:\n  name: home\n")]
    )

    pick = StageSpec(
        stage_type="", name="pick", container_type="serial",
        children=[approach, grasp_ik, allow_tool, close_gripper,
                  attach, allow_surface, lift, forbid])
    place = StageSpec(
        stage_type="", name="place", container_type="serial",
        children=[lower, place_ik, open_gripper, forbid, detach, retreat])

    return StageSpec(
        stage_type="", name="pick and place", container_type="serial",
        children=[
            StageSpec(stage_type="current_state", name="current state",
                      params_yaml=f"require_not_attached: [{OBJECT}]\n"),
            open_gripper,
            StageSpec(stage_type="connect", name="move to pick"),
            pick,
            StageSpec(stage_type="connect", name="move to place"),
            place,
            home,
        ])


def _named_configs() -> dict:
    zeros = {n: 0.0 for n in JOINT_NAMES}
    return {
        "home": {**zeros},
        "grasp_home": {**zeros},
        # Gripper-group configs: only the finger joints are named (as in the
        # real Kortex SRDF), so a MoveTo of the same name leaves the arm at
        # the current pose — the grasp/place config — instead of collapsing
        # it to the straight-arm singularity at full extension.
        "open": {"finger_joint": 0.0},
        "close": {"finger_joint": 0.7},
    }


def _base_scene() -> SceneDiff:
    return SceneDiff().with_object_added(
        ObjectSpec(name=OBJECT, shape="cuboid", dimensions=[0.05] * 3)
    ).with_object_added(
        ObjectSpec(name="table", shape="box", dimensions=[0.5, 0.5, 0.02])
    )


def _executor():
    robot = MockCuroboServer(named=_named_configs(), world=make_pick_world())
    return robot, TaskExecutor(_pick_place_spec(), robot,
                               base_scene=_base_scene(), task_id="pick_place")


def test_pick_place_init_valid():
    robot, ex = _executor()
    assert ex.init(), ex.describe()["comment"]


def test_pick_place_finds_full_solution():
    robot, ex = _executor()
    assert ex.init()
    assert ex.plan()
    assert ex.rank(), "expected at least one full pick&place solution"


def test_pick_place_chain_identity_threading():
    robot, ex = _executor()
    assert ex.init() and ex.plan()
    sol = ex.best()
    chain = ex.flatten_leaves(sol)
    assert len(chain) >= 8
    for prev, nxt in zip(chain, chain[1:]):
        assert prev.end is nxt.start


def test_pick_place_scene_ops_order_and_execute():
    robot, ex = _executor()
    assert ex.init() and ex.plan()
    sol = ex.best()
    results = ex.execute(sol)
    assert results, "expected driven motion segments"

    # attach happened before detach, and both reached the server
    ops = [kind for (kind, _) in robot.world_ops if kind in ("attach", "detach")]
    assert ops == ["attach", "detach"]

    # object was attached and later detached on the server side
    assert robot.attached == set()
    # the winning path terminates at the home configuration
    last = sol.end.joint_state.position
    assert all(abs(p) < 1e-9 for p in last), f"end config not home: {last}"


def test_pick_place_statistics_covers_every_stage():
    robot, ex = _executor()
    assert ex.init() and ex.plan()
    stats = ex.statistics()
    names = {s["stage_name"] for s in stats["stages"]}
    for expected in ("approach", "lower", "lift", "retreat",
                     "move to pick", "move to place", "move home",
                     "attach object", "detach object",
                     "compute pick pose", "compute place pose"):
        assert expected in names, f"{expected!r} missing from statistics"
    assert stats["task_id"] == "pick_place"


def test_pick_place_rejects_attached_object_at_current_state():
    """The require_not_attached predicate (MTC PredicateFilter analog)."""
    robot, ex = _executor()
    robot.attached.add(OBJECT)  # object already attached to the gripper
    assert ex.init()
    assert not ex.plan()  # current state refuses to seed the task
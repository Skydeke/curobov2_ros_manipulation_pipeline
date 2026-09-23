"""Core framework tests: registry, interface adjacency, serial flow, builder.

Validation targets for migration steps 1–2 of the plan: pure-python Stage /
ContainerStage semantics against the MockCuroboServer, plus the declarative
StageSpec -> tree builder.
"""

from __future__ import annotations

import pytest

import curobo_task_constructor.stages  # noqa: F401  (register builtins)
from curobo_task_constructor.core.container import (
    GENERATE_INTERFACE,
    SerialContainer,
    invert_flags,
)
from curobo_task_constructor.core.registry import (
    STAGE_REGISTRY,
    create_stage,
    get_stage_class,
    register_stage,
)
from curobo_task_constructor.core.stage import (
    BACKWARD,
    FORWARD,
    InitStageError,
    InterfaceType,
    PropagatingEitherWay,
)
from curobo_task_constructor.executor import TaskExecutor
from curobo_task_constructor.graph.builder import build_tree
from curobo_task_constructor.graph.spec import StageSpec
from tests.mock_curobo import JOINT_NAMES, MockCuroboServer


# ----------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------
def test_registry_builtins_are_registered():
    for name in ("current_state", "fixed_state", "generate_grasp_pose",
                 "compute_ik", "move_to", "move_relative", "modify_scene",
                 "connect"):
        assert name in STAGE_REGISTRY, f"{name} not registered"


def test_registry_unknown_type_raises():
    with pytest.raises(KeyError, match="unknown stage type"):
        get_stage_class("no_such_stage")


def test_registry_duplicate_registration_raises():
    from curobo_task_constructor.core.stage import GeneratorStage

    @register_stage("collision_dup_name_test")
    class _First(GeneratorStage):
        pass

    with pytest.raises(ValueError, match="already registered"):
        @register_stage("collision_dup_name_test")
        class _Second(GeneratorStage):
            pass


def test_registry_open_extension():
    """New capabilities are staged by decoration alone — the open anti-enum."""

    from curobo_task_constructor.core.stage import (
        GeneratorStage,
        InterfaceState,
    )

    @register_stage("test_extension_stage")
    class _ExtensionStage(GeneratorStage):
        def compute(self):
            self.spawn(InterfaceState(joint_state=self.robot.get_current_joint_state()),
                       comment="extension")

    robot = MockCuroboServer()
    stage = create_stage("test_extension_stage", name="ext", params={})
    stage.init(None, robot)
    stage.run_compute()
    assert stage.solutions and stage.solutions[0].comment == "extension"


# ----------------------------------------------------------------------
# interface flags / invert
# ----------------------------------------------------------------------
def test_invert_flags_swaps_read_write():
    gen = InterfaceType.GENERATOR
    inv = invert_flags(gen.start, gen.end)
    # generator (write both) inverted = read both (CONNECTING-like)
    assert inv == (InterfaceType.CONNECTING.start, InterfaceType.CONNECTING.end)


def test_generate_interface_writes_both():
    s, e = GENERATE_INTERFACE
    assert s.write and not s.read
    assert e.write and not e.read


# ----------------------------------------------------------------------
# adjacency validation (MTC connect rules)
# ----------------------------------------------------------------------
def test_serial_incompatible_interfaces_rejected_at_init():
    """Two generators back-to-back cannot connect: no forward/backward edge."""
    robot = MockCuroboServer()
    spec = StageSpec(stage_type="", name="root", container_type="serial",
                     children=[
                         StageSpec(stage_type="current_state", name="cur"),
                         StageSpec(stage_type="fixed_state", name="home",
                                   params_yaml="goal: home\n"),
                     ])
    ex = TaskExecutor(spec, robot)
    assert not ex.init()
    assert "cannot connect" in ex.describe()["comment"]


def test_connect_only_stage_between_generators_is_valid():
    """generator -> connector -> generator: valid (the pick&place shape)."""
    robot = MockCuroboServer(named={"home": {n: 0.0 for n in JOINT_NAMES}})
    spec = StageSpec(stage_type="", name="root", container_type="serial",
                     children=[
                         StageSpec(stage_type="current_state", name="cur"),
                         StageSpec(stage_type="connect", name="via"),
                         StageSpec(stage_type="fixed_state", name="home",
                                   params_yaml="goal: home\n"),
                     ])
    ex = TaskExecutor(spec, robot)
    assert ex.init()


# ----------------------------------------------------------------------
# serial flow (current_state -> move_to)
# ----------------------------------------------------------------------
def test_serial_simple_flow_reaches_goal():
    robot = MockCuroboServer()
    target = [0.2, 0.3, 0.1, 0.0, 0.0, 0.0, 0.8]
    spec = StageSpec(stage_type="", name="root", container_type="serial",
                     children=[
                         StageSpec(stage_type="current_state", name="cur"),
                         StageSpec(
                             stage_type="move_to", name="to_target",
                             params_yaml="goal:\n  joints: [0.2, 0.3, 0.1, "
                                         "0.0, 0.0, 0.0, 0.8]\n"),
                     ])
    ex = TaskExecutor(spec, robot)
    assert ex.init()
    assert ex.plan()
    sol = ex.best()
    assert sol is not None
    assert sol.cost > 0.0
    end_pos = sol.end.joint_state.position
    assert all(abs(a - b) < 1e-6 for a, b in zip(end_pos, target))

    # executes to the goal and drives exactly one motion segment
    results = ex.execute(sol)
    assert len(results) == 1
    assert len(robot.executed) == 1


def test_serial_flow_named_goal():
    robot = MockCuroboServer(named={"home": {n: 0.0 for n in JOINT_NAMES}})
    spec = StageSpec(stage_type="", name="root", container_type="serial",
                     children=[
                         StageSpec(stage_type="current_state", name="cur"),
                         StageSpec(stage_type="move_to", name="home",
                                   params_yaml="goal:\n  name: home\n"),
                     ])
    ex = TaskExecutor(spec, robot)
    assert ex.init() and ex.plan()
    sol = ex.best()
    assert all(abs(v) < 1e-6 for v in sol.end.joint_state.position)


def test_plan_terminates_when_goal_unreachable():
    """Pull consumption + key dedup guarantee graceful termination."""
    robot = MockCuroboServer(fail_ik=True)  # every pose goal fails
    spec = StageSpec(stage_type="", name="root", container_type="serial",
                     children=[
                         StageSpec(stage_type="current_state", name="cur"),
                         StageSpec(
                             stage_type="move_to", name="unreachable",
                             params_yaml="goal:\n  pose:\n    x: 9.0\n"
                                         "    y: 9.0\n    z: 9.0\n"),
                     ])
    ex = TaskExecutor(spec, robot)
    assert ex.init()
    assert not ex.plan()  # no full solution, and no infinite loop
    assert any(stg.failures for stg in ex.root.subtree_stages())


# ----------------------------------------------------------------------
# builder validation
# ----------------------------------------------------------------------
def test_builder_rejects_unknown_container_type():
    spec = StageSpec(stage_type="", container_type="no_such",
                     children=[StageSpec(stage_type="current_state")])
    with pytest.raises(ValueError, match="unknown container_type"):
        build_tree(spec)


def test_builder_rejects_children_on_plain_stage():
    spec = StageSpec(stage_type="current_state",
                     children=[StageSpec(stage_type="move_to")])
    with pytest.raises(InitStageError, match="cannot take children"):
        build_tree(spec)


def test_builder_wrapper_requires_one_child():
    spec = StageSpec(stage_type="compute_ik",
                     children=[
                         StageSpec(stage_type="generate_grasp_pose"),
                         StageSpec(stage_type="generate_grasp_pose"),
                     ])
    with pytest.raises(InitStageError, match="exactly one child"):
        build_tree(spec)


def test_builder_constructs_wrapper_child():
    spec = StageSpec(stage_type="compute_ik", name="ik",
                     children=[
                         StageSpec(stage_type="generate_grasp_pose",
                                   params_yaml="object: object\n")
                     ])
    root = build_tree(spec)
    assert root.stage_type() == "compute_ik"
    assert root.num_children() == 1
    assert root.children[0].stage_type() == "generate_grasp_pose"


# ----------------------------------------------------------------------
# EitherWay direction resolution
# ----------------------------------------------------------------------
def test_either_way_restrict_direction():
    """A backward stage must be the first child of a serial container and be
    followed by a stage that writes its start (the pick&place approach shape):
    generator -> connect -> sub[backward move, generator wrapper]."""
    robot = MockCuroboServer(named={"home": {n: 0.0 for n in JOINT_NAMES}})
    sub = StageSpec(stage_type="", name="pick-like", container_type="serial",
                    children=[
                        StageSpec(stage_type="move_to", name="approach",
                                  params_yaml="goal:\n  name: home\n"),
                        StageSpec(
                            stage_type="compute_ik", name="ik",
                            children=[StageSpec(
                                stage_type="generate_grasp_pose",
                                params_yaml="object: object\n")]),
                    ])
    spec = StageSpec(stage_type="", name="root", container_type="serial",
                     children=[
                         StageSpec(stage_type="current_state", name="cur"),
                         StageSpec(stage_type="connect", name="link"),
                         sub,
                     ])
    ex = TaskExecutor(spec, robot)
    assert ex.init(), ex.describe()["comment"]
    approach = ex.root.children[-1].children[0]
    assert approach._flow == BACKWARD
    assert approach.required_flags()[0].write  # writes its start (to Connect)

    # a FORWARD-pinned stage in the same slot must be rejected at init
    pinned = StageSpec(stage_type="move_to", name="approach",
                       params_yaml="goal:\n  name: home\n"
                                   "direction: forward\n")
    sub2 = StageSpec(stage_type="", container_type="serial",
                     children=[pinned,
                               StageSpec(
                                   stage_type="compute_ik",
                                   children=[StageSpec(
                                       stage_type="generate_grasp_pose",
                                       params_yaml="object: object\n")])])
    spec2 = StageSpec(stage_type="", container_type="serial",
                      children=[StageSpec(stage_type="current_state"),
                                StageSpec(stage_type="connect"), sub2])
    ex2 = TaskExecutor(spec2, robot)
    assert not ex2.init()


def test_subtree_stages_counts_children():
    spec = StageSpec(stage_type="", container_type="serial",
                     children=[
                         StageSpec(stage_type="current_state"),
                         StageSpec(stage_type="", container_type="serial",
                                   children=[
                                       StageSpec(stage_type="current_state"),
                                       StageSpec(stage_type="move_to"),
                                   ]),
                     ])
    root = build_tree(spec)
    assert [s.name for s in root.subtree_stages()] == \
        [root.name, root.children[0].name, root.children[1].name,
         root.children[1].children[0].name, root.children[1].children[1].name]


# ----------------------------------------------------------------------
# describe/statistics shape (Sec. 7 messages)
# ----------------------------------------------------------------------
def test_describe_and_statistics_shapes():
    robot = MockCuroboServer(named={"home": {n: 0.0 for n in JOINT_NAMES}})
    spec = StageSpec(stage_type="", name="root", container_type="serial",
                     children=[
                         StageSpec(stage_type="current_state", name="cur"),
                         StageSpec(stage_type="move_to", name="home",
                                   params_yaml="goal:\n  name: home\n"),
                     ])
    ex = TaskExecutor(spec, robot)
    ex.init()
    desc = ex.describe()
    assert desc["task_id"] == "task"
    assert desc["valid"] is True
    assert desc["root"]["container_type"] == "serial"
    assert desc["root"]["children"][0]["stage_type"] == "current_state"

    ex.plan()
    stats = ex.statistics()
    stage_names = {s["stage_name"] for s in stats["stages"]}
    assert {"cur", "home", "root"} <= stage_names
    mov = next(s for s in stats["stages"] if s["stage_name"] == "home")
    assert mov["success_count"] >= 1
    assert mov["stage_type"] == "move_to"
    assert any(a["success"] and a["stage_name"] == "home"
               for a in stats["attempts"])
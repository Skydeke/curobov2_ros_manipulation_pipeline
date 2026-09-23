"""Regression: introspection hooks attached after ``init()`` must chain onto
the container threading handlers — never replace them.

node.py attaches its publish hooks to every stage AFTER ``executor.init()``
has already wrapped each child's ``on_solution`` with the container's
chain-lifting handler (``ContainerStage._on_child_solution`` / ``_lift_chain``
in ``core/container.py``). A plain assignment severs that chain: leaves still
emit and publish, but the root never lifts a full solution, so
``executor.plan()`` reports no complete solution even though the server plans
fine. ``core.chain_hook`` is the contract every post-init hook attach point
(the action server's publishers, ComputeIK's collection) relies on.
"""

from __future__ import annotations

from functools import partial

import curobo_task_constructor.stages  # noqa: F401  (register builtins)
from curobo_task_constructor.core.stage import chain_hook
from curobo_task_constructor.executor import TaskExecutor
from curobo_task_constructor.graph.spec import StageSpec
from tests.mock_curobo import JOINT_NAMES, MockCuroboServer


def _return_like_spec() -> StageSpec:
    """Same shape as the orchestrator's retry-return task:
    serial [current_state, modify_scene(allow), move_to]."""
    return StageSpec(stage_type="", name="return", container_type="serial",
                     children=[
                         StageSpec(stage_type="current_state", name="current_state"),
                         StageSpec(
                             stage_type="modify_scene", name="allow",
                             params_yaml="allow_collisions:\n"
                                         "  object: object_0\n"
                                         "  links: [left_inner_finger]\n"
                                         "  enabled: true\n"),
                         StageSpec(
                             stage_type="move_to", name="return",
                             params_yaml="goal:\n  joints: [0.3, 0.4, 0.0, "
                                         "0.1, 0.0, 0.2, 0.5]\n"),
                     ])


def test_chain_hook_preserves_container_threading():
    """The pattern node.py uses: hook every stage AFTER init, compositionally."""
    robot = MockCuroboServer(
        named={"home": {n: 0.0 for n in JOINT_NAMES}},
        world={"object_0": [0.1, 0.2, 0.3]},
    )
    ex = TaskExecutor(_return_like_spec(), robot)
    assert ex.init()

    seen = []

    def record(stg, sol):
        seen.append((stg.name, sol.stage.name))

    for stg in ex.root.subtree_stages():
        stg.on_solution = chain_hook(
            stg.on_solution, partial(record, stg))

    assert ex.plan(), "chaining hooks after init must still lift a root solution"
    assert ex.rank(), "expected at least one complete root solution"
    # every emitted solution reached the introspection hook, including the
    # container-lifted root solutions
    seen_leaves = {name for name, _ in seen}
    assert {"current_state", "allow", "return"} <= seen_leaves
    assert any(stage == "return" for stage, _ in seen)


def test_plain_assign_severs_threading_and_is_rejected():
    """The pre-fix node.py pattern (bare assignment) must show the bug this
    test guards against — attach AFTER init and the root never lifts."""
    robot = MockCuroboServer(
        named={"home": {n: 0.0 for n in JOINT_NAMES}},
        world={"object_0": [0.1, 0.2, 0.3]},
    )
    ex = TaskExecutor(_return_like_spec(), robot)
    assert ex.init()

    for stg in ex.root.subtree_stages():
        stg.on_solution = lambda sol, s=stg: None  # replaces threading handler

    assert not ex.plan()
    assert not ex.rank()
    # leaves still produced their own solutions...
    leaves = [s for s in ex.root.subtree_stages()
              if s.stage_type() in ("current_state", "modify_scene", "move_to")]
    assert all(s.solutions for s in leaves), "leaves emit but root never lifts"
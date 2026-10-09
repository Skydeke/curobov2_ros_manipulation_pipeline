"""StageSpec tree contract: dict/yaml round trip, validation, pre-order.

The StageSpec tree is the local (in-process) task description: planning is
local like MoveIt, so the tree never crosses the wire. Introspection
describes stages with StageDescription/Property messages instead (see
msg_convert + Task._publish_task_description).
"""

import pytest

from curobo_task_constructor.graph.spec import StageSpec, params_from_yaml


def _pick_like_spec() -> StageSpec:
    """A small tree with the same container shapes as the grasp orchestrator:
    serial root -> [current_state, fallbacks(variant i), serial tail]."""
    def variant(i):
        return StageSpec(
            stage_type="", name=f"variant_{i}",
            container_type="serial",
            children=[
                StageSpec(stage_type="move_to", name=f"approach_{i}"),
                StageSpec(stage_type="move_to", name=f"grasp_{i}"),
                StageSpec(stage_type="modify_scene", name=f"attach_{i}"),
            ])

    return StageSpec(
        stage_type="", name="pick", container_type="serial",
        children=[
            StageSpec(stage_type="current_state", name="current_state"),
            StageSpec(stage_type="", name="variants",
                      container_type="fallbacks",
                      children=[variant(0), variant(1)]),
            StageSpec(stage_type="move_to", name="return"),
            StageSpec(stage_type="move_to", name="open"),
        ])


def test_dict_round_trip():
    tree = _pick_like_spec()
    assert StageSpec.from_dict(tree.to_dict()).to_dict() == tree.to_dict()


def test_yaml_round_trip():
    tree = _pick_like_spec()
    assert StageSpec.from_yaml(tree.to_yaml()).to_dict() == tree.to_dict()


def test_preorder_is_root_first_parents_before_children():
    tree = _pick_like_spec()
    order = tree._preorder()
    assert order[0] is tree
    position = {id(node): i for i, node in enumerate(order)}
    for node in order:
        for child in node.children:
            assert position[id(node)] < position[id(child)]
    assert len(order) == 1 + 1 + 1 + 2 * 4 + 1 + 1


def test_validate_rejects_empty_container():
    with pytest.raises(ValueError, match="at least one child"):
        StageSpec(stage_type="", name="empty",
                  container_type="serial", children=[]).validate()


def test_validate_rejects_typeless_leaf():
    with pytest.raises(ValueError, match="stage_type must be non-empty"):
        StageSpec(stage_type="", name="typeless").validate()


def test_validate_accepts_pick_like_tree():
    _pick_like_spec().validate()


def test_params_from_yaml():
    assert params_from_yaml("") == {}
    assert params_from_yaml("  \n") == {}
    assert params_from_yaml("{goal: {joints: [0.5]}}") == {
        "goal": {"joints": [0.5]}}
    assert params_from_yaml("- just\n- a\n- list\n") == {}


def test_leaf_only_task_round_trip():
    spec = StageSpec(stage_type="move_to", name="home")
    assert StageSpec.from_dict(spec.to_dict()).to_dict() == spec.to_dict()

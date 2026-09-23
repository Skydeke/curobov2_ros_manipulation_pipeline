"""Wire-format round trip: python StageSpec tree <-> flat StageSpec[] list.

The msg side of ``curobo_task_constructor_interfaces`` carries the task graph
as a flat ``StageSpec[]`` (MTC-style id/parent_id wiring, root = parent_id ==
id, pre-order) because recursive message types are not buildable in ROS 2.
These tests pin the converter contract that node.py / the grasp orchestrator /
the rviz panel all rely on.
"""

from curobo_task_constructor.graph.spec import StageSpec


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


def test_to_msg_list_is_preorder_with_mtc_root_marker():
    specs = _pick_like_spec().to_msg_list()

    # root is element 0 and marks itself (parent_id == id == 0)
    assert len(specs) == 1 + 1 + 1 + 2 * 4 + 1 + 1  # root, current, variants, 2x (variant+3), return, open
    assert specs[0].name == "pick"
    assert specs[0].id == 0
    assert specs[0].parent_id == 0

    # pre-order: every non-root parent precedes its children (the root's
    # self-marker parent_id == id is a marker, not an edge)
    children_of = {}
    for spec in specs:
        if spec.parent_id != spec.id:
            children_of.setdefault(spec.parent_id, []).append(spec.id)
    expected_top_names = ["current_state", "variants", "return", "open"]
    assert [specs[i].name for i in children_of[0]] == expected_top_names
    variant_ids = children_of[next(
        s.id for s in specs if s.name == "variants")]
    assert len(variant_ids) == 2
    for vid in variant_ids:
        assert specs[vid].name.startswith("variant_")
        assert [specs[cid].name for cid in children_of[vid]] == [
            f"approach_{specs[vid].name[-1]}",
            f"grasp_{specs[vid].name[-1]}",
            f"attach_{specs[vid].name[-1]}",
        ]

    # ids are dense pre-order indices
    assert [s.id for s in specs] == list(range(len(specs)))


def test_round_trip_tree_equality():
    tree = _pick_like_spec()
    assert StageSpec.from_msg_list(tree.to_msg_list()).to_dict() == tree.to_dict()


def test_from_msg_list_rejects_multiple_roots():
    specs = _pick_like_spec().to_msg_list()
    # flip a non-root stage into a second root
    specs[1].parent_id = specs[1].id
    try:
        StageSpec.from_msg_list(specs)
    except ValueError as exc:
        assert "exactly one root" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for two roots")


def test_from_msg_list_rejects_unknown_parent():
    specs = _pick_like_spec().to_msg_list()
    specs[-1].parent_id = 99  # dangling reference
    try:
        StageSpec.from_msg_list(specs)
    except ValueError as exc:
        assert "pre-order" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for unknown parent")


def test_from_msg_list_rejects_duplicate_ids():
    specs = _pick_like_spec().to_msg_list()
    specs[2].id = specs[1].id  # duplicate id
    try:
        StageSpec.from_msg_list(specs)
    except ValueError as exc:
        assert "duplicate stage id" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for duplicate id")


def test_leaf_only_task_round_trip():
    spec = StageSpec(stage_type="move_to", name="home")
    flat = spec.to_msg_list()
    assert len(flat) == 1
    assert flat[0].id == 0 and flat[0].parent_id == 0
    assert StageSpec.from_msg_list(flat).to_dict() == spec.to_dict()


class _BlankMsg:
    """Stand-in for the ROS StageSpec.msg class: a constructor that starts
    every field empty (as rclpy message classes do) plus the six wire fields
    node.py / the rviz panel read back."""

    def __init__(self):
        self.id = None
        self.parent_id = None
        self.stage_type = ""
        self.name = ""
        self.container_type = ""
        self.params_yaml = ""


def test_to_msg_list_copies_fields_to_ros_msg_class():
    """to_msg_list() with a real (blank-starting) msg class must copy every
    field, not just id/parent_id — node.py publishes TaskDescription via
    to_msg_list(StageSpecMsg), and the rviz panel renders Stage/Type from
    those fields. Regression: the fields were dropped, so the panel showed
    empty name/type cells (only tooltips surfaced any text)."""
    tree = _pick_like_spec()
    flat = tree.to_msg_list(_BlankMsg)
    assert len(flat) == 1 + 1 + 1 + 2 * 4 + 1 + 1  # root, current, variants, 2x (variant+3), return, open

    root = flat[0]
    assert root.id == 0 and root.parent_id == 0
    assert root.name == "pick"
    assert root.container_type == "serial"
    assert root.stage_type == ""

    # a leaf: container_type empty, name + stage_type populated
    leaf = next(s for s in flat if s.name == "approach_0")
    assert leaf.stage_type == "move_to"
    assert leaf.container_type == ""
    assert leaf.name == "approach_0"

    # every returned message is the blank-starting class (not the stub)
    assert all(type(s) is _BlankMsg for s in flat)

    # round trip: from_msg_list must recover the original tree from the wire
    assert StageSpec.from_msg_list(flat).to_dict() == tree.to_dict()
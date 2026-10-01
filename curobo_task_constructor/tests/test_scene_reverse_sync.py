"""build_base_scene must rebuild the world from the SERVER, not from a mirror.

`build_base_scene` is the executor's reverse sync: it asks the robot interface
what is in the scene and hands that to the planner as the task's base scene. If
it describes the world wrongly, every plan made against that base scene is
planned against a world that does not exist.

Two defects this pins down, both of which were real:

1. `shape` was hardcoded to "mesh" for every object, so any sphere or cuboid in
   the world was described to the planner as a mesh.
2. The interface kept a local mirror of its OWN `add_object` calls, so it could
   not see an object another client added — and reported a stale pose for a name
   that had been removed and re-added.
"""

from __future__ import annotations

import pytest

import curobo_task_constructor.stages  # noqa: F401  (registers the builtins)
from curobo_task_constructor.core.state import ObjectSpec
from curobo_task_constructor.executor import TaskExecutor
from curobo_task_constructor.graph.spec import StageSpec
from tests.mock_curobo import MockCuroboServer


def _executor(robot) -> TaskExecutor:
    """A TaskExecutor wired to `robot`.

    `build_base_scene` is a pure method over the robot interface, so the spec is
    an empty stage tree — nothing is initialised or planned.
    """
    return TaskExecutor(
        StageSpec(stage_type="move_to", name="root",
                  params_yaml="goal:\n  name: home\n"),
        robot)


def test_base_scene_reports_the_server_shape_not_a_guess():
    robot = MockCuroboServer()
    robot.add_object(ObjectSpec(
        name="ball", shape="sphere",
        pose=_pose(0.5, 0.0, 0.1),
        dimensions=[0.05, 0.05, 0.05],
    ))
    robot.add_object(ObjectSpec(
        name="block", shape="cuboid",
        pose=_pose(0.3, 0.2, 0.05),
        dimensions=[0.2, 0.1, 0.1],
    ))

    scene = _executor(robot).build_base_scene()

    assert scene.objects_added["ball"].shape == "sphere"
    assert scene.objects_added["block"].shape == "cuboid"
    # The regression this file exists for: both were reported as "mesh".
    assert "mesh" not in {s.shape for s in scene.objects_added.values()}


def test_base_scene_carries_the_real_size():
    robot = MockCuroboServer()
    robot.add_object(ObjectSpec(
        name="block", shape="cuboid",
        pose=_pose(0.3, 0.2, 0.05),
        dimensions=[0.2, 0.1, 0.07],
    ))

    spec = _executor(robot).build_base_scene().objects_added["block"]

    assert spec.dimensions == [0.2, 0.1, 0.07]


def test_base_scene_sees_an_object_added_by_someone_else():
    """The interface's own mirror could not do this.

    `add_object` here stands in for any other client writing to the server: the
    reverse sync has to report the whole world, not just what it put there.
    """
    robot = MockCuroboServer()
    robot.add_object(ObjectSpec(name="table", shape="cuboid",
                                pose=_pose(0.0, 0.0, 0.0),
                                dimensions=[1.0, 1.0, 0.05]))

    scene = _executor(robot).build_base_scene()

    assert set(scene.objects_added) == {"table"}
    assert scene.objects_added["table"].shape == "cuboid"


def test_base_scene_is_empty_for_an_empty_world():
    assert _executor(MockCuroboServer()).build_base_scene().objects_added == {}


def test_base_scene_drops_a_removed_object():
    robot = MockCuroboServer()
    robot.add_object(ObjectSpec(name="gone", shape="cuboid",
                                pose=_pose(0.0, 0.0, 0.0)))
    robot.remove_object("gone")

    assert _executor(robot).build_base_scene().objects_added == {}


def test_base_scene_reports_a_re_added_name_with_the_new_pose():
    """A stale local mirror would report the FIRST object's pose here."""
    robot = MockCuroboServer()
    robot.add_object(ObjectSpec(name="slot", shape="cuboid",
                                pose=_pose(1.0, 0.0, 0.0)))
    robot.remove_object("slot")
    robot.add_object(ObjectSpec(name="slot", shape="sphere",
                                pose=_pose(-1.0, 0.0, 0.0)))

    spec = _executor(robot).build_base_scene().objects_added["slot"]

    assert _xyz(spec.pose) == pytest.approx([-1.0, 0.0, 0.0])
    assert spec.shape == "sphere"


def test_base_scene_falls_back_when_the_interface_cannot_describe_objects():
    """An interface implementing only the older two methods still works."""

    class PoseOnly(MockCuroboServer):
        get_object_spec = None  # not offered

    robot = PoseOnly()
    robot.add_object(ObjectSpec(name="thing", shape="cuboid",
                                pose=_pose(0.2, 0.0, 0.0)))

    scene = _executor(robot).build_base_scene()

    assert set(scene.objects_added) == {"thing"}
    assert _xyz(scene.objects_added["thing"].pose) == pytest.approx([0.2, 0.0, 0.0])


# --- helpers ---------------------------------------------------------------


def _pose(x, y, z):
    from curobo_task_constructor.core.geom import Pose3
    return Pose3([x, y, z], [0.0, 0.0, 0.0, 1.0])


def _xyz(pose):
    return [float(pose.position.x), float(pose.position.y), float(pose.position.z)]

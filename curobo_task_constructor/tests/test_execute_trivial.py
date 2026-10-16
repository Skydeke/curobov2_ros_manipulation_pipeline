"""Trivial (no-motion) segments drive nothing.

The planner answers zero-motion legs without solving (see
CuroboServerInterface._trivial_goal), so there is no trajectory to replay for
them. drive_solution_dict must skip the drive (success), while still applying
scene ops and firing progress.
"""

from dataclasses import dataclass, field

from curobo_task_constructor.core.robot import PlanResult
from curobo_task_constructor.core.state import JointStateStub
from curobo_task_constructor.executor import drive_solution_dict


@dataclass
class FakeRobot:
    driven: list = field(default_factory=list)
    ops: list = field(default_factory=list)
    replayed: list = field(default_factory=list)

    def execute(self, request, trajectory):
        self.driven.append(request)
        self.replayed.append(trajectory)
        return PlanResult(True, "ok", trajectory=[
            JointStateStub({"a": 0.0}), JointStateStub({"a": 1.0})])

    def add_object(self, payload):
        self.ops.append(("add", getattr(payload, "name", payload)))

    def remove_object(self, payload):
        self.ops.append(("remove", payload))

    def remove_all_objects(self):
        self.ops.append(("remove_all", None))

    def detach_object(self, payload=None):
        self.ops.append(("detach", payload))

    def attach_object(self, payload):
        self.ops.append(("attach", payload))


def _replay_goal():
    return {
        "goalsets": [{"poses": [], "target_joints": [0.0],
                      "allowed": [], "holds": []}],
        "start": {"name": ["a"], "position": [0.0]},
        "planner": 5,
    }


def _seg(points, replay=True, before=None, after=None):
    seg = {
        "info": {"id": 1, "cost": 0.0, "comment": "",
                 "stage_id": 1, "planner_id": ""},
        "execution_info": {"controller_names": []},
        "trajectory": {"joint_names": ["a"], "points": [list(p) for p in points]},
        "scene_diff": {
            "before": before or {"added": [], "removed": [],
                                 "attached": None, "detached": None},
            "after": after or {"added": [], "removed": [],
                               "attached": None, "detached": None},
        },
        "replay_goal": _replay_goal() if replay else {},
    }
    return seg


def _attach_before(name):
    return {"added": [], "removed": [], "attached": name, "detached": None}


def test_single_waypoint_segment_skips_the_drive():
    robot = FakeRobot()
    progress = []
    d = {"sub_trajectories": [
        _seg([[0.0]], before=_attach_before("object_0")),
        _seg([[0.0], [1.0]]),
    ]}
    results, failed, cancelled = drive_solution_dict(
        robot, d, [],
        progress_callback=lambda i, total, info: progress.append(i))
    assert failed is None
    assert cancelled is False
    assert len(results) == 2
    assert all(r.success for r in results)
    # Only the real motion segment drove; the trivial one did not.
    assert len(robot.driven) == 1
    # Scene ops around the skipped drive still applied...
    assert ("attach", "object_0") in robot.ops
    # ...and the panel still followed the chain.
    assert progress == [0, 1]


def test_empty_trajectory_segment_skips_the_drive():
    robot = FakeRobot()
    d = {"sub_trajectories": [_seg([])]}
    results, failed, cancelled = drive_solution_dict(robot, d, [])
    assert failed is None
    assert len(results) == 1 and results[0].success
    assert robot.driven == []


def test_multi_waypoint_segment_still_drives():
    robot = FakeRobot()
    d = {"sub_trajectories": [_seg([[0.0], [1.0]])]}
    results, failed, cancelled = drive_solution_dict(robot, d, [])
    assert failed is None
    assert len(robot.driven) == 1
    assert results[0].success
    # MTC parity: the segment's OWN waypoints are handed to the drive, so the
    # server streams exactly the validated trajectory instead of re-matching a
    # goal against the trajectory cache.
    assert robot.replayed[0]["points"] == [[0.0], [1.0]]


def test_clear_all_applies_before_the_re_adds():
    """A scene-setup segment clears the world, then re-adds into it.

    The clear is a boolean scene op, not a name list, so it rides the wire
    (remove_all/detach_all) and must be ordered ahead of the adds: re-adding
    onto the previous cycle's world gets every add refused as a duplicate.
    """
    robot = FakeRobot()
    before = {"added": [{"name": "object_0", "shape": "cuboid",
                         "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "qx": 0.0,
                                  "qy": 0.0, "qz": 0.0, "qw": 1.0},
                         "dimensions": [0.1, 0.1, 0.1],
                         "mesh_path": None, "vertices": None,
                         "triangles": None}],
              "removed": [], "attached": None, "detached": None,
              "remove_all": True, "detach_all": True}
    d = {"sub_trajectories": [_seg([[0.0]], before=before)]}
    drive_solution_dict(robot, d, [])
    assert robot.ops == [("detach", None), ("remove_all", None),
                         ("add", "object_0")]

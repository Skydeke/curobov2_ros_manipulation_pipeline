"""Trivial (no-motion) segments drive nothing.

The planner answers zero-motion legs without solving (see
CuroboServerInterface._trivial_goal), so no cached trajectory exists for
them — and a force_cached replay attempt would fail the whole chain on a
cache miss for a motion of zero length. drive_solution_dict must skip the
drive (success), while still applying scene ops and firing progress.
"""

from dataclasses import dataclass, field

from curobo_task_constructor.core.robot import PlanResult
from curobo_task_constructor.core.state import JointStateStub
from curobo_task_constructor.executor import drive_solution_dict


@dataclass
class FakeRobot:
    driven: list = field(default_factory=list)
    ops: list = field(default_factory=list)

    def execute(self, request):
        self.driven.append(request)
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

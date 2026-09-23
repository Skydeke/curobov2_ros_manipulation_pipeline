"""MockCuroboServer — an analytical RobotInterface double (no GPU, no ROS).

Implements enough of the cuRobo server surface for the built-in stages to run
end-to-end:

- a 3-joint planar arm (pan around z + 2-link in the x-z plane) with a
  closed-form FK/IK pair, plus pass-through joints and a ``finger_joint``;
- deterministic ``plan``: linear interpolation start -> each goalset, with a
  per-request waypoint count (the ranking cost proxy);
- ``plan_batch`` records batched calls (the Alternatives optimization);
- scene ops (add/remove/attach/detach) mutate a small world model and are
  logged, so tests can assert what the executor materialized.

Because IK is closed-form both ways, ComputeIK/MoveRelative/MoveTo/Connect all
resolve exactly — the tests exercise the framework, not stochastic math.
"""

from __future__ import annotations

import math
from typing import Optional

from curobo_task_constructor.core.geom import Pose3, pose_to_any
from curobo_task_constructor.core.robot import (
    PlanRequest,
    PlanResult,
    RobotInterface,
)
from curobo_task_constructor.core.robot_config import NamedJointConfig
from curobo_task_constructor.core.state import JointStateStub

#: Canonical joint order: 6 arm joints + finger. The analytic FK only uses
#: the first three; the rest pass through untouched.
JOINT_NAMES = [f"joint_{i}" for i in range(1, 7)] + ["finger_joint"]

#: two link lengths (m)
L1, L2 = 0.10, 0.40


def _positions(joint_state) -> dict:
    return dict(zip(getattr(joint_state, "name", []),
                    getattr(joint_state, "position", [])))


def _xyz(pose) -> list:
    """Tolerate both Pose-stubs (.x/.y/.z) and Pose3 (list-of-3)."""
    p = pose.position
    if hasattr(p, "x"):
        return [float(p.x), float(p.y), float(p.z)]
    return [float(p[0]), float(p[1]), float(p[2])]


def fk_positions(pos: dict) -> list:
    """Tool position [x, y, z] for a joint dict (wrist-agnostic stub)."""
    t1 = pos.get("joint_1", 0.0)
    t2 = pos.get("joint_2", 0.0)
    t3 = pos.get("joint_3", 0.0)
    reach = L1 * math.cos(t2) + L2 * math.cos(t2 + t3)
    z = L1 * math.sin(t2) + L2 * math.sin(t2 + t3)
    return [math.cos(t1) * reach, math.sin(t1) * reach, z]


def ik_positions(target_pos: list, seed: Optional[dict] = None) -> Optional[dict]:
    """Inverse of ``fk_positions`` for the first three joints."""
    px, py, pz = (float(v) for v in target_pos)
    r = math.hypot(px, py)
    c3 = (r * r + pz * pz - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    if c3 < -1.0 or c3 > 1.0:
        return None
    t3 = math.acos(max(-1.0, min(1.0, c3)))
    t2 = math.atan2(pz, r) - math.atan2(L2 * math.sin(t3),
                                        L1 + L2 * math.cos(t3))
    t1 = math.atan2(py, px) if r > 1e-12 else 0.0
    # The analytic solution OVERRIDES the seed's joint_1..3 values (like real
    # cuRobo IK: solved joints replace their seed entries); all other seed
    # joints (joint_4..6, finger) pass through untouched.
    out = dict(seed or {})
    out.update({"joint_1": t1, "joint_2": t2, "joint_3": t3})
    return out


class MockCuroboServer(RobotInterface):
    joint_state_cls = None
    pose_cls = None

    def __init__(self, current: Optional[dict] = None,
                 named: Optional[dict] = None,
                 world: Optional[dict] = None,
                 n_steps: int = 5,
                 fail_ik: bool = False):
        """``current``/``named`` are joint-name->position dicts; ``world`` is
        ``{name: [x, y, z]}`` (identity orientation)."""
        cur = dict(current) if current else {}
        # a nondegenerate default pose outside the straight-arm singularity
        # (must be set BEFORE the zero-fill loop or it's a no-op)
        cur.setdefault("joint_2", math.pi / 6)
        for name in JOINT_NAMES:
            cur.setdefault(name, 0.0)
        self._current = JointStateStub(names=list(JOINT_NAMES),
                                       positions=[cur[n] for n in JOINT_NAMES])
        self.named = {}
        for cfg_name, value in (named or {}).items():
            if isinstance(value, dict):
                self.named[cfg_name] = NamedJointConfig(
                    cfg_name, list(value.keys()),
                    [float(v) for v in value.values()])
            else:
                self.named[cfg_name] = NamedJointConfig(
                    cfg_name, [], [float(v) for v in value])
        self.world = dict(world or {})  # name -> [x, y, z]
        self.attached = set()
        self.world_ops = []  # ("add"|"remove"|"attach"|"detach", name)
        self.executed = []  # PlanRequests driven via execute()
        self.plan_calls = 0
        self.batch_calls = 0
        self.planner = None
        self.n_steps = n_steps
        self.fail_ik = fail_ik

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    def get_current_joint_state(self):
        return self._current

    def get_object_names(self) -> list:
        return sorted(self.world)

    def get_object_pose(self, name: str):
        if name not in self.world:
            return None
        xyz = self.world[name]
        return pose_to_any(Pose3(list(xyz), [0.0, 0.0, 0.0, 1.0]), None)

    def get_named_joint_config(self, name: str):
        try:
            return self.named[name]
        except KeyError:
            raise KeyError(f"named config {name!r} not found") from None

    def get_attached_objects(self) -> list:
        return sorted(self.attached)

    # ------------------------------------------------------------------
    # kinematics
    # ------------------------------------------------------------------
    def fk(self, joint_state, link: Optional[str] = None):
        xyz = fk_positions(_positions(joint_state))
        return pose_to_any(Pose3(xyz, [0.0, 0.0, 0.0, 1.0]), None)

    def ik(self, pose, seed: Optional[dict] = None):
        if self.fail_ik:
            return None
        base = _positions(seed) if seed is not None else None
        joint = ik_positions(_xyz(pose), base)
        if joint is None:
            return None
        cur = _positions(self._current)
        out = {}
        for name in JOINT_NAMES:
            out[name] = joint.get(name, cur.get(name, 0.0))
        return JointStateStub(names=list(JOINT_NAMES),
                              positions=[out[n] for n in JOINT_NAMES])

    # ------------------------------------------------------------------
    # planning
    # ------------------------------------------------------------------
    def set_planner(self, planner) -> None:
        self.planner = planner

    def _goal_joints(self, goalset) -> Optional[list]:
        if getattr(goalset, "target_joint_positions", None):
            return [float(v) for v in goalset.target_joint_positions]
        poses = getattr(goalset, "poses", None)
        if poses:
            joint = self.ik(poses[0])
            return list(getattr(joint, "position", [])) if joint else None
        return None

    def plan(self, request: PlanRequest) -> PlanResult:
        self.plan_calls += 1
        start = request.start_pose if request.start_pose is not None \
            else self._current
        names = list(getattr(start, "name", []) or JOINT_NAMES)
        a = list(getattr(start, "position", []) or [])
        result_names = names
        trajectory = [start]

        def stub(pos):
            return JointStateStub(names=result_names, positions=pos)

        for goalset in getattr(request, "goalsets", []) or []:
            b = self._goal_joints(goalset)
            if b is None:
                return PlanResult(False, "goal unreachable (ik failed)")
            if len(b) != len(a):
                b = b + a[len(b):]
            n = self.n_steps
            for k in range(1, n + 1):
                t = k / n
                trajectory.append(stub(
                    [ai + (bi - ai) * t for ai, bi in zip(a, b)]))
            a = b
        return PlanResult(
            success=True,
            message="ok",
            trajectory=trajectory,
            cost=float(len(trajectory)),
            raw="mock-trajectory-result",
        )

    def plan_batch(self, requests: list) -> list:
        self.batch_calls += 1
        return [self.plan(r) for r in requests]

    def execute(self, request: PlanRequest) -> PlanResult:
        self.executed.append(request)
        return self.plan(request)

    # ------------------------------------------------------------------
    # scene
    # ------------------------------------------------------------------
    def add_object(self, spec) -> bool:
        xyz = _xyz(spec) if spec.pose is not None else None
        self.world[spec.name] = xyz or [0.0, 0.0, 0.5]
        self.world_ops.append(("add", spec.name))
        return True

    def remove_object(self, name: str) -> bool:
        self.world.pop(name, None)
        self.world_ops.append(("remove", name))
        return True

    def remove_all_objects(self) -> None:
        self.world.clear()
        self.attached.clear()

    def attach_object(self, name: str) -> bool:
        self.attached.add(name)
        # the object travels with the flange; keep its world pose available
        # so a later GeneratePlacePose can still read it (the real pipeline
        # estimates it from the gripper transform instead).
        self.world_ops.append(("attach", name))
        return True

    def detach_object(self, name: Optional[str] = None) -> bool:
        if name is not None and name in self.attached:
            self.attached.discard(name)
            self.world_ops.append(("detach", name))
            return True
        return name is None and bool(self.attached)  # detach-all stub


#: A reachable object pose for pick&place tests (the mock arm's reachable
#: shell is 0.09 <= r^2 + z^2 <= 0.25).
OBJECT_POSE = [0.30, 0.0, 0.25]
TABLE_POSE = [0.0, 0.0, 0.0]


def make_pick_world(object_pose=None) -> dict:
    return {"object": list(object_pose or OBJECT_POSE),
            "table": list(TABLE_POSE)}
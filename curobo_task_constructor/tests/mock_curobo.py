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
from curobo_task_constructor.core.state import JointStateStub, ObjectSpec

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


def _quat(pose) -> list:
    """Orientation as [x, y, z, w], from either a message or a Pose3."""
    q = pose.orientation
    if hasattr(q, "x"):
        return [float(q.x), float(q.y), float(q.z), float(q.w)]
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


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
                 fail_ik: bool = False,
                 bow: float = 0.0,
                 winner_index: int = 0):
        """``current``/``named`` are joint-name->position dicts; ``world`` is
        ``{name: [x, y, z]}`` (identity orientation).

        ``bow`` injects that many metres of lateral deviation at mid-path on
        Cartesian (axis-held) goalsets, so the ``cartesian_path`` straightness
        gate can be exercised on both the accept and the reject branch.
        ``winner_index`` is which candidate of a multi-pose goalset the server
        "picks" (the real server reports this through ``selected_goal_index``).
        """
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
        #: Full per-object records behind `world`, the way the real server's
        #: get_scene_objects reports them. `world` stays the position-only view
        #: many tests read directly; this is what get_object_spec answers from,
        #: so a test cannot pass on a description the real interface could not
        #: produce (a pose with no shape, a shape with no size).
        self.records = {
            name: {
                "shape": "cuboid",
                "position": list(xyz),
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "dimensions": [0.1, 0.1, 0.1],
                "mesh_path": "",
            }
            for name, xyz in self.world.items()
        }
        self.attached = set()
        self.world_ops = []  # ("add"|"remove"|"attach"|"detach", name)
        self.executed = []  # PlanRequests driven via execute()
        self.plan_calls = 0
        self.batch_calls = 0
        self.fk_batch_calls = 0
        self.last_hold: list = []  # trajectory_constraints of the last solve
        #: One entry per goalset of every plan() call, in solve order, so a
        #: test can assert the hold of a *particular* stage rather than only
        #: the last one.
        self.holds: list = []
        self.planner = None
        self.n_steps = n_steps
        self.fail_ik = fail_ik
        self.bow = float(bow)
        self.winner_index = int(winner_index)

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    def get_current_joint_state(self):
        return self._current

    def get_object_names(self) -> list:
        return sorted(self.world)

    def get_object_pose(self, name: str):
        rec = self.records.get(name)
        if rec is None:
            return None
        return pose_to_any(
            Pose3(list(rec["position"]), list(rec["orientation"])), None)

    def get_object_spec(self, name: str) -> Optional[ObjectSpec]:
        rec = self.records.get(name)
        if rec is None:
            return None
        return ObjectSpec(
            name=name,
            shape=rec["shape"],
            pose=pose_to_any(
                Pose3(list(rec["position"]), list(rec["orientation"])), None),
            dimensions=list(rec["dimensions"]),
            mesh_path=rec["mesh_path"] or None,
        )

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

    def fk_batch(self, joint_states: list, link: Optional[str] = None) -> list:
        # Stands in for Fk.srv's JointState[] -> Pose[] round trip; the point
        # of the test is that the stage makes ONE call for a whole trajectory.
        self.fk_batch_calls += 1
        return [self.fk(js, link) for js in (joint_states or [])]

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
            joint = self.ik(poses[self._winner(poses)])
            return list(getattr(joint, "position", [])) if joint else None
        return None

    def _winner(self, poses) -> int:
        """Which candidate of a goalset the (simulated) server picks."""
        return min(self.winner_index, len(poses) - 1)

    def _hold_of(self, goalset) -> list:
        """int8[6] whole-path axis holds on a goalset ([] = free)."""
        hold = [int(c) for c in (getattr(goalset, "trajectory_constraints", None)
                                 or [])]
        return hold if len(hold) == 6 else []

    def plan(self, request: PlanRequest) -> PlanResult:
        self.plan_calls += 1
        start = request.start_pose if request.start_pose is not None \
            else self._current
        names = list(getattr(start, "name", []) or JOINT_NAMES)
        a = list(getattr(start, "position", []) or [])
        result_names = names
        trajectory = [start]
        selected: list = []

        def stub(pos):
            return JointStateStub(names=result_names, positions=pos)

        for goalset in getattr(request, "goalsets", []) or []:
            poses = list(getattr(goalset, "poses", None) or [])
            if poses:
                selected.append(self._winner(poses))
            # [theta_x, theta_y, theta_z, x, y, z] where 1 = HOLD that axis
            # along the whole path. The positional tail therefore says which
            # axes are pinned to the goal and which may travel. A constraint
            # that pins nothing or pins everything is no constraint at all ->
            # plain joint interpolation below.
            hold = self._hold_of(goalset)
            self.holds.append(hold)
            self.last_hold = hold
            held = hold[3:]
            is_line = bool(held) and any(held) and not all(held)
            if poses and is_line:
                # A goalset with partial axis holds asks for a STRAIGHT line.
                # The mock honours it the way the real server does: interpolate
                # the tool through Cartesian space and IK each waypoint back,
                # so held axes stay on the segment by construction. ``bow`` is
                # applied LAST (it is the soft cost losing) so the straightness
                # gate has a failing branch to catch.
                p0 = fk_positions(dict(zip(names, a)))
                p1 = _xyz(poses[self._winner(poses)])
                seed = dict(zip(names, a))
                n = self.n_steps
                for k in range(1, n + 1):
                    t = k / n
                    p = [p1[i] if held[i] else p0[i] + (p1[i] - p0[i]) * t
                         for i in range(3)]
                    if self.bow:
                        p[0] += self.bow * math.sin(math.pi * t)
                    joint = ik_positions(p, seed)
                    if joint is None:
                        return PlanResult(False, "goal unreachable (ik failed)")
                    trajectory.append(stub(
                        [joint.get(nm, a[i]) for i, nm in enumerate(names)]))
                a = [float(v) for v in
                     (getattr(trajectory[-1], "position", None) or a)]
                continue
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
        self.last_hold = self._hold_of(
            (getattr(request, "goalsets", []) or [None])[0]) \
            if getattr(request, "goalsets", None) else []
        return PlanResult(
            success=True,
            message="ok",
            trajectory=trajectory,
            # inf, exactly like the rclpy adapter: the framework owns ranking
            # and falls back to its own term (cost::PathLength). A mock that
            # invented a cost here would silently shadow that, and a
            # waypoint-count "cost" would make PathLength untestable.
            cost=float("inf"),
            raw="mock-trajectory-result",
            selected_goal_index=selected,
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
        xyz = _xyz(spec.pose) if spec.pose is not None else None
        self.world[spec.name] = xyz or [0.0, 0.0, 0.5]
        # The full description, so get_object_spec round-trips what was added
        # rather than defaulting every field.
        orient = _quat(spec.pose) if spec.pose is not None else [0.0, 0.0, 0.0, 1.0]
        dims = list(spec.dimensions or [0.0, 0.0, 0.0])
        self.records[spec.name] = {
            "shape": spec.shape or "cuboid",
            "position": list(self.world[spec.name]),
            "orientation": orient,
            "dimensions": (dims + [0.0, 0.0, 0.0])[:3],
            "mesh_path": spec.mesh_path or "",
        }
        self.world_ops.append(("add", spec.name))
        return True

    def remove_object(self, name: str) -> bool:
        self.world.pop(name, None)
        self.records.pop(name, None)
        self.world_ops.append(("remove", name))
        return True

    def remove_all_objects(self) -> None:
        self.world.clear()
        self.records.clear()
        self.attached.clear()
        self.world_ops.append(("remove_all", None))

    def attach_object(self, name: str) -> bool:
        self.attached.add(name)
        # the object travels with the flange; keep its world pose available
        # so a later GeneratePlacePose can still read it (the real pipeline
        # estimates it from the gripper transform instead).
        self.world_ops.append(("attach", name))
        return True

    def detach_object(self, name: Optional[str] = None) -> bool:
        # name None = detach-all (the server's /detach_object Trigger:
        # releases whatever is attached, no name needed).
        if name is None:
            self.attached.clear()
            self.world_ops.append(("detach_all", None))
        else:
            self.attached.discard(name)
            self.world_ops.append(("detach", name))
        return True


#: A reachable object pose for pick&place tests (the mock arm's reachable
#: shell is 0.09 <= r^2 + z^2 <= 0.25).
OBJECT_POSE = [0.30, 0.0, 0.25]
TABLE_POSE = [0.0, 0.0, 0.0]


def make_pick_world(object_pose=None) -> dict:
    return {"object": list(object_pose or OBJECT_POSE),
            "table": list(TABLE_POSE)}
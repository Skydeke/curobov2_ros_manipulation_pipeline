"""The thing that flows between stages: ``InterfaceState`` and its scene diff.

An ``InterfaceState`` holds everything a cuRobo solve needs to continue from:
the joint configuration, a scene diff relative to the task's *base scene*
(objects added/removed, the currently-attached object, currently-disabled
collision pairs), an optional cost-so-far for ranking, and a free-form
``meta`` dict that lets stages pass per-state payloads (e.g. a sampled
target pose from GenerateGraspPose to ComputeIK).

This is the cuRobo analog of MTC's planning-scene based interface state: a
later stage learns what an earlier stage did without re-deriving it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ObjectSpec:
    """Description of an object to (re-)introduce into the base scene."""

    name: str
    shape: str = "cuboid"  # cuboid | sphere | cylinder | capsule | mesh
    pose: Optional[Any] = None  # Pose-like (message stub or ROS Pose)
    dimensions: Optional[list] = None  # [dx, dy, dz] / [r, h] / [r]
    mesh_path: Optional[str] = None  # absolute path for MESH type
    vertices: Optional[list] = None  # inline vertices (MESH, MTC Sec 6b fix)
    triangles: Optional[list] = None  # flat triangle index buffer (MESH)


@dataclass
class SceneDiff:
    """Scene changes *relative to the task's base scene*.

    A scene diff is cumulative: each state's diff describes the complete
    delta from the base scene, so a later stage knows everything an earlier
    stage did (object now attached, collisions disabled, ...).
    """

    objects_added: dict = field(default_factory=dict)  # name -> ObjectSpec
    objects_removed: list = field(default_factory=list)  # names
    attached_object: Optional[str] = None  # currently attached object name
    detached_object: Optional[str] = None  # object detached by this diff
    disabled_collisions: set = field(default_factory=set)  # (object, link)

    # ------------------------------------------------------------------
    # Mutation helpers — each returns a NEW cumulative diff.
    # ------------------------------------------------------------------
    def with_object_added(self, spec: ObjectSpec) -> "SceneDiff":
        out = self.copy()
        out.objects_added[spec.name] = spec
        if spec.name in out.objects_removed:
            out.objects_removed.remove(spec.name)
        return out

    def with_object_removed(self, name: str) -> "SceneDiff":
        out = self.copy()
        if name not in out.objects_removed:
            out.objects_removed.append(name)
        out.objects_added.pop(name, None)
        return out

    def with_attached(self, name: str) -> "SceneDiff":
        out = self.copy()
        out.attached_object = name
        out.detached_object = None
        # A picked object leaves the world and joins the hand.
        out.objects_added.pop(name, None)
        return out

    def with_detached(self, name: str) -> "SceneDiff":
        out = self.copy()
        out.detached_object = name
        if out.attached_object == name:
            out.attached_object = None
        return out

    def with_collisions(self, object_name: str, links, enabled: bool) -> "SceneDiff":
        out = self.copy()
        for link in links:
            key = (object_name, link)
            if enabled:
                out.disabled_collisions.add(key)
            else:
                out.disabled_collisions.discard(key)
        return out

    def allowed_collision_links(self, object_name: str) -> list:
        """Links currently disabled w.r.t. ``object_name`` (for goalsets)."""
        return sorted(link for (obj, link) in self.disabled_collisions
                      if obj == object_name)

    def all_allowed_links(self) -> list:
        return sorted({link for (_, link) in self.disabled_collisions})

    def copy(self) -> "SceneDiff":
        return SceneDiff(
            objects_added=dict(self.objects_added),
            objects_removed=list(self.objects_removed),
            attached_object=self.attached_object,
            detached_object=self.detached_object,
            disabled_collisions=set(self.disabled_collisions),
        )

    def key(self) -> tuple:
        """Deterministic identity key (scene is part of an InterfaceState's
        identity: two states with identical joints but a different scene —
        e.g. object now attached — are different states)."""
        added = tuple(sorted((k, repr(v)) for k, v in self.objects_added.items()))
        removed = tuple(sorted(self.objects_removed))
        disabled = tuple(sorted((o, l) for o, l in self.disabled_collisions))
        return (added, removed, self.attached_object, self.detached_object, disabled)

    def is_empty(self) -> bool:
        return not (self.objects_added or self.objects_removed or
                    self.attached_object or self.detached_object or
                    self.disabled_collisions)

    # ------------------------------------------------------------------
    # Materialization deltas — what must be pushed to the curobo server
    # when a solution chain is executed.
    # ------------------------------------------------------------------
    def ops_to_materialize(self) -> list:
        """Ordered list of (kind, payload) ops to apply on the server.

        ``kind`` in {"add", "remove", "attach", "detach"}; the executor walks
        a solution chain and applies each stage's delta in order.
        """
        ops = []
        for name, spec in self.objects_added.items():
            ops.append(("add", spec))
        for name in self.objects_removed:
            ops.append(("remove", name))
        if self.detached_object:
            ops.append(("detach", self.detached_object))
        if self.attached_object:
            ops.append(("attach", self.attached_object))
        return ops


@dataclass
class InterfaceState:
    """A configuration + scene the next stage can continue from.

    ``joint_state`` is a JointStateLike — an object with ``name``
    (list[str]) and ``position`` (list[float]); a real
    ``sensor_msgs/msg/JointState`` in ROS deployments, a simple stub in the
    unit tests.
    """

    joint_state: Any
    scene: Optional[SceneDiff] = field(default_factory=SceneDiff)
    cost: float = 0.0  # accumulated cost-so-far (for ranking)
    meta: dict = field(default_factory=dict)  # free-form per-state payload

    def clone(self, **overrides) -> "InterfaceState":
        base = dict(joint_state=self.joint_state, scene=self.scene,
                    cost=self.cost, meta=dict(self.meta))
        base.update(overrides)
        return InterfaceState(**base)

    def with_cost(self, cost: float) -> "InterfaceState":
        out = self.clone()
        out.cost = cost
        return out

    def key(self):
        """Semantic identity for dedup across compute passes.

        Includes the scene diff and per-state meta: two states with the same
        joint angles never have the same scene (an attach changes the scene
        at constant joints), and a grasp-pose candidate in ``meta`` makes a
        state distinct even when its IK solution coincides. Positions are
        rounded to 5 decimals so numerically-identical solves converge.
        """
        js = self.joint_state
        pos = tuple(round(p, 5) for p in (getattr(js, "position", None) or ()))
        scene = self.scene.key() if self.scene is not None else None
        meta = tuple(sorted((k, repr(v)) for k, v in self.meta.items()))
        return (pos, scene, meta)


def joint_positions(joint_state: Any) -> list:
    return list(getattr(joint_state, "position", []) or [])


def make_joint_state(joint_state_cls, names: list, positions: list) -> Any:
    """Build a JointStateLike from names+positions.

    ``joint_state_cls`` may be None → a lightweight stub is used (core alone
    must run with no ROS dependency).
    """
    if joint_state_cls is None:
        return JointStateStub(dict(zip(names, positions)))
    js = joint_state_cls()
    js.header = type(js).__class__  # placeholder, replaced by adapter if needed
    js.name = list(names)
    js.position = [float(p) for p in positions]
    return js


class JointStateStub:
    """Minimal sensor_msgs/JointState stand-in for the ROS-free core/tests."""

    def __init__(self, name_to_position: Optional[dict] = None,
                 names: Optional[list] = None, positions: Optional[list] = None):
        if name_to_position is not None:
            self.name = list(name_to_position.keys())
            self.position = [float(v) for v in name_to_position.values()]
        else:
            self.name = list(names or [])
            self.position = [float(v) for v in (positions or [])]
        self.velocity = []
        self.effort = []
        self.header = None

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"JointStateStub({dict(zip(self.name, self.position))})"
"""Minimal geometric helpers (ROS-free).

Poses/quaternions are duck-typed objects carrying ``position`` (x/y/z) and
``orientation`` (x/y/z/w), i.e. ``geometry_msgs/msg/Pose`` in a ROS
deployment and simple stubs in unit tests. Only the operations the built-in
stages need are implemented — no full 3D math library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Pose3:
    """Plain float pose used inside the ROS-free core."""

    position: list  # [x, y, z]
    orientation: list  # [x, y, z, w]

    @classmethod
    def from_any(cls, pose: Any) -> "Pose3":
        p = pose.position
        q = pose.orientation
        return cls([float(p.x), float(p.y), float(p.z)],
                   [float(q.x), float(q.y), float(q.z), float(q.w)])

    def copy(self) -> "Pose3":
        return Pose3(list(self.position), list(self.orientation))


def quat_rotate_vector(q: list, v: list) -> list:
    """Rotate vector v by unit quaternion q=[x,y,z,w] (Hamilton convention)."""
    x, y, z, w = q
    # t = 2 * cross(q.xyz, v)
    tx = 2.0 * (y * v[2] - z * v[1])
    ty = 2.0 * (z * v[0] - x * v[2])
    tz = 2.0 * (x * v[1] - y * v[0])
    # v' = v + w * t + cross(q.xyz, t)
    return [
        v[0] + w * tx + (y * tz - z * ty),
        v[1] + w * ty + (z * tx - x * tz),
        v[2] + w * tz + (x * ty - y * tx),
    ]


def quat_from_axis_angle(axis: list, angle: float) -> list:
    """Unit quaternion for a rotation of `angle` rad about `axis`."""
    a = math.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)
    if a < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    half = angle / 2.0
    s = math.sin(half) / a
    return [axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)]


def quat_multiply(a: list, b: list) -> list:
    """Quaternion product a * b (both [x,y,z,w])."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def pose_translate(pose: Pose3, delta: list) -> Pose3:
    """Translate pose by delta (expressed in the pose's own frame)."""
    d = quat_rotate_vector(pose.orientation, delta)
    out = pose.copy()
    out.position[0] += d[0]
    out.position[1] += d[1]
    out.position[2] += d[2]
    return out


def pose_to_any(pose: Pose3, pose_cls: Optional[Any] = None) -> Any:
    """Convert a Pose3 back into a message-like object.

    ``pose_cls`` defaults to an internal lightweight Pose stub so the core
    stays ROS-free; the ROS adapter passes ``geometry_msgs.msg.Pose``.
    """
    if pose_cls is not None:
        out = pose_cls()
        out.position.x = pose.position[0]
        out.position.y = pose.position[1]
        out.position.z = pose.position[2]
        out.orientation.x = pose.orientation[0]
        out.orientation.y = pose.orientation[1]
        out.orientation.z = pose.orientation[2]
        out.orientation.w = pose.orientation[3]
        return out
    return _PoseStub(*pose.position, *pose.orientation)


class _PoseStub:
    """Duck-typed Pose stand-in for tests (position/orientation with attrs)."""

    def __init__(self, x=0.0, y=0.0, z=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        self.position = _Vec3(x, y, z)
        self.orientation = _Quat(qx, qy, qz, qw)

    def __repr__(self):  # pragma: no cover - debugging aid
        return (f"PoseStub(pos=({self.position.x}, {self.position.y}, "
                f"{self.position.z}))")


class _Vec3:
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class _Quat:
    def __init__(self, x, y, z, w):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)
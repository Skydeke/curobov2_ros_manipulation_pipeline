#!/usr/bin/env python3
"""Frame resolution for pose-carrying requests (server side).

Convention (mirrors ``SetMask.srv``): every pose input carries the TF frame
it is expressed in; ``""`` (or the base frame itself) means the robot base
frame and needs no lookup. Any other frame is resolved into the configured
base link via the node's TF buffer. RViz clients send fixed-frame
coordinates; the server owns every transform, so RViz stays frame-dumb
(no ``base_link`` parameters on the RViz side).

All helpers are pure ``geometry_msgs/Pose`` in / out; quaternion order in
the math below is (w, x, y, z), converted at the ROS boundary.
"""

import math

import numpy as np
import rclpy.time
from rclpy.duration import Duration
from geometry_msgs.msg import Pose

try:
    from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
    _TF_ERRORS = (LookupException, ConnectivityException, ExtrapolationException)
except Exception:  # noqa: BLE001 - tf2_ros present wherever this is used
    _TF_ERRORS = (Exception,)


def _normalize_quat_wxyz(q):
    """Normalise a (w, x, y, z) quaternion; identity for an all-zero one."""
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return 1.0, 0.0, 0.0, 0.0
    return w / n, x / n, y / n, z / n


def _quat_to_matrix(q):
    """Rotation matrix for the (w, x, y, z) quaternion ``q``."""
    w, x, y, z = _normalize_quat_wxyz(q)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def _quat_multiply(a, b):
    """Hamilton product of two (w, x, y, z) quaternions (``a`` o ``b``)."""
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    return (
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
        a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
        a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
        a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0)


def apply_transform(pose: Pose, transform) -> Pose:
    """Apply a ``geometry_msgs/TransformStamped`` to a ``Pose`` (pure math).

    Returns a new ``Pose``: ``out = transform * pose``.
    """
    t = transform.transform.translation
    r = transform.transform.rotation
    tq = (r.w, r.x, r.y, r.z)
    rot = _quat_to_matrix(tq)
    p = np.array([pose.position.x, pose.position.y, pose.position.z],
                 dtype=np.float64)
    out_p = np.array([t.x, t.y, t.z], dtype=np.float64) + rot @ p
    q = (pose.orientation.w, pose.orientation.x,
         pose.orientation.y, pose.orientation.z)
    out_q = _normalize_quat_wxyz(_quat_multiply(tq, q))
    out = Pose()
    out.position.x, out.position.y, out.position.z = (float(v) for v in out_p)
    out.orientation.w, out.orientation.x, out.orientation.y, out.orientation.z = out_q
    return out


def get_transform(tf_buffer, target_frame: str, source_frame: str,
                  timeout_sec: float = 0.5):
    """Look up ``target_frame <- source_frame`` (latest available transform).

    Raises ``RuntimeError`` with a human-readable message when the lookup
    fails, so service/action callbacks can report it instead of solving in
    the wrong frame.
    """
    try:
        return tf_buffer.lookup_transform(
            target_frame, source_frame, rclpy.time.Time(),
            timeout=Duration(seconds=float(timeout_sec)))
    except _TF_ERRORS as e:
        raise RuntimeError(
            f"cannot transform '{source_frame}' -> '{target_frame}': {e}")


def resolve_to_base(pose: Pose, frame_id: str, base_link: str, tf_buffer,
                    timeout_sec: float = 0.5) -> Pose:
    """Resolve one pose into the base frame (``""`` / base passes through)."""
    src = frame_id or base_link
    if src == base_link:
        return pose
    if tf_buffer is None:
        raise RuntimeError(
            f"cannot transform '{src}' -> '{base_link}': no TF buffer")
    return apply_transform(
        pose, get_transform(tf_buffer, base_link, src, timeout_sec))


def resolve_poses_to_base(poses, frame_id: str, base_link: str, tf_buffer,
                          timeout_sec: float = 0.5):
    """Resolve a pose list into the base frame with a single TF lookup."""
    src = frame_id or base_link
    if src == base_link:
        return list(poses)
    if tf_buffer is None:
        raise RuntimeError(
            f"cannot transform '{src}' -> '{base_link}': no TF buffer")
    tf = get_transform(tf_buffer, base_link, src, timeout_sec)
    return [apply_transform(p, tf) for p in poses]

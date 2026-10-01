#!/usr/bin/env python3
"""Single source of truth for the perception cameras.

The raw depth streams that feed the perception chain (robot-segmentation filter
-> mapper) are described by ONE ROS-parameter block — node params instead of a
per-repo ``cameras.yaml`` file. Every camera field is an ARRAY indexed by
``camera_index``. ``CameraSystemManager`` declares and reads the block to build
the mapper's ``DepthMapCameraStrategy`` per camera; ``RobotSegmentation``
consumes the same fields for its own subscriptions, so the two can never point
at different topics or frames again.

Which cameras feed which pipeline is a per-camera ``camera_purpose``:
``'all'`` (default) -- the camera feeds BOTH the mapper ESDF and (when the
node's ``enable_robot_segmentation`` master switch is on) the robot
segmentation; ``'esdf'`` -- mapper only; ``'segmentation'`` -- segmentation
only. This lets e.g. a fixed scene camera map the world while a wrist camera is
used purely to keep the arm out of its own way.
"""

from dataclasses import dataclass
from typing import List, Optional

# Assumed rate when ``camera_frame_rate_hz`` is not given. Deliberately HIGH:
# this value normalizes the TSDF decay (see ObstacleManager._resolve_time_decay).
# Overestimating the rate pushes `time_decay` toward 1.0 -> forgetting is too
# slow, obstacles linger, which is the conservative direction for collision.
# Underestimating clears the map faster than reality -> real obstacles vanish,
# which is the dangerous direction. So err high.
DEFAULT_CAMERA_FRAME_RATE_HZ = 30.0

# Purposes a camera can feed. 'all' is the DEFAULT: unless you say otherwise a
# camera is used for everything it can be.
PURPOSE_ESDF = 'esdf'
PURPOSE_SEGMENTATION = 'segmentation'
PURPOSE_ALL = 'all'


def masked_depth_topic(depth_topic: str) -> str:
    """Masked-output topic of a camera's robot segmentation.

    The leaf segment of the raw depth topic is replaced by ``masked_depth``:
    ``/kortex_vision/depth/image`` -> ``/kortex_vision/depth/masked_depth``.
    A bare leaf (no directory) maps to ``masked_depth``. Returns '' for an
    empty/unset topic.
    """
    topic = depth_topic.rstrip('/') if depth_topic else ''
    if not topic:
        return ''
    idx = topic.rfind('/')
    if idx < 0:
        return 'masked_depth'
    return f'{topic[:idx]}/masked_depth'


@dataclass
class PerceptionCameraCfg:
    """One perception camera, resolved from the node's ``camera_*`` params.

    Shared by the mapper's camera strategy (``CameraSystemManager``) and the
    in-server robot segmentation (``RobotSegmentation``), so they always agree
    on the raw depth topic, the camera-info topic and the integration frame.

    Attributes:
        camera_index: Position of this camera among the ``camera_*`` arrays.
        name: Strategy name used in logs and as the CameraContext key.
        depth_topic: RAW depth stream. Input of the robot-segmentation filter,
            and the mapper's input when this camera is not segmented.
        camera_info_topic: ``CameraInfo`` topic carrying the intrinsics (used
            when ``intrinsics`` is empty).
        frame_id: Raw optical frame of the depth stream (TF fallback).
        intrinsics: 3x3 K as a 9-element row-major list, or None -> read once
            from ``camera_info_topic`` at startup.
        extrinsics: 7-element pose [x, y, z, qw, qx, qy, qz] in the base frame,
            or None -> resolved per frame via TF (base frame -> frame).
        frame_rate_hz: Declared publication rate of the depth stream; drives
            the mapper TSDF decay normalisation.
        purpose: What this camera feeds ('all' | 'esdf' | 'segmentation').
        masked_output_topic: Where the robot segmentation republishes this
            camera's masked depth stream (only meaningful when segmented).
            Derived from ``depth_topic``: its leaf segment is replaced by
            ``masked_depth`` (``masked_depth_topic``).
        mapper_topic: What the mapper's camera strategy subscribes to — the
            masked output topic when this camera is segmented and the master
            switch is on, else the raw ``depth_topic``.
    """

    camera_index: int = 0
    name: str = ''
    depth_topic: str = ''
    camera_info_topic: str = ''
    frame_id: str = ''
    intrinsics: Optional[List[float]] = None
    extrinsics: Optional[List[float]] = None
    frame_rate_hz: float = DEFAULT_CAMERA_FRAME_RATE_HZ
    purpose: str = PURPOSE_ALL
    masked_output_topic: str = ''
    mapper_topic: str = ''

    @property
    def for_esdf(self) -> bool:
        """True when this camera feeds the mapper (purpose 'all' or 'esdf')."""
        return self.purpose in (PURPOSE_ESDF, PURPOSE_ALL)

    @property
    def for_segmentation(self) -> bool:
        """True when this camera is robot-segmented (purpose 'all' or 'segmentation')."""
        return self.purpose in (PURPOSE_SEGMENTATION, PURPOSE_ALL)

    # The per-camera ``camera_*`` param block, one array entry per camera.
    # Declared once (by CameraSystemManager) and read back by the segmenter too.
    PARAM_TOPIC = 'camera_topic'
    PARAM_CAMERA_INFO = 'camera_info_topic'
    PARAM_FRAME = 'camera_frame'
    PARAM_INTRINSICS = 'camera_intrinsics'
    PARAM_EXTRINSICS = 'camera_extrinsics'
    PARAM_FRAME_RATE = 'camera_frame_rate_hz'
    PARAM_PURPOSE = 'camera_purpose'

    @classmethod
    def declare(cls, node) -> None:
        """Declare the ``camera_*`` array params with their defaults, if absent.

        Defaults are non-empty STRING/DOUBLE arrays (e.g. ``['']``, ``[0.0]``)
        so the declared type is unambiguous to rclpy and stays compatible with
        the launch file's array override.
        """
        _declare_param(node, cls.PARAM_TOPIC, [''])
        _declare_param(node, cls.PARAM_CAMERA_INFO, [''])
        _declare_param(node, cls.PARAM_FRAME, [''])
        _declare_param(node, cls.PARAM_INTRINSICS, [''])
        _declare_param(node, cls.PARAM_EXTRINSICS, [''])
        _declare_param(node, cls.PARAM_FRAME_RATE, [0.0])
        _declare_param(node, cls.PARAM_PURPOSE, [PURPOSE_ALL])

    @classmethod
    def num_cameras(cls, node) -> int:
        """Number of configured cameras = length of the ``camera_topic`` array.

        A scalar (single-camera) value counts as one. An empty topic array
        entry still occupies a slot (the camera simply stays inactive).
        """
        if not node.has_parameter(cls.PARAM_TOPIC):
            return 0
        try:
            value = node.get_parameter(cls.PARAM_TOPIC).value
        except Exception:
            return 0
        if isinstance(value, list):
            return len(value)
        return 1 if value else 0

    @classmethod
    def from_node(cls, node, camera_index: int = 0) -> 'PerceptionCameraCfg':
        """Read this camera's entry of the ``camera_*`` arrays and derive its
        mapper input topic. Never raises: missing entries fall back to their
        defaults (no topic -> inactive camera).

        Args:
            node: The ROS2 node carrying the parameters.
            camera_index: Index into the ``camera_*`` arrays for this camera.
        """
        depth_topic = _entry(node, cls.PARAM_TOPIC, camera_index, '') or ''
        camera_info_topic = (_entry(node, cls.PARAM_CAMERA_INFO, camera_index, '')
                             or '')
        frame_id = _entry(node, cls.PARAM_FRAME, camera_index, '') or ''
        intrinsics = _parse_floats(
            _entry(node, cls.PARAM_INTRINSICS, camera_index, ''))
        extrinsics = _parse_floats(
            _entry(node, cls.PARAM_EXTRINSICS, camera_index, ''))
        rate_raw = _entry(node, cls.PARAM_FRAME_RATE, camera_index, 0.0)
        try:
            rate = float(rate_raw or 0.0)
        except (TypeError, ValueError):
            rate = 0.0
        if rate <= 0.0:
            rate = DEFAULT_CAMERA_FRAME_RATE_HZ

        purpose = (_entry(node, cls.PARAM_PURPOSE, camera_index, PURPOSE_ALL)
                   or PURPOSE_ALL).lower().strip()
        if purpose not in (PURPOSE_ALL, PURPOSE_ESDF, PURPOSE_SEGMENTATION):
            purpose = PURPOSE_ALL

        # Masked output topic is DERIVED from the camera's own raw depth topic:
        # the leaf segment is stripped and republished as `masked_depth` in the
        # same directory (e.g. /kortex_vision/depth/image ->
        # /kortex_vision/depth/masked_depth), so every segmented camera has a
        # unique, per-camera output with no extra configuration.
        masked_output_topic = masked_depth_topic(depth_topic)

        # The mapper consumes the masked output while this camera is segmented
        # AND the master switch is on, otherwise the raw depth stream directly.
        # Derived here so the camera and the segmenter agree by construction.
        if (bool(_param(node, 'enable_robot_segmentation', True))
                and depth_topic and purpose in (PURPOSE_SEGMENTATION, PURPOSE_ALL)):
            mapper_topic = masked_output_topic
        else:
            mapper_topic = depth_topic

        return cls(
            camera_index=camera_index,
            name=f'depth_camera_{camera_index}',
            depth_topic=depth_topic,
            camera_info_topic=camera_info_topic,
            frame_id=frame_id,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            frame_rate_hz=rate,
            purpose=purpose,
            masked_output_topic=masked_output_topic,
            mapper_topic=mapper_topic,
        )


def _declare_param(node, name, default):
    if not node.has_parameter(name):
        node.declare_parameter(name, default)


def _param(node, name, default=None):
    if not node.has_parameter(name):
        return default
    try:
        return node.get_parameter(name).value
    except Exception:
        return default


def _entry(node, name, index, default):
    """Entry ``index`` of the array param ``name``; falls back to ``default``.

    A scalar value (single-camera convenience) is treated as a 1-entry array.
    Emptiness of a string entry means "unset".
    """
    if not node.has_parameter(name):
        return default
    try:
        value = node.get_parameter(name).value
    except Exception:
        return default
    if not isinstance(value, list):
        value = [value]
    try:
        entry = value[index]
    except IndexError:
        return default
    if isinstance(entry, str) and not entry.strip():
        return default
    if entry is None:
        return default
    return entry


def _parse_floats(text):
    """Parse a comma-separated float string; None when empty/unparsable.

    Accepts 'fx,fy,cx,cy' (4) or the full row-major K 'fx,0,cx,0,fy,cy,0,0,1'
    for intrinsics, and 'x,y,z,qw,qx,qy,qz' (7) for extrinsics. Bad entries
    degrade to None (read from camera_info / TF) rather than crash launch.
    """
    if text is None or isinstance(text, (list, tuple)):
        return None
    if isinstance(text, float) or isinstance(text, int):
        return None
    try:
        parts = [p.strip() for p in text.split(',') if p.strip()]
    except AttributeError:
        return None
    if not parts:
        return None
    try:
        floats = [float(p) for p in parts]
    except ValueError:
        return None
    return floats
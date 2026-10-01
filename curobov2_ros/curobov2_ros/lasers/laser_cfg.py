#!/usr/bin/env python3
"""Single source of truth for the perception LiDAR/laser sensors.

The range-image streams that feed the Mapper TSDF alongside the depth cameras
are described by one per-sensor ROS-parameter block — node params instead of a
per-repo ``lasers.yaml`` file.  Every laser field is an ARRAY indexed by
``laser_index``.  ``LaserSystemManager`` declares and reads the block to build
the ``PointCloudLaserStrategy`` per sensor; the parameters are consumed by no
other component (unlike cameras, lasers have no robot-segmentation sidecar).

The range image H/W and the Mapper lidar integration are governed by the
**global** ``laser_image_height`` / ``laser_image_width`` params (non-array):
curobo requires a single shared resolution for all sensors, and the Mapper's
projective buffer is sized once from these values.

Assumptions
-----------
- Point cloud data arrives as ``sensor_msgs/PointCloud2``.
- All configured lasers share the same ``laser_image_height`` ×
  ``laser_image_width`` (validated by ``LaserSystemManager``).
"""

from dataclasses import dataclass
from typing import List, Optional

# Deliberately HIGH: normalises the TSDF decay (see ObstacleManager._resolve_time_decay).
# Overestimating the rate pushes time_decay toward 1.0 → forgetting is too slow,
# obstacles linger — the conservative direction for collision. Underestimating
# clears the map faster than reality → real obstacles vanish → dangerous.
DEFAULT_LASER_FRAME_RATE_HZ = 10.0


@dataclass
class PerceptionLaserCfg:
    """One perception LiDAR/laser, resolved from the node's ``laser_*`` params.

    Attributes:
        laser_index: Position of this laser among the ``laser_*`` arrays.
        name: Strategy name used in logs and as the LaserContext key.
        topic: ``PointCloud2`` topic for this laser.
        frame_id: Sensor TF frame (TF lookup: ``base_frame → frame_id``).
        extrinsics: 7-element pose [x, y, z, qw, qx, qy, qz] in the base
            frame, or ``None`` → resolved per frame via TF.
        laser_type: ROS message type of ``topic`` — ``'pointcloud'``
            (``sensor_msgs/PointCloud2``, default) or ``'scan'``
            (``sensor_msgs/LaserScan``).
        frame_rate_hz: Declared publication rate of the point cloud; drives
            the mapper TSDF decay normalisation.
        range_min_m: Minimum valid range (m); points closer are discarded.
        range_max_m: Maximum valid range (m); points farther are discarded.
        elevation_min_rad: Lower bound of the vertical field-of-view (rad).
            For planar scans (``laser_image_height == 1``), both min and max
            must be equal (set to 0.0).
        elevation_max_rad: Upper bound of the vertical field-of-view (rad).
    """

    laser_index: int = 0
    name: str = ''
    topic: str = ''
    frame_id: str = ''
    extrinsics: Optional[List[float]] = None
    laser_type: str = 'pointcloud'
    frame_rate_hz: float = DEFAULT_LASER_FRAME_RATE_HZ
    range_min_m: float = 0.1
    range_max_m: float = 10.0
    elevation_min_rad: float = -1.5707963  # -π/2
    elevation_max_rad: float = 1.5707963   # +π/2

    # ---- Parameter names (single source of truth) ----

    PARAM_TOPIC = 'laser_topic'
    PARAM_FRAME = 'laser_frame'
    PARAM_EXTRINSICS = 'laser_extrinsics'
    PARAM_LASER_TYPE = 'laser_type'
    PARAM_FRAME_RATE = 'laser_frame_rate_hz'
    PARAM_RANGE_MIN = 'laser_range_min_m'
    PARAM_RANGE_MAX = 'laser_range_max_m'
    PARAM_ELEVATION_MIN = 'laser_elevation_min_rad'
    PARAM_ELEVATION_MAX = 'laser_elevation_max_rad'

    # Global params (non-array) — declared alongside the per-sensor block so
    # everything is declared in one place.
    PARAM_IMAGE_HEIGHT = 'laser_image_height'
    PARAM_IMAGE_WIDTH = 'laser_image_width'

    @classmethod
    def declare(cls, node) -> None:
        """Declare the ``laser_*`` array params **and** global H/W with defaults.

        Defaults are non-empty arrays (e.g. ``['']``, ``[0.0]``) so the
        declared type is unambiguous to rclpy and compatible with the launch
        file's array override.
        """
        _declare_param(node, cls.PARAM_TOPIC, [''])
        _declare_param(node, cls.PARAM_FRAME, [''])
        _declare_param(node, cls.PARAM_EXTRINSICS, [''])
        _declare_param(node, cls.PARAM_LASER_TYPE, ['pointcloud'])
        _declare_param(node, cls.PARAM_FRAME_RATE, [0.0])
        _declare_param(node, cls.PARAM_RANGE_MIN, [0.0])
        _declare_param(node, cls.PARAM_RANGE_MAX, [0.0])
        _declare_param(node, cls.PARAM_ELEVATION_MIN, [0.0])
        _declare_param(node, cls.PARAM_ELEVATION_MAX, [0.0])

        # Global H/W — int params; 0 = no lidar integration.
        if not node.has_parameter(cls.PARAM_IMAGE_HEIGHT):
            node.declare_parameter(cls.PARAM_IMAGE_HEIGHT, 0)
        if not node.has_parameter(cls.PARAM_IMAGE_WIDTH):
            node.declare_parameter(cls.PARAM_IMAGE_WIDTH, 0)

    @classmethod
    def num_lasers(cls, node) -> int:
        """Number of configured lasers = length of the ``laser_topic`` array.

        A scalar (single-laser) value counts as one. An empty topic entry
        still occupies a slot (the laser simply stays inactive).
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
    def get_image_height(cls, node) -> int:
        """Global ``laser_image_height`` param (0 = disabled)."""
        if not node.has_parameter(cls.PARAM_IMAGE_HEIGHT):
            return 0
        try:
            return int(node.get_parameter(cls.PARAM_IMAGE_HEIGHT).value)
        except Exception:
            return 0

    @classmethod
    def get_image_width(cls, node) -> int:
        """Global ``laser_image_width`` param (0 = disabled)."""
        if not node.has_parameter(cls.PARAM_IMAGE_WIDTH):
            return 0
        try:
            return int(node.get_parameter(cls.PARAM_IMAGE_WIDTH).value)
        except Exception:
            return 0

    @classmethod
    def from_node(cls, node, laser_index: int = 0) -> 'PerceptionLaserCfg':
        """Read this laser's entry of the ``laser_*`` arrays.  Never raises:
        missing entries fall back to their defaults (no topic → inactive laser).

        Args:
            node: The ROS2 node carrying the parameters.
            laser_index: Index into the ``laser_*`` arrays for this laser.
        """
        topic = _entry(node, cls.PARAM_TOPIC, laser_index, '') or ''
        frame_id = _entry(node, cls.PARAM_FRAME, laser_index, '') or ''
        extrinsics = _parse_floats(
            _entry(node, cls.PARAM_EXTRINSICS, laser_index, ''))
        laser_type = _entry(node, cls.PARAM_LASER_TYPE, laser_index, 'pointcloud') or 'pointcloud'
        if laser_type not in ('pointcloud', 'scan'):
            laser_type = 'pointcloud'

        rate_raw = _entry(node, cls.PARAM_FRAME_RATE, laser_index, 0.0)
        try:
            rate = float(rate_raw or 0.0)
        except (TypeError, ValueError):
            rate = 0.0
        if rate <= 0.0:
            rate = DEFAULT_LASER_FRAME_RATE_HZ

        range_min = _float_or(node, cls.PARAM_RANGE_MIN, laser_index, 0.1)
        range_max = _float_or(node, cls.PARAM_RANGE_MAX, laser_index, 10.0)
        elev_min = _float_or(node, cls.PARAM_ELEVATION_MIN, laser_index, -1.5707963)
        elev_max = _float_or(node, cls.PARAM_ELEVATION_MAX, laser_index, 1.5707963)

        return cls(
            laser_index=laser_index,
            name=f'laser_{laser_index}',
            topic=topic,
            frame_id=frame_id,
            extrinsics=extrinsics,
            laser_type=laser_type,
            frame_rate_hz=rate,
            range_min_m=range_min,
            range_max_m=range_max,
            elevation_min_rad=elev_min,
            elevation_max_rad=elev_max,
        )


# -------------------------------------------------------------------
# Helpers (shared with laser_strategy if needed)
# -------------------------------------------------------------------

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

    A scalar value (single-laser convenience) is treated as a 1-entry array.
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


def _float_or(node, name, index, default):
    """Read a float array entry, returning *default* on any failure."""
    raw = _entry(node, name, index, default)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _parse_floats(text):
    """Parse a comma-separated float string; ``None`` when empty/unparsable.

    Accepts ``x,y,z,qw,qx,qy,qz`` (7) for extrinsics.
    Bad entries degrade to ``None`` (fall back to TF) rather than crash launch.
    """
    if text is None or isinstance(text, (list, tuple)):
        return None
    if isinstance(text, (float, int)):
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

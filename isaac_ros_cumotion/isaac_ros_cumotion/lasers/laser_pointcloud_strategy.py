#!/usr/bin/env python3

import math
import threading
from collections import deque

import numpy as np
from sensor_msgs.msg import PointCloud2

from isaac_ros_cumotion.lasers.laser_strategy import LaserStrategy


def finite_range_mask(xyz: np.ndarray, range_min: float, range_max: float):
    """Reference port: ``examples/reference/lidar_volumetric_mapping.py``.

    Returns:
        mask: bool (N,) — finite points (xyz and range) within ``[range_min,
            range_max]`` (inclusive).
        ranges: float32 (N,) — Euclidean distance of every input point.
    """
    ranges = np.linalg.norm(xyz, axis=1).astype(np.float32)
    mask = np.isfinite(xyz).all(axis=1) & np.isfinite(ranges)
    mask &= (ranges >= range_min) & (ranges <= range_max)
    return mask, ranges


def make_range_image(
    xyz: np.ndarray,
    *,
    image_height: int,
    image_width: int,
    range_min: float,
    range_max: float,
    elevation_min_rad: float,
    elevation_max_rad: float,
):
    """Project (N, 3) points into a LiDAR range image.

    Reference port of ``examples/reference/lidar_volumetric_mapping.py``
    ``make_range_image`` (geometry-only: no color / intensity / features, so the
    RGB buffer stays zero).  Column u is azimuthal, row is elevation; the
    nearest point wins each bilinear candidate bin.

    Returns:
        range_image: float32 (H, W) — Euclidean distance, 0.0 = no measurement.
        rgb_image: uint8 (H, W, 3) — zeros (geometry-only integration).
        projected_pixels: int — number of points that landed in a bin.
    """
    h, w = image_height, image_width

    mask, ranges = finite_range_mask(xyz, range_min, range_max)
    xy_norm = np.linalg.norm(xyz[:, :2], axis=1)
    elevation = np.arctan2(xyz[:, 2], xy_norm)
    mask &= (elevation >= elevation_min_rad) & (elevation <= elevation_max_rad)

    if not mask.any():
        return (
            np.zeros((h, w), dtype=np.float32),
            np.zeros((h, w, 3), dtype=np.uint8),
            0,
        )

    points = xyz[mask]
    ranges = ranges[mask]
    elevation = elevation[mask]
    source_idx = np.arange(points.shape[0], dtype=np.int64)

    azimuth = np.arctan2(points[:, 1], points[:, 0])
    u = np.mod((azimuth + math.pi) * (w / (2.0 * math.pi)), w)
    if h == 1:
        v = np.zeros_like(u)
    else:
        v = (
            (elevation_max_rad - elevation)
            * ((h - 1.0) / (elevation_max_rad - elevation_min_rad))
        )

    u0 = np.floor(u).astype(np.int64)
    u0 = np.mod(u0, w)
    u1 = np.mod(u0 + 1, w)
    if h == 1:
        v0 = np.zeros_like(u0)
        candidate_flat = np.concatenate((u0, u1))
        candidate_source = np.concatenate((source_idx, source_idx))
    else:
        v0 = np.floor(v).astype(np.int64)
        v0 = np.clip(v0, 0, h - 1)
        v1 = np.clip(v0 + 1, 0, h - 1)
        candidate_flat = np.concatenate(
            (
                v0 * w + u0,
                v0 * w + u1,
                v1 * w + u0,
                v1 * w + u1,
            )
        )
        candidate_source = np.concatenate((source_idx, source_idx, source_idx, source_idx))

    candidate_ranges = ranges[candidate_source]
    order = np.lexsort((candidate_ranges, candidate_flat))
    flat_sorted = candidate_flat[order]
    first = np.empty(flat_sorted.shape[0], dtype=bool)
    first[0] = True
    first[1:] = flat_sorted[1:] != flat_sorted[:-1]
    winner_candidates = order[first]
    winner_source = candidate_source[winner_candidates]
    winner_flat = candidate_flat[winner_candidates]

    range_flat = np.zeros(h * w, dtype=np.float32)
    range_flat[winner_flat] = ranges[winner_source]

    return (
        range_flat.reshape(h, w),
        np.zeros((h, w, 3), dtype=np.uint8),
        int(winner_source.size),
    )


class PointCloudLaserStrategy(LaserStrategy):
    """Subscribes to a ``sensor_msgs/PointCloud2`` lidar topic, converts the
    decoded points into a LiDAR range image (reference-port of
    ``examples/reference/lidar_volumetric_mapping.py``) and hands the result to
    the ``LaserContext`` batch integration worker.

    All pre-integration work (decode, TF lookup, projection) runs on a dedicated
    worker thread, mirroring ``DepthMapCameraStrategy``; the worker always
    consumes the FRESHEST frame (bounded drop-oldest deque), so a slow
    projector never builds a stale backlog.
    """

    def __init__(
        self,
        node,
        laser_name,
        topic,
        frame_id='',
        extrinsics=None,
        context=None,
        image_height: int = 1,
        image_width: int = 360,
        range_min_m: float = 0.1,
        range_max_m: float = 10.0,
        elevation_min_rad: float = 0.0,
        elevation_max_rad: float = 0.0,
        callback_group=None,
    ):
        super().__init__(node, laser_name, topic, frame_id, extrinsics)

        self._context = context
        self._image_height = image_height
        self._image_width = image_width
        self._range_min = range_min_m
        self._range_max = range_max_m
        self._elevation_min = elevation_min_rad
        self._elevation_max = elevation_max_rad

        self.sub = node.create_subscription(
            PointCloud2, topic, self._callback_pointcloud, 1,
            callback_group=callback_group)

        # CPU work (decode + TF + projection) on a dedicated worker so the
        # executor never blocks on a large point cloud.
        self._integrate_queue: deque = deque(maxlen=1)
        self._integrate_stop = threading.Event()
        self._integrate_thread = threading.Thread(
            target=self._integrate_loop, name=f'{laser_name}_integrate', daemon=True)
        self._integrate_thread.start()

        node.get_logger().info(
            f"[{self.name}] PointCloudLaserStrategy subscribed to {topic} "
            f"(HxW={image_height}x{image_width}, "
            f"range=[{range_min_m:.2f}, {range_max_m:.2f}] m)")

    # ------------------------------------------------------------------
    # Subscriber callback (CPU-only, runs on the executor thread)
    # ------------------------------------------------------------------

    def _callback_pointcloud(self, msg: PointCloud2):
        """Decode PointCloud2 into numpy (M, 3) xyz and resolve the TF pose."""
        try:
            import ros2_numpy as rnp

            arr = rnp.point_cloud2.pointcloud2_to_array(msg)
            if arr.shape[0] == 0:
                return
            xyz = np.column_stack([arr['x'], arr['y'], arr['z']]).astype(np.float32)

            # Resolve the sensor pose as a plain CPU 7-list; GPU conversion is
            # left to the context (TF buffer is not thread-safe).
            ok, pose_list = self._resolve_pose_7(msg.header.stamp)
            if not ok:
                return

            self._integrate_queue.append((xyz, pose_list))

        except Exception as e:
            self.node.get_logger().error(f'[{self.name}] PointCloud callback error: {e}')

    # ------------------------------------------------------------------
    # Dedicated worker (projection only — no CUDA on this thread)
    # ------------------------------------------------------------------

    def _integrate_loop(self):
        """Pop the freshest queued cloud, project it, publish to the context."""
        while True:
            try:
                xyz, pose_list = self._integrate_queue.pop()
            except IndexError:
                if self._integrate_stop.wait(0.005):
                    return
                continue

            try:
                range_img, rgb_img, projected = make_range_image(
                    xyz,
                    image_height=self._image_height,
                    image_width=self._image_width,
                    range_min=self._range_min,
                    range_max=self._range_max,
                    elevation_min_rad=self._elevation_min,
                    elevation_max_rad=self._elevation_max,
                )
                if self._context is not None:
                    self._context.publish_frame(
                        self.name, range_img, rgb_img, pose_list)
            except Exception as e:
                self.node.get_logger().error(f'[{self.name}] Projection failed: {e}')

    def destroy(self):
        """Signal the integration worker to stop."""
        self._integrate_stop.set()
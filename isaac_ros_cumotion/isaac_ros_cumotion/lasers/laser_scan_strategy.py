#!/usr/bin/env python3

import threading
from collections import deque

import numpy as np
from sensor_msgs.msg import LaserScan

from isaac_ros_cumotion.lasers.laser_strategy import LaserStrategy
from isaac_ros_cumotion.lasers.laser_pointcloud_strategy import make_range_image


class LaserScanLaserStrategy(LaserStrategy):
    """Subscribes to a ``sensor_msgs/LaserScan`` (planar 2D) lidar topic and
    hands the result to the ``LaserContext`` batch integration worker.

    A LaserScan is a 1-row "cloud": N samples at uniform angular steps in the
    sensor frame. Each valid range is unwrapped into a sensor-frame point
    ``[r·cos(θ), r·sin(θ), 0]`` and fed through the exact same
    ``make_range_image`` reference projection as ``PointCloudLaserStrategy``, so
    both message types share one semantics. The scan lands on the single
    elevation row if the shared grid is planar (H==1) or on whatever row the
    sensor's configured elevation range maps 0 rad to when H > 1.

    Like ``PointCloudLaserStrategy``, all pre-integration work (decode, TF
    lookup, projection) runs on a dedicated worker thread using the FRESHEST
    frame (bounded drop-oldest deque).
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
            LaserScan, topic, self._callback_laserscan, 1,
            callback_group=callback_group)

        # CPU work (decode + TF + projection) on a dedicated worker so the
        # executor never blocks (mirrors PointCloudLaserStrategy).
        self._integrate_queue: deque = deque(maxlen=1)
        self._integrate_stop = threading.Event()
        self._integrate_thread = threading.Thread(
            target=self._integrate_loop, name=f'{laser_name}_integrate', daemon=True)
        self._integrate_thread.start()

        node.get_logger().info(
            f"[{self.name}] LaserScanLaserStrategy subscribed to {topic} "
            f"(HxW={image_height}x{image_width}, "
            f"range=[{range_min_m:.2f}, {range_max_m:.2f}] m)")

    # ------------------------------------------------------------------
    # Subscriber callback (CPU-only, runs on the executor thread)
    # ------------------------------------------------------------------

    def _callback_laserscan(self, msg: LaserScan):
        """Unwrap the scan's ranges into sensor-frame (M, 3) points."""
        try:
            ranges = np.asarray(msg.ranges, dtype=np.float32)
            if ranges.size == 0:
                return
            # Drop invalid (NaN/±inf) range samples up front — building xyz
            # from them would push non-finite points into the projector.
            finite = np.isfinite(ranges)
            valid = ranges[finite]
            if valid.size == 0:
                return

            angles = (
                msg.angle_min
                + np.arange(ranges.size, dtype=np.float32) * msg.angle_increment
            )[finite]
            xyz = np.column_stack([
                valid * np.cos(angles),
                valid * np.sin(angles),
                np.zeros(valid.size, dtype=np.float32),
            ]).astype(np.float32)

            ok, pose_list = self._resolve_pose_7(msg.header.stamp)
            if not ok:
                return

            self._integrate_queue.append((xyz, pose_list))

        except Exception as e:
            self.node.get_logger().error(f'[{self.name}] LaserScan callback error: {e}')

    # ------------------------------------------------------------------
    # Dedicated worker (projection only — no CUDA on this thread)
    # ------------------------------------------------------------------

    def _integrate_loop(self):
        """Pop the freshest queued scan, project it, publish to the context."""
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
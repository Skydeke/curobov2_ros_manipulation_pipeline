#!/usr/bin/env python3

import threading
from typing import Dict, List, Optional

import torch
import numpy as np

from curobo.types import Pose, LidarObservation
from isaac_ros_cumotion.lasers.laser_strategy import LaserStrategy
from isaac_ros_cumotion.lasers.laser_pointcloud_strategy import PointCloudLaserStrategy
from isaac_ros_cumotion.lasers.laser_scan_strategy import LaserScanLaserStrategy


class LaserContext:
    """Manages the LiDAR/laser strategies and their shared Mapper integration.

    Each ``LaserStrategy`` (PointCloud2 or LaserScan) converts its data into a
    range image (reference port of
    ``examples/reference/lidar_volumetric_mapping.py``) and publishes the
    freshest frame via ``publish_frame()``.  A single batch thread here merges the freshest frame of EVERY configured laser into one
    ``LidarObservation`` — leading dimension ``num_lasers``, exactly what the
    Mapper's ``lidar_num_sensors`` expects — and calls
    ``mapper.integrate(lidar_observation=...)`` under the node's ``gpu_lock``.

    The batch thread wakes on every publication and integrates whenever *any*
    laser has new data, reusing the last known frame from the others (drop-
    oldest semantics: the newest frame per laser always overwrites the old).
    Integration starts only once every configured laser has delivered at least
    one frame; until then a warning lists the missing sensors.
    """

    def __init__(self, node, image_height: int, image_width: int):
        """
        Args:
            node: ROS2 node.
            image_height: Global range-image height (shared by all lasers).
            image_width: Global range-image width (shared by all lasers).
        """
        self.node = node
        self.image_height = image_height
        self.image_width = image_width
        self._device = torch.device('cuda')

        self.lasers: Dict[str, LaserStrategy] = {}
        self.laser_frame_rates: Dict[str, float] = {}
        self._static_poses_7: Dict[str, Optional[List[float]]] = {}
        # Resolved PerceptionLaserCfg per laser (set by LaserSystemManager).
        self._laser_cfgs: Dict[str, 'PerceptionLaserCfg'] = {}

        # Per-laser freshest projected frame + publication counter.
        self._frames: Dict[str, tuple] = {}
        self._frame_versions: Dict[str, int] = {}
        self._last_integrated_versions: Dict[str, int] = {}

        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._batch_thread = threading.Thread(
            target=self._integrate_loop, name='laser_batch_integrate', daemon=True)
        self._batch_thread.start()

    def add_laser(
        self,
        name: str,
        laser_type: str,
        topic: str,
        frame_id: str,
        extrinsics=None,
        frame_rate_hz: float = 10.0,
        range_min_m: float = 0.1,
        range_max_m: float = 10.0,
        elevation_min_rad: float = 0.0,
        elevation_max_rad: float = 0.0,
        callback_group=None,
        **kwargs,
    ):
        """Create and register one laser strategy.

        Args:
            name: Unique laser identifier.
            laser_type: ``'pointcloud'`` (``sensor_msgs/PointCloud2``) or
                ``'scan'`` (``sensor_msgs/LaserScan``).
            topic: Sensor topic.
            frame_id: Sensor TF frame.
            extrinsics: 7-element pose list [x,y,z,qw,qx,qy,qz] or ``None``.
            frame_rate_hz: Declared publication rate (decay normalisation).
            range_min_m / range_max_m: Valid range bounds for the range image.
            elevation_min_rad / elevation_max_rad: Elevation FOV bounds.
            callback_group: ``rclpy`` callback group for the subscription.
        """
        if laser_type not in ('pointcloud', 'scan'):
            self.node.get_logger().error(
                f"Unknown/unsupported laser type: {laser_type} "
                f"(supported: 'pointcloud' (PointCloud2) or 'scan' (LaserScan))")
            return

        if laser_type == 'scan':
            strategy = LaserScanLaserStrategy(
                node=self.node,
                laser_name=name,
                topic=topic,
                frame_id=frame_id,
                extrinsics=extrinsics,
                context=self,
                image_height=self.image_height,
                image_width=self.image_width,
                range_min_m=range_min_m,
                range_max_m=range_max_m,
                elevation_min_rad=elevation_min_rad,
                elevation_max_rad=elevation_max_rad,
                callback_group=callback_group,
            )
        else:
            strategy = PointCloudLaserStrategy(
                node=self.node,
                laser_name=name,
                topic=topic,
                frame_id=frame_id,
                extrinsics=extrinsics,
                context=self,
                image_height=self.image_height,
                image_width=self.image_width,
                range_min_m=range_min_m,
                range_max_m=range_max_m,
                elevation_min_rad=elevation_min_rad,
                elevation_max_rad=elevation_max_rad,
                callback_group=callback_group,
            )

        self.lasers[name] = strategy
        self.laser_frame_rates[name] = float(frame_rate_hz)

        # Static pose (7-list) for the batch assembler when the strategy worker
        # sends ``pose_list=None`` (static-extrinsics mode).
        static = strategy.camera_pose_static
        if static is not None:
            self._static_poses_7[name] = (
                [float(static.position[0, 0]), float(static.position[0, 1]),
                 float(static.position[0, 2]),
                 float(static.quaternion[0, 0]), float(static.quaternion[0, 1]),
                 float(static.quaternion[0, 2]), float(static.quaternion[0, 3])])
        else:
            self._static_poses_7[name] = None

        self._frames[name] = None
        self._frame_versions[name] = -1  # no valid frame yet
        self._last_integrated_versions[name] = -2

        self.node.get_logger().info(
            f"Added laser strategy '{name}' of type '{laser_type}', "
            f"rate={frame_rate_hz} Hz")

    def set_laser_cfgs(self, cfgs: dict):
        """Store the resolved ``PerceptionLaserCfg`` objects keyed by name.

        Used by the batch assembler for per-laser ``valid_range_m`` /
        ``elevation_range_rad``.
        """
        self._laser_cfgs = dict(cfgs)

    def publish_frame(self, name: str, range_img: np.ndarray,
                      rgb_img: np.ndarray, pose_7: Optional[List[float]]):
        """Register the freshest projected frame for a laser and wake the
        batch thread (drop-oldest: the previous frame for this laser is lost)."""
        with self._cond:
            self._frames[name] = (range_img, rgb_img, pose_7)
            self._frame_versions[name] = self._frame_versions.get(name, -1) + 1
            self._cond.notify()

    def get_total_frame_rate_hz(self) -> float:
        """Sum of the declared laser publication rates (Hz).

        Used by ``ObstacleManager._resolve_time_decay``: the TSDF decay fires
        once per ``integrate()`` call, so the combined camera+laser rate is the
        effective decay rate.
        """
        return float(sum(self.laser_frame_rates.values()))

    def get_laser_names(self) -> List[str]:
        return list(self.lasers.keys())

    # ------------------------------------------------------------------
    # Batch integration (single thread — all lasers, one integrate() call)
    # ------------------------------------------------------------------

    def _integrate_loop(self):
        """Assemble a batched ``LidarObservation`` from every laser's freshest
        frame and push it into the Mapper under ``gpu_lock``."""
        while True:
            with self._cond:
                while not self._stop.is_set():
                    any_new = any(
                        self._frame_versions.get(n, -1)
                        > self._last_integrated_versions.get(n, -2)
                        for n in self._frame_versions)
                    if any_new:
                        break
                    self._cond.wait(timeout=0.05)
                if self._stop.is_set():
                    return
                # Snapshot under the lock (strategy workers may mutate _frames).
                frames = {n: self._frames[n] for n in self._frames}
                versions = {n: self._frame_versions[n] for n in self._frame_versions}

            try:
                self._maybe_integrate(frames, versions)
            except Exception as e:
                self.node.get_logger().error(f'Laser batch integrate failed: {e}')

    def _maybe_integrate(self, frames: dict, versions: dict):
        """Integrate when all lasers have ≥1 frame and at least one is new."""
        laser_names = list(self.lasers.keys())
        if not laser_names:
            return

        missing = [n for n in laser_names if frames.get(n) is None]
        if missing:
            self.node.get_logger().warn(
                f'Waiting for laser(s) {missing}: perception inactive until '
                f'all configured lasers have published at least once.',
                throttle_duration_sec=5.0)
            return

        any_new = any(
            versions.get(n, -1) > self._last_integrated_versions.get(n, -2)
            for n in laser_names)
        if not any_new:
            return

        mapper = getattr(self.node, 'mapper', None)
        if mapper is None:
            return

        gpu_lock = getattr(self.node, 'gpu_lock', None)
        if gpu_lock is not None and not gpu_lock.acquire(blocking=False):
            return  # CUDA graph capture in progress — drop this batch

        try:
            self._build_and_integrate(laser_names, frames, mapper)
            self._last_integrated_versions = dict(versions)
        finally:
            if gpu_lock is not None:
                gpu_lock.release()

    def _build_and_integrate(self, laser_names: list, frames: dict, mapper):
        """Stack the per-laser frames into a batched ``LidarObservation`` and
        ``mapper.integrate(lidar_observation=...)``."""
        N = len(laser_names)
        H, W = self.image_height, self.image_width

        range_np = np.zeros((N, H, W), dtype=np.float32)
        rgb_np = np.zeros((N, H, W, 3), dtype=np.uint8)
        positions = np.zeros((N, 3), dtype=np.float32)
        quaternions = np.zeros((N, 4), dtype=np.float32)
        valid_range = np.zeros((N, 2), dtype=np.float32)
        elevation_range = np.zeros((N, 2), dtype=np.float32)

        for i, name in enumerate(laser_names):
            range_img, rgb_img, pose_7 = frames[name]
            range_np[i] = range_img
            rgb_np[i] = rgb_img

            if pose_7 is not None:
                pose = pose_7
            elif self._static_poses_7.get(name) is not None:
                pose = self._static_poses_7[name]
            else:
                self.node.get_logger().error(
                    f'[{name}] No pose available — using identity.')
                pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            positions[i] = [pose[0], pose[1], pose[2]]
            quaternions[i] = [pose[3], pose[4], pose[5], pose[6]]

            cfg = self._laser_cfgs.get(name)
            if cfg is not None:
                valid_range[i] = [cfg.range_min_m, cfg.range_max_m]
                elevation_range[i] = [cfg.elevation_min_rad, cfg.elevation_max_rad]

        range_tensor = torch.from_numpy(range_np).to(
            device=self._device, dtype=torch.float32)
        rgb_tensor = torch.from_numpy(rgb_np).to(
            device=self._device, dtype=torch.uint8)
        valid_range_t = torch.from_numpy(valid_range).to(
            device=self._device, dtype=torch.float32)
        elevation_range_t = torch.from_numpy(elevation_range).to(
            device=self._device, dtype=torch.float32)

        pose = Pose(
            position=torch.from_numpy(positions).to(self._device),
            quaternion=torch.from_numpy(quaternions).to(self._device),
            normalize_rotation=True,
        )

        observation = LidarObservation(
            range_image=range_tensor,
            rgb_image=rgb_tensor,
            pose=pose,
            valid_range_m=valid_range_t,
            elevation_range_rad=elevation_range_t,
        )
        mapper.integrate(lidar_observation=observation)

    def destroy(self):
        """Signal the batch integration thread to stop."""
        self._stop.set()
        with self._cond:
            self._cond.notify()
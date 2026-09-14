#!/usr/bin/env python3

from abc import ABC
import rclpy
import torch
import numpy as np
from scipy.spatial.transform import Rotation

from curobo.types import Pose, DeviceCfg

from tf2_ros import Buffer, TransformException, TransformListener


class LaserStrategy(ABC):
    """Abstract base class for LiDAR / laser strategies.

    Mirrors the camera ``CameraStrategy`` base: owns a TF buffer for resolving
    the sensor pose in the planner frame, a static-extrinsics fallback parsed
    from the ``laser_extrinsics`` parameter, and the shared ``_resolve_pose_7``
    helper that both the PointCloud2 and LaserScan strategies use to turn the
    sensor pose into a plain CPU 7-list (GPU conversion is left to the context,
    where the CUDA tensors are assembled).
    """

    def __init__(self, node, laser_name, topic, frame_id, extrinsics=None):
        """
        Args:
            node: ROS2 node.
            laser_name: Unique identifier for this laser.
            topic: Sensor data topic for this laser.
            frame_id: Sensor TF frame (``base_frame → frame_id`` gives pose).
            extrinsics: 7-element list [x,y,z,qw,qx,qy,qz] or ``None`` (TF).
        """
        self.node = node
        self.name = laser_name
        self._topic = topic
        self._frame_id = frame_id
        self.tensor_args = DeviceCfg(device='cuda', dtype=torch.float32)
        self._device = torch.device('cuda')

        # Robot base frame for the sensor-pose TF lookup. Read from the node's
        # 'base_link' param (same convention as DepthMapCameraStrategy).
        self.base_frame = (
            node.get_parameter('base_link').get_parameter_value().string_value
            if node.has_parameter('base_link') else 'base_0')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)

        self.camera_pose_static = self._parse_extrinsics(extrinsics)
        if self.camera_pose_static is not None:
            node.get_logger().info(
                f"[{self.name}] Using static extrinsics from config file")

    def _parse_extrinsics(self, extrinsics):
        """Parse ``laser_extrinsics`` config to a cuRobo ``Pose``.

        Accepts a 7-element list ``[x, y, z, qw, qx, qy, qz]`` or a 4×4
        transformation matrix (list of lists).  Returns ``None`` for
        ``None``/empty (will use TF at runtime).
        """
        if extrinsics is None or extrinsics == []:
            return None
        try:
            if isinstance(extrinsics, list) and len(extrinsics) == 4 and isinstance(extrinsics[0], list):
                matrix = np.array(extrinsics, dtype=np.float64)
                position = matrix[:3, 3].tolist()
                rot = Rotation.from_matrix(matrix[:3, :3])
                quat_scipy = rot.as_quat()  # [x, y, z, w]
                pose_list = position + [quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]]
                pose = Pose.from_list(pose_list, device_cfg=self.tensor_args)
                self.node.get_logger().info(
                    f"[{self.name}] Loaded extrinsics from 4x4 matrix: "
                    f"pos=[{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
                return pose

            elif isinstance(extrinsics, list) and len(extrinsics) == 7:
                quat = extrinsics[3:]
                quat_norm = sum(q**2 for q in quat) ** 0.5
                if abs(quat_norm - 1.0) > 0.01:
                    self.node.get_logger().warn(
                        f"[{self.name}] Quaternion norm {quat_norm:.4f} != 1.0; normalising.")
                    quat = [q / quat_norm for q in quat]
                    extrinsics = extrinsics[:3] + quat
                pose = Pose.from_list(extrinsics, device_cfg=self.tensor_args)
                self.node.get_logger().info(
                    f"[{self.name}] Loaded extrinsics from config: "
                    f"pos=[{extrinsics[0]:.3f}, {extrinsics[1]:.3f}, {extrinsics[2]:.3f}]")
                return pose
            else:
                self.node.get_logger().error(
                    f"[{self.name}] extrinsics must be 7-element list or 4x4 matrix, "
                    f"got {type(extrinsics)}")
                return None
        except Exception as e:
            self.node.get_logger().error(f"[{self.name}] Error parsing extrinsics: {e}")
            return None

    def _resolve_pose_7(self, stamp, warn_on_failure: bool = True):
        """Resolve the sensor pose as a plain CPU 7-list [x,y,z,qw,qx,qy,qz].

        Returns ``(ok, pose_7)``:
        - Static-extrinsics mode → ``(True, None)``: the pose is left to the
          context's stored static pose, so the worker sends ``None``.
        - TF mode, transform found → ``(True, pose_7)``.
        - TF mode, transform missing → ``(False, None)``: the caller MUST drop
          this frame (no integration). The stamp lookup falls back to the
          latest transform, as the sensor stamp routinely runs a few ms ahead
          of TF's newest data.

        Args:
            stamp: ``builtin_interfaces.Time`` (``msg.header.stamp``).
            warn_on_failure: Log a throttled warning on a missing transform.
        """
        if self.camera_pose_static is not None:
            return True, None

        if not self._frame_id:
            if warn_on_failure:
                self.node.get_logger().warn(
                    f'[{self.name}] No laser_frame set and no static extrinsics: '
                    'cannot resolve the sensor pose. Dropping the frame '
                    '(no integration).',
                    throttle_duration_sec=2.0)
            return False, None

        try:
            t = self.tf_buffer.lookup_transform(
                self.base_frame, self._frame_id, rclpy.time.Time.from_msg(stamp))
        except TransformException:
            try:
                t = self.tf_buffer.lookup_transform(
                    self.base_frame, self._frame_id, rclpy.time.Time())
            except TransformException as ex:
                if warn_on_failure:
                    self.node.get_logger().warn(
                        f'[{self.name}] Cannot transform {self.base_frame} → '
                        f'{self._frame_id}: {ex}. Dropping the frame '
                        f'(no integration).',
                        throttle_duration_sec=2.0)
                return False, None

        if t is None or t.transform is None:
            if warn_on_failure:
                self.node.get_logger().warn(
                    f'[{self.name}] NULL transform {self.base_frame} → '
                    f'{self._frame_id}: dropping the frame (no integration).',
                    throttle_duration_sec=2.0)
            return False, None

        pos = t.transform.translation
        rot = t.transform.rotation
        # ROS/geometry_msg and cuRobo use the same [x, y, z, w] convention;
        # Pose expects [x, y, z, qw, qx, qy, qz].
        return True, [pos.x, pos.y, pos.z, rot.w, rot.x, rot.y, rot.z]

    def set_update_callback(self, callback):
        self._update_callback = callback

    def _update_callback(self, msg):
        pass
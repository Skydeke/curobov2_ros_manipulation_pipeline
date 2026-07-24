"""Robot segmentation — folded from ``robot_segmenter.py``.

Shares the one ``MotionPlanner``'s kinematics instead of building a second
``Kinematics`` instance. Uses its own callback group so that the GPU
segmentation pipeline does not block motion planning.
"""

from __future__ import annotations

import time
from typing import List, Optional

from curobo.perception import RobotSegmenter
from curobo.types import CameraObservation, DeviceCfg
from curobo.types import JointState as CuJointState
from curobo.types import Pose as CuPose

import numpy as np

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import torch

from .context import CuroboContext


class RobotSegmentationIntegration:
    """Depth-image robot segmentation using the shared ``Kinematics``.

    Subscribes to depth, camera info, and joint state topics; publishes
    robot masks and filtered (robot-removed) depth images.
    """

    def __init__(self, node: Node, context: CuroboContext, cb_group=None):
        self._node = node
        self._context = context

        # Parameters
        self._cuda_device = int(node.get_parameter("cuda_device").value)
        self._distance_threshold = float(node.get_parameter("distance_threshold").value)
        self._tf_lookup_duration = float(node.get_parameter("tf_lookup_duration").value)
        self._filter_speckles = bool(node.get_parameter("filter_speckles_in_mask").value)
        self._max_speckle_size = int(node.get_parameter("max_filtered_speckles_size").value)
        time_sync_slop = float(node.get_parameter("time_sync_slop").value)

        depth_image_topics = list(node.get_parameter("depth_image_topics").value)
        depth_camera_info_topics = list(node.get_parameter("depth_camera_info_topics").value)
        self._joint_states_topic = node.get_parameter("joint_states_topic").value

        self._num_cameras = len(depth_image_topics)

        # Build RobotSegmenter from SHARED kinematics
        kinematics = context.motion_planner.kinematics
        self._device_cfg = DeviceCfg(device=torch.device("cuda", self._cuda_device))
        self._segmenter = RobotSegmenter(
            kinematics,
            distance_threshold=self._distance_threshold,
            use_cuda_graph=False,
        )
        self._segmenter._ops_dtype = torch.float32
        self._base_frame = kinematics.base_link

        # Camera state
        self._depth_intrinsics: List[Optional[np.ndarray]] = [None] * self._num_cameras
        self._camera_frames: List[Optional[str]] = [None] * self._num_cameras
        self._robot_pose_camera: List[Optional[CuPose]] = [None] * self._num_cameras
        self._image_h: Optional[int] = None
        self._image_w: Optional[int] = None

        # TF
        self._tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=60.0))
        self._tf_listener = TransformListener(self._tf_buffer, node)

        # Callback group (shares motion planner's group to prevent concurrent
        # CUDA graph capture on the same device).
        self._cb_group = cb_group or MutuallyExclusiveCallbackGroup()

        # Camera info subscribers
        self._info_subs = []
        for idx in range(self._num_cameras):
            topic = depth_camera_info_topics[idx] if idx < len(depth_camera_info_topics) else None
            if topic:
                self._info_subs.append(
                    node.create_subscription(
                        CameraInfo, topic,
                        lambda msg, i=idx: self._camera_info_cb(msg, i),
                        10, callback_group=self._cb_group,
                    )
                )

        # TODO(agent): if PyNITROS is available and required by downstream
        # consumers, switch these to PyNitrosSubscriber/PyNitrosPublisher.
        # Using standard sensor_msgs/Image for now.
        import cv2
        from cv_bridge import CvBridge
        from message_filters import ApproximateTimeSynchronizer, Subscriber as MfSub

        self._cv_bridge = CvBridge()

        self._depth_subs = [
            MfSub(node, Image, topic, callback_group=self._cb_group)
            for topic in depth_image_topics
        ]
        self._js_sub = MfSub(node, JointState, self._joint_states_topic, callback_group=self._cb_group)

        self._synchronizer = ApproximateTimeSynchronizer(
            self._depth_subs + [self._js_sub],
            queue_size=10,
            slop=time_sync_slop,
        )
        self._synchronizer.registerCallback(self._process_frames)

        # Publishers (standard ROS 2 Image)
        self._mask_pubs = []
        self._world_depth_pubs = []
        for idx in range(self._num_cameras):
            self._mask_pubs.append(
                node.create_publisher(Image, f"/cumotion/depth_{idx+1}/robot_mask", 10)
            )
            self._world_depth_pubs.append(
                node.create_publisher(Image, f"/cumotion/depth_{idx+1}/world_depth", 10)
            )

        node.get_logger().info(
            f"RobotSegmentationIntegration — {self._num_cameras} camera(s), "
            f"shared kinematics from MotionPlanner"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _camera_info_cb(self, msg: CameraInfo, idx: int):
        self._depth_intrinsics[idx] = np.array(msg.k, dtype=np.float32).reshape(3, 3)
        self._camera_frames[idx] = msg.header.frame_id
        if self._image_h is None:
            self._image_h = msg.height
            self._image_w = msg.width

    def _process_frames(self, *msgs):
        if not all(k is not None for k in self._depth_intrinsics):
            return

        depth_buffers = []
        camera_headers = []
        js_buffer = None
        timestamp = None
        image_encoding = None

        for msg in msgs:
            if isinstance(msg, Image):
                depth = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                enc = msg.encoding
                if enc == "32FC1":
                    depth_t = torch.from_numpy(depth.astype(np.float32))
                    depth_t = 1000.0 * depth_t  # m → mm for projection rays
                elif enc == "16UC1":
                    depth_t = torch.from_numpy(depth.astype(np.float32))
                else:
                    self._node.get_logger().warn(f"Unsupported depth encoding: {enc}")
                    return
                depth_buffers.append(depth_t)
                camera_headers.append(msg.header)
                image_encoding = enc
            elif isinstance(msg, JointState):
                js_buffer = {"joint_names": msg.name, "position": list(msg.position)}
                timestamp = msg.header.stamp

        if timestamp is None or len(camera_headers) == 0:
            return

        # Look up camera poses
        for i in range(self._num_cameras):
            frame = self._camera_frames[i]
            if frame is None:
                return
            try:
                t = self._tf_buffer.lookup_transform(
                    self._base_frame, frame,
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=self._tf_lookup_duration),
                )
                self._robot_pose_camera[i] = CuPose.from_list([
                    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z,
                    t.transform.rotation.w, t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z,
                ])
            except TransformException as ex:
                self._node.get_logger().warn(f"TF lookup {frame} → {self._base_frame}: {ex}")
                return

        if any(p is None for p in self._robot_pose_camera):
            return

        poses_cat = CuPose.cat(self._robot_pose_camera)
        depth_image = torch.stack(depth_buffers).to(device="cuda", dtype=torch.float32)
        intrinsics = np.stack(self._depth_intrinsics)

        if not self._segmenter.ready:
            intrinsics_t = torch.from_numpy(intrinsics).to(
                device="cuda", dtype=torch.float32
            ).view(self._num_cameras, 3, 3)
            init_obs = CameraObservation(
                depth_image=depth_image, intrinsics=intrinsics_t,
            )
            self._segmenter.update_camera_projection(init_obs)
            self._node.get_logger().info("Robot segmenter projection initialised")

        cam_obs = CameraObservation(depth_image=depth_image, pose=poses_cat)
        js_pos = np.array(js_buffer["position"])
        j_names = js_buffer["joint_names"]

        q = CuJointState.from_numpy(
            joint_names=j_names, position=js_pos, device_cfg=self._device_cfg,
        ).unsqueeze(0)
        q = self._segmenter.kinematics.get_active_js(q)

        depth_mask, segmented_depth = self._segmenter.get_robot_mask_from_active_js(cam_obs, q)
        depth_mask = (depth_mask * 255).to(torch.uint8)

        for idx in range(self._num_cameras):
            self._publish_images(
                idx, depth_mask, segmented_depth, image_encoding, camera_headers,
            )

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def _publish_images(self, idx, depth_mask, segmented_depth, encoding, headers):
        mask = depth_mask[idx].cpu().numpy()
        world_depth = segmented_depth[idx].cpu().numpy()

        if self._filter_speckles:
            import cv2
            invalid = world_depth <= 0.0
            combined = np.logical_or(mask, invalid).astype(np.uint8) * 255
            filtered = cv2.filterSpeckles(combined, 255, self._max_speckle_size, 0)[0]
            mask = filtered
            world_depth[filtered.astype(bool)] = 0.0

        mask_msg = self._cv_bridge.cv2_to_imgmsg(mask, encoding="mono8")
        mask_msg.header = headers[idx]
        self._mask_pubs[idx].publish(mask_msg)

        if encoding == "32FC1":
            world_depth = world_depth / 1000.0
        elif encoding == "16UC1":
            world_depth = world_depth.astype(np.uint16)

        depth_msg = self._cv_bridge.cv2_to_imgmsg(world_depth, encoding=encoding)
        depth_msg.header = headers[idx]
        self._world_depth_pubs[idx].publish(depth_msg)

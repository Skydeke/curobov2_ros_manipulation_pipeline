import numpy as np
import torch
from cv_bridge import CvBridge, CvBridgeError
from rclpy import time as rclpy_time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header

from curobo.types import JointState

from isaac_ros_cumotion.cameras.camera_cfg import masked_depth_topic
from isaac_ros_cumotion.cameras.camera_strategy import CameraStrategy


class RobotSegmentationCameraStrategy(CameraStrategy):
    """One camera's robot-segmentation stream: masks the robot (and any
    user-defined mask shapes) out of the raw depth stream and republishes the
    result BEFORE it reaches the perception Mapper, so the arm and its mounting
    never become ESDF voxels.

    Folds the in-server segmentation filter into the per-camera strategy model:
    it is registered through the same ``CameraContext.add_camera`` as the
    mapper's ``DepthMapCameraStrategy``, so a segmented camera is just another
    camera strategy — but it consumes the CAMERA'S RAW depth topic and its
    output topic is DERIVED from that topic (the leaf segment is replaced by
    ``masked_depth``, e.g. ``/kortex_vision/depth/image`` ->
    ``/kortex_vision/depth/masked_depth``). The owning node derives the
    mapper's input topic to be this output, so the two never disagree about
    which stream is which.

    The strategy resolves the robot's pose from the node's shared RobotContext
    (joint state) and Kinematics (FK for the collision spheres) and runs one
    masking stream per camera; the set_mask / remove_mask services and the
    masks dict are owned by the component facade and SHARED across threads.
    """

    def __init__(self, node, camera_name, topic, camera_info_topic,
                 frame_id='', intrinsics=None, extrinsics=None,
                 robot_context=None, kin_model=None, base_frame=None,
                 ops_dtype=None, device=None,
                 distance_threshold=0.05, mask_margin=0.0, masks=None):
        """
        Initialize a robot-segmentation camera strategy.

        Args:
            node: ROS2 node for creating subscriptions and logging.
            camera_name: Name of the camera (CameraContext key).
            topic: RAW depth topic of this camera, e.g. /kortex_vision/depth/image.
            camera_info_topic: CameraInfo topic carrying the intrinsics.
            frame_id: Map/mask frame for this camera (the frame the depth is
                masked in; '' -> the depth message's own frame_id).
            intrinsics: Optional camera intrinsics (unused by the masker;
                kept for CameraContext signature parity).
            extrinsics: Optional camera extrinsics (same).
            robot_context: Shared RobotContext providing the joint pose at the
                depth frame's capture time.
            kin_model: Shared Kinematics used to FK the collision spheres.
            base_frame: Robot base frame for the distance test / mask frames;
                None -> resolved from the node's 'base_link' param.
            ops_dtype: torch dtype (default float32).
            device: torch device (default cuda if available, else cpu).
            distance_threshold: Min distance (m) to a robot collision sphere
                for a depth point to be kept.
            mask_margin: Inflation (m) added to every mask shape's extents.
            masks: Shared dict of user-defined mask shapes (the facade's
                ``self._masks``), mutated by the set_mask / remove_mask
                services; every strategy sees the same shapes.
        """
        super().__init__(node, camera_name, topic, camera_info_topic,
                         frame_id, intrinsics, extrinsics)

        self._ops_dtype = ops_dtype or torch.float32
        if device is not None:
            self._device = device
        else:
            self._device = (torch.device('cuda') if torch.cuda.is_available()
                            else torch.device('cpu'))

        self.robot_context = robot_context
        self._kin_model = kin_model
        self._robot_base_frame = base_frame or self._fallback_base_frame()
        # The frame the depth is masked in is the configured camera frame
        # (`camera_frame`, e.g. curobo_frame in the kortex launch) — same frame
        # the mapper integrates in. There is no separate correction-frame param.
        self._distance_threshold = float(distance_threshold)
        self._mask_margin = float(mask_margin)

        # Shared (by reference) across threads and across all camera streams.
        self._masks = masks if masks is not None else {}

        # Output topic derived from the raw camera topic: strip the leaf
        # segment, publish as `masked_depth` in the same directory.
        self.output_topic = masked_depth_topic(topic)

        # Per-camera stream state.
        self.camera_info = None
        self.depth_frame_id = None
        self.encoding = '16UC1'

        self.sub_camera_info = self.node.create_subscription(
            CameraInfo, camera_info_topic, self.callback_camera_info, 1)
        self.sub_depth = self.node.create_subscription(
            Image, topic, self.callback_depth, 1)
        self.publisher = self.node.create_publisher(Image, self.output_topic, 10)
        self.robot_pointcloud_pub = self.node.create_publisher(
            PointCloud2, self._debug_topic(topic), 10)

        self.bridge = CvBridge()

        self.node.get_logger().info(
            f"[{self.name}] robot segmentation: {topic} -> {self.output_topic} "
            f"(frame={self._frame_id or '?'})")

    def destroy(self):
        for sub in (self.sub_depth, self.sub_camera_info):
            if sub is not None:
                self.node.destroy_subscription(sub)
        for pub in (self.publisher, self.robot_pointcloud_pub):
            if pub is not None:
                self.node.destroy_publisher(pub)

    def _fallback_base_frame(self):
        """Base frame for the robot-distance test when none is injected.

        Prefers the node's shared 'base_link' parameter (ConfigManager's
        configurable base frame); falls back to 'base_link' only if the node
        has no such parameter.
        """
        if self.node.has_parameter('base_link'):
            try:
                return self.node.get_parameter(
                    'base_link').get_parameter_value().string_value
            except Exception:
                pass
        return 'base_link'

    def _debug_topic(self, depth_topic):
        """Per-camera debug-cloud topic: the raw topic's directory with the
        leaf replaced by ``robot_pointcloud_debug``."""
        idx = depth_topic.rfind('/')
        if idx < 0:
            return 'robot_pointcloud_debug'
        return f'{depth_topic[:idx]}/robot_pointcloud_debug'

    # ---- Depth stream callbacks ----

    def callback_camera_info(self, msg):
        self.camera_info = msg

    def callback_depth(self, msg):
        """Mask the robot out of a depth frame and republish the result.

        Runs at the camera rate (frame arrival) for each segmented stream. All
        GPU work — including the frame's CPU->GPU transfer — is under the owning
        node's gpu_lock so a CUDA-graph capture can't race it. An illegal kernel
        launch into a capturing stream invalidates the capture AND the stream,
        so the transfer must not happen before the lock is held. Any CUDA error
        that still slips through only drops this frame: an exception escaping a
        subscription callback is re-raised by rclpy's executor and kills the
        whole node.
        """
        try:
            if msg.encoding == '16UC1':
                depth = self.bridge.imgmsg_to_cv2(msg, '16UC1').astype(np.float32) / 1000.0
            elif msg.encoding == '32FC1':
                depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')
            else:
                self.node.get_logger().warn(
                    f'Unsupported depth encoding: {msg.encoding}')
                return
        except CvBridgeError as e:
            self.node.get_logger().error(f'CvBridge Error: {e}')
            return

        if self.camera_info is None:
            return

        self.depth_frame_id = msg.header.frame_id
        self.encoding = msg.encoding

        joint_pose = self._joint_pose_at(msg.header.stamp)
        if len(joint_pose) != len(self._kin_model.joint_names):
            self.node.get_logger().warn(
                f'Joint pose has {len(joint_pose)} values, expected '
                f'{len(self._kin_model.joint_names)} (model DOF) - skipping '
                'this frame',
                throttle_duration_sec=5.0)
            return

        gpu_lock = getattr(self.node, 'gpu_lock', None)
        if gpu_lock is not None and not gpu_lock.acquire(blocking=False):
            self.node.get_logger().debug(
                'robot_segmentation frame skipped (GPU capture in progress)',
                throttle_duration_sec=2.0)
            return
        try:
            depth = torch.from_numpy(depth).to(
                dtype=self._ops_dtype, device=self._device)
            q = torch.tensor(joint_pose, dtype=self._ops_dtype, device=self._device)
            masked = self._mask_depth_image(depth, q, msg.header.stamp)
            self.publisher.publish(
                self.depth_tensor_to_image_msg(masked, msg.header.stamp))
        except Exception as e:
            self.node.get_logger().debug(
                f'robot_segmentation frame dropped ({e})',
                throttle_duration_sec=2.0)
        finally:
            if gpu_lock is not None:
                gpu_lock.release()

    def _joint_pose_at(self, stamp):
        """Joint position at the depth frame's capture time, or the live pose.

        The FK'd collision spheres can only cover what the camera actually saw
        if the joint state matches the image's capture time. The active control
        strategy keeps a timestamped feedback buffer for this query; the live
        pose is used while that buffer is still filling.
        """
        stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        pose = self.robot_context.get_joint_pose_at(stamp_ns)
        if pose is None:
            pose = self.robot_context.get_joint_pose()
            self.node.get_logger().debug(
                'No time-synced joint feedback yet - masking with live pose',
                throttle_duration_sec=5.0)
        return pose

    # ---- Masking pipeline ----

    def _mask_depth_image(self, depth_image, q, stamp):
        """Return the depth image with robot (and mask-shape) pixels zeroed.

        Every kept pixel retains its exact original depth: the keep/drop mask
        is scattered back onto the image through each point's original pixel
        coordinate, avoiding a lossy point-cloud -> depth round trip.
        """
        H, W = depth_image.shape
        k = self.camera_info.k
        intrinsics = {'fx': k[0], 'fy': k[4], 'cx': k[2], 'cy': k[5]}
        points, u, v = self.depth_to_pointcloud(depth_image, intrinsics)
        keep = self._mask_pointcloud(points, q, stamp)  # (N,) True = not robot

        keep_img = torch.zeros(H * W, dtype=torch.bool, device=self._device)
        keep_img[(v.long() * W) + u.long()] = keep
        keep_img = keep_img.view(H, W)
        return torch.where(keep_img, depth_image, torch.zeros_like(depth_image))

    def depth_to_pointcloud(self, depth_image, intrinsics):
        """Convert a (H, W) depth image (meters) to a (N, 3) camera-frame point
        cloud. Also returns the original pixel coordinate of every valid point.
        """
        H, W = depth_image.shape
        v_coords, u_coords = torch.meshgrid(
            torch.arange(H, dtype=self._ops_dtype, device=self._device),
            torch.arange(W, dtype=self._ops_dtype, device=self._device),
            indexing='ij')
        valid = (depth_image > 0) & ~torch.isnan(depth_image) & ~torch.isinf(depth_image)

        u = u_coords[valid]
        v = v_coords[valid]
        z = depth_image[valid]
        x = (u - intrinsics['cx']) * z / intrinsics['fx']
        y = (v - intrinsics['cy']) * z / intrinsics['fy']
        return torch.stack([x, y, z], dim=1), u, v

    def _mask_pointcloud(self, point_cloud, q, stamp):
        """Return (N,) bool: True = keep (not the robot or a mask shape)."""
        q = q.unsqueeze(0) if q.ndim == 1 else q
        js = JointState(position=q, joint_names=self._kin_model.joint_names)
        spheres = self._kin_model.compute_kinematics(js).robot_spheres.reshape(-1, 4)

        # Distance to the robot spheres, judged in the base frame. The camera is
        # wrist-mounted, so the cloud is transformed at the image's capture time.
        points_base = self._transform_points_to_base(point_cloud, stamp)
        if points_base is None:
            # No TF at the capture instant - mask everything rather than risk
            # integrating the robot. A dropped frame is harmless; an unmasked
            # robot is a permanent voxel error.
            return torch.zeros(point_cloud.shape[0], dtype=torch.bool,
                               device=self._device)

        dist = (torch.norm(points_base.unsqueeze(1) - spheres[:, :3].unsqueeze(0),
                           dim=2) - spheres[:, 3].unsqueeze(0))
        keep = dist.min(dim=1).values > self._distance_threshold

        # Also drop points inside user-defined mask shapes (e.g. a grasped
        # object) so they never reach the mapper / ESDF.
        inside = self._shape_inside_mask(points_base, stamp)
        if inside is not None:
            keep = keep & ~inside

        # Publish the points masked OUT as robot, in the base frame, for debug.
        robot_points = points_base[~keep]
        if robot_points.shape[0] > 0:
            self.robot_pointcloud_pub.publish(self._create_pointcloud2_msg(
                robot_points, self._robot_base_frame,
                self.node.get_clock().now().to_msg()))
        return keep

    def _transform_points_to_base(self, points, stamp):
        """Transform (N, 3) camera-frame points into the robot base frame.

        The source frame is the configured camera frame (`camera_frame`) —
        the same frame the mapper integrates in — falling back to the depth
        message's own frame_id when the camera frame is unset.
        """
        source = self._frame_id or self.depth_frame_id
        if source is None:
            return None
        T = self._tf_matrix(self._robot_base_frame, source, stamp)
        if T is None:
            return None
        ones = torch.ones((points.shape[0], 1), dtype=self._ops_dtype,
                          device=self._device)
        homog = torch.cat([points, ones], dim=1)
        return (homog @ T.T)[:, :3]

    def _tf_matrix(self, target_frame, source_frame, stamp=None):
        """4x4 transform ``target <- source`` from TF, or None.

        Looks up at ``stamp`` (the data's capture time): for a wrist-mounted
        camera the transform moves with the arm, so the latest pose would place
        past pixels at the wrong world location. The image stamp routinely runs
        a few ms ahead of TF's newest data, so a miss silently falls back to
        the latest transform (only a few ms stale). None only if that also
        fails.
        """
        when = (rclpy_time.Time.from_msg(stamp) if stamp is not None
                else rclpy_time.Time())
        try:
            tf = self.tf_buffer.lookup_transform(target_frame, source_frame, when)
        except Exception:
            if stamp is None:
                self.node.get_logger().warn(
                    f'TF {source_frame} -> {target_frame} unavailable',
                    throttle_duration_sec=2.0)
                return None
            try:
                tf = self.tf_buffer.lookup_transform(
                    target_frame, source_frame, rclpy_time.Time())
            except Exception:
                self.node.get_logger().warn(
                    f'TF {source_frame} -> {target_frame} unavailable',
                    throttle_duration_sec=2.0)
                return None

        t = tf.transform.translation
        r = tf.transform.rotation
        R = Rotation.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
        T = torch.eye(4, dtype=self._ops_dtype, device=self._device)
        T[:3, :3] = torch.tensor(R, dtype=self._ops_dtype, device=self._device)
        T[:3, 3] = torch.tensor([t.x, t.y, t.z],
                                dtype=self._ops_dtype, device=self._device)
        return T

    def _pose_matrix(self, pos, quat):
        """4x4 transform from a position (3,) and a scipy quaternion [x, y, z, w]."""
        R = Rotation.from_quat(quat).as_matrix()
        T = torch.eye(4, dtype=self._ops_dtype, device=self._device)
        T[:3, :3] = torch.tensor(R, dtype=self._ops_dtype, device=self._device)
        T[:3, 3] = pos
        return T

    def _shape_inside_mask(self, points_base, stamp):
        """(N,) bool: True where a base-frame point lies inside ANY mask shape.
        Returns None when no masks are defined. Each mask is expressed in its
        own ``frame_id`` (resolved at the same capture time so a mask attached
        to a moving frame follows the arm).
        """
        if not self._masks:
            return None

        N = points_base.shape[0]
        ones = torch.ones((N, 1), dtype=self._ops_dtype, device=self._device)
        homog = torch.cat([points_base, ones], dim=1)
        inside_any = torch.zeros(N, dtype=torch.bool, device=self._device)

        for m in self._masks.values():
            T_mask = self._pose_matrix(m['pos'], m['quat'])
            if m['frame_id']:
                T_base = self._tf_matrix(self._robot_base_frame, m['frame_id'], stamp)
                if T_base is None:
                    continue  # TF missing this cycle -> skip only this mask
                T_mask = T_base @ T_mask
            local = (homog @ torch.inverse(T_mask).T)[:, :3]
            inside = self._inside_shape(local, m)
            if inside is not None:
                inside_any |= inside
        return inside_any

    def _inside_shape(self, local, m):
        """Analytic point-inside test in a shape's local frame (+ mask margin)."""
        t = self._mask_margin
        dims = m['dims']
        typ = m['type']

        if typ == 0:  # cuboid, half-extents
            hx, hy, hz = dims[0] / 2 + t, dims[1] / 2 + t, dims[2] / 2 + t
            return ((local[:, 0].abs() <= hx)
                    & (local[:, 1].abs() <= hy)
                    & (local[:, 2].abs() <= hz))
        if typ == 1:  # sphere
            return torch.norm(local, dim=1) <= dims[0] + t
        if typ == 2:  # cylinder, centered, axis z, z in [-h/2, h/2]
            r, hz = dims[0] + t, dims[1] / 2 + t
            return ((local[:, 2].abs() <= hz)
                    & (torch.norm(local[:, :2], dim=1) <= r))
        if typ == 3:  # capsule segment [0,0,0]->[0,0,h] along +z
            r, h = dims[0] + t, dims[1]
            z_clamped = local[:, 2].clamp(0.0, h)
            dz = local[:, 2] - z_clamped
            return (torch.sqrt(local[:, 0] ** 2 + local[:, 1] ** 2 + dz ** 2) <= r)
        if typ == 4:
            self.node.get_logger().warn(
                'MESH mask not supported analytically yet - skipped. '
                '(future: curobo WorldMeshCollision zero-radius point SDF)',
                throttle_duration_sec=5.0)
        return None

    # ---- Message builders ----

    def depth_tensor_to_image_msg(self, depth_tensor, stamp):
        """Convert a depth tensor to an Image message.

        The encoding matches the source stream (16UC1 -> uint16 mm, 32FC1 ->
        float32 m) so the masked output is byte-compatible with the input.

        The header carries the *capture* stamp of the source frame, not publish
        time: consumers (the mapper) must resolve transforms at the same instant
        the mask was evaluated, otherwise the masked pixels and the integrated
        cloud drift apart to different world poses.
        """
        depth_np = depth_tensor.cpu().numpy()
        if self.encoding == '32FC1':
            msg = self.bridge.cv2_to_imgmsg(depth_np.astype(np.float32),
                                            encoding='32FC1')
        else:
            msg = self.bridge.cv2_to_imgmsg((depth_np * 1000.0).astype(np.uint16),
                                            encoding='16UC1')
        msg.header.stamp = stamp
        msg.header.frame_id = self.depth_frame_id
        return msg

    def _create_pointcloud2_msg(self, points, frame_id, timestamp):
        """Create a PointCloud2 (XYZ float32) message from a (N, 3) tensor."""
        points_np = points.cpu().numpy().astype('<f4')
        msg = PointCloud2()
        msg.header = Header()
        msg.header.stamp = timestamp
        msg.header.frame_id = frame_id
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * points_np.shape[0]
        msg.is_dense = True
        msg.height = 1
        msg.width = points_np.shape[0]
        msg.data = points_np.tobytes()
        return msg
import threading
import time as time_module
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose as RosPose
from isaac_ros_cumotion_interfaces.srv import GetEsdf
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA, Float32MultiArray, Header, MultiArrayDimension
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import Marker
import torch

from curobo.perception import FilterDepth, Mapper, MapperCfg
from curobo.types import CameraObservation
from curobo.types import Pose as CuPose


class MapperNode(Node):

    def __init__(self):
        super().__init__("curobo_mapper_node")

        # --- TSDF parameters ---
        self.declare_parameter("voxel_size", 0.02)
        self.declare_parameter("esdf_voxel_size", 0.02)
        self.declare_parameter("extent_meters_xyz", [5.0, 5.0, 5.0])
        self.declare_parameter("grid_center_m", [0.0, 0.0, 0.0])
        self.declare_parameter("depth_minimum_distance", 0.05)
        self.declare_parameter("depth_maximum_distance", 5.0)
        self.declare_parameter("truncation_distance", -1.0)
        self.declare_parameter("minimum_tsdf_weight", 1.0)
        self.declare_parameter("decay_factor", 0.3)
        self.declare_parameter("block_size", 2)
        self.declare_parameter("roughness", 3.0)
        self.declare_parameter("num_cameras", 1)
        self.declare_parameter("device", "cuda:0")

        # --- ROS wiring parameters ---
        self.declare_parameter("robot_base_frame", "base_link")
        self.declare_parameter(
            "depth_image_topics",
            ["/camera_1/aligned_depth_to_color/image_raw"],
        )
        self.declare_parameter(
            "depth_camera_info_topics",
            ["/camera_1/color/camera_info"],
        )
        self.declare_parameter("tf_lookup_duration", 0.05)
        self.declare_parameter("filter_depth", True)
        self.declare_parameter("integrate_rate_hz", 20.0)
        self.declare_parameter(
            "esdf_service_name", "/curobo_mapper/get_esdf_and_gradient"
        )
        self.declare_parameter("camera_correction_frame", "curobo_frame")
        self.declare_parameter("publish_debug_voxels", False)
        self.declare_parameter("debug_voxel_publish_rate_hz", 1.0)

        # Read parameters (image_height/width come from CameraInfo at runtime)
        self._vs = self.get_parameter("voxel_size").value
        self._esdf_vs = self.get_parameter("esdf_voxel_size").value
        self._extent = list(self.get_parameter("extent_meters_xyz").value)
        self._grid_center = list(self.get_parameter("grid_center_m").value)
        self._depth_min = self.get_parameter("depth_minimum_distance").value
        self._depth_max = self.get_parameter("depth_maximum_distance").value
        td = self.get_parameter("truncation_distance").value
        self._trunc_dist = td if td > 0.0 else self._vs * 4.0
        self._min_weight = self.get_parameter("minimum_tsdf_weight").value
        self._decay = self.get_parameter("decay_factor").value
        self._block_size = self.get_parameter("block_size").value
        self._roughness = self.get_parameter("roughness").value
        self._num_cameras = self.get_parameter("num_cameras").value
        self._device_str = self.get_parameter("device").value
        self._device = torch.device(self._device_str)
        self._robot_base_frame = self.get_parameter("robot_base_frame").value
        depth_image_topics = list(self.get_parameter("depth_image_topics").value)
        depth_camera_info_topics = list(
            self.get_parameter("depth_camera_info_topics").value
        )
        self._tf_lookup_duration = self.get_parameter("tf_lookup_duration").value
        self._filter_depth_enabled = self.get_parameter("filter_depth").value
        integrate_rate = self.get_parameter("integrate_rate_hz").value
        esdf_service_name = self.get_parameter("esdf_service_name").value
        self._integrate_period = 1.0 / max(integrate_rate, 1.0)

        # State for lazy mapper init
        self._image_height: Optional[int] = None
        self._image_width: Optional[int] = None
        self._cam_info_received: List[bool] = [False] * self._num_cameras

        # --- Threading ---
        self._data_lock = threading.Lock()

        # --- TF ---
        self._tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=60.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # --- Camera state ---
        self._depth_intrinsics: List[Optional[np.ndarray]] = [None] * self._num_cameras
        self._camera_frames: List[Optional[str]] = [None] * self._num_cameras
        self._latest_depth: List[Optional[torch.Tensor]] = [None] * self._num_cameras

        # --- Mapper (created lazily) ---
        self._mapper: Optional[Mapper] = None
        self._depth_filter: Optional[FilterDepth] = None
        self._integrate_timer = None
        self._last_voxel_grid = None

        # Subscriptions
        for i in range(self._num_cameras):
            topic = depth_image_topics[i] if i < len(depth_image_topics) else None
            if topic:
                self.create_subscription(
                    Image, topic, lambda msg, idx=i: self._depth_cb(msg, idx), 10
                )
            info_topic = (
                depth_camera_info_topics[i]
                if i < len(depth_camera_info_topics)
                else None
            )
            if info_topic:
                self.create_subscription(
                    CameraInfo,
                    info_topic,
                    lambda msg, idx=i: self._cam_info_cb(msg, idx),
                    10,
                )

        # --- ESDF service ---
        self._esdf_service = self.create_service(
            GetEsdf,
            esdf_service_name,
            self._esdf_cb,
        )

        # --- Debug voxel publisher ---
        self._curobo_frame = self.get_parameter("camera_correction_frame").value
        self._publish_debug_voxels = self.get_parameter("publish_debug_voxels").value
        debug_voxel_rate = self.get_parameter("debug_voxel_publish_rate_hz").value
        self._debug_voxel_pub = self.create_publisher(
            Marker,
            "/curobo_mapper/debug_voxels",
            1,
        )
        if self._publish_debug_voxels:
            self._debug_voxel_timer = self.create_timer(
                1.0 / max(debug_voxel_rate, 0.1),
                self._publish_debug_marker,
            )
        else:
            self._debug_voxel_timer = None

        self.get_logger().info(
            f"cuRobo Mapper node started — {self._num_cameras} camera(s), "
            f"waiting for CameraInfo..."
        )

    # ------------------------------------------------------------------
    # Lazy mapper initialization
    # ------------------------------------------------------------------

    def _try_init_mapper(self):
        if self._image_height is None or self._image_width is None:
            return
        if not all(self._cam_info_received):
            return
        if self._mapper is not None:
            return

        H = self._image_height
        W = self._image_width
        self.get_logger().info(f"Initializing mapper with image size {H}×{W}")

        config = MapperCfg(
            voxel_size=self._vs,
            esdf_voxel_size=self._esdf_vs,
            extent_meters_xyz=tuple(self._extent),
            extent_esdf_meters_xyz=tuple(self._extent),
            grid_center=torch.tensor(self._grid_center, dtype=torch.float32),
            truncation_distance=self._trunc_dist,
            depth_minimum_distance=self._depth_min,
            depth_maximum_distance=self._depth_max,
            minimum_tsdf_weight=self._min_weight,
            decay_factor=self._decay,
            roughness=self._roughness,
            num_cameras=self._num_cameras,
            image_height=H,
            image_width=W,
            device=self._device_str,
            block_size=self._block_size,
        )
        self._mapper = Mapper(config)

        if self._filter_depth_enabled:
            self._depth_filter = FilterDepth(
                image_shape=(H, W),
                depth_minimum_distance=self._depth_min,
                depth_maximum_distance=self._depth_max,
                flying_pixel_threshold=0.5,
                bilateral_kernel_size=3,
            )
        else:
            self._depth_filter = None

        self._integrate_timer = self.create_timer(
            self._integrate_period, self._integrate_frames
        )

        self.get_logger().info(
            f"cuRobo Mapper — {self._mapper.memory_usage_mb():.1f} MB, "
            f"extent={self._extent}, "
            f"voxel_size={self._vs}, esdf_voxel_size={self._esdf_vs}"
        )

    # ------------------------------------------------------------------
    # Camera callbacks
    # ------------------------------------------------------------------

    def _cam_info_cb(self, msg: CameraInfo, idx: int):
        with self._data_lock:
            if self._image_height is None:
                self._image_height = msg.height
                self._image_width = msg.width
            self._depth_intrinsics[idx] = np.array(msg.k, dtype=np.float32).reshape(
                3, 3
            )
            self._camera_frames[idx] = msg.header.frame_id
            self._cam_info_received[idx] = True
        self._try_init_mapper()

    def _depth_cb(self, msg: Image, idx: int):
        if msg.encoding == "32FC1":
            depth = np.frombuffer(msg.data, dtype=np.float32).reshape(
                msg.height, msg.width
            )
        elif msg.encoding in ("16UC1", "mono16"):
            depth = (
                np.frombuffer(msg.data, dtype=np.uint16)
                .reshape(msg.height, msg.width)
                .astype(np.float32)
                / 1000.0
            )
        else:
            self.get_logger().warn(f"Unsupported depth encoding: {msg.encoding}")
            return
        with self._data_lock:
            self._latest_depth[idx] = torch.from_numpy(np.ascontiguousarray(depth)).to(
                device=self._device,
                dtype=torch.float32,
            )

    # ------------------------------------------------------------------
    # TF helper
    # ------------------------------------------------------------------

    def _lookup_camera_pose(self, camera_frame: str) -> Optional[CuPose]:
        try:
            t = self._tf_buffer.lookup_transform(
                self._robot_base_frame,
                self._curobo_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=self._tf_lookup_duration),
            )
            return CuPose.from_list(
                [
                    t.transform.translation.x,
                    t.transform.translation.y,
                    t.transform.translation.z,
                    t.transform.rotation.w,
                    t.transform.rotation.x,
                    t.transform.rotation.y,
                    t.transform.rotation.z,
                ]
            )
        except TransformException as ex:
            self.get_logger().warn(
                f"TF lookup {camera_frame} → {self._robot_base_frame}: {ex}"
            )
            return None

    # ------------------------------------------------------------------
    # Depth integration (periodic)
    # ------------------------------------------------------------------

    def _integrate_frames(self):
        if self._mapper is None:
            return

        with self._data_lock:
            intrinsics = list(self._depth_intrinsics)
            depths = list(self._latest_depth)
            frames = list(self._camera_frames)

        if any(d is None for d in depths):
            return
        if any(k is None for k in intrinsics):
            return
        if any(f is None for f in frames):
            return

        depth_list: List[torch.Tensor] = []
        intrinsics_list: List[torch.Tensor] = []
        pose_list: List[CuPose] = []

        for i in range(self._num_cameras):
            pose = self._lookup_camera_pose(frames[i])
            if pose is None:
                return
            depth_list.append(depths[i])
            intrinsics_list.append(
                torch.from_numpy(intrinsics[i]).to(
                    device=self._device, dtype=torch.float32
                )
            )
            pose_list.append(pose)

        if self._num_cameras == 1:
            depth_batched = depth_list[0].unsqueeze(0)
            intrinsics_batched = intrinsics_list[0].unsqueeze(0)
            pose_batched = pose_list[0]
        else:
            depth_batched = torch.stack(depth_list)
            intrinsics_batched = torch.stack(intrinsics_list)
            pos = torch.cat([p.position.view(1, 3) for p in pose_list])
            quat = torch.cat([p.quaternion.view(1, 4) for p in pose_list])
            pose_batched = CuPose(position=pos, quaternion=quat)

        if self._depth_filter is not None:
            depth_batched = torch.nan_to_num(
                depth_batched, nan=0.0, posinf=0.0, neginf=0.0
            )
            filtered, _ = self._depth_filter(depth_batched)
            depth_batched = filtered

        rgb_dummy = torch.zeros(
            depth_batched.shape[0],
            depth_batched.shape[1],
            depth_batched.shape[2],
            3,
            dtype=torch.uint8,
            device=self._device,
        )
        obs = CameraObservation(
            name="mapper_camera",
            depth_image=depth_batched,
            rgb_image=rgb_dummy,
            intrinsics=intrinsics_batched,
            pose=pose_batched,
        )
        try:
            self._mapper.integrate(obs)
        except Exception as e:
            self.get_logger().error(f"Integrate failed: {e}")

    # ------------------------------------------------------------------
    # ESDF service callback
    # ------------------------------------------------------------------

    def _esdf_cb(self, request, response):
        if self._mapper is None:
            response.success = False
            return response

        t0 = time_module.perf_counter()

        # 1. Clear requested regions
        if request.aabbs_to_clear_min_m:
            for i in range(len(request.aabbs_to_clear_min_m)):
                m = request.aabbs_to_clear_min_m[i]
                s = request.aabbs_to_clear_size_m[i]
                self._mapper.clear_region(
                    [m.x, m.y, m.z],
                    [m.x + s.x, m.y + s.y, m.z + s.z],
                )
        if request.spheres_to_clear_center_m:
            for i in range(len(request.spheres_to_clear_center_m)):
                c = request.spheres_to_clear_center_m[i]
                r = request.spheres_to_clear_radius_m[i]
                self._mapper.clear_region(
                    [c.x - r, c.y - r, c.z - r],
                    [c.x + r, c.y + r, c.z + r],
                )

        # 2. Compute ESDF
        voxel_grid = self._mapper.compute_esdf()
        self._last_voxel_grid = voxel_grid

        if voxel_grid.feature_tensor is None:
            response.success = False
            return response

        # 3. VoxelGrid → GetEsdf response
        ft = voxel_grid.feature_tensor
        grid_shape, low, high = voxel_grid.get_grid_shape()
        nx, ny, nz = grid_shape
        total = nx * ny * nz
        if ft.numel() != total:
            self.get_logger().warn(
                f"ESDF shape mismatch: feature_tensor={ft.shape[0]}, grid={nx}×{ny}×{nz}={total}"
            )
            response.success = False
            return response

        vs = float(voxel_grid.voxel_size)
        # Grid origin = grid center + low corner offset (in robot_base_frame)
        ox = float(voxel_grid.pose[0]) + float(low[0])
        oy = float(voxel_grid.pose[1]) + float(low[1])
        oz = float(voxel_grid.pose[2]) + float(low[2])

        response.success = True
        response.voxel_size_m = vs
        response.origin_m = Point(x=ox, y=oy, z=oz)

        esdf_data = ft.cpu().numpy().flatten().astype(np.float32)
        response.esdf_and_gradients = Float32MultiArray()
        dims_layout = [
            MultiArrayDimension(label="x", size=nx, stride=ny * nz),
            MultiArrayDimension(label="y", size=ny, stride=nz),
            MultiArrayDimension(label="z", size=nz, stride=1),
        ]
        response.esdf_and_gradients.layout.dim = dims_layout
        response.esdf_and_gradients.layout.data_offset = 0
        response.esdf_and_gradients.data = esdf_data.tolist()

        dt = time_module.perf_counter() - t0
        self.get_logger().info(
            f"ESDF — shape=({nx},{ny},{nz}), vs={voxel_grid.voxel_size:.4f}, "
            f"origin=({ox:.3f},{oy:.3f},{oz:.3f}), "
            f"min={ft.min().item():.4f}, max={ft.max().item():.4f}, "
            f"dt={dt*1000:.1f}ms"
        )
        return response

    # ------------------------------------------------------------------
    # Debug voxel publisher
    # ------------------------------------------------------------------

    def _publish_debug_marker(self):
        if self._last_voxel_grid is None:
            return
        vg = self._last_voxel_grid
        if vg.feature_tensor is None:
            return
        ft = vg.feature_tensor
        grid_shape, low, high = vg.get_grid_shape()
        nx, ny, nz = grid_shape
        vs = float(vg.voxel_size)
        esdf_np = ft.cpu().numpy()
        cx = float(vg.pose[0])
        cy = float(vg.pose[1])
        cz = float(vg.pose[2])

        stamp = self.get_clock().now().to_msg()
        points = []
        colors = []
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    val = float(esdf_np[ix, iy, iz])
                    if abs(val) > 10.0:
                        continue
                    if val >= vs * 2:
                        continue
                    x = cx + low[0] + (ix + 0.5) * vs
                    y = cy + low[1] + (iy + 0.5) * vs
                    z = cz + low[2] + (iz + 0.5) * vs
                    points.append(Point(x=x, y=y, z=z))
                    if val < 0:
                        colors.append(ColorRGBA(r=1.0, g=0.2, b=0.2, a=0.6))
                    else:
                        colors.append(ColorRGBA(r=0.2, g=1.0, b=0.2, a=0.6))

        if not points:
            return

        msg = Marker()
        msg.header = Header(frame_id=self._robot_base_frame)
        msg.header.stamp = stamp
        msg.ns = "esdf_voxels"
        msg.id = 0
        msg.type = Marker.CUBE_LIST
        msg.action = Marker.ADD
        msg.scale.x = vs
        msg.scale.y = vs
        msg.scale.z = vs
        msg.pose.orientation.w = 1.0
        msg.points = points
        msg.colors = colors
        self._debug_voxel_pub.publish(msg)

        self.get_logger().info(
            f"Published {len(points)} debug voxels "
            f"(max={float(ft.max().item()):.2f}, "
            f"min={float(ft.min().item()):.2f})"
        )


def main(args=None):
    rclpy.init(args=args)
    node = MapperNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down cuRobo mapper.")
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()

import threading
import time as time_module
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose as RosPose
from isaac_ros_cumotion_interfaces.srv import GetEsdf
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import (
    ColorRGBA,
    Float32MultiArray,
    Header,
    MultiArrayDimension,
    String,
)
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import Marker
import torch
import torch.nn.functional as F

from curobo.perception import FilterDepth, Mapper, MapperCfg
from curobo.types import CameraObservation
from curobo.types import Pose as CuPose

RADIO_MODEL_NAME = "c-radio_v3-B"


class CRadioInference:
    def __init__(self, device: str = "cuda:0", text_adaptor_name: Optional[str] = None):
        self.device = device
        adaptor_names = [text_adaptor_name] if text_adaptor_name else None
        hub_version = RADIO_MODEL_NAME.strip().lower()
        self.model = (
            torch.hub.load(
                "NVlabs/RADIO",
                "radio_model",
                source="github",
                version=hub_version,
                progress=True,
                skip_validation=True,
                adaptor_names=adaptor_names,
            )
            .eval()
            .to(device)
        )
        self.patch_size = int(getattr(self.model, "patch_size", 16))

        self.text_adaptor = None
        self.tokenizer = None
        self._encode_text_fn = None
        if text_adaptor_name is not None:
            self.text_adaptor = self._resolve_text_adaptor(text_adaptor_name)
            self.tokenizer = getattr(self.text_adaptor, "tokenizer", None)
            self._encode_text_fn = getattr(self.text_adaptor, "encode_text", None)

    def _resolve_text_adaptor(self, adaptor_name: str):
        for attr in ("adaptors", "adapters", "_adaptors"):
            registry = getattr(self.model, attr, None)
            if registry is not None and adaptor_name in registry:
                return registry[adaptor_name]
        available = {}
        for attr in ("adaptors", "adapters", "_adaptors"):
            registry = getattr(self.model, attr, None)
            if registry is not None:
                available[attr] = list(registry.keys())
        raise RuntimeError(
            f"Could not find adaptor '{adaptor_name}' on RADIO model. "
            f"Available: {available or 'none'}"
        )

    def _project_through_text_adaptor(self, features: torch.Tensor) -> torch.Tensor:
        if self.text_adaptor is None:
            raise RuntimeError("No text adaptor loaded")
        for attr in ("head_mlp", "feat_mlp", "head"):
            sub = getattr(self.text_adaptor, attr, None)
            if sub is not None and callable(sub):
                return sub(features)
        if callable(self.text_adaptor):
            try:
                out = self.text_adaptor(features)
            except TypeError:
                summary = features.mean(dim=0, keepdim=True)
                out = self.text_adaptor(summary, features.unsqueeze(0))
                if isinstance(out, tuple):
                    out = out[1]
                    if out.dim() == 3:
                        out = out[0]
            if isinstance(out, tuple):
                out = out[1] if len(out) > 1 else out[0]
            return out
        raise RuntimeError(
            f"Adaptor {type(self.text_adaptor).__name__} has no known entry point"
        )

    @torch.inference_mode()
    def encode_text(self, text) -> torch.Tensor:
        if self.tokenizer is None or self._encode_text_fn is None:
            raise RuntimeError("No text adaptor loaded")
        if isinstance(text, str):
            text = [text]
        tokens = self.tokenizer(text)
        if hasattr(tokens, "to"):
            tokens = tokens.to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            try:
                feats = self._encode_text_fn(tokens, normalize=True)
            except TypeError:
                feats = self._encode_text_fn(tokens)
                feats = F.normalize(feats, dim=-1)
        return feats

    @torch.inference_mode()
    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self._project_through_text_adaptor(features)
            out = F.normalize(out, dim=-1)
        return out

    @torch.inference_mode()
    def extract_patch_features(self, rgb_uint8: torch.Tensor) -> torch.Tensor:
        H, W = rgb_uint8.shape[:2]
        target_h, target_w = self.model.get_nearest_supported_resolution(H, W)
        img = rgb_uint8.permute(2, 0, 1).float() / 255.0
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        output = self.model(img)
        if isinstance(output, dict):
            output = output["backbone"]
        features = getattr(output, "features", None)
        if features is None:
            _, features = output
        ps = self.patch_size
        return features[0].view(target_h // ps, target_w // ps, -1).contiguous()


def _pca_basis(
    centered: torch.Tensor, prev_basis: Optional[torch.Tensor] = None
) -> torch.Tensor:
    N, D = centered.shape
    device, dtype = centered.device, centered.dtype
    basis = torch.zeros((D, 3), device=device, dtype=dtype)
    for i in range(min(D, 3)):
        basis[i, i] = 1.0
    if D > 0 and N > 0:
        try:
            cov = centered.T @ centered
            if N > 1:
                cov = cov / (N - 1)
            cov = 0.5 * (cov + cov.T)
            _, eigvecs = torch.linalg.eigh(cov)
            rank = min(3, D)
            basis[:, :rank] = eigvecs[:, -rank:].flip(dims=[1]).to(dtype)
        except RuntimeError:
            pass
    if (
        prev_basis is not None
        and prev_basis.shape == basis.shape
        and torch.isfinite(prev_basis).all()
    ):
        prev = prev_basis.to(device=device, dtype=dtype)
        signs = torch.where((basis * prev).sum(dim=0) >= 0, 1.0, -1.0).to(basis)
        basis = basis * signs
    return basis


def pca_colorize_tensor(
    feats_flat: torch.Tensor,
    prev_basis: Optional[torch.Tensor] = None,
    low_pct: float = 0.02,
    high_pct: float = 0.98,
) -> Tuple[torch.Tensor, torch.Tensor]:
    flat = feats_flat.float()
    N, D = flat.shape
    colors = torch.zeros((N, 3), device=flat.device, dtype=torch.uint8)
    valid_rows = torch.isfinite(flat).all(dim=1)
    valid = flat[valid_rows]
    centered = valid - valid.mean(dim=0, keepdim=True) if valid.shape[0] > 0 else valid
    if (
        prev_basis is not None
        and prev_basis.shape == (D, 3)
        and torch.isfinite(prev_basis).all()
    ):
        basis = prev_basis.to(device=flat.device, dtype=flat.dtype)
    else:
        basis = _pca_basis(centered)
    if valid.shape[0] == 0 or D == 0:
        return colors, basis
    proj = centered @ basis
    lo = torch.quantile(proj, low_pct, dim=0)
    hi = torch.quantile(proj, high_pct, dim=0)
    spread = hi - lo
    scaled = ((proj - lo) / spread.clamp(min=1e-6)).clamp(0.0, 1.0)
    scaled = torch.where(
        spread.unsqueeze(0) > 1e-6, scaled, torch.full_like(scaled, 0.5)
    )
    colors[valid_rows] = (scaled * 255.0).to(torch.uint8)
    return colors, basis


class MapperNode(Node):

    def __init__(self):
        super().__init__("curobo_mapper_node")

        # --- TSDF parameters ---
        self.declare_parameter("voxel_size", 0.02)
        self.declare_parameter("esdf_voxel_size", 0.02)
        self.declare_parameter("grid_size_m", [5.0, 5.0, 5.0])
        self.declare_parameter("grid_center_m", [0.0, 0.0, 0.0])
        self.declare_parameter("depth_minimum_distance", 0.05)
        self.declare_parameter("depth_maximum_distance", 5.0)
        self.declare_parameter("truncation_distance", -1.0)
        self.declare_parameter("minimum_tsdf_weight", 1.0)
        self.declare_parameter("decay_factor", 1.0)
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
        self.declare_parameter("camera_correction_frame", ["curobo_frame"])
        self.declare_parameter("publish_debug_voxels", False)
        self.declare_parameter("debug_voxel_publish_rate_hz", 1.0)
        self.declare_parameter(
            "rgb_image_topics",
            ["/kortex_vision/color/image"],
        )
        self.declare_parameter("enable_feature_mapping", True)
        self.declare_parameter("enable_text_query", True)
        self.declare_parameter("text_query_top_k", 500)
        self.declare_parameter("text_query_min_score", 0.05)
        self.declare_parameter("max_publish_voxels", 100000)

        # Read parameters (image_height/width come from CameraInfo at runtime)
        self._vs = self.get_parameter("voxel_size").value
        self._esdf_vs = self.get_parameter("esdf_voxel_size").value
        self._extent = list(self.get_parameter("grid_size_m").value)
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
        rgb_image_topics = list(self.get_parameter("rgb_image_topics").value)
        self._tf_lookup_duration = self.get_parameter("tf_lookup_duration").value
        self._filter_depth_enabled = self.get_parameter("filter_depth").value
        self._enable_feature_mapping = self.get_parameter(
            "enable_feature_mapping"
        ).value
        self._enable_text_query = self.get_parameter("enable_text_query").value
        self._text_query_top_k = self.get_parameter("text_query_top_k").value
        self._text_query_min_score = self.get_parameter("text_query_min_score").value
        self._max_publish_voxels = self.get_parameter("max_publish_voxels").value
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
        self._latest_rgb: List[Optional[torch.Tensor]] = [None] * self._num_cameras

        # --- Mapper (created lazily) ---
        self._mapper: Optional[Mapper] = None
        self._depth_filter: Optional[FilterDepth] = None
        self._integrate_timer = None

        # --- Feature mapping ---
        self._feature_model: Optional[CRadioInference] = None
        self._feature_h: int = 0
        self._feature_w: int = 0
        self._feature_dim: int = 0
        self._pca_basis: Optional[torch.Tensor] = None
        self._cached_feat: Optional[torch.Tensor] = None
        self._cached_feat_shape: Optional[tuple] = None

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
            rgb_topic = rgb_image_topics[i] if i < len(rgb_image_topics) else None
            if rgb_topic:
                self.create_subscription(
                    Image, rgb_topic, lambda msg, idx=i: self._rgb_cb(msg, idx), 10
                )

        # --- ESDF service ---
        self._esdf_service = self.create_service(
            GetEsdf,
            esdf_service_name,
            self._esdf_cb,
        )

        # --- Debug voxel / surface publisher ---
        curobo_frames = self.get_parameter("camera_correction_frame").value
        self._curobo_frames: List[str] = list(curobo_frames) if curobo_frames else []
        self._publish_debug_voxels = self.get_parameter("publish_debug_voxels").value
        debug_voxel_rate = self.get_parameter("debug_voxel_publish_rate_hz").value
        self._colored_surface_pub = self.create_publisher(
            PointCloud2,
            "/curobo_mapper/colored_surface",
            1,
        )
        self._features_pca_pub = self.create_publisher(
            PointCloud2,
            "/curobo_mapper/features_pca",
            1,
        )
        self._feature_pca_image_pub = self.create_publisher(
            Image,
            "/curobo_mapper/feature_pca_image",
            1,
        )
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

        # --- Text query (semantic search) ---
        self._matched_features_pub = self.create_publisher(
            PointCloud2,
            "/curobo_mapper/matched_features",
            1,
        )
        if self._enable_feature_mapping and self._enable_text_query:
            self._text_query_sub = self.create_subscription(
                String, "/curobo_mapper/text_query", self._text_query_cb, 10
            )
            self.get_logger().info(
                "Text query enabled — subscribe to /curobo_mapper/text_query"
            )

        self.get_logger().info(
            f"cuRobo Mapper node started — {self._num_cameras} camera(s), "
            f"waiting for CameraInfo..."
        )

    # ------------------------------------------------------------------
    # Lazy mapper initialization
    # ------------------------------------------------------------------

    def _determine_feature_dim(self, H: int, W: int) -> int:
        if not self._enable_feature_mapping:
            self._feature_dim = 0
            return 0
        if self._feature_model is None:
            self.get_logger().info(
                f"Loading C-RADIO ({RADIO_MODEL_NAME}) via NVlabs/RADIO torch.hub..."
            )
            text_adaptor = "siglip2" if self._enable_text_query else None
            self._feature_model = CRadioInference(
                device=self._device_str, text_adaptor_name=text_adaptor
            )
            probe = self._feature_model.extract_patch_features(
                torch.zeros((H, W, 3), dtype=torch.uint8, device=self._device)
            )
            self._feature_h = probe.shape[0]
            self._feature_w = probe.shape[1]
            self._feature_dim = probe.shape[-1]
            self.get_logger().info(
                f"C-RADIO loaded: feature_dim={self._feature_dim}, "
                f"grid={self._feature_h}×{self._feature_w}"
            )
        return self._feature_dim

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

        feature_dim = self._determine_feature_dim(H, W)

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
            feature_dim=feature_dim,
            feature_grid_height=self._feature_h if feature_dim > 0 else None,
            feature_grid_width=self._feature_w if feature_dim > 0 else None,
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

    def _rgb_cb(self, msg: Image, idx: int):
        if msg.encoding == "rgb8":
            rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3
            )
        elif msg.encoding == "bgr8":
            rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3
            )[:, :, ::-1]
        elif msg.encoding == "rgba8":
            rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 4
            )[:, :, :3]
        elif msg.encoding == "bgra8":
            rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 4
            )[:, :, 2::-1]
        else:
            self.get_logger().warn(f"Unsupported RGB encoding: {msg.encoding}")
            return
        with self._data_lock:
            self._latest_rgb[idx] = torch.from_numpy(np.ascontiguousarray(rgb)).to(
                device=self._device,
                dtype=torch.uint8,
            )

    # ------------------------------------------------------------------
    # TF helper
    # ------------------------------------------------------------------

    def _lookup_camera_pose(
        self, camera_index: int, camera_frame: str
    ) -> Optional[CuPose]:
        target_frame = (
            self._curobo_frames[camera_index]
            if camera_index < len(self._curobo_frames)
            and self._curobo_frames[camera_index]
            else camera_frame
        )
        try:
            t = self._tf_buffer.lookup_transform(
                self._robot_base_frame,
                target_frame,
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
                f"TF lookup {target_frame} → {self._robot_base_frame}: {ex}"
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
            rgbs = list(self._latest_rgb)
            frames = list(self._camera_frames)

        if any(d is None for d in depths):
            return
        if any(k is None for k in intrinsics):
            return
        if any(f is None for f in frames):
            return

        depth_list: List[torch.Tensor] = []
        rgb_list: List[torch.Tensor] = []
        intrinsics_list: List[torch.Tensor] = []
        pose_list: List[CuPose] = []

        for i in range(self._num_cameras):
            pose = self._lookup_camera_pose(i, frames[i])
            if pose is None:
                return
            depth_list.append(depths[i])
            intrinsics_list.append(
                torch.from_numpy(intrinsics[i]).to(
                    device=self._device, dtype=torch.float32
                )
            )
            pose_list.append(pose)

            if rgbs[i] is not None:
                rgb_list.append(rgbs[i])
            else:
                rgb_list.append(
                    torch.zeros(
                        depths[i].shape[0],
                        depths[i].shape[1],
                        3,
                        dtype=torch.uint8,
                        device=self._device,
                    )
                )

        if self._num_cameras == 1:
            depth_batched = depth_list[0].unsqueeze(0)
            rgb_batched = rgb_list[0].unsqueeze(0)
            intrinsics_batched = intrinsics_list[0].unsqueeze(0)
            pose_batched = pose_list[0]
        else:
            depth_batched = torch.stack(depth_list)
            rgb_batched = torch.stack(rgb_list)
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

        if self._feature_model is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feat = self._feature_model.extract_patch_features(rgb_batched[0])
            self._cached_feat = feat
            self._cached_feat_shape = rgb_batched[0].shape[:2]

            obs = CameraObservation(
                name="mapper_camera",
                depth_image=depth_batched,
                rgb_image=rgb_batched,
                intrinsics=intrinsics_batched,
                pose=pose_batched,
                feature_grid=feat.to(dtype=torch.float16).contiguous().unsqueeze(0),
            )
        else:
            obs = CameraObservation(
                name="mapper_camera",
                depth_image=depth_batched,
                rgb_image=rgb_batched,
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
        self.get_logger().debug(
            f"ESDF — shape=({nx},{ny},{nz}), vs={voxel_grid.voxel_size:.4f}, "
            f"origin=({ox:.3f},{oy:.3f},{oz:.3f}), "
            f"min={ft.min().item():.4f}, max={ft.max().item():.4f}, "
            f"dt={dt*1000:.1f}ms"
        )
        return response

    # ------------------------------------------------------------------
    # Text query (semantic search)
    # ------------------------------------------------------------------

    def _text_query_cb(self, msg: String):
        if self._mapper is None or self._feature_model is None:
            self.get_logger().warn("Mapper or feature model not ready for text query")
            return

        text = msg.data.strip()
        if not text:
            self._matched_features_pub.publish(PointCloud2())
            self.get_logger().info("Empty text query — cleared matched features")
            return

        self.get_logger().info(f"Text query: '{text}'")

        try:
            text_emb = self._feature_model.encode_text(text)
        except Exception as e:
            self.get_logger().error(f"Text encoding failed: {e}")
            return

        self.get_logger().info(
            f"text_emb norm={text_emb[0].norm():.4f}, "
            f"min={text_emb[0].min():.4f}, max={text_emb[0].max():.4f}"
        )

        # Quick per-image match diagnostic using cached features
        feat = self._cached_feat
        if feat is not None:
            Hp, Wp, D = feat.shape
            flat = feat.reshape(-1, D).float()
            projected = self._feature_model.project_features(flat)
            projected_norm = projected.norm(dim=1)
            img_scores = projected @ text_emb[0]
            self.get_logger().info(
                f"Per-image scores: min={img_scores.min():.4f} "
                f"max={img_scores.max():.4f} "
                f"mean={img_scores.mean():.4f} "
                f"std={img_scores.std():.4f} "
                f"proj_norm min={projected_norm.min():.6f} "
                f"proj_norm max={projected_norm.max():.6f}"
            )

        try:
            matched = self._mapper.extract_matching_feature_voxels(
                feature_vector=text_emb[0],
                top_k=self._text_query_top_k,
                minimum_score=self._text_query_min_score,
                surface_only=True,
                feature_projector=self._feature_model.project_features,
            )
        except Exception as e:
            self.get_logger().error(f"Feature matching failed: {e}")
            return

        if matched is None or len(matched.voxels) == 0:
            self.get_logger().info("No matching voxels found")
            self._matched_features_pub.publish(PointCloud2())
            return

        centers = matched.voxels.centers

        n = len(centers)
        max_voxels = self._max_publish_voxels
        if n > max_voxels:
            stride = int(np.ceil(n / max_voxels))
            centers = centers[::stride]
            n = len(centers)

        centers_np = centers.cpu().numpy()

        colors_np = np.zeros((n, 3), dtype=np.uint8)
        colors_np[:, 1] = 255
        colors_np[:, 2] = 255

        cloud = PointCloud2()
        cloud.header = Header(frame_id=self._robot_base_frame)
        cloud.header.stamp = self.get_clock().now().to_msg()
        cloud.height = 1
        cloud.width = n
        cloud.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.point_step = 16
        cloud.row_step = n * 16
        cloud.is_bigendian = False
        cloud.is_dense = True

        packed = np.zeros((n, 4), dtype=np.float32)
        packed[:, 0] = centers_np[:, 0]
        packed[:, 1] = centers_np[:, 1]
        packed[:, 2] = centers_np[:, 2]
        rgb_packed = (
            (colors_np[:, 0].astype(np.uint32) << 16)
            | (colors_np[:, 1].astype(np.uint32) << 8)
            | colors_np[:, 2].astype(np.uint32)
        )
        packed[:, 3] = rgb_packed.view(np.float32)
        cloud.data = packed.tobytes()
        self._matched_features_pub.publish(cloud)

        top_score = float(matched.block_scores[0].item())
        self.get_logger().info(
            f"Matched {n} voxels, "
            f"top score={top_score:.3f}, "
            f"blocks={len(matched.block_scores)}"
        )
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Debug voxel / surface publisher
    # ------------------------------------------------------------------

    def _publish_debug_marker(self):
        if self._mapper is None:
            return

        voxels = self._mapper.extract_occupied_voxels(surface_only=True)
        if len(voxels) == 0:
            return

        centers = voxels.centers
        colors_uint8 = voxels.colors_uint8()

        max_voxels = self._max_publish_voxels
        if len(centers) > max_voxels:
            stride = int(len(centers) / max_voxels)
            if stride > 1:
                centers = centers[::stride]
                colors_uint8 = colors_uint8[::stride]

        centers_np = centers.cpu().numpy()
        colors_np = colors_uint8.cpu().numpy()
        n = len(centers_np)

        # Publish colored surface as PointCloud2
        cloud = PointCloud2()
        cloud.header = Header(frame_id=self._robot_base_frame)
        cloud.header.stamp = self.get_clock().now().to_msg()
        cloud.height = 1
        cloud.width = n
        cloud.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.point_step = 16
        cloud.row_step = n * 16
        cloud.is_bigendian = False
        cloud.is_dense = True
        packed = np.zeros((n, 4), dtype=np.float32)
        packed[:, 0] = centers_np[:, 0]
        packed[:, 1] = centers_np[:, 1]
        packed[:, 2] = centers_np[:, 2]
        rgb_packed = (
            (colors_np[:, 0].astype(np.uint32) << 16)
            | (colors_np[:, 1].astype(np.uint32) << 8)
            | colors_np[:, 2].astype(np.uint32)
        )
        packed[:, 3] = rgb_packed.view(np.float32)
        cloud.data = packed.tobytes()
        self._colored_surface_pub.publish(cloud)

        # Publish PCA-colored feature point cloud
        if self._feature_model is not None and self._feature_dim > 0:
            block_features = voxels.block_data.features_normalized()

            # Fit PCA basis from per-image features (clean, dense) instead of
            # block features (noisy per-block with small block_size).
            feat = self._cached_feat
            if feat is None:
                with self._data_lock:
                    rgb_cached = self._latest_rgb[0]
                if rgb_cached is not None:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        feat = self._feature_model.extract_patch_features(rgb_cached)
            if feat is not None:
                Hp, Wp, D = feat.shape
                feat_flat = feat.reshape(-1, D).float()
                image_colors, self._pca_basis = pca_colorize_tensor(
                    feat_flat, prev_basis=self._pca_basis,
                )
                H, W = self._cached_feat_shape or (Hp * 14, Wp * 14)
                pca_img = image_colors.view(Hp, Wp, 3)
                pca_img_t = pca_img.permute(2, 0, 1).unsqueeze(0).float() / 255.0
                pca_img_up = F.interpolate(pca_img_t, size=(H, W), mode="bilinear", align_corners=False)
                pca_img_up = (pca_img_up[0].permute(1, 2, 0) * 255.0).to(torch.uint8).cpu().numpy()
                img_msg = Image()
                img_msg.header = Header(frame_id=self._robot_base_frame)
                img_msg.header.stamp = self.get_clock().now().to_msg()
                img_msg.height = H
                img_msg.width = W
                img_msg.encoding = "rgb8"
                img_msg.is_bigendian = False
                img_msg.step = W * 3
                img_msg.data = pca_img_up.tobytes()
                self._feature_pca_image_pub.publish(img_msg)

                # Project block features onto the same basis
                colors_pca, _ = pca_colorize_tensor(
                    block_features, prev_basis=self._pca_basis,
                )
                feat_centers = voxels.centers
                feat_colors = colors_pca[voxels.block_idx_per_voxel]
                if len(feat_centers) > max_voxels:
                    step_f = int(len(feat_centers) / max_voxels)
                    feat_centers = feat_centers[::step_f]
                    feat_colors = feat_colors[::step_f]
                fcn = feat_centers.cpu().numpy()
                fcl = feat_colors.cpu().numpy()
                fn = len(fcn)
                if fn > 0:
                    fcloud = PointCloud2()
                    fcloud.header = Header(frame_id=self._robot_base_frame)
                    fcloud.header.stamp = self.get_clock().now().to_msg()
                    fcloud.height = 1
                    fcloud.width = fn
                    fcloud.fields = [
                        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
                    ]
                    fcloud.point_step = 16
                    fcloud.row_step = fn * 16
                    fcloud.is_bigendian = False
                    fcloud.is_dense = True
                    fpacked = np.zeros((fn, 4), dtype=np.float32)
                    fpacked[:, 0] = fcn[:, 0]
                    fpacked[:, 1] = fcn[:, 1]
                    fpacked[:, 2] = fcn[:, 2]
                    frgb_packed = (
                        (fcl[:, 0].astype(np.uint32) << 16)
                        | (fcl[:, 1].astype(np.uint32) << 8)
                        | fcl[:, 2].astype(np.uint32)
                    )
                    fpacked[:, 3] = frgb_packed.view(np.float32)
                    fcloud.data = fpacked.tobytes()
                    self._features_pca_pub.publish(fcloud)


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

from isaac_ros_cumotion.cameras.camera_cfg import PerceptionCameraCfg
from isaac_ros_cumotion.cameras.camera_context import CameraContext


class CameraSystemManager:
    """Owns the perception cameras: maps the ``camera_*`` parameter block onto
    the mapper's camera strategies.

    Responsible for:
    - Declaring/reading the perception-camera params (one array entry per
      camera, see PerceptionCameraCfg) — replaces the old per-repo
      ``cameras.yaml`` file, whose fields had to be hand-synced with the
      robot_segmentation params.
    - Creating the CameraContext and one DepthMapCameraStrategy per camera that
      feeds the shared Mapper. Which cameras feed the mapper is the per-camera
      ``camera_purpose``; each strategy subscribes to the robot-segmentation
      output when that camera is segmented, to the raw depth stream otherwise.
    - Exposing the resolved ``camera_cfgs`` so the in-server RobotSegmentation
      consumes the SAME camera identity (raw topic, camera-info topic,
      camera frame) for the cameras marked to be segmented.
    """

    def __init__(self, node):
        """
        Initialize the camera system manager.

        Args:
            node: ROS2 node instance
        """
        self.node = node
        self.camera_context = None
        self.camera_cfgs = []
        self.camera_cfg = None
        self._configure()

    def _configure(self):
        """Declare and read the camera params, then build the camera strategies.

        The cameras are described entirely by the ``camera_*`` params
        (PerceptionCameraCfg) — there is no config file anymore. ``camera_cfgs``
        is exposed so the robot-segmentation component can consume the SAME
        camera identity (raw topic, camera-info topic, camera frame).
        """
        PerceptionCameraCfg.declare(self.node)
        num = PerceptionCameraCfg.num_cameras(self.node)

        esdf_cfg = None
        for i in range(num):
            cfg = PerceptionCameraCfg.from_node(self.node, camera_index=i)
            self.camera_cfgs.append(cfg)
            if not cfg.depth_topic:
                continue

            roles = []
            if cfg.for_esdf:
                roles.append('esdf')
            if cfg.for_segmentation:
                roles.append('segmentation')
            self.node.get_logger().info(
                f"Camera '{cfg.name}' (index {i}): raw={cfg.depth_topic}, "
                f"purpose={cfg.purpose} [{'+'.join(roles)}], "
                f"mapper_input={cfg.mapper_topic}, "
                f"frame_rate_hz={cfg.frame_rate_hz:.1f}, frame={cfg.frame_id}")

            if cfg.for_esdf:
                esdf_cfg = cfg
                if self.camera_context is None:
                    self.camera_context = CameraContext(self.node)
                self.camera_context.add_camera(
                    camera_name=cfg.name,
                    camera_type='depth_camera',
                    topic=cfg.mapper_topic,
                    camera_info=cfg.camera_info_topic,
                    frame_id=cfg.frame_id,
                    intrinsics=cfg.intrinsics,
                    extrinsics=cfg.extrinsics,
                    frame_rate_hz=cfg.frame_rate_hz,
                    camera_index=cfg.camera_index,
                )

        self.camera_cfg = self.camera_cfgs[0] if self.camera_cfgs else None
        if esdf_cfg is None:
            self.node.get_logger().warn(
                "No camera feeds the Mapper: set 'camera_topic' (with a "
                "camera_purpose of 'all' or 'esdf') at launch. The legacy "
                "cameras_config_file YAML has been removed.")

    def get_camera_context(self):
        """
        Get the camera context.

        Returns:
            CameraContext instance or None if no camera feeds the Mapper
        """
        return self.camera_context
from isaac_ros_cumotion.lasers.laser_cfg import PerceptionLaserCfg
from isaac_ros_cumotion.lasers.laser_context import LaserContext


class LaserSystemManager:
    """Owns the perception LiDAR/laser sensors: maps the ``laser_*`` parameter
    block onto the mapper's laser strategies (mirrors ``CameraSystemManager``).

    Responsibilities:
    - Declaring/reading the ``laser_*`` params (one array entry per laser) plus
      the global range-image resolution (``laser_image_height`` /
      ``laser_image_width``, 0 = disabled).
    - Building a ``LaserContext`` and one ``PointCloudLaserStrategy`` per laser
      that feeds the shared Mapper.  curobo sizes the lidar projective buffer
      once from ``lidar_num_sensors`` and a single (H, W): all configured lasers
      share the range-image resolution.
    - Exposing the activated laser count, the shared resolution and the resolved
      per-laser configs so ``ObstacleManager`` can build ``MapperCfg`` with
      ``lidar_num_sensors`` and the pipeline's ``setup_perception`` can size the
      mapper even when no camera is present.
    """

    def __init__(self, node):
        """
        Args:
            node: ROS2 node instance.
        """
        self.node = node
        self.laser_context = None
        self.laser_cfgs = []
        self.num_lasers = 0
        self.lidar_image_height = 0
        self.lidar_image_width = 0
        self._configure()

    def _configure(self):
        """Declare and read the laser params, then build the laser strategies."""
        PerceptionLaserCfg.declare(self.node)
        num = PerceptionLaserCfg.num_lasers(self.node)
        if num < 1:
            self.node.get_logger().info("No lasers configured (laser_topic unset).")
            return

        # Keep only slots with a real topic — an empty topic slot is "no laser".
        cfgs = []
        for i in range(num):
            cfg = PerceptionLaserCfg.from_node(self.node, laser_index=i)
            if cfg.topic:
                cfgs.append(cfg)

        if not cfgs:
            self.node.get_logger().info(
                "No lasers configured (all laser_topic entries empty).")
            return

        h = PerceptionLaserCfg.get_image_height(self.node)
        w = PerceptionLaserCfg.get_image_width(self.node)
        if h < 1 or w < 1:
            self.node.get_logger().error(
                f"{len(cfgs)} laser(s) configured but 'laser_image_height' / "
                f"'laser_image_width' are not positive ({h}×{w}). LiDAR "
                "integration disabled — set both to the range-image resolution "
                "(e.g. 1×360 for a 2D scan, 64×1024 for a 3D lidar).")
            return

        self.lidar_image_height = h
        self.lidar_image_width = w
        self.laser_context = LaserContext(self.node, image_height=h, image_width=w)

        cfgs_by_name = {}
        for cfg in cfgs:
            # Planar scans (H == 1) integrate a single elevation row: curobo
            # requires elevation_range_rad[:, 0] == elevation_range_rad[:, 1].
            if h == 1:
                cfg.elevation_min_rad = 0.0
                cfg.elevation_max_rad = 0.0
                self.node.get_logger().info(
                    f"Laser '{cfg.name}' is planar (H=1): forcing elevation "
                    "range to [0, 0] rad.")

            cfgs_by_name[cfg.name] = cfg
            self.node.get_logger().info(
                f"Laser '{cfg.name}' (index {cfg.laser_index}): "
                f"type={cfg.laser_type}, topic={cfg.topic}, "
                f"frame={cfg.frame_id}, range=[{cfg.range_min_m:.2f}, "
                f"{cfg.range_max_m:.2f}] m, elevation=[{cfg.elevation_min_rad:.3f}, "
                f"{cfg.elevation_max_rad:.3f}] rad, rate={cfg.frame_rate_hz:.1f} Hz, "
                f"image={w}x{h}")

            self.laser_context.add_laser(
                name=cfg.name,
                laser_type=cfg.laser_type,
                topic=cfg.topic,
                frame_id=cfg.frame_id,
                extrinsics=cfg.extrinsics,
                frame_rate_hz=cfg.frame_rate_hz,
                range_min_m=cfg.range_min_m,
                range_max_m=cfg.range_max_m,
                elevation_min_rad=cfg.elevation_min_rad,
                elevation_max_rad=cfg.elevation_max_rad,
                callback_group=getattr(
                    self.node, '_perception_callback_group', None),
            )

        self.laser_context.set_laser_cfgs(cfgs_by_name)
        self.laser_cfgs = list(cfgs_by_name.values())
        self.num_lasers = len(self.laser_context.lasers)

    def get_laser_context(self):
        """Get the laser context (None when no laser is active)."""
        return self.laser_context
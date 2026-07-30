"""Thin ROS 2 node class for the unified cuRobo server.

All business logic lives in the handler modules (``motion.py``, ``ik.py``,
…); this class only wires parameters, builds the one ``MotionPlanner``, and
registers action/service servers.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from typing import Dict, Optional, Set

from curobo.logging import setup_logger as curobo_setup_logger
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Scene, VoxelGrid as CuVoxelGrid
from curobo.types import DeviceCfg
from curobo.types import JointState as CuJointState

import rclpy
from rclpy.action import ActionServer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from isaac_ros_cumotion.update_kinematics import get_robot_config
from isaac_ros_cumotion_interfaces.action import AttachObject, ControlMPC, OptimizeTrajectory, PlanGrasp, PlanMotion, RetargetMotion
from isaac_ros_cumotion_interfaces.srv import CheckCollision, ComputeFK, ComputeIK, GetEsdf, PublishStaticPlanningScene, UpdateWorld
from isaac_ros_cumotion.curobo_server.utils import (
    get_grid_center,
    get_grid_min_corner,
    get_grid_size,
    is_grid_valid,
    load_grid_corners_from_workspace_file,
)

from sensor_msgs.msg import JointState
from visualization_msgs.msg import MarkerArray
import torch

from isaac_ros_cumotion.util import get_spheres_marker
from moveit_msgs.action import MoveGroup

from .context import CuroboContext
from . import motion as motion_handler
from . import trajopt as trajopt_handler
from .mpc import MPCIntegration
from . import retargeter as retargeter_handler
from . import ik as ik_handler
from . import attach as attach_handler
from . import collision as collision_handler
from . import fk as fk_handler
from . import world as world_handler
from . import moveit_bridge as moveit_bridge_handler
from . import segmentation as segmentation_handler
from .segmentation import RobotSegmentationIntegration
from . import mapping as mapping_handler
from .mapping import MapperIntegration


class CuroboServerNode(Node):

    def __init__(self):
        super().__init__("curobo_server_node", allow_undeclared_parameters=True)

        # ------------------------------------------------------------------
        # Declare parameters (merged from cumotion_planner.py,
        # cumotion_goal_set_planner.py, mapper_node.py, robot_segmenter.py,
        # and attach_object_server.py)
        # ------------------------------------------------------------------
        self._declare_all_parameters()

        # ------------------------------------------------------------------
        # Build the one MotionPlanner instance
        # ------------------------------------------------------------------
        motion_planner, grid_size_m, esdf_voxel_size, raw_grid_shape, esdf_scene = (
            self._build_motion_planner()
        )
        self._motion_planner = motion_planner
        self._esdf_voxel_size = esdf_voxel_size
        self._grid_size_m = grid_size_m
        self._raw_grid_shape = raw_grid_shape

        # ------------------------------------------------------------------
        # Shared context passed to every handler
        # ------------------------------------------------------------------
        # NOTE: named _curobo_ctx, NOT _context, to avoid overwriting
        # Node._context (rclpy's internal rcl context handle).
        self._curobo_ctx = CuroboContext(
            motion_planner=motion_planner,
            esdf_scene=esdf_scene,
            device=str(motion_planner.device_cfg.device),
            logger=self.get_logger(),
            node=self,
        )

        # ------------------------------------------------------------------
        # Guard against concurrent GPU access
        # ------------------------------------------------------------------
        self._planner_busy = False
        self._lock = threading.Lock()
        self._motion_planner_cb_group = MutuallyExclusiveCallbackGroup()

        # ------------------------------------------------------------------
        # Depth-to-ESDF mapper (folded from mapper_node.py; own callback group)
        # ------------------------------------------------------------------
        if self.get_parameter("enable_mapper").value:
            self._mapper_integration = MapperIntegration(
                self, self._curobo_ctx, planner_cb_group=self._motion_planner_cb_group,
            )
        else:
            self._mapper_integration = None
            self.get_logger().info("Mapper disabled (enable_mapper=false)")

        # ------------------------------------------------------------------
        # Robot segmentation (folded from robot_segmenter.py; own callback group)
        # ------------------------------------------------------------------
        if self.get_parameter("enable_segmenter").value:
            self._segmentation_integration = RobotSegmentationIntegration(
                self, self._curobo_ctx, cb_group=self._motion_planner_cb_group,
            )
        else:
            self._segmentation_integration = None

        # ------------------------------------------------------------------
        # ROS wiring
        # ------------------------------------------------------------------
        self._js_buffer: Optional[dict] = None
        self._joint_states_topic = (
            self.get_parameter("joint_states_topic").get_parameter_value().string_value
        )
        self._joint_state_sub = self.create_subscription(
            JointState, self._joint_states_topic, self._js_callback, 10
        )

        # ------------------------------------------------------------------
        # Collision sphere visualization publisher
        # ------------------------------------------------------------------
        self._sphere_topic = self.get_parameter("debug_robot_topic").value
        self._sphere_publisher = self.create_publisher(
            MarkerArray, self._sphere_topic, 10,
        )
        sphere_hz = self.get_parameter("publish_robot_spheres_hz").value
        self._sphere_timer = self.create_timer(
            1.0 / sphere_hz, self._publish_robot_spheres,
            callback_group=self._motion_planner_cb_group,
        )
        self._sphere_rgb = [0.0, 1.0, 0.0, 1.0]  # green

        # ------------------------------------------------------------------
        # MPC integration (stateful, in motion_planner callback group)
        # ------------------------------------------------------------------
        self._mpc_integration = MPCIntegration(
            self, self._curobo_ctx, self._lock, self._motion_planner_cb_group,
        )

        # Action servers (all in the same mutually-exclusive group)
        self._plan_motion_server = ActionServer(
            self, PlanMotion, "cumotion/plan_motion",
            self._on_plan_motion, callback_group=self._motion_planner_cb_group,
        )
        self._plan_grasp_server = ActionServer(
            self, PlanGrasp, "cumotion/plan_grasp",
            self._on_plan_grasp, callback_group=self._motion_planner_cb_group,
        )
        self._attach_object_server = ActionServer(
            self, AttachObject, "cumotion/attach_object",
            self._on_attach_object, callback_group=self._motion_planner_cb_group,
        )
        self._optimize_trajectory_server = ActionServer(
            self, OptimizeTrajectory, "cumotion/optimize_trajectory",
            self._on_optimize_trajectory, callback_group=self._motion_planner_cb_group,
        )
        self._control_mpc_server = ActionServer(
            self, ControlMPC, "cumotion/mpc/control",
            self._on_control_mpc, callback_group=self._motion_planner_cb_group,
        )
        self._retarget_motion_server = ActionServer(
            self, RetargetMotion, "cumotion/retarget_motion",
            self._on_retarget_motion, callback_group=self._motion_planner_cb_group,
        )

        # MoveIt compatibility: accept moveit_msgs::action::MoveGroup at
        # cumotion/move_group (the C++ CumotionMoveGroupClient sends here)
        self._move_group_server = ActionServer(
            self, MoveGroup, "cumotion/move_group",
            self._on_move_group, callback_group=self._motion_planner_cb_group,
        )

        # Service servers
        self._compute_ik_srv = self.create_service(
            ComputeIK, "cumotion/compute_ik",
            self._on_compute_ik, callback_group=self._motion_planner_cb_group,
        )
        self._compute_fk_srv = self.create_service(
            ComputeFK, "cumotion/compute_fk",
            self._on_compute_fk, callback_group=self._motion_planner_cb_group,
        )
        self._check_collision_srv = self.create_service(
            CheckCollision, "cumotion/check_collision",
            self._on_check_collision, callback_group=self._motion_planner_cb_group,
        )
        self._update_world_srv = self.create_service(
            UpdateWorld, "cumotion/update_world",
            self._on_update_world, callback_group=self._motion_planner_cb_group,
        )
        self._get_esdf_srv = self.create_service(
            GetEsdf, "cumotion/update_esdf",
            self._on_get_esdf, callback_group=self._motion_planner_cb_group,
        )
        self._static_scene_srv = self.create_service(
            PublishStaticPlanningScene, "cumotion/publish_static_scene",
            self._on_publish_static_scene, callback_group=self._motion_planner_cb_group,
        )

        self.get_logger().info("CuroboServerNode initialised — one MotionPlanner for all capabilities")

    # ------------------------------------------------------------------
    # Parameter declarations
    # ------------------------------------------------------------------

    def _declare_all_parameters(self):
        # Robot / model
        self.declare_parameter("robot", "ur5e.yml")
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("yml_file_path", "")
        self.declare_parameter("tool_frame", "")

        # Motion planning
        self.declare_parameter("time_dilation_factor", 0.5)
        self.declare_parameter("max_attempts", 100)
        self.declare_parameter("num_graph_seeds", 32)
        self.declare_parameter("num_trajopt_seeds", 4)
        self.declare_parameter("include_trajopt_retract_seed", True)
        self.declare_parameter("num_trajopt_time_steps", 32)
        self.declare_parameter("num_trajopt_noisy_seeds", 2)
        self.declare_parameter("trajopt_seed_ratio", '{"linear": 0.5, "bias": 0.5}')
        self.declare_parameter("trajopt_finetune_iters", 400)
        self.declare_parameter("interpolation_dt", 0.025)
        # max_goalset = number of alternative goal poses for the same problem.
        #   The solver tries each and picks the best.  Shapes [..., N, ...]
        #   nested inside the batch dim.
        self.declare_parameter("max_goalset", 12)
        # max_batch_size = number of independent problems (different start/goal
        #   pairs) solved in parallel on GPU.  Shapes [batch, ...].
        #   batch=1 → SolveMode.SINGLE, batch>1 → SolveMode.BATCH.
        self.declare_parameter("max_batch_size", 1)
        self.declare_parameter("ik_optimizer_config", "")
        self.declare_parameter("trajopt_optimizer_config", "")
        self.declare_parameter("enable_cuda_graph", True)
        self.declare_parameter("collision_cache_mesh", 20)
        self.declare_parameter("collision_cache_cuboid", 20)

        # ESDF / world
        self.declare_parameter("esdf_voxel_size", 0.05)
        self.declare_parameter("read_esdf_world", False)
        self.declare_parameter("publish_curobo_world_as_voxels", False)
        self.declare_parameter("add_ground_plane", False)
        self.declare_parameter("publish_voxel_size", 0.05)
        self.declare_parameter("max_publish_voxels", 50000)
        self.declare_parameter("workspace_file_path", "")
        self.declare_parameter("grid_center_m", [0.0, 0.0, 0.0])
        self.declare_parameter("grid_size_m", [2.0, 2.0, 2.0])
        self.declare_parameter("update_esdf_on_request", True)
        self.declare_parameter("use_aabb_on_request", True)
        self.declare_parameter("clear_robot_spheres_from_esdf", True)
        self.declare_parameter("robot_esdf_clearing_padding", 0.0)

        # ROS wiring
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("enable_curobo_debug_mode", False)
        self.declare_parameter("override_moveit_scaling_factors", False)

        # Static planning scene
        self.declare_parameter("moveit_collision_objects_scene_file", "")

        # Mapper parameters (from mapper_node.py)
        self.declare_parameter("enable_mapper", True)
        self.declare_parameter("tsdf_voxel_size", 0.02)
        self.declare_parameter("depth_minimum_distance", 0.05)
        self.declare_parameter("depth_maximum_distance", 5.0)
        self.declare_parameter("truncation_distance", -1.0)
        self.declare_parameter("minimum_tsdf_weight", 1.0)
        self.declare_parameter("decay_factor", 1.0)
        self.declare_parameter("block_size", 2)
        self.declare_parameter("roughness", 3.0)
        self.declare_parameter("num_cameras", 1)
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("robot_base_frame", "base_link")
        self.declare_parameter("depth_image_topics", ["/camera_1/aligned_depth_to_color/image_raw"])
        self.declare_parameter("depth_camera_info_topics", ["/camera_1/color/camera_info"])
        self.declare_parameter("rgb_image_topics", ["/kortex_vision/color/image"])
        self.declare_parameter("tf_lookup_duration", 5.0)
        self.declare_parameter("filter_depth", True)
        self.declare_parameter("integrate_rate_hz", 20.0)
        self.declare_parameter("camera_correction_frame", ["curobo_frame"])
        self.declare_parameter("publish_debug_voxels", False)
        self.declare_parameter("debug_voxel_publish_rate_hz", 1.0)
        self.declare_parameter("enable_feature_mapping", True)
        self.declare_parameter("enable_text_query", True)
        self.declare_parameter("text_query_top_k", 500)
        self.declare_parameter("text_query_min_score", 0.05)

        # Collision sphere visualization
        self.declare_parameter("debug_robot_topic", "/cumotion/robot_segmenter/robot_spheres")
        self.declare_parameter("publish_robot_spheres_hz", 30.0)

        # Robot segmentation parameters (from robot_segmenter.py)
        self.declare_parameter("enable_segmenter", False)
        self.declare_parameter("cuda_device", 0)
        self.declare_parameter("distance_threshold", 0.2)
        self.declare_parameter("time_sync_slop", 0.1)
        self.declare_parameter("filter_speckles_in_mask", False)
        self.declare_parameter("max_filtered_speckles_size", 1250)

        # Object attachment parameters (from attach_object_server.py)
        self.declare_parameter("object_link_name", "attached_object")
        self.declare_parameter("object_attachment_gripper_frame_name", "grasp_frame")
        self.declare_parameter("object_attachment_n_spheres", 100)
        self.declare_parameter("search_radius", 0.2)
        self.declare_parameter("surface_sphere_radius", 0.01)

        level = logging.INFO if self.get_parameter("enable_curobo_debug_mode").value else logging.WARNING
        curobo_setup_logger("info" if self.get_parameter("enable_curobo_debug_mode").value else "warning", "curobo")
        if hasattr(logging, "lastResort") and logging.lastResort is not None:
            logging.lastResort.setLevel(level)

    # ------------------------------------------------------------------
    # MotionPlanner construction (reuses cumotion_planner.py logic)
    # ------------------------------------------------------------------

    def _build_motion_planner(self):
        device_cfg = DeviceCfg()
        robot_file = self.get_parameter("robot").value
        urdf_path = self.get_parameter("urdf_path").value or None
        yml_path = self.get_parameter("yml_file_path").value or None
        if yml_path:
            robot_file = yml_path

        read_esdf = self.get_parameter("read_esdf_world").value
        publish_voxels = self.get_parameter("publish_curobo_world_as_voxels").value
        esdf_vs = float(self.get_parameter("esdf_voxel_size").value)
        grid_size_m = list(self.get_parameter("grid_size_m").value)
        grid_center_m = list(self.get_parameter("grid_center_m").value)
        workspace_file = self.get_parameter("workspace_file_path").value
        collision_cache_cuboid = int(self.get_parameter("collision_cache_cuboid").value)
        collision_cache_mesh = int(self.get_parameter("collision_cache_mesh").value)

        if os.path.exists(workspace_file):
            self.get_logger().info(f"Loading grid from workspace file: {workspace_file}")
            min_corner, max_corner = load_grid_corners_from_workspace_file(workspace_file)
            grid_size_m = list(get_grid_size(min_corner, max_corner, esdf_vs))
            grid_center_m = list(get_grid_center(min_corner, grid_size_m))

        if is_grid_valid(grid_size_m, esdf_vs):
            self.get_logger().fatal("Grid has zero voxels in some dimension")
            raise SystemExit(1)

        raw_grid_shape = [round(d / esdf_vs) for d in grid_size_m]

        world_file = None
        if read_esdf or publish_voxels:
            grid_shape = [s + 1 for s in raw_grid_shape]
            num_voxels = math.prod(grid_shape)
            world_file = Scene.create({
                "voxel": {
                    "world_voxel": {
                        "dims": [s * esdf_vs for s in grid_shape],
                        "pose": [0, 0, 0, 1, 0, 0, 0],
                        "voxel_size": esdf_vs,
                        "feature_dtype": torch.float16,
                        "feature_tensor": torch.zeros(num_voxels, dtype=torch.float16, device="cuda"),
                    },
                },
            })

        robot_config = get_robot_config(
            robot_file=robot_file, urdf_file_path=urdf_path, logger=self.get_logger(),
        )

        create_kwargs = dict(
            robot=robot_config,
            scene_model=world_file,
            collision_cache={"obb": collision_cache_cuboid, "mesh": collision_cache_mesh},
            num_ik_seeds=int(self.get_parameter("num_graph_seeds").value),
            num_trajopt_seeds=int(self.get_parameter("num_trajopt_seeds").value),
            device_cfg=device_cfg,
            use_cuda_graph=bool(self.get_parameter("enable_cuda_graph").value),
            max_goalset=int(self.get_parameter("max_goalset").value),
            max_batch_size=int(self.get_parameter("max_batch_size").value),
        )
        ik_opt = self.get_parameter("ik_optimizer_config").value
        if ik_opt:
            create_kwargs["ik_optimizer_configs"] = ik_opt
        trajopt_opt = self.get_parameter("trajopt_optimizer_config").value
        if trajopt_opt:
            create_kwargs["trajopt_optimizer_configs"] = trajopt_opt

        cfg = MotionPlannerCfg.create(**create_kwargs)
        motion_planner = MotionPlanner(cfg)
        motion_planner.trajopt_solver.config.interpolation_dt = float(self.get_parameter("interpolation_dt").value)

        if not self.get_parameter("add_ground_plane").value:
            motion_planner.clear_scene_cache()

        motion_planner.warmup(enable_graph=True)
        self.get_logger().info("MotionPlanner warmup complete")

        return motion_planner, grid_size_m, esdf_vs, raw_grid_shape, world_file

    # ------------------------------------------------------------------
    # Joint state buffer (used by multiple handlers)
    # ------------------------------------------------------------------

    def _js_callback(self, msg):
        self._js_buffer = {
            "joint_names": msg.name,
            "position": msg.position,
            "velocity": msg.velocity,
        }

    def _publish_robot_spheres(self):
        if self._sphere_publisher.get_subscription_count() == 0:
            return
        if self._js_buffer is None:
            return
        kin = self._curobo_ctx.motion_planner.kinematics
        names = self._js_buffer["joint_names"]
        pos = self._js_buffer["position"]
        active = kin.joint_names
        try:
            indices = [names.index(n) for n in active]
        except ValueError:
            return
        device = self._curobo_ctx.motion_planner.device_cfg.device
        q = CuJointState.from_position(
            position=torch.tensor([pos[i] for i in indices],
                                  dtype=torch.float32, device=device).unsqueeze(0),
            joint_names=active,
        )
        ks = kin.compute_kinematics(q)
        spheres = ks.robot_spheres.squeeze(1).cpu().numpy()
        t = self.get_clock().now().to_msg()
        m_arr = get_spheres_marker(
            spheres[0], kin.base_link, t, rgb=self._sphere_rgb,
        )
        self._sphere_publisher.publish(m_arr)

    # ------------------------------------------------------------------
    # Action / Service callbacks — each delegates to the handler module
    # ------------------------------------------------------------------

    def _on_plan_motion(self, goal_handle):
        return motion_handler.handle_plan_motion(
            self._curobo_ctx, goal_handle, self._js_buffer, self._lock, self._motion_planner,
        )

    def _on_plan_grasp(self, goal_handle):
        return motion_handler.handle_plan_grasp(
            self._curobo_ctx, goal_handle, self._js_buffer, self._lock, self._motion_planner,
        )

    def _on_attach_object(self, goal_handle):
        return attach_handler.handle_attach_object(
            self._curobo_ctx, goal_handle, self._js_buffer, self._lock,
        )

    def _on_optimize_trajectory(self, goal_handle):
        return trajopt_handler.handle_optimize_trajectory(
            self._curobo_ctx, goal_handle, self._js_buffer, self._lock, self._motion_planner,
        )

    def _on_control_mpc(self, goal_handle):
        return self._mpc_integration.handle_control_mpc(
            self._curobo_ctx, goal_handle, self._lock, self._motion_planner,
        )

    def _on_retarget_motion(self, goal_handle):
        return retargeter_handler.handle_retarget_motion(
            self._curobo_ctx, goal_handle, self._lock, self._motion_planner,
        )

    def _on_move_group(self, goal_handle):
        return moveit_bridge_handler.handle_move_group_action(
            self._curobo_ctx, goal_handle, self._js_buffer, self._lock,
        )

    def _on_compute_ik(self, request, response):
        return ik_handler.handle_compute_ik(
            self._curobo_ctx, request, response, self._lock,
        )

    def _on_compute_fk(self, request, response):
        return fk_handler.handle_compute_fk(
            self._curobo_ctx, request, response, self._lock,
        )

    def _on_check_collision(self, request, response):
        return collision_handler.handle_check_collision(
            self._curobo_ctx, request, response,
        )

    def _on_update_world(self, request, response):
        return world_handler.handle_update_world(
            self._curobo_ctx, request, response,
        )

    def _on_get_esdf(self, request, response):
        return mapping_handler.handle_get_esdf(
            self._curobo_ctx, request, response,
        )

    def _on_publish_static_scene(self, request, response):
        return world_handler.handle_publish_static_scene(
            self._curobo_ctx, request, response,
        )


def main(args=None):
    rclpy.init(args=args)
    node = CuroboServerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down CuroboServerNode")
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()

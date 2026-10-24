// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/core/unified_planner_node.py (M2 subset)
#include "curobo_core/core/unified_planner_node.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

#include <chrono>

#include <tf2/exceptions.h>
#include <tf2/time.h>

#include "curobo_core/core/frame_utils.hpp"
#include "curobo_core/core/gpu_runtime.hpp"
#include "curobo_core/planners/reactive_controller.hpp"
#include "curobo_runtime/runtime.hpp"

namespace curobo_core {

UnifiedPlannerNode::UnifiedPlannerNode(const rclcpp::NodeOptions &options)
    : rclcpp::Node("unified_planner", options) {
  declare_parameters();
  setup();
}

UnifiedPlannerNode::~UnifiedPlannerNode() {
  // The Runtime/interpreter outlives the node (C1: one interpreter per
  // process).
}

// PORT: unified_planner_node.py parameter declarations (M2 subset)
void UnifiedPlannerNode::declare_parameters() {
  declare_parameter<std::string>("robot", "emulator");
  declare_parameter<std::string>("robot_config_file", "");
  // PORT: unified_planner_node.py warmup_ik / warmup_fk (default true)
  declare_parameter<bool>("warmup_ik", true);
  declare_parameter<bool>("warmup_fk", true);
  // PORT: unified_planner_node.py planning parameters (subset used by M3).
  declare_parameter<int>("num_ik_seeds", 32);
  declare_parameter<int>("num_trajopt_seeds", 12);
  declare_parameter<bool>("use_cuda_graph", true);
  declare_parameter<double>("collision_activation_distance", 0.025);
  declare_parameter<double>("interpolation_dt", 0.025);
  declare_parameter<int>("max_batch_size", 1);
  // PORT: config_wrapper_motion._resolve_max_goalset default is 16 (launches
  // may pin it; the deployment does).
  declare_parameter<int>("max_goalset", 1); // PORT: node default
  declare_parameter<int>("collision_cache_cuboid", 32);
  declare_parameter<int>("collision_cache_mesh", 4);
  // PORT: the benchmark's planner envelope + the params it reads/writes.
  declare_parameter<int>("max_attempts", 2);
  // PORT: the Python planner read this from its per-call config with a 1
  // fallback (joint_space_planner.py / single_planner.py:
  // `config.get("enable_graph_attempt", 1)`). `_get_planner_config()` never
  // populated the key, so 1 is the reference value. cuRobo consults the PRM
  // graph planner on attempts where `current_attempt >= enable_graph_attempt`.
  declare_parameter<int>("enable_graph_attempt", 1);
  declare_parameter<bool>("reset_seed_per_plan", false);
  declare_parameter<bool>("load_dynamics", false);
  declare_parameter<double>("robot_payload_mass", 0.0);
  declare_parameter<std::string>("obstacle_collision_mode", "cuboid");
  declare_parameter<double>("time_dilation_factor", 1.0);
  declare_parameter<double>("voxel_size", 0.05);
  declare_parameter<std::vector<double>>("mapper_extent_xyz", {2.0, 2.0, 2.0});
  declare_parameter<std::vector<double>>("mapper_grid_center", {0.0, 0.0, 0.0});
  // PORT: obstacle_manager source defaults. 0 = AUTO: the CameraSystemManager
  // probes the configured cameras' camera_info topics and sizes the Mapper to
  // the LARGEST resolution any of them reports (the depth strategies resample
  // the others to it). Set a positive value to pin the resolution instead.
  declare_parameter<int>("mapper_image_width", 0);
  declare_parameter<int>("mapper_image_height", 0);
  declare_parameter<double>("sparse_voxel_publish_rate", 7.0);
  declare_parameter<bool>("publish_plan_debug_image", false);
  declare_parameter<bool>("publish_collision_spheres", true);
  declare_parameter<bool>("publish_workspace_visualisation", true);
  // PORT: MPC / reactive parameters (design section 7.5).
  declare_parameter<double>("convergence_threshold", 0.01);
  declare_parameter<int>("max_mpc_iterations", 1000);
  declare_parameter<double>("mpc_step_dt", 0.03);
  declare_parameter<int>("mpc_horizon_steps", 30);
  declare_parameter<int>("mpc_warm_start_iters", 5);
  declare_parameter<int>("mpc_cold_start_iters", 10);
  declare_parameter<std::string>("mpc_solver_type", "mppi_acceleration");
  declare_parameter<int>("mpc_mppi_num_particles", 400);
  declare_parameter<double>("mpc_vel_feedback_alpha", 1.0);
  declare_parameter<double>("mpc_command_interval", 0.24);
  // PORT: retarget_* (MotionRetargeter teleop).
  declare_parameter<double>("retarget_position_weight", 1.0);
  declare_parameter<double>("retarget_orientation_weight", 1.0);
  declare_parameter<bool>("retarget_use_mpc", false);
  // PORT: robot description / config manager (design section 7.5).
  declare_parameter<std::string>("urdf_path", "");
  declare_parameter<std::string>("asset_root_path", "");
  declare_parameter<std::string>("world_file", "");
  declare_parameter<std::string>("cameras_config_file", "");
  declare_parameter<std::string>("planner_type", "classic");
  declare_parameter<bool>("use_mapper", true);
  declare_parameter<bool>("node_is_available", true);
  declare_parameter<int>("warmup_ik_batch_size", 1);
  declare_parameter<int>("warmup_fk_batch_size", 1);
  // PORT: IK / solver tunables.
  declare_parameter<bool>("ik_collision_check", true);
  declare_parameter<bool>("ik_use_lm_seed", true);
  declare_parameter<bool>("torch_sync", false);
  declare_parameter<bool>("gpu_exclusive_plans", true);
  declare_parameter<bool>("exit_on_gpu_fault", true);
  declare_parameter<bool>("enable_curobo_debug_mode", false);
  declare_parameter<int>("graph_new_nodes_per_iteration", 20);
  declare_parameter<int>("graph_max_path_finding_iterations", 10);
  declare_parameter<int>("graph_max_nodes", 20000);
  // PORT: Mapper / ESDF tunables.
  declare_parameter<double>("decay_half_life_s", 0.7);
  declare_parameter<double>("esdf_refresh_min_interval", 2.0);
  // How often the collision model is refreshed from the unpinned lock joints
  // (lock_joints: {joint: null}, e.g. the PTU gimbal). Cheap: only calls the
  // bridge when a value actually changed.
  declare_parameter<double>("unpinned_lock_update_period", 0.1);
  // A push costs a full model rewrite in the bridge, so ignore movements below
  // this (rad/m) - encoder noise on a held joint is not a change of the
  // mechanism. 0 restores exact comparison.
  declare_parameter<double>("unpinned_lock_deadband", 1.0e-4);
  declare_parameter<bool>("push_esdf_to_solvers", true);
  declare_parameter<double>("mapper_stats_period", 60.0);
  declare_parameter<double>("curobo_mesh_cache_log_interval", 5.0);
  declare_parameter<double>("mapper_voxel_size", 0.02);
  declare_parameter<double>("mapper_depth_min", 0.1);
  declare_parameter<double>("mapper_depth_max", 5.0);
  declare_parameter<double>("mapper_truncation_distance", 0.04);
  declare_parameter<double>("mapper_minimum_tsdf_weight", 0.1);
  declare_parameter<int>("mapper_block_size", 8);
  declare_parameter<std::string>("mapper_seeding_method", "gather");
  declare_parameter<int>("mapper_max_support_pixels_per_block_camera", 8);
  // PORT: enable_robot_segmentation (default true) + the masking parameters.
  declare_parameter<bool>("enable_robot_segmentation", true);
  declare_parameter<double>("robot_segmentation_mask_margin", 0.0);
  declare_parameter<double>("robot_segmentation_distance_threshold", 0.05);
}

// PORT: unified_planner_node.py:__init__ solver setup (M2 subset)
void UnifiedPlannerNode::setup() {
  curobo_runtime::Runtime &runtime = curobo_runtime::Runtime::instance();
  if (!runtime.started()) {
    runtime.start();
  }
  bridge_ = &runtime.bridge();
  handle_ = runtime.handle();

  // Enable cuRobo's CUDA-graph RESET (CUDA 12+). Without it
  // SolverCore.reset_cuda_graph() - reached from reset_shape() when the solve
  // structure changes, e.g. a pose<->cspace switch - raises and the stale graph
  // is replayed instead of being re-captured, which faults the CUDA context.
  try {
    bridge_->set_global_flags(
        true, get_parameter("enable_curobo_debug_mode").as_bool());
  } catch (const std::exception &e) {
    RCLCPP_WARN(get_logger(), "could not enable CUDA-graph reset: %s",
                e.what());
  }

  // PORT: unified_planner_node.__init__ - create the GPU lanes BEFORE any
  // camera or solver, so they exist when the first callback fires. Bound to the
  // shared Runtime barrier (see the header): a lane job takes it shared inside
  // GpuLane::run_job, and the capture/rebuild sites still take it exclusive.
  {
    curobo_runtime::GpuLaneOptions lane_opts;
    lane_opts.on_poisoned = [this](curobo_runtime::GpuLane *lane,
                                   std::exception_ptr error) {
      on_gpu_poisoned(lane, error);
    };
    auto &gpu_barrier = runtime.barrier();
    auto make_lane = [&](const char *name) {
      return std::make_unique<curobo_runtime::GpuLane>(name, gpu_barrier,
                                                       bridge_, lane_opts);
    };
    lane_reactive_ = make_lane("reactive");
    lane_plan_ = make_lane("plan");
    lane_kin_ = make_lane("kin");
    lane_perception_ = make_lane("perception");
    lane_viz_ = make_lane("viz");
  }

  // PORT: robot_description.load_robot_description. The descriptor (robots/
  // <robot>.yaml) selects the control strategy + its params, the base link and
  // the cuRobo config, and carries the `home` named joint config; without it
  // the node falls back to its parameter defaults (emulator / base_0).
  curobo_runtime::RobotDescriptionOutcome desc;
  try {
    desc = bridge_->robot_description(
        get_parameter("robot").as_string(),
        get_parameter("robot_config_file").as_string());
  } catch (const std::exception &e) {
    RCLCPP_WARN(get_logger(), "robot descriptor load failed: %s", e.what());
  }
  if (desc.found) {
    RCLCPP_INFO(get_logger(), "Loaded robot descriptor '%s' (%s)",
                desc.name.c_str(), desc.path.c_str());
  }

  // base_link: descriptor default unless overridden at launch.
  if (!has_parameter("base_link")) {
    declare_parameter<std::string>(
        "base_link",
        (desc.found && !desc.base_link.empty()) ? desc.base_link : "base_0");
  }
  base_link_ = get_parameter("base_link").as_string();

  // control_strategy: descriptor default unless overridden at launch.
  if (!has_parameter("control_strategy")) {
    declare_parameter<std::string>(
        "control_strategy",
        (desc.found && !desc.strategy.empty()) ? desc.strategy : "emulator");
  }

  std::string robot_config_file =
      get_parameter("robot_config_file").as_string();
  if (robot_config_file.empty() && desc.found) {
    robot_config_file = desc.curobo_config;
  }

  if (!robot_config_file.empty()) {
    // PORT: ConfigManager._translate_xrdf - xrdf configs need the urdf_path /
    // asset_root_path parameters (they are not inside the xrdf file).
    // The robot-masking knobs ride along here because robot_segment() reads
    // them from this handle's registry entry: without them the bridge silently
    // falls back to its 0.05 / 0.0 defaults and
    // robot_segmentation_distance_threshold / _mask_margin were accepted
    // (declared) but never took effect.
    std::ostringstream model_cfg;
    model_cfg
        << "{\"robot_config_file\":\"" << robot_config_file << "\""
        << ",\"urdf_path\":\"" << get_parameter("urdf_path").as_string() << "\""
        << ",\"asset_root_path\":\""
        << get_parameter("asset_root_path").as_string()
        << "\""
        // PORT: RobotModelManager.set_torque_mode - the pinocchio
        // inverse-dynamics model is only built for torque-limited runs.
        // Without this the bridge attached it unconditionally, adding a
        // joint_torque (RNEA) term to every rollout iteration.
        << ",\"load_dynamics\":"
        << (get_parameter("load_dynamics").as_bool() ? "true" : "false")
        << ",\"seg_distance_threshold\":"
        << get_parameter("robot_segmentation_distance_threshold").as_double()
        << ",\"seg_mask_margin\":"
        << get_parameter("robot_segmentation_mask_margin").as_double() << "}";
    try {
      bridge_->robot_model_build(handle_, model_cfg.str());
    } catch (const std::exception &e) {
      RCLCPP_FATAL(get_logger(),
                   "Failed to build the robot model from '%s':\n  %s\n"
                   "Fix the robot config / parameters above and relaunch.",
                   robot_config_file.c_str(), e.what());
      throw;
    }
    const auto limits = bridge_->joint_limits(handle_);
    joint_names_ = limits.joint_names;
    // `lock_joints: {joint: null}` - held out of the cspace but NOT pinned: the
    // collision model must follow them from the joint state (see
    // push_unpinned_lock_joints), e.g. the PTU gimbal.
    unpinned_lock_joint_names_ = limits.unpinned_lock_joint_names;
    if (!unpinned_lock_joint_names_.empty()) {
      for (const auto & name : unpinned_lock_joint_names_) {
        unpinned_lock_joint_list_ +=
            (unpinned_lock_joint_list_.empty() ? "" : ", ") + name;
      }
      RCLCPP_INFO(
          get_logger(),
          "Collision model tracks %zu unpinned lock joint(s) from the joint "
          "state: %s", unpinned_lock_joint_names_.size(),
          unpinned_lock_joint_list_.c_str());
    }

    // PORT: ConfigWrapperMotion.set_motion_gen_config (M3 subset). Build the
    // shared MotionPlanner with the frozen recipe.
    std::ostringstream cfg;
    cfg << "{\"num_ik_seeds\":" << get_parameter("num_ik_seeds").as_int()
        << ",\"num_trajopt_seeds\":"
        << get_parameter("num_trajopt_seeds").as_int() << ",\"use_cuda_graph\":"
        << (resolve_use_cuda_graph(this) ? "true" : "false")
        << ",\"collision_activation_distance\":"
        << get_parameter("collision_activation_distance").as_double()
        << ",\"interpolation_dt\":"
        << get_parameter("interpolation_dt").as_double()
        << ",\"max_batch_size\":" << get_parameter("max_batch_size").as_int()
        << ",\"max_goalset\":" << get_parameter("max_goalset").as_int()
        // The plan-request attempt budget. The bridge records it so its
        // post-world-push CUDA-graph re-capture warms exactly the paths a plan
        // replays (and skips the PRM warmup when `enable_graph_attempt >=
        // max_attempts`, i.e. when no attempt can reach the graph planner).
        << ",\"max_attempts\":" << get_parameter("max_attempts").as_int()
        << ",\"enable_graph_attempt\":"
        << get_parameter("enable_graph_attempt").as_int()
        << ",\"obstacle_collision_mode\":\""
        << get_parameter("obstacle_collision_mode").as_string() << "\""
        << ",\"collision_cache\":{\"cuboid\":"
        << get_parameter("collision_cache_cuboid").as_int() << ",\"mesh\":"
        << get_parameter("collision_cache_mesh").as_int()
        // PORT: obstacle_manager.collision_cache["voxel"] - WITHOUT this the
        // solver allocates no voxel storage and the first ESDF world push
        // raises "Voxel cache not initialized".
        << ",\"voxel\":{\"layers\":1,\"dims\":[";
    const auto cache_extent =
        get_parameter("mapper_extent_xyz").as_double_array();
    for (std::size_t i = 0; i < cache_extent.size(); ++i) {
      cfg << (i ? "," : "") << cache_extent[i];
    }
    // Three closers: voxel object, collision_cache object, top-level object.
    cfg << "],\"voxel_size\":" << get_parameter("voxel_size").as_double()
        << "}}}";
    // PORT: unified_planner_node "Warming up {type} planner..." / "  ->
    // Initializing MotionPlanner..." / "{type} planner ready".
    RCLCPP_INFO(get_logger(), "Warming up %s planner...",
                get_parameter("planner_type").as_string().c_str());
    RCLCPP_INFO(get_logger(), "  -> Initializing MotionPlanner...");
    // PORT: the build (warmup) captures a CUDA graph -> hold the exclusive GPU
    // barrier (matches the other builders and the config-service rebuild path).
    {
      auto gpu_guard =
          curobo_runtime::Runtime::instance().barrier().exclusive();
      bridge_->motion_planner_build(handle_, cfg.str());
    }
    RCLCPP_INFO(get_logger(), "%s planner ready",
                get_parameter("planner_type").as_string().c_str());

    // PORT: ObstacleManager._build_mapper (node mapper_* parameters). The cfg
    // itself is built by configure_mapper_from_params() (below) so that the
    // `set_workspace` service can rebuild it from the CURRENT parameters rather
    // than a cfg baked once at startup.
  }

  config_services_ =
      std::make_unique<ConfigServices>(this, bridge_, handle_, &joint_names_);
  // `set_workspace` changes mapper_extent_xyz/mapper_grid_center: the Mapper
  // grid must be rebuilt to the new box and the marker redrawn, so hand the
  // node's own rebuild + the marker publisher to the service.
  config_services_->set_mapper_reconfigure(
      [this]() { configure_mapper_from_params(); });
  config_services_->set_workspace_republish([this]() {
    if (ros_service_manager_) {
      ros_service_manager_->publish_mapper_workspace();
    }
  });
  config_services_->set_integration_frame_provider(
      [this]() { return integration_frame(); });
  obstacle_manager_ = std::make_unique<ObstacleManager>(bridge_, handle_);
  // PORT: unified_planner_node._esdf_refresh_min_interval (open-loop throttle).
  obstacle_manager_->set_esdf_refresh_min_interval(
      get_parameter("esdf_refresh_min_interval").as_double());
  // A rebuild that could not push its world must not be mistaken for a fresh
  // perception world: invalidate so the next plan rebuilds + re-pushes.
  config_services_->set_world_invalidate([this]() {
    if (obstacle_manager_) {
      obstacle_manager_->invalidate_world();
    }
  });
  ros_service_manager_ = std::make_unique<RosServiceManager>(
      this, bridge_, handle_, obstacle_manager_.get());
  // PORT: ros_service_manager.init_services - the GPU-backed viz timers run on
  // lane_viz and drop the cycle when it is busy.
  ros_service_manager_->set_viz_lane(lane_viz_.get());
  ros_service_manager_->set_integration_frame_provider(
      [this]() { return integration_frame(); });
  ros_service_manager_->init_services();

  // PORT: ConfigWrapperMotion.camera_system_manager / laser_system_manager.
  // Both are no-ops with no `camera_topic` / `laser_topic` configured, and
  // their config classes declare the perception parameter blocks on this node.
  // A depth integrate bumps the perception epoch (drives should_refresh_world).
  // PORT: unified_planner_node.__init__ - the depth/lidar integrates run on
  // lane_perception (one owner thread for the shared Mapper) and drop the frame
  // when the lane is busy, instead of every strategy worker taking the barrier
  // inline and integrating concurrently.
  // PORT: unified_planner_node._perception_callback_group - a dedicated group
  // so long generate_trajectory service callbacks cannot starve the camera
  // streams (see the header). Reentrant: the two feeds may decode in parallel;
  // the GPU barrier serializes the actual bridge calls.
  perception_callback_group_ = create_callback_group(
      rclcpp::CallbackGroupType::Reentrant);

  camera_system_manager_ = std::make_unique<CameraSystemManager>(
      this, bridge_, handle_, lane_perception_.get(), [this]() {
        if (obstacle_manager_) {
          obstacle_manager_->bump_perception_epoch();
        }
      },
      perception_callback_group_);
  // PORT: unified_planner_node.__init__ - the lidar integrates share
  // lane_perception with the depth integrates (one Mapper owner thread).
  laser_system_manager_ = std::make_unique<LaserSystemManager>(
      this, bridge_, handle_, lane_perception_.get());

  // PORT: obstacle_manager._resolve_time_decay - cuRobo applies decay_factor
  // once per integrate() call, so anchor it on a HALF-LIFE and the combined
  // camera+laser rate; 1.0 means "no decay" and the map accumulates without
  // bound (edge noise crosses the occupancy threshold forever, so the collision
  // world fills up).
  {
    const double total_rate =
        (camera_system_manager_ && camera_system_manager_->get_camera_context()
             ? camera_system_manager_->get_camera_context()
                   ->get_total_frame_rate_hz()
             : 0.0) +
        (laser_system_manager_ && laser_system_manager_->get_laser_context()
             ? laser_system_manager_->get_laser_context()
                   ->get_total_frame_rate_hz()
             : 0.0);
    const double half_life = get_parameter("decay_half_life_s").as_double();
    double decay = 1.0;
    if (half_life <= 0.0) {
      RCLCPP_INFO(
          get_logger(),
          "decay_half_life_s <= 0: TSDF decay disabled (voxels never fade).");
    } else if (total_rate <= 0.0) {
      RCLCPP_WARN(get_logger(),
                  "No camera frame rate available: cannot convert "
                  "decay_half_life_s=%gs into a "
                  "per-integrate factor. Falling back to no decay (the map "
                  "will accumulate). "
                  "Declare 'camera_frame_rate_hz' in the camera params.",
                  half_life);
    } else {
      decay = std::pow(0.5, 1.0 / (total_rate * half_life));
    }
    // The mapper's depth-projection resolution: the CameraSystemManager has
    // already resolved it (explicit mapper_image_width/height, else the largest
    // configured camera, else the default) and its depth strategies resample to
    // the same value, so the two always agree.
    int mapper_w = get_parameter("mapper_image_width").as_int();
    int mapper_h = get_parameter("mapper_image_height").as_int();
    if (camera_system_manager_ != nullptr) {
      if (camera_system_manager_->mapper_image_width() > 0) {
        mapper_w = camera_system_manager_->mapper_image_width();
      }
      if (camera_system_manager_->mapper_image_height() > 0) {
        mapper_h = camera_system_manager_->mapper_image_height();
      }
    }
    if (mapper_w <= 0) {
      mapper_w = DEFAULT_MAPPER_IMAGE_WIDTH;
    }
    if (mapper_h <= 0) {
      mapper_h = DEFAULT_MAPPER_IMAGE_HEIGHT;
    }
    mapper_image_w_ = mapper_w;
    mapper_image_h_ = mapper_h;
    mapper_decay_ = decay;
    // Depth is integrated around the robot base: `mapper_grid_center` is
    // base-relative, so the Mapper box always sits in front of the robot and
    // the voxels ride with it. See integration_frame() in camera_cfg.cpp.
    configure_mapper_from_params();
    RCLCPP_INFO(get_logger(),
                "Mapper configured: %dx%d, decay_factor=%.6f (half_life=%.2fs, "
                "total_frame_rate=%.1fHz)",
                mapper_w, mapper_h, decay, half_life, total_rate);
    if (ros_service_manager_) {
      ros_service_manager_->publish_mapper_workspace();
    }
  }

  // PORT: unified_planner_node depth-map robot segmentation (gated by
  // enable_robot_segmentation). Only cameras with a depth topic and a
  // camera_purpose of 'all' or 'segmentation' take part; its mask services are
  // registered on the shared RosServiceManager (node-namespaced).
  if (get_parameter("enable_robot_segmentation").as_bool()) {
    std::vector<PerceptionCameraCfg> seg_cfgs;
    for (const auto &cfg : camera_system_manager_->camera_cfgs()) {
      if (cfg.for_segmentation() && !cfg.depth_topic.empty()) {
        seg_cfgs.push_back(cfg);
      }
    }
    if (seg_cfgs.empty()) {
      RCLCPP_WARN(
          get_logger(),
          "enable_robot_segmentation requires at least one camera with a depth "
          "topic and a camera_purpose of 'all' or 'segmentation': set "
          "'camera_topic' (and camera_info_topic) at launch. Segmentation "
          "disabled.");
    } else {
      robot_segmentation_ = std::make_unique<RobotSegmentation>(
          this, bridge_, handle_, seg_cfgs, [this](std::int64_t stamp_ns) {
            // PORT: _joint_pose_at - the capture-time pose wins; the live pose
            // is the fallback while the timestamped feedback buffer fills.
            if (robot_context_ != nullptr) {
              const auto pose_at = robot_context_->get_joint_pose_at(stamp_ns);
              if (pose_at.has_value() && !pose_at->empty()) {
                return *pose_at;
              }
              const auto live = robot_context_->get_joint_pose();
              if (!live.empty()) {
                return live;
              }
            }
            const std::vector<float> pose = current_pose();
            return std::vector<double>(pose.begin(), pose.end());
          },
          perception_callback_group_);
      ros_service_manager_->register_robot_segmentation(
          robot_segmentation_.get());
    }
  }

  fk_services_ =
      std::make_unique<FKServices>(this, bridge_, handle_, joint_names_);
  ik_services_ =
      std::make_unique<IKServices>(this, bridge_, handle_, base_link_);
  // PORT: ik_services._on_lane / fk_services._on_lane - both solve services
  // share lane_kin, so FK and IK never issue kinematics kernels concurrently.
  fk_services_->set_lane(lane_kin_.get());
  ik_services_->set_lane(lane_kin_.get());
  collision_distance_ = std::make_unique<CollisionDistance>(
      this, bridge_, handle_, [this]() { return current_pose(); });
  attachment_services_ = std::make_unique<AttachmentServices>(
      this, bridge_, handle_, obstacle_manager_.get(),
      [this]() { return current_pose(); });

  // PORT: reachability_services.ReachabilityServices (/generate_rm).
  if (tf_buffer_ == nullptr) {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    // Populated by this node's executor thread, not the caller: required for a
    // lookup timeout to be honoured (see camera_strategy.cpp).
    tf_buffer_->setUsingDedicatedThread(true);
  }
  reachability_services_ = std::make_unique<ReachabilityServices>(
      this, ik_services_.get(), base_link_, tf_buffer_);

  // Create the open-loop planning services (generate_trajectory is the
  // readiness signal the record/replay harness waits for), then the robot
  // context and the SendTrajectory action over it.
  planner_services_ = std::make_unique<PlannerServices>(
      this, bridge_, handle_, joint_names_, obstacle_manager_.get());
  // PORT: unified_planner_node.lane_plan - the plan body runs on the plan lane.
  planner_services_->set_plan_lane(lane_plan_.get());
  // PORT: unified_planner_node.lane_reactive - MPC/retarget servo jobs.
  planner_services_->set_reactive_lane(lane_reactive_.get());
  // PORT: unified_planner_node.__init__ - warm the CONFIGURED planner type
  // instead of always hard-coding "classic" (the manager ctor default).
  planner_services_->manager().set_current_planner(
      get_parameter("planner_type").as_string());
  // PORT: frame_utils.resolve_poses_to_base - the node owns the TF buffer, so
  // the plan path resolves a goalset's non-base frame through it.
  planner_services_->set_pose_resolver(
      [this](const std::vector<geometry_msgs::msg::Pose> &poses,
             const std::string &frame) {
        return resolve_poses_to_base(poses, frame, base_link_,
                                     tf_buffer_.get());
      });
  // PORT: robot_context - the control strategy comes from `control_strategy`,
  // its params from the descriptor's strategy_params (falling back to the
  // emulator's joint_states topic), and the seed pose from `home`.
  std::map<std::string, std::string> strategy_params = desc.strategy_params;
  if (strategy_params.find("joint_states_topic") == strategy_params.end() &&
      get_parameter("control_strategy").as_string() == "emulator") {
    strategy_params["joint_states_topic"] = "/emulator/joint_states";
  }
  std::vector<double> default_home(joint_names_.size(), 0.0);
  if (desc.found && desc.home_names.size() == joint_names_.size()) {
    for (std::size_t i = 0; i < joint_names_.size(); ++i) {
      const auto it = std::find(desc.home_names.begin(), desc.home_names.end(),
                                joint_names_[i]);
      if (it != desc.home_names.end()) {
        default_home[i] = desc.home_positions[static_cast<std::size_t>(
            it - desc.home_names.begin())];
      }
    }
  }
  robot_context_ = std::make_unique<RobotContext>(
      this, get_parameter("interpolation_dt").as_double(), joint_names_,
      get_parameter("control_strategy").as_string(), strategy_params,
      default_home);
  // The unpinned lock joints (lock_joints: {joint: null}) ride the control
  // strategy's joint-state topic but are not part of the cspace pose. The
  // strategy forwards the raw message; a low-rate timer pushes the values into
  // the collision model on the perception lane. Nothing is wired up when the
  // robot has none.
  if (!unpinned_lock_joint_names_.empty()) {
    unpinned_lock_deadband_ = get_parameter("unpinned_lock_deadband").as_double();
    robot_context_->set_joint_state_callback(
        [this](const sensor_msgs::msg::JointState &msg) {
          handle_joint_state_for_locks(msg);
        });
    unpinned_lock_timer_ = create_wall_timer(
        std::chrono::duration<double>(
            get_parameter("unpinned_lock_update_period").as_double()),
        [this]() {push_unpinned_lock_joints();});
  }
  // PORT: single_planner holds a reference to the RobotContext so a
  // compute-only plan can be staged into the command buffer for preview
  // (GhostStrategy publishes it on <node>/trajectory without executing it).
  planner_services_->set_robot_context(robot_context_.get());
  action_server_ = std::make_unique<TrajectoryActionServer>(
      this, planner_services_.get(), robot_context_.get(), joint_names_);

  // PORT: the collision-sphere viz reads the robot's live joint pose.
  ros_service_manager_->set_pose_provider([this]() { return current_pose(); });
  planner_services_->set_current_pose_provider(
      [this]() { return current_pose(); });

  // PORT: unified_planner_node.mpc_goal_callback - live reactive-goal updates.
  if (tf_buffer_ == nullptr) {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    // Populated by this node's executor thread, not the caller: required for a
    // lookup timeout to be honoured (see camera_strategy.cpp).
    tf_buffer_->setUsingDedicatedThread(true);
  }
  // PORT: unified_planner_node._publish_tool_pose_callback (1 Hz, TF-driven).
  tool_pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
      std::string(get_name()) + "/current_tool_pose", 10);
  tool_pose_timer_ = create_wall_timer(std::chrono::seconds(1),
                                       [this]() { publish_tool_pose(); });
  mpc_goal_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      std::string(get_name()) + "/mpc_goal", 10,
      [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        handle_mpc_goal(*msg);
      });

  // PORT: the node's startup default warmup. The Python server warms both up in
  // __init__, so the recorded /warmup_ik and /warmup_fk replies are
  // "... already ready"; reproduce that here.
  if (get_parameter("warmup_ik").as_bool()) {
    try {
      RCLCPP_INFO(
          get_logger(), "%s",
          ik_services_->warmup(get_parameter("warmup_ik_batch_size").as_int())
              .c_str());
    } catch (const std::exception &e) {
      RCLCPP_WARN(get_logger(), "IK startup warmup failed: %s", e.what());
    }
  }
  if (get_parameter("warmup_fk").as_bool()) {
    try {
      RCLCPP_INFO(
          get_logger(), "%s",
          fk_services_->warmup(get_parameter("warmup_fk_batch_size").as_int())
              .c_str());
    } catch (const std::exception &e) {
      RCLCPP_WARN(get_logger(), "FK startup warmup failed: %s", e.what());
    }
  }

  RCLCPP_INFO(get_logger(), "UnifiedPlannerNode (M2 services) ready");
}

// Rebuild the Mapper (TSDF/ESDF) from the current mapper_* parameters. Used at
// startup and by the `set_workspace` service, which changes mapper_extent_xyz /
// mapper_grid_center and must reallocate the Mapper's grid to the new box.
// Needs a positive image size (0 == auto never reaches here: the startup path
// resolves it first).
void UnifiedPlannerNode::configure_mapper_from_params() {
  // PORT: ObstacleManager._build_mapper / _load_perception_params. Built fresh
  // from the node's mapper_* parameters every time, so set_workspace (which
  // rewrites mapper_extent_xyz / mapper_grid_center) reallocates the grid to
  // the new box. The TSDF voxel size is `mapper_voxel_size`; the ESDF size is
  // `voxel_size`.
  std::ostringstream mapper_cfg;
  mapper_cfg
      << "{\"voxel_size\":" << get_parameter("mapper_voxel_size").as_double()
      << ",\"depth_min\":" << get_parameter("mapper_depth_min").as_double()
      << ",\"depth_max\":" << get_parameter("mapper_depth_max").as_double()
      << ",\"truncation_distance\":"
      << get_parameter("mapper_truncation_distance").as_double()
      << ",\"minimum_tsdf_weight\":"
      << get_parameter("mapper_minimum_tsdf_weight").as_double()
      << ",\"block_size\":" << get_parameter("mapper_block_size").as_int()
      << ",\"seeding_method\":\""
      << get_parameter("mapper_seeding_method").as_string() << "\""
      << ",\"max_support_pixels_per_block_camera\":"
      << get_parameter("mapper_max_support_pixels_per_block_camera").as_int()
      << ",\"grid_center\":[";
  const auto center = get_parameter("mapper_grid_center").as_double_array();
  for (std::size_t i = 0; i < center.size(); ++i) {
    mapper_cfg << (i ? "," : "") << center[i];
  }
  mapper_cfg << "],\"extent_xyz\":[";
  const auto extent = get_parameter("mapper_extent_xyz").as_double_array();
  for (std::size_t i = 0; i < extent.size(); ++i) {
    mapper_cfg << (i ? "," : "") << extent[i];
  }
  mapper_cfg << "],\"esdf_voxel_size\":"
             << get_parameter("voxel_size").as_double()
             << ",\"image_width\":" << mapper_image_w_
             << ",\"image_height\":" << mapper_image_h_
             << ",\"decay_factor\":" << mapper_decay_ << "}";
  // The mapper's Warp kernels load lazily; the recycle/decay kernels only load
  // once frames need block recycling - long after startup and often while the
  // plan lane is capturing a CUDA graph, where the load fails ("misaligned
  // address" in wp_cuda_load_module) and poisons the CUDA context. Configure
  // and warm the mapper under the exclusive GPU barrier so every module is
  // compiled and resident before any planning.
  {
    auto gpu_guard = curobo_runtime::Runtime::instance().barrier().exclusive();
    bridge_->configure_mapper(handle_, mapper_cfg.str());
    bridge_->warmup_mapper(handle_);
  }
}

// Frame the depth is integrated in: always the configured robot base. Exposed
// so the mapper_workspace marker and the workspace service report the same
// frame the voxels are in (see integration_frame() in camera_cfg.cpp).
std::string UnifiedPlannerNode::integration_frame() { return base_link_; }

// PORT: unified_planner_node.mpc_goal_callback
void UnifiedPlannerNode::handle_mpc_goal(
    const geometry_msgs::msg::PoseStamped &msg) {
  TrajectoryPlanner *planner =
      planner_services_ ? planner_services_->manager().current() : nullptr;
  auto *reactive = dynamic_cast<ReactiveController *>(planner);
  if (reactive == nullptr) {
    RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Received reactive goal but current planner is %s - goal ignored",
        planner != nullptr ? planner->get_planner_name().c_str() : "none");
    return;
  }
  geometry_msgs::msg::Pose pose;
  try {
    pose = resolve_to_base(msg.pose, msg.header.frame_id, base_link_,
                           tf_buffer_.get());
  } catch (const std::exception &e) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                         "Reactive goal dropped (%s)", e.what());
    return;
  }
  reactive->set_live_goal({pose.position.x, pose.position.y, pose.position.z,
                           pose.orientation.w, pose.orientation.x,
                           pose.orientation.y, pose.orientation.z});
  RCLCPP_DEBUG(get_logger(),
               "Reactive goal updated from topic: [%.3f, %.3f, %.3f]",
               pose.position.x, pose.position.y, pose.position.z);
}

// PORT: unified_planner_node._publish_tool_pose_callback
void UnifiedPlannerNode::publish_tool_pose() {
  if (tool_pose_pub_ == nullptr || tf_buffer_ == nullptr) {
    return;
  }
  try {
    const auto frames = bridge_->tool_frames(handle_);
    if (frames.empty()) {
      return;
    }
    // The tool frame is the same frame /ik_batch solves for.
    const auto tf =
        tf_buffer_->lookupTransform(base_link_, frames[0], tf2::TimePointZero);
    geometry_msgs::msg::PoseStamped msg;
    msg.header.stamp = get_clock()->now();
    msg.header.frame_id = base_link_;
    // Pose of the tool frame origin == the transform applied to identity.
    msg.pose = apply_transform(geometry_msgs::msg::Pose(), tf);
    tool_pose_pub_->publish(msg);
  } catch (const std::exception &) {
    // TF/solver not ready yet; retry next tick.
  }
}

// The control strategy forwards its raw joint-state message, before it reduces
// it to the cspace pose. Keep the latest value of every unpinned lock joint.
void UnifiedPlannerNode::handle_joint_state_for_locks(
    const sensor_msgs::msg::JointState &msg) {
  if (unpinned_lock_joint_names_.empty() || msg.name.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lk(unpinned_lock_mutex_);
  for (const auto &name : unpinned_lock_joint_names_) {
    const auto it = std::find(msg.name.begin(), msg.name.end(), name);
    if (it == msg.name.end()) {
      continue;
    }
    const auto index = static_cast<std::size_t>(it - msg.name.begin());
    if (index < msg.position.size()) {
      unpinned_lock_positions_[name] = msg.position[index];
    }
  }
}

// Push the latest unpinned lock values into the collision model. Submitted to
// lane_perception at PRIO_PERCEPTION (drop-if-busy) - the same lane the depth
// integrates use - so it is ordered with the rest of the perception work and can
// never queue up behind a plan. The bridge rewrites the joints' baked transforms
// in place (no model rebuild, no solver rebuild) and this returns early when
// nothing moved, so the periodic call is essentially free at rest.
void UnifiedPlannerNode::push_unpinned_lock_joints() {
  if (unpinned_lock_joint_names_.empty()) {
    return;
  }
  std::map<std::string, double> latest;
  {
    std::lock_guard<std::mutex> lk(unpinned_lock_mutex_);
    if (unpinned_lock_positions_.empty()) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 10000,
          "the collision model tracks the unpinned lock joints [%s] but none of "
          "them has appeared on the joint-state topic yet - the model keeps "
          "their configured values and the collision geometry does not follow "
          "the real mechanism", unpinned_lock_joint_list_.c_str());
      return;
    }
    bool changed =
        unpinned_lock_positions_.size() != pushed_unpinned_lock_positions_.size();
    if (!changed) {
      for (const auto &[name, value] : unpinned_lock_positions_) {
        const auto it = pushed_unpinned_lock_positions_.find(name);
        if (it == pushed_unpinned_lock_positions_.end() ||
          std::abs(it->second - value) > unpinned_lock_deadband_) {
          changed = true;
          break;
        }
      }
    }
    if (!changed) {
      return;
    }
    latest = unpinned_lock_positions_;
  }
  std::ostringstream json;
  json << "{";
  bool first = true;
  for (const auto &[name, value] : latest) {
    json << (first ? "" : ",") << "\"" << name << "\":" << value;
    first = false;
  }
  json << "}";
  const std::string payload = json.str();
  auto job = [this, payload, latest]() {
    try {
      bridge_->robot_model_update_lock_joints(handle_, payload);
      std::lock_guard<std::mutex> lk(unpinned_lock_mutex_);
      pushed_unpinned_lock_positions_ = latest;
    } catch (const std::exception &e) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "Failed to update the unpinned lock joints: %s", e.what());
    }
  };
  if (lane_perception_ != nullptr) {
    curobo_runtime::SubmitOptions opts;
    opts.prio = curobo_runtime::PRIO_PERCEPTION;
    lane_perception_->submit_if_idle(opts, std::move(job));
  } else {
    job();
  }
}

// PORT: unified_planner_node.py emulator live pose (RobotContext at M3/M5)
std::vector<float> UnifiedPlannerNode::current_pose() const {
  // PORT: the emulator's live joint pose (RobotContext owns it once built).
  if (robot_context_ != nullptr) {
    const auto pose = robot_context_->get_joint_pose();
    if (!pose.empty()) {
      return std::vector<float>(pose.begin(), pose.end());
    }
  }
  return std::vector<float>(joint_names_.size(), 0.0f);
}

// PORT: unified_planner_node._on_gpu_poisoned - a lane hit a fatal CUDA error.
// The context is unusable: fail loudly, then exit rather than keep serving.
void UnifiedPlannerNode::on_gpu_poisoned(curobo_runtime::GpuLane *lane,
                                         std::exception_ptr error) {
  std::string what = "unknown error";
  try {
    std::rethrow_exception(error);
  } catch (const std::exception &e) {
    what = e.what();
  } catch (...) {
  }
  RCLCPP_FATAL(get_logger(), "GPU lane '%s' poisoned: %s",
               lane != nullptr ? lane->name().c_str() : "?", what.c_str());
  const bool exit_flag = !has_parameter("exit_on_gpu_fault") ||
                         get_parameter("exit_on_gpu_fault").as_bool();
  if (!exit_flag) {
    return;
  }
  RCLCPP_FATAL(
      get_logger(),
      "exit_on_gpu_fault=true: stopping instead of serving a corrupt context");
  rclcpp::shutdown();
  std::fflush(nullptr);
  std::_Exit(EXIT_FAILURE);
}

} // namespace curobo_core

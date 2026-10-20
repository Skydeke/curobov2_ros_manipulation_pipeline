// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of the config/cache/joint-lock service callbacks.
#include "curobo_core/core/config_services.hpp"
#include "curobo_core/core/gpu_runtime.hpp"

#include <algorithm>
#include <sstream>

#include <std_srvs/srv/trigger.hpp>

#include "curobo_runtime/runtime.hpp"
#include "curobo_core/cameras/camera_cfg.hpp"

namespace curobo_core
{

ConfigServices::ConfigServices(
  rclcpp::Node * node, curobo_runtime::IBridge * bridge,
  curobo_runtime::RobotHandle handle, std::vector<std::string> * joint_names)
: node_(node), bridge_(bridge), handle_(handle), joint_names_(joint_names)
{
  const std::string name = node_->get_name();
  cache_srv_ = node_->create_service<curobov2_ros_interfaces::srv::SetCollisionCache>(
    name + "/set_collision_cache",
    std::bind(
      &ConfigServices::handle_set_collision_cache, this,
      std::placeholders::_1, std::placeholders::_2));
  motion_gen_srv_ = node_->create_service<std_srvs::srv::Trigger>(
    name + "/update_motion_gen_config",
    std::bind(
      &ConfigServices::handle_update_motion_gen_config, this,
      std::placeholders::_1, std::placeholders::_2));
  locks_srv_ = node_->create_service<curobov2_ros_interfaces::srv::SetJointLocks>(
    name + "/set_joint_locks",
    std::bind(
      &ConfigServices::handle_set_joint_locks, this,
      std::placeholders::_1, std::placeholders::_2));
  get_workspace_srv_ = node_->create_service<curobov2_ros_interfaces::srv::GetWorkspace>(
    name + "/get_workspace",
    std::bind(
      &ConfigServices::handle_get_workspace, this,
      std::placeholders::_1, std::placeholders::_2));
  set_workspace_srv_ = node_->create_service<curobov2_ros_interfaces::srv::SetWorkspace>(
    name + "/set_workspace",
    std::bind(
      &ConfigServices::handle_set_workspace, this,
      std::placeholders::_1, std::placeholders::_2));

  // Seed the lock state from the model.
  const auto limits = bridge_->joint_limits(handle_);
  for (std::size_t i = 0; i < limits.locked_joint_names.size(); ++i) {
    locks_[limits.locked_joint_names[i]] = limits.locked_positions[i];
  }
}

std::map<std::string, int> ConfigServices::collision_cache() const
{
  return {
    {"cuboid", node_->get_parameter("collision_cache_cuboid").as_int()},
    {"mesh", node_->get_parameter("collision_cache_mesh").as_int()},
    {"voxel", voxel_layers_},
  };
}

// PORT: ConfigWrapperMotion.set_motion_gen_config
void ConfigServices::rebuild_planner()
{
  std::ostringstream cfg;
  cfg << "{\"num_ik_seeds\":" << node_->get_parameter("num_ik_seeds").as_int()
      << ",\"num_trajopt_seeds\":" << node_->get_parameter("num_trajopt_seeds").as_int()
      << ",\"use_cuda_graph\":"
      << (resolve_use_cuda_graph(node_) ? "true" : "false")
      << ",\"collision_activation_distance\":"
      << node_->get_parameter("collision_activation_distance").as_double()
      << ",\"interpolation_dt\":" << node_->get_parameter("interpolation_dt").as_double()
      << ",\"max_batch_size\":" << node_->get_parameter("max_batch_size").as_int()
      << ",\"max_goalset\":" << node_->get_parameter("max_goalset").as_int()
      << ",\"obstacle_collision_mode\":\""
      << node_->get_parameter("obstacle_collision_mode").as_string() << "\""
      << ",\"collision_cache\":{\"cuboid\":"
      << node_->get_parameter("collision_cache_cuboid").as_int()
      << ",\"mesh\":" << node_->get_parameter("collision_cache_mesh").as_int()
      // PORT: obstacle_manager.collision_cache["voxel"] (layers 0 == disabled).
      << ",\"voxel\":{\"layers\":" << voxel_layers_ << ",\"dims\":[";
  const auto cache_extent = node_->has_parameter("mapper_extent_xyz") ?
    node_->get_parameter("mapper_extent_xyz").as_double_array() :
    std::vector<double>{2.0, 2.0, 2.0};
  for (std::size_t i = 0; i < cache_extent.size(); ++i) {
    cfg << (i ? "," : "") << cache_extent[i];
  }
  // Three closers: voxel object, collision_cache object, top-level object.
  cfg << "],\"voxel_size\":" << node_->get_parameter("voxel_size").as_double() << "}}}";
  // PORT: MotionPlanner build (warmup) captures a CUDA graph; it must hold the
  // exclusive GPU barrier or a concurrent lane's work invalidates the capture
  // ("operation failed due to a previous error during capture"). The startup
  // build is also wrapped, matching the other builders (mpc/retarget).
  auto gpu_guard = curobo_runtime::Runtime::instance().barrier().exclusive();
  bridge_->motion_planner_build(handle_, cfg.str());
  // PORT: config_wrapper_motion.update_world_config - push the CURRENT scene
  // into the freshly built solvers. A rebuild starts from an EMPTY scene (so the
  // new cache always fits), and the plan path's freshness gate will NOT re-push
  // it (the scene epoch is unchanged), so without this the solvers plan with NO
  // obstacles at all. The benchmark sizes the collision cache once per scene, so
  // every scene's first problems were planned straight through the world - the
  // "success rate is too high" against the native leg.
  bool pushed = false;
  try {
    pushed = bridge_->world_push(handle_);
  } catch (const std::exception &) {
    // A push must never turn a cache/config change into a hard error.
    pushed = false;
  }
  if (!pushed && world_invalidate_) {
    // The freshly built planner is on an empty world. Do not let the freshness
    // gate consider it current, or the next plan collides against nothing.
    world_invalidate_();
  }
}

// PORT: obstacle_manager.set_collision_cache
void ConfigServices::handle_set_collision_cache(
  const std::shared_ptr<curobov2_ros_interfaces::srv::SetCollisionCache::Request> request,
  std::shared_ptr<curobov2_ros_interfaces::srv::SetCollisionCache::Response> response)
{
  try {
    const int cur_cuboid = node_->get_parameter("collision_cache_cuboid").as_int();
    const int cur_mesh = node_->get_parameter("collision_cache_mesh").as_int();
    const int new_cuboid = request->obb >= 0 ? static_cast<int>(request->obb) : cur_cuboid;
    const int new_mesh = request->mesh >= 0 ? static_cast<int>(request->mesh) : cur_mesh;
    // PORT: obstacle_manager.set_collision_cache - refuse a shrink below the
    // loaded scene BEFORE touching anything: a smaller cache is built against an
    // empty scene, and the re-push of the oversized world then fails
    // ("Cannot load N cuboids"), leaving the solvers blind.
    const auto counts = bridge_->collision_cache_bucket_counts(handle_);
    if (counts.cuboid > new_cuboid || counts.mesh > new_mesh) {
      std::ostringstream msg;
      msg << "refusing collision-cache resize to cuboid=" << new_cuboid
          << " mesh=" << new_mesh << ": the current scene holds " << counts.cuboid
          << " cuboid slot(s) and " << counts.mesh << " mesh slot(s). Remove "
          << "objects (or clear the scene) so it fits, then retry - the cache "
          << "and the solvers are unchanged.";
      response->success = false;
      response->message = msg.str();
      response->obb_cache = cur_cuboid;
      response->mesh_cache = cur_mesh;
      response->blox_cache = voxel_layers_;
      RCLCPP_WARN(node_->get_logger(), "%s", response->message.c_str());
      return;
    }
    if (request->obb >= 0) {
      node_->set_parameter(rclcpp::Parameter("collision_cache_cuboid", new_cuboid));
    }
    if (request->mesh >= 0) {
      node_->set_parameter(rclcpp::Parameter("collision_cache_mesh", new_mesh));
    }
    // PORT: blox 0 = disable (no voxel storage -> no ESDF avoidance),
    // > 0 = new layer count, < 0 = leave unchanged.
    if (request->blox == 0) {
      voxel_layers_ = 0;
    } else if (request->blox > 0) {
      voxel_layers_ = static_cast<int>(request->blox);
    }
    rebuild_planner();
    response->success = true;
    response->message = "Collision cache updated";
    response->obb_cache = node_->get_parameter("collision_cache_cuboid").as_int();
    response->mesh_cache = node_->get_parameter("collision_cache_mesh").as_int();
    response->blox_cache = voxel_layers_;
    RCLCPP_INFO(
      node_->get_logger(), "Collision cache set: obb=%d, mesh=%d, blox=%d",
      response->obb_cache, response->mesh_cache, response->blox_cache);
  } catch (const std::exception & e) {
    response->success = false;
    response->message = std::string("set_collision_cache failed: ") + e.what();
  }
}

// PORT: config_wrapper_motion.set_motion_gen_config service
void ConfigServices::handle_update_motion_gen_config(
  const std::shared_ptr<std_srvs::srv::Trigger::Request>,
  std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
  try {
    rebuild_planner();
    response->success = true;
    response->message = "Motion planner config set";
    RCLCPP_INFO(node_->get_logger(), "Motion planner config set");
  } catch (const std::exception & e) {
    response->success = false;
    response->message = std::string("set_motion_gen_config failed: ") + e.what();
  }
}

// PORT: robot_model_manager.set_lock_joints / update_lock_joints
void ConfigServices::handle_set_joint_locks(
  const std::shared_ptr<curobov2_ros_interfaces::srv::SetJointLocks::Request> request,
  std::shared_ptr<curobov2_ros_interfaces::srv::SetJointLocks::Response> response)
{
  std::map<std::string, double> requested = locks_;
  if (request->restore) {
    // PORT: robot_model_manager.set_lock_joints(None) - drop the override and
    // reload the robot YAML's own lock_joints. The YAML's values are NOT
    // readable from the live model (it reports whatever override is in force),
    // so the restore is a rebuild with an EMPTY lock map: that is what makes
    // the bridge re-parse the YAML's own lock_joints. Reading the live limits
    // back before the rebuild (as this used to) just echoed the override being
    // dropped, so every restore reported "already match" and changed nothing.
    if (locks_.empty()) {
      // No override in force: nothing to drop (matches the reference's
      // idempotence - a restore of an unmodified model costs no rebuild).
      const auto limits = bridge_->joint_limits(handle_);
      for (std::size_t i = 0; i < limits.locked_joint_names.size(); ++i) {
        response->joint_names.push_back(limits.locked_joint_names[i]);
        response->positions.push_back(limits.locked_positions[i]);
      }
      response->success = true;
      response->message = "No joint-lock override in force";
      return;
    }
    try {
      bridge_->robot_model_rebuild_locks(handle_, "{}");
      locks_.clear();
      const auto limits = bridge_->joint_limits(handle_);
      for (std::size_t i = 0; i < limits.locked_joint_names.size(); ++i) {
        locks_[limits.locked_joint_names[i]] = limits.locked_positions[i];
      }
      if (joint_names_ != nullptr) {
        *joint_names_ = limits.joint_names;
      }
      rebuild_planner();
    } catch (const std::exception & e) {
      response->success = false;
      response->message = std::string("Joint-lock restore failed: ") + e.what();
      return;
    }
    for (const auto & [name, value] : locks_) {
      response->joint_names.push_back(name);
      response->positions.push_back(value);
    }
    response->success = true;
    response->message = "Joint locks restored from the robot YAML - solvers rebuilt (blocking, ~20s)";
    RCLCPP_INFO(
      node_->get_logger(), "Joint locks restored from the robot YAML (%zu joint(s))",
      locks_.size());
    return;
  }
  {
    if (request->joint_names.size() != request->positions.size()) {
      response->success = false;
      response->message = "Got " + std::to_string(request->joint_names.size()) +
        " joint_names but " + std::to_string(request->positions.size()) +
        " positions — one value per named joint (or send restore: true to drop the "
        "override entirely)";
      return;
    }
    for (std::size_t i = 0; i < request->joint_names.size(); ++i) {
      requested[request->joint_names[i]] = request->positions[i];
    }
    for (const auto & name : request->unlock_joint_names) {
      requested.erase(name);
    }
  }

  // Idempotent: same effective set -> no rebuild.
  bool same = requested.size() == locks_.size();
  if (same) {
    for (const auto & [name, value] : requested) {
      auto it = locks_.find(name);
      if (it == locks_.end() || it->second != value) {
        same = false;
        break;
      }
    }
  }
  if (same) {
    response->success = true;
    response->message = "Joint locks already match the requested state";
  } else {
    std::ostringstream json;
    json << "{";
    bool first = true;
    for (const auto & [name, value] : requested) {
      json << (first ? "" : ",") << "\"" << name << "\":" << value;
      first = false;
    }
    json << "}";
    try {
      bridge_->robot_model_rebuild_locks(handle_, json.str());
      locks_ = requested;
      if (joint_names_ != nullptr) {
        *joint_names_ = bridge_->joint_limits(handle_).joint_names;
      }
      rebuild_planner();
      response->success = true;
      response->message = "Joint locks updated - solvers rebuilt (blocking, ~20s)";
    } catch (const std::exception & e) {
      response->success = false;
      response->message = std::string("Joint-lock change failed: ") + e.what();
      return;
    }
  }

  for (const auto & [name, value] : locks_) {
    response->joint_names.push_back(name);
    response->positions.push_back(value);
  }
}

// A Mapper exists when use_mapper is on AND at least one depth/laser feed is
// configured. The camera_*/laser_* blocks are declared by the perception
// managers, so a missing block means "not configured".
bool ConfigServices::perception_active() const
{
  if (node_->has_parameter("use_mapper") &&
    !node_->get_parameter("use_mapper").as_bool())
  {
    return false;
  }
  const auto any_set = [this](const char * name) {
      if (!node_->has_parameter(name)) {
        return false;
      }
      for (const auto & s : node_->get_parameter(name).as_string_array()) {
        if (!s.empty()) {
          return true;
        }
      }
      return false;
    };
  return any_set("camera_topic") || any_set("laser_topic");
}

// Read-only view of the Mapper's TSDF/ESDF box. Pure parameter read (no solver,
// no GPU), so the RViz Context tab can poll it.
void ConfigServices::handle_get_workspace(
  const std::shared_ptr<curobov2_ros_interfaces::srv::GetWorkspace::Request>,
  std::shared_ptr<curobov2_ros_interfaces::srv::GetWorkspace::Response> response)
{
  try {
    response->frame_id = integration_frame_provider_ ? integration_frame_provider_() :
      integration_frame(node_);
    if (node_->has_parameter("mapper_grid_center")) {
      response->center = node_->get_parameter("mapper_grid_center").as_double_array();
    }
    if (node_->has_parameter("mapper_extent_xyz")) {
      response->extent = node_->get_parameter("mapper_extent_xyz").as_double_array();
    }
    if (node_->has_parameter("voxel_size")) {
      response->voxel_size = node_->get_parameter("voxel_size").as_double();
    }
    response->perception_active = perception_active();
    response->success = true;
    response->message = response->perception_active ?
      "Mapper workspace (perception active)" : "Mapper workspace (no camera/laser feed)";
  } catch (const std::exception & e) {
    response->success = false;
    response->message = std::string("get_workspace failed: ") + e.what();
    RCLCPP_ERROR(node_->get_logger(), "%s", response->message.c_str());
  }
}

// Apply a new box centre/size: update mapper_grid_center / mapper_extent_xyz,
// rebuild the Mapper grid to the new box (dropping the integrated TSDF/ESDF),
// rebuild the solvers (their voxel collision cache is sized from
// mapper_extent_xyz) and redraw the wireframe. An empty list leaves that field
// unchanged; the response always echoes the values actually in force.
void ConfigServices::handle_set_workspace(
  const std::shared_ptr<curobov2_ros_interfaces::srv::SetWorkspace::Request> request,
  std::shared_ptr<curobov2_ros_interfaces::srv::SetWorkspace::Response> response)
{
  const auto echo = [this, &response]() {
      response->frame_id = integration_frame_provider_ ? integration_frame_provider_() :
        integration_frame(node_);
      if (node_->has_parameter("mapper_grid_center")) {
        response->center = node_->get_parameter("mapper_grid_center").as_double_array();
      }
      if (node_->has_parameter("mapper_extent_xyz")) {
        response->extent = node_->get_parameter("mapper_extent_xyz").as_double_array();
      }
      if (node_->has_parameter("voxel_size")) {
        response->voxel_size = node_->get_parameter("voxel_size").as_double();
      }
      response->perception_active = perception_active();
    };
  try {
    if ((!request->center.empty() && request->center.size() != 3) ||
      (!request->extent.empty() && request->extent.size() != 3))
    {
      echo();
      response->success = false;
      response->message = "center/extent must be empty (unchanged) or exactly 3 values";
      return;
    }
    if (!node_->has_parameter("mapper_grid_center") ||
      !node_->has_parameter("mapper_extent_xyz"))
    {
      echo();
      response->success = false;
      response->message = "mapper_grid_center / mapper_extent_xyz are not declared";
      return;
    }
    const std::vector<double> cur_center =
      node_->get_parameter("mapper_grid_center").as_double_array();
    const std::vector<double> cur_extent =
      node_->get_parameter("mapper_extent_xyz").as_double_array();
    std::vector<double> new_center = cur_center;
    std::vector<double> new_extent = cur_extent;
    if (request->center.size() == 3) {
      new_center = request->center;
    }
    if (request->extent.size() == 3) {
      for (double e : request->extent) {
        if (!(e > 0.0)) {
          echo();
          response->success = false;
          response->message = "extent values must all be > 0";
          return;
        }
      }
      new_extent = request->extent;
    }
    const bool same = new_center == cur_center && new_extent == cur_extent;
    if (!same) {
      node_->set_parameter(rclcpp::Parameter("mapper_grid_center", new_center));
      node_->set_parameter(rclcpp::Parameter("mapper_extent_xyz", new_extent));
      // Reallocate the Mapper to the new box first (the solvers' voxel cache is
      // then sized from the new mapper_extent_xyz by rebuild_planner).
      if (mapper_reconfigure_) {
        mapper_reconfigure_();
      }
      rebuild_planner();
    }
    if (workspace_republish_) {
      workspace_republish_();
    }
    echo();
    response->success = true;
    response->message = same ? "workspace already matches the request" :
      "workspace updated - Mapper and solvers rebuilt (blocking, ~20s)";
    std::ostringstream log;
    log << "set_workspace: center=[";
    for (std::size_t i = 0; i < response->center.size(); ++i) {
      log << (i ? "," : "") << response->center[i];
    }
    log << "] extent=[";
    for (std::size_t i = 0; i < response->extent.size(); ++i) {
      log << (i ? "," : "") << response->extent[i];
    }
    log << "] (" << (same ? "unchanged" : "rebuilt") << ")";
    RCLCPP_INFO(node_->get_logger(), "%s", log.str().c_str());
  } catch (const std::exception & e) {
    response->success = false;
    response->message = std::string("set_workspace failed: ") + e.what();
    RCLCPP_ERROR(node_->get_logger(), "%s", response->message.c_str());
  }
}

}  // namespace curobo_core

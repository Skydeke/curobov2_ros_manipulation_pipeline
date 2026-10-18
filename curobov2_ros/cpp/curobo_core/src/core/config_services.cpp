// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of the config/cache/joint-lock service callbacks.
#include "curobo_core/core/config_services.hpp"

#include <algorithm>
#include <sstream>

#include <std_srvs/srv/trigger.hpp>

#include "curobo_runtime/runtime.hpp"

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
      << (node_->get_parameter("use_cuda_graph").as_bool() ? "true" : "false")
      << ",\"collision_activation_distance\":"
      << node_->get_parameter("collision_activation_distance").as_double()
      << ",\"interpolation_dt\":" << node_->get_parameter("interpolation_dt").as_double()
      << ",\"max_batch_size\":" << node_->get_parameter("max_batch_size").as_int()
      << ",\"max_goalset\":" << node_->get_parameter("max_goalset").as_int()
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
    // Drop the override and reload the robot YAML's own lock_joints.
    const auto limits = bridge_->joint_limits(handle_);
    requested.clear();
    for (std::size_t i = 0; i < limits.locked_joint_names.size(); ++i) {
      requested[limits.locked_joint_names[i]] = limits.locked_positions[i];
    }
  } else {
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

}  // namespace curobo_core

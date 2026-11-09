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
  // Native artifacts are immutable, but a lock_joints change maps to another
  // pre-exported artifact bundle (same robot/arch, different baked gripper
  // geometry). The benchmark's per-dataset files are exactly that case, so
  // /set_joint_locks stays available and the CppBridge reloads the matching
  // bundle instead of silently keeping the launch artifact.
  get_workspace_srv_ = node_->create_service<curobov2_ros_interfaces::srv::GetWorkspace>(
    name + "/get_workspace",
    std::bind(
      &ConfigServices::handle_get_workspace, this,
      std::placeholders::_1, std::placeholders::_2));
  set_joint_locks_srv_ =
    node_->create_service<curobov2_ros_interfaces::srv::SetJointLocks>(
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
      << (resolve_use_cuda_graph(node_) ? "true" : "false")
      << ",\"collision_activation_distance\":"
      << node_->get_parameter("collision_activation_distance").as_double()
      << ",\"interpolation_dt\":" << node_->get_parameter("interpolation_dt").as_double()
      << ",\"max_batch_size\":" << node_->get_parameter("max_batch_size").as_int()
      << ",\"max_goalset\":" << node_->get_parameter("max_goalset").as_int()
      << ",\"obstacle_collision_mode\":\""
      << node_->get_parameter("obstacle_collision_mode").as_string() << "\""
      << ",\"check_effort_limits\":"
      << (node_->get_parameter("load_dynamics").as_bool() ? "true" : "false")
      << ",\"collision_cache\":{\"cuboid\":"
      << node_->get_parameter("collision_cache_cuboid").as_int()
      << ",\"mesh\":" << node_->get_parameter("collision_cache_mesh").as_int()
      // PORT: obstacle_manager.collision_cache["voxel"] (layers 0 == disabled).
      << ",\"voxel\":{\"layers\":" << voxel_layers_ << ",\"dims\":[";
  const auto cache_extent = node_->has_parameter("mapper_extent_xyz") ?
    node_->get_parameter("mapper_extent_xyz").as_double_array() :
    std::vector<double>{2.0, 2.0, 2.0};
  const double voxel_size = node_->get_parameter("voxel_size").as_double();
  // The mapper rounds its grid up, so an extent that divides exactly into
  // voxels can still produce one more cell per axis than this cache. Allocate
  // two cells of margin so world_push always fits.
  const double margin = voxel_size > 0.0 ? 2.0 * voxel_size : 0.0;
  for (std::size_t i = 0; i < cache_extent.size(); ++i) {
    cfg << (i ? "," : "") << (cache_extent[i] + margin);
  }
  // Three closers: voxel object, collision_cache object, top-level object.
  cfg << "],\"voxel_size\":" << voxel_size << "}}}";
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

// PORT: config_wrapper_motion.set_motion_gen_config
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

}  // namespace curobo_core

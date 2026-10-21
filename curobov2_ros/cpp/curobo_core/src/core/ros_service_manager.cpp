// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of the M2 subset of curobov2_ros/core/ros_service_manager.py.
#include "curobo_runtime/runtime.hpp"
#include "curobo_core/core/ros_service_manager.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <string>
#include <utility>
#include <vector>

#include <std_srvs/srv/trigger.hpp>

#include "curobo_core/core/robot_segmentation.hpp"

namespace curobo_core
{

RosServiceManager::RosServiceManager(
  rclcpp::Node * node, curobo_runtime::IBridge * bridge,
  curobo_runtime::RobotHandle handle, ObstacleManager * obstacle_manager)
: node_(node), bridge_(bridge), handle_(handle), obstacle_manager_(obstacle_manager)
{
}

namespace
{
Pose7 pose_from_msg(const geometry_msgs::msg::Pose & msg)
{
  Pose7 pose;
  pose << msg.position.x, msg.position.y, msg.position.z,
    msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z;
  return pose;
}
}  // namespace

// PORT: the AddObject request -> spec extraction
SceneObjectSpec RosServiceManager::spec_from_add_object(
  const curobov2_ros_interfaces::srv::AddObject::Request & request)
{
  SceneObjectSpec spec;
  spec.type = request.type;
  spec.name = request.name;
  spec.mesh_file_path = request.mesh_file_path;
  spec.pose = pose_from_msg(request.pose);
  spec.dims = {request.dimensions.x, request.dimensions.y, request.dimensions.z};
  spec.color = {request.color.r, request.color.g, request.color.b, request.color.a};
  for (const auto & v : request.vertices) {
    spec.vertices.push_back({v.x, v.y, v.z});
  }
  spec.triangles = request.triangles;
  return spec;
}

// PORT: the SceneObject msg -> spec extraction (set_objects)
SceneObjectSpec RosServiceManager::spec_from_scene_object(
  const curobov2_ros_interfaces::msg::SceneObject & object)
{
  SceneObjectSpec spec;
  spec.type = object.type;
  spec.name = object.name;
  spec.mesh_file_path = object.mesh_file_path;
  spec.pose = pose_from_msg(object.pose);
  spec.dims = {object.dimensions.x, object.dimensions.y, object.dimensions.z};
  spec.color = {object.color.r, object.color.g, object.color.b, object.color.a};
  for (const auto & v : object.vertices) {
    spec.vertices.push_back({v.x, v.y, v.z});
  }
  spec.triangles = object.triangles;
  return spec;
}

// PORT: ros_service_manager.init_services (M2 subset)
void RosServiceManager::init_services()
{
  const std::string name = node_->get_name();
  add_object_srv_ = node_->create_service<curobov2_ros_interfaces::srv::AddObject>(
    name + "/add_object",
    std::bind(
      &RosServiceManager::handle_add_object, this,
      std::placeholders::_1, std::placeholders::_2));
  remove_object_srv_ = node_->create_service<curobov2_ros_interfaces::srv::RemoveObject>(
    name + "/remove_object",
    std::bind(
      &RosServiceManager::handle_remove_object, this,
      std::placeholders::_1, std::placeholders::_2));
  remove_all_srv_ = node_->create_service<std_srvs::srv::Trigger>(
    name + "/remove_all_objects",
    std::bind(
      &RosServiceManager::handle_remove_all_objects, this,
      std::placeholders::_1, std::placeholders::_2));
  set_objects_srv_ = node_->create_service<curobov2_ros_interfaces::srv::SetObjects>(
    name + "/set_objects",
    std::bind(
      &RosServiceManager::handle_set_objects, this,
      std::placeholders::_1, std::placeholders::_2));
  get_obstacles_srv_ = node_->create_service<std_srvs::srv::Trigger>(
    name + "/get_obstacles",
    std::bind(
      &RosServiceManager::handle_get_obstacles, this,
      std::placeholders::_1, std::placeholders::_2));
  get_scene_objects_srv_ =
    node_->create_service<curobov2_ros_interfaces::srv::GetSceneObjects>(
    name + "/get_scene_objects",
    std::bind(
      &RosServiceManager::handle_get_scene_objects, this,
      std::placeholders::_1, std::placeholders::_2));
  get_joint_info_srv_ = node_->create_service<curobov2_ros_interfaces::srv::GetJointInfo>(
    name + "/get_joint_info",
    std::bind(
      &RosServiceManager::handle_get_joint_info, this,
      std::placeholders::_1, std::placeholders::_2));
  get_voxel_grid_srv_ = node_->create_service<curobov2_ros_interfaces::srv::GetVoxelGrid>(
    name + "/get_voxel_grid",
    std::bind(
      &RosServiceManager::handle_get_voxel_grid, this,
      std::placeholders::_1, std::placeholders::_2));
  clear_voxel_map_srv_ = node_->create_service<std_srvs::srv::Trigger>(
    name + "/clear_voxel_map",
    std::bind(
      &RosServiceManager::handle_clear_voxel_map, this,
      std::placeholders::_1, std::placeholders::_2));
  set_link_collision_srv_ =
    node_->create_service<curobov2_ros_interfaces::srv::SetLinkCollision>(
    name + "/set_link_collision",
    std::bind(
      &RosServiceManager::handle_set_link_collision, this,
      std::placeholders::_1, std::placeholders::_2));
  set_collision_spheres_srv_ = node_->create_service<std_srvs::srv::SetBool>(
    name + "/set_collision_spheres_enabled",
    std::bind(
      &RosServiceManager::handle_set_collision_spheres_enabled, this,
      std::placeholders::_1, std::placeholders::_2));

  // PORT: ros_service_manager viz publishers (mapper workspace + sparse voxel).
  workspace_viz_enabled_ = node_->has_parameter("publish_workspace_visualisation") ?
    node_->get_parameter("publish_workspace_visualisation").as_bool() : true;
  workspace_viz_pub_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>(
    name + "/mapper_workspace", rclcpp::QoS(1).transient_local());
  publish_mapper_workspace();

  sparse_voxel_pub_ =
    node_->create_publisher<curobov2_ros_interfaces::msg::SparseVoxelGrid>(
    name + "/voxel_grid_sparse", rclcpp::QoS(10).transient_local());
  scene_obstacle_pub_ =
    node_->create_publisher<visualization_msgs::msg::MarkerArray>(
    name + "/scene_obstacles", 10);
  publish_scene_obstacles();

  // PORT: ros_service_manager.publish_collision_spheres (timer, gated).
  collision_spheres_pub_ =
    node_->create_publisher<visualization_msgs::msg::MarkerArray>(
    name + "/collision_spheres", 10);
  // Both viz timers live in their own mutually exclusive group. In the default
  // group they share it with every service callback, and a single (multi-second)
  // generate_trajectory call then starves the 200 ms sphere timer for the whole
  // plan burst -- the gripper/robot spheres simply stop updating while the
  // orchestrator keeps replanning.
  viz_callback_group_ = node_->create_callback_group(
    rclcpp::CallbackGroupType::MutuallyExclusive);
  const bool sphere_viz = node_->has_parameter("publish_collision_spheres") ?
    node_->get_parameter("publish_collision_spheres").as_bool() : true;
  if (sphere_viz) {
    collision_spheres_timer_ = node_->create_wall_timer(
      std::chrono::milliseconds(200),
      std::bind(&RosServiceManager::publish_collision_spheres, this),
      viz_callback_group_);
  }
  const double sparse_rate = node_->has_parameter("sparse_voxel_publish_rate") ?
    node_->get_parameter("sparse_voxel_publish_rate").as_double() : 7.0;
  if (sparse_rate > 0.0) {
    sparse_voxel_timer_ = node_->create_wall_timer(
      std::chrono::duration<double>(1.0 / sparse_rate),
      std::bind(&RosServiceManager::publish_sparse_voxel_grid, this),
      viz_callback_group_);
  }
}

// PORT: ros_service_manager._callback_add_object
void RosServiceManager::handle_add_object(
  const std::shared_ptr<curobov2_ros_interfaces::srv::AddObject::Request> request,
  std::shared_ptr<curobov2_ros_interfaces::srv::AddObject::Response> response)
{
  std::string message;
  const bool ok = obstacle_manager_->add_object(spec_from_add_object(*request), message);
  response->success = ok;
  response->message = message;
  if (!ok) {
    RCLCPP_ERROR(node_->get_logger(), "%s", message.c_str());
  }
}

// PORT: ros_service_manager._callback_remove_object
void RosServiceManager::handle_remove_object(
  const std::shared_ptr<curobov2_ros_interfaces::srv::RemoveObject::Request> request,
  std::shared_ptr<curobov2_ros_interfaces::srv::RemoveObject::Response> response)
{
  std::string message;
  response->success = obstacle_manager_->remove_object(request->name, message);
  response->message = message;
}

// PORT: ros_service_manager._callback_remove_all_objects
void RosServiceManager::handle_remove_all_objects(
  const std::shared_ptr<std_srvs::srv::Trigger::Request>,
  std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
  std::string message;
  response->success = obstacle_manager_->remove_all_objects(message);
  response->message = message;
}

// PORT: ros_service_manager._callback_set_objects
void RosServiceManager::handle_set_objects(
  const std::shared_ptr<curobov2_ros_interfaces::srv::SetObjects::Request> request,
  std::shared_ptr<curobov2_ros_interfaces::srv::SetObjects::Response> response)
{
  std::vector<SceneObjectSpec> specs;
  specs.reserve(request->objects.size());
  for (const auto & object : request->objects) {
    specs.push_back(spec_from_scene_object(object));
  }
  std::string message;
  const bool ok = obstacle_manager_->set_objects(specs, request->clear_first, message);
  response->success = ok;
  response->message = message;
  if (!ok) {
    RCLCPP_ERROR(node_->get_logger(), "%s", message.c_str());
  }
}

// PORT: ros_service_manager._callback_get_obstacles
void RosServiceManager::handle_get_obstacles(
  const std::shared_ptr<std_srvs::srv::Trigger::Request>,
  std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
  response->message = obstacle_manager_->get_obstacles();
  response->success = true;
}

// PORT: ros_service_manager._callback_get_scene_objects
void RosServiceManager::handle_get_scene_objects(
  const std::shared_ptr<curobov2_ros_interfaces::srv::GetSceneObjects::Request>,
  std::shared_ptr<curobov2_ros_interfaces::srv::GetSceneObjects::Response> response)
{
  try {
    const SceneView view = obstacle_manager_->get_scene_objects();
    for (std::size_t i = 0; i < view.names.size(); ++i) {
      response->names.push_back(view.names[i]);
      response->types.push_back(static_cast<std::int8_t>(view.types[i]));
      response->poses.push_back(to_ros_pose(view.poses[i]));
      geometry_msgs::msg::Vector3 dims;
      dims.x = view.dimensions[i][0];
      dims.y = view.dimensions[i][1];
      dims.z = view.dimensions[i][2];
      response->dimensions.push_back(dims);
      std_msgs::msg::ColorRGBA color;
      color.r = view.colors[i][0];
      color.g = view.colors[i][1];
      color.b = view.colors[i][2];
      color.a = view.colors[i][3];
      response->colors.push_back(color);
      response->mesh_file_paths.push_back(view.mesh_file_paths[i]);
      response->attached.push_back(view.attached[i]);
    }
    response->success = true;
    response->message = std::to_string(view.names.size()) + " object(s)";
  } catch (const std::exception & exc) {
    response->success = false;
    response->message = std::string("get_scene_objects failed: ") + exc.what();
  }
}

// PORT: ros_service_manager._callback_get_joint_info
void RosServiceManager::handle_get_joint_info(
  const std::shared_ptr<curobov2_ros_interfaces::srv::GetJointInfo::Request>,
  std::shared_ptr<curobov2_ros_interfaces::srv::GetJointInfo::Response> response)
{
  try {
    const auto limits = bridge_->joint_limits(handle_);
    response->joint_names = limits.joint_names;
    const std::size_t dof = limits.joint_names.size();
    for (std::size_t j = 0; j < dof; ++j) {
      response->lower_limits.push_back(limits.positions.data[j]);
      response->upper_limits.push_back(limits.positions.data[dof + j]);
    }
    response->locked_joint_names = limits.locked_joint_names;
    response->locked_positions = limits.locked_positions;
    response->success = true;
    response->message = std::to_string(dof) + " joints in cspace";
    if (!limits.locked_joint_names.empty()) {
      response->message += ", " + std::to_string(limits.locked_joint_names.size()) + " locked";
    }
  } catch (const std::exception & exc) {
    response->success = false;
    response->message = std::string("get_joint_info failed: ") + exc.what();
  }
}

// PORT: ros_service_manager._callback_get_voxel_grid (dense nav2 VoxelGrid)
void RosServiceManager::handle_get_voxel_grid(
  const std::shared_ptr<curobov2_ros_interfaces::srv::GetVoxelGrid::Request>,
  std::shared_ptr<curobov2_ros_interfaces::srv::GetVoxelGrid::Response> response)
{
  try {
    // PORT: ros_service_manager._gpu_guard - the dense grid rasterises analytic
    // primitives on the GPU and then host-syncs; without the exclusive barrier
    // it can launch into an in-flight CUDA-graph capture and invalidate it.
    auto gpu_guard = curobo_runtime::Runtime::instance().barrier().exclusive();
    const auto grid = obstacle_manager_->get_voxel_grid();
    auto & out = response->voxel_grid;
    // PORT: get_voxel_grid sets no header (frame_id/stamp stay at their message
    // defaults) - only resolutions/origin/size/data.
    if (grid.shape.size() == 3) {
      out.size_x = static_cast<std::uint32_t>(grid.shape[0]);
      out.size_y = static_cast<std::uint32_t>(grid.shape[1]);
      out.size_z = static_cast<std::uint32_t>(grid.shape[2]);
    }
    out.resolutions.x = grid.resolution;
    out.resolutions.y = grid.resolution;
    out.resolutions.z = grid.resolution;
    out.origin.x = static_cast<float>(grid.origin[0]);
    out.origin.y = static_cast<float>(grid.origin[1]);
    out.origin.z = static_cast<float>(grid.origin[2]);
    out.data.reserve(grid.data.size());
    for (std::uint8_t v : grid.data) {
      out.data.push_back(v);
    }
  } catch (const std::exception & e) {
    RCLCPP_ERROR(node_->get_logger(), "get_voxel_grid failed: %s", e.what());
  }
}

// PORT: ros_service_manager.register_robot_segmentation
void RosServiceManager::register_robot_segmentation(RobotSegmentation * segmentation)
{
  segmentation_ = segmentation;
  const std::string name = node_->get_name();
  set_mask_srv_ = node_->create_service<curobov2_ros_interfaces::srv::SetMask>(
    name + "/set_mask",
    std::bind(
      &RobotSegmentation::handle_set_mask, segmentation_,
      std::placeholders::_1, std::placeholders::_2));
  remove_mask_srv_ = node_->create_service<curobov2_ros_interfaces::srv::RemoveObject>(
    name + "/remove_mask",
    std::bind(
      &RobotSegmentation::handle_remove_mask, segmentation_,
      std::placeholders::_1, std::placeholders::_2));
  RCLCPP_INFO(
    node_->get_logger(),
    "Registered robot segmentation services on %s/set_mask and %s/remove_mask",
    name.c_str(), name.c_str());
}

// PORT: unified_planner_node.clear_voxel_map_callback (dynamic channel only)
void RosServiceManager::handle_clear_voxel_map(
  const std::shared_ptr<std_srvs::srv::Trigger::Request>,
  std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
  int n_blocks = 0;
  bool refreshed = false;
  try {
    // PORT: unified_planner_node.clear_voxel_map_callback - `with self.gpu_lock:`
    // around the TSDF clear AND the ESDF rebuild (both are CUDA ops), then
    // update_all_solvers_world() so the solvers drop the cleared voxels. Without
    // the lock the pair raced an in-flight capture; without the push the solvers
    // kept the old voxels (clear_dynamic_voxels does not bump the epoch, so the
    // freshness gate would not re-push either).
    auto gpu_guard = curobo_runtime::Runtime::instance().barrier().exclusive();
    n_blocks = obstacle_manager_->clear_dynamic_voxels();
    refreshed = obstacle_manager_->refresh_esdf();
    if (refreshed) {
      bridge_->world_push(handle_);
      obstacle_manager_->mark_world_pushed();
    }
  } catch (const std::exception & e) {
    response->success = false;
    response->message = std::string("clear_voxel_map failed: ") + e.what();
    RCLCPP_ERROR(node_->get_logger(), "%s", response->message.c_str());
    return;
  }
  response->success = true;
  response->message = "Cleared dynamic voxel channel (" + std::to_string(n_blocks) +
    " block(s))" + (refreshed ? "" : "; WARNING: perception ESDF not refreshed");
  RCLCPP_INFO(node_->get_logger(), "%s", response->message.c_str());
}

namespace
{
// Python list repr, e.g. ['joint_1', 'joint_2'] (matches the oracle wording).
std::string py_list(const std::vector<std::string> & values)
{
  std::string out = "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    out += (i ? ", '" : "'");
    out += values[i];
    out += "'";
  }
  out += "]";
  return out;
}
}  // namespace

// PORT: robot_model_manager.set_link_collision_callback
void RosServiceManager::handle_set_link_collision(
  const std::shared_ptr<curobov2_ros_interfaces::srv::SetLinkCollision::Request> request,
  std::shared_ptr<curobov2_ros_interfaces::srv::SetLinkCollision::Response> response)
{
  std::vector<std::string> link_names(request->link_names.begin(), request->link_names.end());
  std::vector<std::string> unknown;
  try {
    // PORT: robot_model_manager.set_link_collision_callback - `with
    // gpu.exclusive():` because it mutates kinematics tensors shared by every
    // solver.
    auto gpu_guard = curobo_runtime::Runtime::instance().barrier().exclusive();
    const auto outcome = bridge_->set_link_collision(handle_, link_names, request->enabled);
    const std::vector<std::string> & applied = outcome.applied;
    unknown = outcome.unknown;
    response->applied_links = applied;
    response->unknown_links = unknown;

    if (!unknown.empty()) {
      RCLCPP_WARN(
        node_->get_logger(), "set_link_collision: unknown links %s",
        py_list(unknown).c_str());
    }
    if (!applied.empty()) {
      const char * state = request->enabled ? "enabled" : "disabled";
      RCLCPP_INFO(
        node_->get_logger(), "Collision spheres %s for: %s", state,
        py_list(applied).c_str());
      response->success = true;
      response->message = std::string("Collision ") + state + " for " + py_list(applied);
    } else {
      response->success = false;
      response->message = "No valid links found in: " + py_list(link_names);
    }
  } catch (const std::exception & e) {
    response->success = false;
    response->message = std::string("set_link_collision failed: ") + e.what();
    RCLCPP_ERROR(node_->get_logger(), "%s", response->message.c_str());
  }
}

// PORT: ros_service_manager._callback_set_collision_spheres_enabled
void RosServiceManager::handle_set_collision_spheres_enabled(
  const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
  std::shared_ptr<std_srvs::srv::SetBool::Response> response)
{
  collision_spheres_enabled_ = request->data;
  const char * state = request->data ? "enabled" : "disabled";
  RCLCPP_INFO(node_->get_logger(), "Collision sphere publishing %s", state);
  response->success = true;
  response->message = std::string("Collision spheres ") + state;
}

// PORT: ros_service_manager.publish_mapper_workspace
void RosServiceManager::publish_mapper_workspace()
{
  if (!workspace_viz_enabled_ || workspace_viz_pub_ == nullptr) {
    return;
  }
  try {
    std::vector<double> extent, center;
    if (node_->has_parameter("mapper_extent_xyz")) {
      extent = node_->get_parameter("mapper_extent_xyz").as_double_array();
    }
    if (node_->has_parameter("mapper_grid_center")) {
      center = node_->get_parameter("mapper_grid_center").as_double_array();
    }
    if (extent.size() < 3 || center.size() < 3) {
      return;
    }
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = integration_frame_provider_ ? integration_frame_provider_() :
      integration_frame(node_);
    marker.header.stamp = node_->get_clock()->now();
    marker.ns = "mapper_workspace";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::LINE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.01;
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 1.0;
    marker.color.a = 1.0;
    std::array<geometry_msgs::msg::Point, 8> corners;
    std::size_t idx = 0;
    for (int sx : {-1, 1}) {
      for (int sy : {-1, 1}) {
        for (int sz : {-1, 1}) {
          geometry_msgs::msg::Point p;
          p.x = center[0] + sx * extent[0] / 2.0;
          p.y = center[1] + sy * extent[1] / 2.0;
          p.z = center[2] + sz * extent[2] / 2.0;
          corners[idx++] = p;
        }
      }
    }
    const std::pair<int, int> edges[] = {
      {0, 1}, {0, 2}, {0, 4}, {1, 3}, {1, 5}, {2, 3},
      {2, 6}, {3, 7}, {4, 5}, {4, 6}, {5, 7}, {6, 7}};
    for (const auto & [i, j] : edges) {
      marker.points.push_back(corners[i]);
      marker.points.push_back(corners[j]);
    }
    visualization_msgs::msg::MarkerArray array;
    array.markers.push_back(marker);
    workspace_viz_pub_->publish(array);
  } catch (const std::exception & e) {
    RCLCPP_WARN_THROTTLE(
      node_->get_logger(), *node_->get_clock(), 5000,
      "Could not build mapper workspace marker from params: %s", e.what());
  }
}

// PORT: ros_service_manager._publish_sparse_voxel_grid -> obstacle_manager.
// PORT: ros_service_manager._publish_sparse_voxel_grid - lane_viz.submit_if_idle
// (or a non-blocking gpu_lock) and SKIP this cycle when it is busy, instead of
// blocking the 7 Hz timer on an in-flight capture.
bool RosServiceManager::submit_viz_job(std::function<void()> job)
{
  if (lane_viz_ != nullptr) {
    curobo_runtime::SubmitOptions opts;
    opts.prio = curobo_runtime::PRIO_BULK;   // rate-limited viz work
    return lane_viz_->submit_if_idle(opts, std::move(job)).has_value();
  }
  auto gpu_guard = curobo_runtime::Runtime::instance().barrier().shared();
  job();
  return true;
}

void RosServiceManager::publish_sparse_voxel_grid()
{
  if (sparse_voxel_pub_ == nullptr) {
    return;
  }
  if (!submit_viz_job([this]() {publish_sparse_voxel_grid_job();})) {
    RCLCPP_DEBUG_THROTTLE(
      node_->get_logger(), *node_->get_clock(), 5000,
      "viz lane busy - skipping this voxel grid publish cycle");
  }
}

// PORT: the GPU body of _publish_sparse_voxel_grid (runs on the viz lane).
void RosServiceManager::publish_sparse_voxel_grid_job()
{
  try {
    const auto grid = obstacle_manager_->get_voxel_grid();
    if (grid.shape.size() != 3) {
      return;
    }
    auto msg = std::make_unique<curobov2_ros_interfaces::msg::SparseVoxelGrid>();
    msg->header.stamp = node_->get_clock()->now();
    msg->header.frame_id = node_->has_parameter("base_link") ?
      node_->get_parameter("base_link").as_string() : "base_link";
    msg->origin.x = static_cast<float>(grid.origin[0]);
    msg->origin.y = static_cast<float>(grid.origin[1]);
    msg->origin.z = static_cast<float>(grid.origin[2]);
    msg->resolution = static_cast<float>(grid.resolution);
    msg->size_x = static_cast<std::uint32_t>(grid.shape[0]);
    msg->size_y = static_cast<std::uint32_t>(grid.shape[1]);
    msg->size_z = static_cast<std::uint32_t>(grid.shape[2]);
    msg->occupied_indices = ObstacleManager::occupied_indices(grid);
    sparse_voxel_pub_->publish(std::move(msg));
  } catch (const std::exception & e) {
    // PORT: skip this cycle rather than publish a partial grid.
    RCLCPP_WARN_THROTTLE(
      node_->get_logger(), *node_->get_clock(), 5000,
      "Voxel grid computation failed (%s) - skipping this publish cycle", e.what());
  }
}

// PORT: ros_service_manager.publish_scene_obstacles
void RosServiceManager::publish_scene_obstacles()
{
  if (scene_obstacle_pub_ == nullptr) {
    return;
  }
  const SceneView view = obstacle_manager_->get_scene_objects();
  visualization_msgs::msg::MarkerArray array;
  visualization_msgs::msg::Marker clear;
  // Empty namespace on purpose: rviz's DELETEALL is namespace-scoped.
  clear.action = visualization_msgs::msg::Marker::DELETEALL;
  array.markers.push_back(clear);

  const std::string frame = node_->has_parameter("base_link") ?
    node_->get_parameter("base_link").as_string() : "base_link";
  int mid = 0;
  for (std::size_t i = 0; i < view.names.size(); ++i) {
    const int type = i < view.types.size() ? view.types[i] : CUBOID;
    if (type == MESH) {
      continue;   // PORT: only cuboid/sphere/cylinder/capsule get markers here
    }
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame;
    marker.header.stamp = node_->get_clock()->now();
    marker.ns = "scene_obstacles";
    marker.id = mid++;
    marker.type = (type == CUBOID) ? visualization_msgs::msg::Marker::CUBE :
      (type == SPHERE) ? visualization_msgs::msg::Marker::SPHERE :
      visualization_msgs::msg::Marker::CYLINDER;
    marker.action = visualization_msgs::msg::Marker::ADD;
    const auto & pose = view.poses[i];
    marker.pose.position.x = pose[0];
    marker.pose.position.y = pose[1];
    marker.pose.position.z = pose[2];
    marker.pose.orientation.w = pose[3];
    marker.pose.orientation.x = pose[4];
    marker.pose.orientation.y = pose[5];
    marker.pose.orientation.z = pose[6];
    const auto & dims = view.dimensions[i];
    if (type == CUBOID) {
      marker.scale.x = dims[0];
      marker.scale.y = dims[1];
      marker.scale.z = dims[2];
    } else {
      const double d = 2.0 * dims[0];
      marker.scale.x = marker.scale.y = d;
      marker.scale.z = (type == SPHERE) ? d : dims[1];
    }
    const bool attached = i < view.attached.size() && view.attached[i];
    if (attached) {
      marker.color.r = 0.5;
      marker.color.g = 0.5;
      marker.color.b = 0.5;
      marker.color.a = 0.35;   // attached -> grey, translucent
    } else {
      marker.color.r = 1.0;
      marker.color.g = 0.5;
      marker.color.b = 0.0;
      marker.color.a = 0.6;    // active obstacle -> orange
    }
    array.markers.push_back(marker);
  }
  scene_obstacle_pub_->publish(array);
}

// PORT: ros_service_manager.publish_collision_spheres
void RosServiceManager::publish_collision_spheres()
{
  if (collision_spheres_pub_ == nullptr || !collision_spheres_enabled_ ||
    pose_provider_ == nullptr)
  {
    return;
  }
  // PORT: ros_service_manager.publish_collision_spheres - lane_viz.submit_if_idle
  // (drop this cycle when busy), not a blocking shared hold on the timer.
  if (!submit_viz_job([this]() {publish_collision_spheres_job();})) {
    RCLCPP_DEBUG_THROTTLE(
      node_->get_logger(), *node_->get_clock(), 5000,
      "viz lane busy - skipping this collision-sphere publish cycle");
  }
}

// PORT: the GPU body of publish_collision_spheres (runs on the viz lane).
void RosServiceManager::publish_collision_spheres_job()
{
  try {
    const std::vector<float> q = pose_provider_();
    if (q.empty()) {
      return;
    }
    const auto sphere_set = bridge_->collision_spheres_with_attached(handle_, q);
    const auto & spheres = sphere_set.spheres;
    if (spheres.shape.size() != 2 || spheres.shape[1] != 4) {
      return;
    }
    // Per-sphere scene distance for the collision colouring; absent -> all green.
    std::vector<float> distances;
    try {
      curobo_runtime::TensorF32 q_tensor;
      q_tensor.shape = {1, static_cast<std::int64_t>(q.size())};
      q_tensor.data = q;
      distances = bridge_->collision_distance(handle_, q_tensor);
    } catch (const std::exception &) {
      distances.clear();
    }

    visualization_msgs::msg::MarkerArray array;
    visualization_msgs::msg::Marker clear;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    array.markers.push_back(clear);

    const std::string frame = node_->has_parameter("base_link") ?
      node_->get_parameter("base_link").as_string() : "base_link";
    // Link-relative payload spheres (newer bridge). Their frame is the URDF
    // link the payload is fixed to, so a trajectory preview can FK that link
    // per waypoint and keep the object in the gripper.
    const bool has_attached_local =
      !sphere_set.attach_frame.empty() &&
      sphere_set.attached_local.shape.size() == 2 &&
      sphere_set.attached_local.shape[1] == 4 &&
      sphere_set.attached_local.shape[0] == spheres.shape[0];
    const std::int64_t n = spheres.shape[0];
    for (std::int64_t i = 0; i < n; ++i) {
      const std::size_t base = static_cast<std::size_t>(i) * 4;
      const float radius = spheres.data[base + 3];
      if (radius <= 0.0f) {
        continue;   // disabled sphere (radius -100)
      }
      visualization_msgs::msg::Marker marker;
      // PORT: the attached-object spheres carry their own namespace so a
      // consumer can show the grasped object separately.
      const bool attached = static_cast<std::size_t>(i) < sphere_set.attached.size() &&
        sphere_set.attached[static_cast<std::size_t>(i)] != 0;
      const bool link_relative = attached && has_attached_local &&
        sphere_set.attached_local.data[base + 3] > 0.0f;
      marker.header.frame_id = link_relative ? sphere_set.attach_frame : frame;
      marker.header.stamp = node_->get_clock()->now();
      marker.ns = attached ? "attached_spheres" : "collision_spheres";
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.id = static_cast<int>(i);
      const auto & pos = link_relative ? sphere_set.attached_local.data
                                       : spheres.data;
      marker.pose.position.x = pos[base + 0];
      marker.pose.position.y = pos[base + 1];
      marker.pose.position.z = pos[base + 2];
      marker.pose.orientation.w = 1.0;
      marker.scale.x = marker.scale.y = marker.scale.z = radius * 2.0;
      marker.color.a = 0.5;
      const bool colliding = static_cast<std::size_t>(i) < distances.size() &&
        distances[static_cast<std::size_t>(i)] < 0.0f;
      if (colliding) {
        marker.color.r = 1.0;   // colliding = red
      } else if (attached) {
        marker.color.b = 1.0;   // attached object, not colliding = blue
      } else {
        marker.color.g = 1.0;   // collision-free = green
      }
      array.markers.push_back(marker);
    }
    // PORT: the Python names the offending link + obstacle(s); the attribution
    // pass is not ported, so report the colliding sphere indices only.
    if (std::any_of(distances.begin(), distances.end(),
        [](float d) {return d < 0.0f;}))
    {
      std::string indices;
      for (std::size_t i = 0; i < distances.size(); ++i) {
        if (distances[i] < 0.0f) {
          indices += (indices.empty() ? "" : ", ") + std::to_string(i);
        }
      }
      RCLCPP_DEBUG_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 5000,
        "collision_spheres: sphere(s) in collision (indices: %s)", indices.c_str());
    }
    collision_spheres_pub_->publish(array);
  } catch (const std::exception & e) {
    // A viz failure must not break the node, but it must not be silent either:
    // an exception thrown by the bridge (e.g. a stale/incompatible Python
    // helper) disables ALL sphere output, which is invisible in RViz.
    RCLCPP_WARN_THROTTLE(
      node_->get_logger(), *node_->get_clock(), 5000,
      "collision_spheres publish failed: %s", e.what());
  }
}

}  // namespace curobo_core

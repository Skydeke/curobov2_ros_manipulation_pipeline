// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of the GetCollisionDistance service in
// curobov2_ros/core/collision_distance.py.
#include "curobo_core/core/collision_distance.hpp"

#include <cstdint>

#include "curobo_runtime/runtime.hpp"

namespace curobo_core
{

CollisionDistance::CollisionDistance(
  rclcpp::Node * node, curobo_runtime::IBridge * bridge,
  curobo_runtime::RobotHandle handle, std::function<std::vector<float>()> current_pose)
: node_(node), bridge_(bridge), handle_(handle), current_pose_(std::move(current_pose))
{
  srv_ = node_->create_service<curobov2_ros_interfaces::srv::GetCollisionDistance>(
    std::string(node_->get_name()) + "/get_collision_distance",
    std::bind(
      &CollisionDistance::handle_get_collision_distance, this,
      std::placeholders::_1, std::placeholders::_2));
}

// PORT: collision_distance._compute_sphere_distance
std::vector<float> CollisionDistance::computeSphereDistance()
{
  const std::vector<float> q = current_pose_ ? current_pose_() : std::vector<float>{};
  if (q.empty()) {
    return {};
  }
  curobo_runtime::TensorF32 tensor;
  tensor.shape = {1, static_cast<std::int64_t>(q.size())};
  tensor.data = q;
  try {
    // PORT: ros_service_manager._gpu_guard - FK + sphere distance + a host sync,
    // so it must not run inside an in-flight CUDA-graph capture.
    auto gpu_guard = curobo_runtime::Runtime::instance().barrier().exclusive();
    return bridge_->collision_distance(handle_, tensor);
  } catch (const std::exception & e) {
    RCLCPP_WARN_THROTTLE(
      node_->get_logger(), *node_->get_clock(), 5000,
      "get_collision_distance failed: %s", e.what());
    return {};
  }
}

// PORT: collision_distance._compute_sphere_distance (response fill)
void CollisionDistance::handle_get_collision_distance(
  const std::shared_ptr<curobov2_ros_interfaces::srv::GetCollisionDistance::Request>,
  std::shared_ptr<curobov2_ros_interfaces::srv::GetCollisionDistance::Response> response)
{
  const std::vector<float> distances = computeSphereDistance();
  if (distances.empty()) {
    response->nb_sphere = 0;
    response->data.clear();
    return;
  }
  response->nb_sphere = static_cast<std::int8_t>(distances.size());
  response->data = distances;
}

}  // namespace curobo_core

// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/core/collision_distance.py
//
// The GetCollisionDistance service wrapper (PORT _compute_sphere_distance): the
// per-sphere scene distances at the robot's current configuration come from the
// bridge; this class only fills the response.
#ifndef CUROBO_CORE__CORE__COLLISION_DISTANCE_HPP_
#define CUROBO_CORE__CORE__COLLISION_DISTANCE_HPP_

#include <functional>
#include <memory>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "curobov2_ros_interfaces/srv/get_collision_distance.hpp"
#include "curobo_runtime/bridge.hpp"

namespace curobo_core
{

class CollisionDistance
{
public:
  CollisionDistance(
    rclcpp::Node * node, curobo_runtime::IBridge * bridge,
    curobo_runtime::RobotHandle handle, std::function<std::vector<float>()> current_pose);

  // PORT: collision_distance._compute_sphere_distance
  std::vector<float> computeSphereDistance();

private:
  void handle_get_collision_distance(
    const std::shared_ptr<curobov2_ros_interfaces::srv::GetCollisionDistance::Request> request,
    std::shared_ptr<curobov2_ros_interfaces::srv::GetCollisionDistance::Response> response);

  rclcpp::Node * node_;
  curobo_runtime::IBridge * bridge_;
  curobo_runtime::RobotHandle handle_;
  std::function<std::vector<float>()> current_pose_;
  rclcpp::Service<curobov2_ros_interfaces::srv::GetCollisionDistance>::SharedPtr srv_;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__CORE__COLLISION_DISTANCE_HPP_

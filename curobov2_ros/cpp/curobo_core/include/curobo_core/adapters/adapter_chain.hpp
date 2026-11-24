// SPDX-License-Identifier: Apache-2.0
#ifndef CUROBO_CORE__ADAPTERS__ADAPTER_CHAIN_HPP_
#define CUROBO_CORE__ADAPTERS__ADAPTER_CHAIN_HPP_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose.hpp>
#include <pluginlib/class_loader.hpp>
#include <rclcpp/rclcpp.hpp>

#include "curobo_core/adapters/planning_request_adapter.hpp"
#include "curobo_runtime/bridge.hpp"

namespace curobo_core
{
namespace adapters
{

/// Loads pluginlib PlanningRequestAdapter plugins from the `request_adapters`
/// parameter and runs them in order. The chain is generic: it knows only the
/// cspace names/limits from the robot model, never a robot-specific joint name.
class AdapterChain
{
public:
  AdapterChain(
    rclcpp::Node * node, curobo_runtime::IBridge * bridge,
    curobo_runtime::RobotHandle handle);
  ~AdapterChain();

  bool enabled() const {return !adapters_.empty();}

  bool adapt(
    curobov2_ros_interfaces::msg::TrajectoryGoal & goal,
    const std::vector<float> & start,
    const std::function<std::vector<geometry_msgs::msg::Pose>(
      const std::vector<geometry_msgs::msg::Pose> &, const std::string &)> & pose_resolver,
    std::string & error);

private:
  rclcpp::Node * node_;
  std::vector<std::string> joint_names_;
  std::vector<double> lower_;
  std::vector<double> upper_;
  std::vector<std::uint8_t> continuous_;
  std::unique_ptr<pluginlib::ClassLoader<PlanningRequestAdapter>> loader_;
  std::vector<std::shared_ptr<PlanningRequestAdapter>> adapters_;
};

}  // namespace adapters
}  // namespace curobo_core

#endif  // CUROBO_CORE__ADAPTERS__ADAPTER_CHAIN_HPP_

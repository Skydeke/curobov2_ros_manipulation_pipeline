// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/robot/robot_registry.py.
#include "curobo_core/robot/robot_registry.hpp"

#include <algorithm>

#include "curobo_core/robot/emulator_strategy.hpp"
#include "curobo_core/robot/joint_pose_strategy.hpp"
#include "curobo_core/robot/joint_speed_strategy.hpp"

namespace curobo_core
{

namespace
{
std::map<std::string, bool> & registry()
{
  static std::map<std::string, bool> kRegistry = {
    {"emulator", true},
    {"joint_speed", true},
    {"joint_pose", true},
  };
  return kRegistry;
}
}  // namespace

// PORT: create_strategy
std::unique_ptr<JointCommandStrategy> RobotRegistry::create(
  const std::string & key, rclcpp::Node * node, double dt,
  std::vector<std::string> joint_names, std::map<std::string, std::string> params,
  std::vector<double> default_home)
{
  if (registry().find(key) == registry().end()) {
    return nullptr;
  }
  if (key == "emulator") {
    return std::make_unique<EmulatorStrategy>(
      node, dt, std::move(joint_names), std::move(params), std::move(default_home));
  }
  if (key == "joint_speed") {
    return std::make_unique<JointSpeedStrategy>(
      node, dt, std::move(joint_names), std::move(params));
  }
  if (key == "joint_pose") {
    return std::make_unique<JointPoseStrategy>(
      node, dt, std::move(joint_names), std::move(params));
  }
  // PORT: ghost_strategy exists but is not in the registry (as in Python).
  return nullptr;
}

// PORT: available_strategies
std::vector<std::string> RobotRegistry::available_strategies()
{
  std::vector<std::string> keys;
  for (const auto & [key, _] : registry()) {
    keys.push_back(key);
  }
  std::sort(keys.begin(), keys.end());
  return keys;
}

// PORT: register_strategy
void RobotRegistry::register_strategy(const std::string & key)
{
  registry()[key] = true;
}

}  // namespace curobo_core

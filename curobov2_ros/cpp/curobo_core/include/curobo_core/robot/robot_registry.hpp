// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/robot/robot_registry.py
//
// Registry of joint-command CONTROL STRATEGIES. The Python lazy import becomes a
// factory map. Strategies are created by create_strategy().
#ifndef CUROBO_CORE__ROBOT__ROBOT_REGISTRY_HPP_
#define CUROBO_CORE__ROBOT__ROBOT_REGISTRY_HPP_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "curobo_core/robot/joint_control_strategy.hpp"

namespace curobo_core
{

class RobotRegistry
{
public:
  // PORT: create_strategy (returns nullptr for an unknown key).
  static std::unique_ptr<JointCommandStrategy> create(
    const std::string & key, rclcpp::Node * node, double dt,
    std::vector<std::string> joint_names, std::map<std::string, std::string> params = {},
    std::vector<double> default_home = {});

  // PORT: available_strategies
  static std::vector<std::string> available_strategies();

  // PORT: register_strategy
  static void register_strategy(const std::string & key);
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__ROBOT__ROBOT_REGISTRY_HPP_

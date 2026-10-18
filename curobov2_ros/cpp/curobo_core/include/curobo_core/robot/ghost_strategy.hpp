// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/robot/ghost_strategy.py
//
// Visualization-only strategy: publishes the commanded trajectory on
// `<node>/trajectory` for the RViz preview. The rest of the JointCommandStrategy
// contract is inert.
#ifndef CUROBO_CORE__ROBOT__GHOST_STRATEGY_HPP_
#define CUROBO_CORE__ROBOT__GHOST_STRATEGY_HPP_

#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "curobo_core/robot/joint_control_strategy.hpp"

namespace curobo_core
{

class GhostStrategy : public JointCommandStrategy
{
public:
  GhostStrategy(
    rclcpp::Node * node, double dt, std::vector<std::string> joint_names,
    std::map<std::string, std::string> params = {});

  // PORT: GhostStrategy.send_trajectrory
  void send_trajectrory() override;
  // PORT: GhostStrategy.get_joint_pose
  std::vector<double> get_joint_pose() override {return {};}
  // PORT: GhostStrategy.stop_robot
  void stop_robot() override;
  // PORT: GhostStrategy.get_progression
  double get_progression() override;

private:
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr pub_command_;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__ROBOT__GHOST_STRATEGY_HPP_

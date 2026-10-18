// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/robot/joint_pose_strategy.py
//
// Joint-POSE control: command target positions (a JointTrajectory whose points
// carry positions only; the driver interpolates). Same message type and topics as
// joint_speed, but no velocity/acceleration setpoints.
#ifndef CUROBO_CORE__ROBOT__JOINT_POSE_STRATEGY_HPP_
#define CUROBO_CORE__ROBOT__JOINT_POSE_STRATEGY_HPP_

#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float32.hpp>

#include "curobo_core/robot/joint_control_strategy.hpp"

namespace curobo_core
{

class JointPoseStrategy : public JointCommandStrategy
{
public:
  JointPoseStrategy(
    rclcpp::Node * node, double dt, std::vector<std::string> joint_names,
    std::map<std::string, std::string> params = {});

  // PORT: JointPoseStrategy.send_trajectrory
  void send_trajectrory() override;
  // PORT: JointPoseStrategy.get_joint_pose
  std::vector<double> get_joint_pose() override;
  // PORT: JointPoseStrategy.stop_robot
  void stop_robot() override;
  // PORT: JointPoseStrategy.get_progression
  double get_progression() override;

  // PORT: callback_trajectory_state / callback_joint_pose
  void callback_trajectory_state(const std_msgs::msg::Float32 & msg);
  void callback_joint_pose(const sensor_msgs::msg::JointState & msg);

private:
  std::vector<double> joint_pose_;
  std::vector<std::string> expected_joint_names_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr sub_trajectory_state_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_joint_state_;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__ROBOT__JOINT_POSE_STRATEGY_HPP_

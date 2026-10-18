// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/robot/joint_speed_strategy.py
//
// Joint-SPEED control: stream a full JointTrajectory (positions + velocities +
// accelerations) to the robot's FollowJointTrajectory action; an external
// driver/bridge turns it into speed commands. Adds a hard acceleration clamp
// against the REAL measured velocity and a terminal deceleration ramp.
#ifndef CUROBO_CORE__ROBOT__JOINT_SPEED_STRATEGY_HPP_
#define CUROBO_CORE__ROBOT__JOINT_SPEED_STRATEGY_HPP_

#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float32.hpp>

#include "curobo_core/core/diagnostics.hpp"
#include "curobo_core/robot/joint_control_strategy.hpp"

namespace curobo_core
{

class JointSpeedStrategy : public JointCommandStrategy
{
public:
  JointSpeedStrategy(
    rclcpp::Node * node, double dt, std::vector<std::string> joint_names,
    std::map<std::string, std::string> params = {});

  // PORT: JointSpeedStrategy.send_trajectrory
  void send_trajectrory() override;
  // PORT: JointSpeedStrategy.get_joint_pose / get_joint_velocity
  std::vector<double> get_joint_pose() override;
  std::vector<double> get_joint_velocity();
  // PORT: JointSpeedStrategy.stop_robot
  void stop_robot() override;
  // PORT: JointSpeedStrategy.get_progression
  double get_progression() override;

  // PORT: callback_trajectory_state / callback_joint_pose
  void callback_trajectory_state(const std_msgs::msg::Float32 & msg);
  void callback_joint_pose(const sensor_msgs::msg::JointState & msg);

  // PORT: _clamp_velocities / _taper_terminal_velocity (exposed for tests)
  std::vector<std::vector<double>> clamp_velocities(
    const std::vector<std::vector<double>> & vel_list);
  std::vector<std::vector<double>> taper_terminal_velocity(
    std::vector<std::vector<double>> vel_clamped);

private:
  bool debug_csv_enabled();
  void debug_csv_init();
  void debug_csv_close();
  void debug_csv_write(const std::vector<std::vector<double>> & vel_clamped);

  double max_accel_rad_s2_ = 0.0;
  double terminal_taper_dt_ = 0.6;
  double accel_clamp_dt_ = 0.0;

  std::vector<double> joint_pose_;
  std::vector<double> joint_velocity_;
  std::vector<std::string> expected_joint_names_;

  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr sub_trajectory_state_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_joint_state_;

  std::shared_ptr<DiagCsv> debug_csv_;
  double debug_csv_t0_ = 0.0;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__ROBOT__JOINT_SPEED_STRATEGY_HPP_

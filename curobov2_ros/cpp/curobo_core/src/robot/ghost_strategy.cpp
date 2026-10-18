// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/robot/ghost_strategy.py.
#include "curobo_core/robot/ghost_strategy.hpp"

#include <cmath>

#include <builtin_interfaces/msg/duration.hpp>

namespace curobo_core
{

GhostStrategy::GhostStrategy(
  rclcpp::Node * node, double dt, std::vector<std::string> joint_names,
  std::map<std::string, std::string> params)
: JointCommandStrategy(node, dt, std::move(joint_names), std::move(params))
{
  // PORT: namespaced under the owning node so the preview topic tracks it.
  pub_command_ = node_->create_publisher<trajectory_msgs::msg::JointTrajectory>(
    std::string(node_->get_name()) + "/trajectory", 10);
}

// PORT: GhostStrategy.send_trajectrory
void GhostStrategy::send_trajectrory()
{
  trajectory_msgs::msg::JointTrajectory msg;
  {
    std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
    robot_state_ = RobotState::kRunning;
    msg.joint_names = joint_names_;
    const double stamp_dt = dilated_dt();
    for (std::size_t i = 0; i < position_command_.size(); ++i) {
      trajectory_msgs::msg::JointTrajectoryPoint point;
      point.positions = position_command_[i];
      if (i < vel_command_.size()) {
        point.velocities = vel_command_[i];
      }
      if (i < accel_command_.size()) {
        point.accelerations = accel_command_[i];
      }
      const double t = stamp_dt * static_cast<double>(i);
      point.time_from_start.sec = static_cast<std::int32_t>(t);
      point.time_from_start.nanosec =
        static_cast<std::uint32_t>((t - std::floor(t)) * 1e9);
      msg.points.push_back(std::move(point));
    }
  }
  pub_command_->publish(msg);
}

// PORT: GhostStrategy.stop_robot
void GhostStrategy::stop_robot()
{
  std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
  robot_state_ = RobotState::kStopped;
}

// PORT: GhostStrategy.get_progression
double GhostStrategy::get_progression()
{
  std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
  return trajectory_progression_;
}

}  // namespace curobo_core

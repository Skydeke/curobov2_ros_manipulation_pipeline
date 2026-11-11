// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/robot/joint_speed_strategy.py.
#include "curobo_core/robot/joint_speed_strategy.hpp"

#include <algorithm>
#include <cmath>

namespace curobo_core
{

JointSpeedStrategy::JointSpeedStrategy(
  rclcpp::Node * node, double dt, std::vector<std::string> joint_names,
  std::map<std::string, std::string> params)
: JointCommandStrategy(node, dt, std::move(joint_names), std::move(params))
{
  auto state = params_.find("state_topic");
  if (state != params_.end() && !state->second.empty()) {
    sub_trajectory_state_ = node_->create_subscription<std_msgs::msg::Float32>(
      state->second, 10,
      [this](const std_msgs::msg::Float32::SharedPtr msg) {
        callback_trajectory_state(*msg);
      });
  }
  auto joint_states = params_.find("joint_states_topic");
  if (joint_states != params_.end() && !joint_states->second.empty()) {
    sub_joint_state_ = node_->create_subscription<sensor_msgs::msg::JointState>(
      joint_states->second, 10,
      [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
        callback_joint_pose(*msg);
      });
  }
  joint_pose_.assign(dof(), 0.0);
  joint_velocity_.assign(dof(), 0.0);
  expected_joint_names_ = joint_names_;
}

// PORT: JointSpeedStrategy.send_trajectrory
void JointSpeedStrategy::send_trajectrory()
{
  trajectory_msgs::msg::JointTrajectory msg;
  std::vector<std::vector<double>> positions;
  {
    std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
    robot_state_ = RobotState::kRunning;
    msg.joint_names = joint_names_;
    if (position_command_.empty()) {
      trajectory_progression_ = 1.0;
    }
    const double speed_scale = time_dilation_scale();
    auto velocities = vel_command_;
    if (speed_scale != 1.0) {
      for (auto & row : velocities) {
        for (double & v : row) {
          v *= speed_scale;
        }
      }
    }
    // Acceleration is d2p/dt2, so it follows the velocity scale squared.
    const double accel_scale = speed_scale * speed_scale;
    if (accel_scale != 1.0) {
      for (auto & row : accel_command_) {
        for (double & a : row) {
          a *= accel_scale;
        }
      }
    }
    const double stamp_dt = dilated_dt();
    for (std::size_t i = 0; i < position_command_.size(); ++i) {
      trajectory_msgs::msg::JointTrajectoryPoint point;
      point.positions = position_command_[i];
      point.velocities = i < velocities.size() ? velocities[i] :
        std::vector<double>(this->dof(), 0.0);
      point.accelerations = i < accel_command_.size() ? accel_command_[i] :
        std::vector<double>(this->dof(), 0.0);
      const double t = stamp_dt * static_cast<double>(i);
      point.time_from_start.sec = static_cast<std::int32_t>(t);
      point.time_from_start.nanosec =
        static_cast<std::uint32_t>((t - std::floor(t)) * 1e9);
      positions.push_back(point.positions);
      msg.points.push_back(std::move(point));
    }
    position_command_.clear();
    vel_command_.clear();
    accel_command_.clear();
    trajectory_progression_ = 0.0;
  }
  send_as_action(msg, positions);
}

// PORT: JointSpeedStrategy.get_joint_pose / get_joint_velocity
std::vector<double> JointSpeedStrategy::get_joint_pose()
{
  std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
  return joint_pose_;
}

std::vector<double> JointSpeedStrategy::get_joint_velocity()
{
  std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
  return joint_velocity_;
}

// PORT: JointSpeedStrategy.stop_robot
void JointSpeedStrategy::stop_robot()
{
  {
    std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
    vel_command_.clear();
    position_command_.clear();
    accel_command_.clear();
    command_index_ = 0;
    trajectory_progression_ = 0.0;
    robot_state_ = RobotState::kStopped;
  }
  clear_execution_timer();
  cancel_action();
}

// PORT: JointSpeedStrategy.callback_trajectory_state
void JointSpeedStrategy::callback_trajectory_state(const std_msgs::msg::Float32 & msg)
{
  mark_progression_feedback(msg.data);
}

// PORT: JointSpeedStrategy.callback_joint_pose
void JointSpeedStrategy::callback_joint_pose(const sensor_msgs::msg::JointState & msg)
{
  notify_joint_state(msg);
  std::vector<std::size_t> remap;
  if (!expected_joint_names_.empty() && !msg.name.empty()) {
    bool ok = true;
    for (const auto & n : expected_joint_names_) {
      auto it = std::find(msg.name.begin(), msg.name.end(), n);
      if (it == msg.name.end()) {
        ok = false;
        break;
      }
      remap.push_back(static_cast<std::size_t>(std::distance(msg.name.begin(), it)));
    }
    if (!ok) {
      remap.clear();
    }
  }

  // A message that names none of the expected arm joints and is too short to
  // hold them is another publisher's (gripper, head, ...). Reading it
  // positionally overwrote joint_pose_ and fed a short/foreign vector into the
  // feedback buffer, so the segmenter placed the robot spheres at a wrong pose.
  if (remap.empty() && !expected_joint_names_.empty() &&
    msg.position.size() < expected_joint_names_.size())
  {
    return;
  }

  const std::size_t n = dof() != 0u ? dof() : msg.position.size();
  std::vector<double> feedback;
  {
    std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
    if (!remap.empty()) {
      joint_pose_.clear();
      joint_velocity_.clear();
      for (std::size_t idx : remap) {
        joint_pose_.push_back(idx < msg.position.size() ? msg.position[idx] : 0.0);
        if (!msg.velocity.empty()) {
          joint_velocity_.push_back(idx < msg.velocity.size() ? msg.velocity[idx] : 0.0);
        }
      }
      joint_names_ = expected_joint_names_;
      feedback = joint_pose_;
    } else {
      joint_pose_.assign(
        msg.position.begin(), msg.position.begin() + std::min(n, msg.position.size()));
      if (!msg.velocity.empty()) {
        joint_velocity_.assign(
          msg.velocity.begin(), msg.velocity.begin() + std::min(n, msg.velocity.size()));
      }
      if (!msg.name.empty()) {
        joint_names_.assign(
          msg.name.begin(), msg.name.begin() + std::min(n, msg.name.size()));
      }
      feedback = joint_pose_;
    }
  }
  const bool have_stamp = msg.header.stamp.sec != 0 || msg.header.stamp.nanosec != 0;
  if (have_stamp) {
    record_joint_feedback(
      static_cast<std::int64_t>(msg.header.stamp.sec) * 1000000000 +
      static_cast<std::int64_t>(msg.header.stamp.nanosec),
      feedback);
  }
}

// PORT: JointSpeedStrategy.get_progression
double JointSpeedStrategy::get_progression()
{
  return action_progression();
}

}  // namespace curobo_core

// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/robot/joint_pose_strategy.py.
#include "curobo_core/robot/joint_pose_strategy.hpp"

#include <algorithm>
#include <cmath>

namespace curobo_core
{

JointPoseStrategy::JointPoseStrategy(
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
  expected_joint_names_ = joint_names_;
}

// PORT: JointPoseStrategy.send_trajectrory
void JointPoseStrategy::send_trajectrory()
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
    const double stamp_dt = dilated_dt();
    for (std::size_t i = 0; i < position_command_.size(); ++i) {
      trajectory_msgs::msg::JointTrajectoryPoint point;
      point.positions = position_command_[i];   // positions only
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

// PORT: JointPoseStrategy.get_joint_pose
std::vector<double> JointPoseStrategy::get_joint_pose()
{
  std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
  return joint_pose_;
}

// PORT: JointPoseStrategy.stop_robot
void JointPoseStrategy::stop_robot()
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

// PORT: JointPoseStrategy.callback_trajectory_state
void JointPoseStrategy::callback_trajectory_state(const std_msgs::msg::Float32 & msg)
{
  mark_progression_feedback(msg.data);
}

// PORT: JointPoseStrategy.callback_joint_pose
void JointPoseStrategy::callback_joint_pose(const sensor_msgs::msg::JointState & msg)
{
  notify_joint_state(msg);
  // Prefer a NAME-based remap against the canonical cspace arm joints when the
  // message names every one of them (order-agnostic; drops interleaved non-arm
  // joints). Falls back to a positional read when the names do not line up.
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
      for (std::size_t idx : remap) {
        joint_pose_.push_back(idx < msg.position.size() ? msg.position[idx] : 0.0);
      }
      joint_names_ = expected_joint_names_;
      feedback = joint_pose_;
    } else {
      joint_pose_.assign(
        msg.position.begin(), msg.position.begin() + std::min(n, msg.position.size()));
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

// PORT: JointPoseStrategy.get_progression
double JointPoseStrategy::get_progression()
{
  return action_progression();
}

}  // namespace curobo_core

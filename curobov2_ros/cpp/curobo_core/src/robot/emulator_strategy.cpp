// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/robot/emulator_strategy.py.
#include "curobo_core/robot/emulator_strategy.hpp"

#include <chrono>

#include <sensor_msgs/msg/joint_state.hpp>

namespace curobo_core
{

EmulatorStrategy::EmulatorStrategy(
  rclcpp::Node * node, double dt, std::vector<std::string> joint_names,
  std::map<std::string, std::string> params, std::vector<double> default_home)
: JointCommandStrategy(node, dt, std::move(joint_names), std::move(params))
{
  auto it = params_.find("joint_states_topic");
  joint_states_topic_ = it != params_.end() ? it->second : "/emulator/joint_states";
  pub_joint_states_ =
    node_->create_publisher<sensor_msgs::msg::JointState>(joint_states_topic_, 10);

  // PORT: seed at the config's default (home) pose, NOT zeros (a zero start
  // state is a cspace violation for the franka). Falls back to zeros.
  current_joint_positions_ = default_home.size() == dof() ?
    default_home : std::vector<double>(dof(), 0.0);

  publish_state(joint_names_, current_joint_positions_, std::vector<double>(dof(), 0.0));
  RCLCPP_INFO(
    node_->get_logger(), "Emulator strategy initialized - Publishing to %s",
    joint_states_topic_.c_str());
}

// PORT: EmulatorStrategy.__del__
EmulatorStrategy::~EmulatorStrategy()
{
  stop_execution_.store(true);
  if (execution_thread_.joinable()) {
    execution_thread_.join();
  }
}

// PORT: EmulatorStrategy._publish_state
void EmulatorStrategy::publish_state(
  const std::vector<std::string> & names, const std::vector<double> & positions,
  const std::vector<double> & velocities)
{
  const auto now = node_->get_clock()->now();
  sensor_msgs::msg::JointState msg;
  msg.header.stamp = now;
  msg.name = names;
  msg.position = positions;
  msg.velocity = velocities;
  pub_joint_states_->publish(msg);
  // The emulator is the joint-state authority: mirror into the shared feedback
  // buffer for time-synchronized queries (robot segmentation).
  record_joint_feedback(now.nanoseconds(), positions);
}

// PORT: EmulatorStrategy._apply_immediate
void EmulatorStrategy::apply_immediate(long index)
{
  stop_execution_.store(true);
  std::vector<double> positions;
  std::vector<double> velocities;
  std::vector<std::string> names;
  {
    std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
    const long n = static_cast<long>(position_command_.size());
    const long i = index < 0 ? n + index : index;
    if (n == 0 || i < 0 || i >= n) {
      return;
    }
    positions = position_command_[static_cast<std::size_t>(i)];
    velocities = vel_command_.empty() ? std::vector<double>(positions.size(), 0.0) :
      vel_command_[static_cast<std::size_t>(i)];
    names = joint_names_;
    current_joint_positions_ = positions;
    robot_state_ = RobotState::kRunning;
    trajectory_progression_ = 1.0;
  }
  publish_state(names, positions, velocities);
}

// PORT: EmulatorStrategy.send_trajectrory
void EmulatorStrategy::send_trajectrory()
{
  std::size_t n_points = 0;
  {
    std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
    n_points = position_command_.size();
  }
  if (n_points == 0) {
    RCLCPP_WARN(node_->get_logger(), "No trajectory to execute");
    std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
    trajectory_progression_ = 1.0;
    return;
  }
  if (n_points == 1) {
    apply_immediate(0);
    return;
  }
  if (execution_thread_.joinable()) {
    // A previous multi-point playback is still running: apply the target point
    // immediately instead of threading through a buffer that would be killed.
    apply_immediate(-1);
    return;
  }

  stop_execution_.store(true);
  std::vector<std::vector<double>> positions;
  std::vector<std::vector<double>> velocities;
  std::vector<std::string> names;
  int epoch = 0;
  {
    std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
    positions = position_command_;
    velocities = vel_command_.empty() ?
      std::vector<std::vector<double>>(positions.size(),
        std::vector<double>(dof(), 0.0)) : vel_command_;
    names = joint_names_;
    robot_state_ = RobotState::kRunning;
    trajectory_progression_ = 0.0;
    command_index_ = 0;
    epoch = buffer_epoch_;
  }
  stop_execution_.store(false);
  execution_thread_ = std::thread(
    &EmulatorStrategy::execute_trajectory, this, positions, velocities, names, epoch);
}

// PORT: EmulatorStrategy._execute_trajectory
void EmulatorStrategy::execute_trajectory(
  std::vector<std::vector<double>> positions, std::vector<std::vector<double>> velocities,
  std::vector<std::string> names, int epoch)
{
  try {
    const std::size_t total = positions.size();
    std::size_t index = 0;
    while (index < total && !stop_execution_.load()) {
      {
        std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
        if (buffer_epoch_ != epoch) {
          return;   // preempted by a newer command
        }
        current_joint_positions_ = positions[index];
        index += 1;
        command_index_ = index;
        trajectory_progression_ =
          static_cast<double>(index) / static_cast<double>(total);
      }
      publish_state(names, positions[index - 1], velocities[index - 1]);
      std::this_thread::sleep_for(std::chrono::duration<double>(dilated_dt()));
    }
    std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
    if (buffer_epoch_ == epoch) {
      robot_state_ = RobotState::kIdle;
      if (!stop_execution_.load()) {
        trajectory_progression_ = 1.0;
      }
    }
  } catch (const std::exception & e) {
    RCLCPP_ERROR(node_->get_logger(), "Emulator execution error: %s", e.what());
    std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
    if (buffer_epoch_ == epoch) {
      robot_state_ = RobotState::kError;
    }
  }
}

// PORT: EmulatorStrategy.get_joint_pose
std::vector<double> EmulatorStrategy::get_joint_pose()
{
  std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
  return current_joint_positions_;
}

// PORT: EmulatorStrategy.stop_robot
void EmulatorStrategy::stop_robot()
{
  RCLCPP_INFO(node_->get_logger(), "Emulator: Stopping trajectory execution");
  stop_execution_.store(true);
  std::vector<std::string> names;
  std::vector<double> positions;
  {
    std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
    vel_command_.clear();
    position_command_.clear();
    accel_command_.clear();
    command_index_ = 0;
    trajectory_progression_ = 0.0;
    robot_state_ = RobotState::kStopped;
    buffer_epoch_ += 1;
    names = joint_names_;
    positions = current_joint_positions_;
  }
  publish_state(names, positions, std::vector<double>(names.size(), 0.0));
}

// PORT: EmulatorStrategy.get_progression
double EmulatorStrategy::get_progression()
{
  std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
  return trajectory_progression_;
}

// PORT: EmulatorStrategy.wait_for_execution_complete
bool EmulatorStrategy::wait_for_execution_complete(double /*timeout_s*/)
{
  if (execution_thread_.joinable()) {
    execution_thread_.join();
  }
  return true;
}

}  // namespace curobo_core

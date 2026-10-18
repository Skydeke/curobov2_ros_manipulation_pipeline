// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/robot/joint_speed_strategy.py.
#include "curobo_core/robot/joint_speed_strategy.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>

namespace curobo_core
{

namespace
{
double steady_now()
{
  return std::chrono::duration<double>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}
constexpr double kPi = 3.14159265358979323846;
double degrees(double rad) {return rad * 180.0 / kPi;}
}  // namespace

JointSpeedStrategy::JointSpeedStrategy(
  rclcpp::Node * node, double dt, std::vector<std::string> joint_names,
  std::map<std::string, std::string> params)
: JointCommandStrategy(node, dt, std::move(joint_names), std::move(params))
{
  auto num = [this](const char * key, double fallback) {
      auto it = params_.find(key);
      return it != params_.end() ? std::stod(it->second) : fallback;
    };
  max_accel_rad_s2_ = num("max_joint_accel_dps", 45.0) * kPi / 180.0;
  terminal_taper_dt_ = num("terminal_taper_dt", 0.6);
  accel_clamp_dt_ = num("accel_clamp_dt", dt_);

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

// PORT: JointSpeedStrategy._clamp_velocities
std::vector<std::vector<double>> JointSpeedStrategy::clamp_velocities(
  const std::vector<std::vector<double>> & vel_list)
{
  const double max_dv = max_accel_rad_s2_ * accel_clamp_dt_;
  std::vector<std::vector<double>> clamped;
  std::vector<double> prev;
  {
    std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
    prev = joint_velocity_;
  }
  for (const auto & v : vel_list) {
    std::vector<double> row;
    row.reserve(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
      const double p = i < prev.size() ? prev[i] : 0.0;
      row.push_back(p + std::max(-max_dv, std::min(max_dv, v[i] - p)));
    }
    clamped.push_back(row);
    prev = row;
  }
  return clamped;
}

// PORT: JointSpeedStrategy._taper_terminal_velocity
std::vector<std::vector<double>> JointSpeedStrategy::taper_terminal_velocity(
  std::vector<std::vector<double>> vel_clamped)
{
  if (vel_clamped.empty()) {
    return vel_clamped;
  }
  const std::size_t n = vel_clamped.size();
  const std::size_t dof = this->dof();
  if (n < 2) {
    vel_clamped[n - 1] = std::vector<double>(dof, 0.0);
    return vel_clamped;
  }
  const double dt = dilated_dt();
  if (dt <= 0.0) {
    vel_clamped[n - 1] = std::vector<double>(dof, 0.0);
    return vel_clamped;
  }
  const std::size_t k = std::max<std::size_t>(
    1, std::min<std::size_t>(
      n - 1, static_cast<std::size_t>(std::llround(terminal_taper_dt_ / dt))));
  for (std::size_t e = 0; e < k; ++e) {
    const double factor = static_cast<double>(e) / static_cast<double>(k);
    auto & row = vel_clamped[n - 1 - e];
    for (double & v : row) {
      v *= factor;
    }
  }
  vel_clamped[n - 1] = std::vector<double>(dof, 0.0);
  {
    std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
    if (accel_command_.size() == n) {
      accel_command_[n - 1] = std::vector<double>(dof, 0.0);
    }
  }
  return vel_clamped;
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
    auto vel_clamped = clamp_velocities(vel_command_);
    vel_clamped = taper_terminal_velocity(vel_clamped);
    if (!vel_command_.empty()) {
      debug_csv_write(vel_clamped);
    }
    const double stamp_dt = dilated_dt();
    for (std::size_t i = 0; i < position_command_.size(); ++i) {
      trajectory_msgs::msg::JointTrajectoryPoint point;
      point.positions = position_command_[i];
      point.velocities = i < vel_clamped.size() ? vel_clamped[i] :
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
  debug_csv_close();
}

// PORT: JointSpeedStrategy.callback_trajectory_state
void JointSpeedStrategy::callback_trajectory_state(const std_msgs::msg::Float32 & msg)
{
  mark_progression_feedback(msg.data);
}

// PORT: JointSpeedStrategy.callback_joint_pose
void JointSpeedStrategy::callback_joint_pose(const sensor_msgs::msg::JointState & msg)
{
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

// PORT: JointSpeedStrategy._debug_csv_enabled/_init/_close/_write
bool JointSpeedStrategy::debug_csv_enabled()
{
  if (!node_->has_parameter("joint_speed_debug_csv")) {
    node_->declare_parameter("joint_speed_debug_csv", false);
  }
  return node_->get_parameter("joint_speed_debug_csv").as_bool();
}

void JointSpeedStrategy::debug_csv_init()
{
  debug_csv_ = open_diag_csv(node_, "speedj_publish");
  if (debug_csv_ == nullptr) {
    return;
  }
  debug_csv_->write_header_once({
    "t_s", "n_points", "real_vel_max_dps", "point0_max_dps",
    "point0_vs_real_accel_dps2", "unclamped_point0_max_dps",
    "clamp_active_j", "intra_batch_accel_max_dps2"});
  debug_csv_t0_ = steady_now();
}

void JointSpeedStrategy::debug_csv_close()
{
  if (debug_csv_ != nullptr) {
    debug_csv_->close();
  }
  debug_csv_.reset();
}

void JointSpeedStrategy::debug_csv_write(
  const std::vector<std::vector<double>> & vel_clamped)
{
  if (!debug_csv_enabled()) {
    return;
  }
  if (debug_csv_ == nullptr) {
    debug_csv_init();
  }
  if (debug_csv_ == nullptr) {
    return;
  }
  std::vector<double> real, p0, raw0;
  {
    std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
    real = joint_velocity_;
    p0 = vel_clamped.empty() ? std::vector<double>(dof(), 0.0) : vel_clamped[0];
    raw0 = vel_command_.empty() ? std::vector<double>(dof(), 0.0) : vel_command_[0];
  }
  double accel_p0 = 0.0;
  for (std::size_t i = 0; i < p0.size(); ++i) {
    const double r = i < real.size() ? real[i] : 0.0;
    accel_p0 = std::max(accel_p0, std::abs(p0[i] - r));
  }
  accel_p0 = accel_clamp_dt_ != 0.0 ? degrees(accel_p0 / accel_clamp_dt_) : 0.0;
  std::vector<std::string> clamp_active;
  double raw0_max = 0.0;
  for (std::size_t i = 0; i < p0.size(); ++i) {
    const double u = i < raw0.size() ? raw0[i] : 0.0;
    raw0_max = std::max(raw0_max, std::abs(u));
    if (std::abs(p0[i] - u) > 1e-9) {
      clamp_active.push_back(std::to_string(i));
    }
  }
  double intra = 0.0;
  for (std::size_t i = 1; i < vel_clamped.size(); ++i) {
    double d = 0.0;
    for (std::size_t j = 0; j < vel_clamped[i].size(); ++j) {
      d = std::max(d, std::abs(vel_clamped[i][j] - vel_clamped[i - 1][j]));
    }
    intra = std::max(intra,
      accel_clamp_dt_ != 0.0 ? degrees(d / accel_clamp_dt_) : 0.0);
  }
  double real_max = 0.0, p0_max = 0.0;
  for (double v : real) {
    real_max = std::max(real_max, std::abs(v));
  }
  for (double v : p0) {
    p0_max = std::max(p0_max, std::abs(v));
  }
  std::ostringstream active;
  active << "[";
  for (std::size_t i = 0; i < clamp_active.size(); ++i) {
    active << (i ? "," : "") << clamp_active[i];
  }
  active << "]";
  std::ostringstream t;
  t << (steady_now() - debug_csv_t0_);
  debug_csv_->writerow({
    t.str(), std::to_string(vel_clamped.size()),
    std::to_string(degrees(real_max)), std::to_string(degrees(p0_max)),
    std::to_string(accel_p0), std::to_string(degrees(raw0_max)),
    active.str(), std::to_string(intra)});
}

}  // namespace curobo_core

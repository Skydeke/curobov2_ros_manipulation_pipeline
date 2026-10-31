// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/robot/joint_control_strategy.py
//
// Base class for a joint-command CONTROL MODE: command buffers, buffer epoch,
// the timestamped joint-feedback buffer, time-dilation re-timing, and the
// FollowJointTrajectory action client path. No GPU.
#ifndef CUROBO_CORE__ROBOT__JOINT_CONTROL_STRATEGY_HPP_
#define CUROBO_CORE__ROBOT__JOINT_CONTROL_STRATEGY_HPP_

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

namespace curobo_core
{

// PORT: joint_control_strategy.RobotState
enum class RobotState { kRunning = 0, kStopped = 1, kReady = 2, kError = 3, kIdle = 4 };

// PORT: joint_control_strategy.JointCommandStrategy
class JointCommandStrategy
{
public:
  JointCommandStrategy(
    rclcpp::Node * node, double dt, std::vector<std::string> joint_names,
    std::map<std::string, std::string> params = {});
  virtual ~JointCommandStrategy() = default;

  // PORT: set_command - defensive copies, returns the new buffer epoch.
  int set_command(
    const std::vector<std::string> & joint_names,
    const std::vector<std::vector<double>> & vel_command,
    const std::vector<std::vector<double>> & accel_command,
    const std::vector<std::vector<double>> & position_command);

  // PORT: get_joint_name
  std::vector<std::string> get_joint_name();

  // PORT: mark_progression_feedback
  void mark_progression_feedback(double value);

  // PORT: _dilated_dt (time_dilation_factor)
  double dilated_dt() const;

  // PORT: mark_execution_start / clear_execution_timer
  void mark_execution_start(
    int num_points, const std::vector<std::vector<double>> & positions = {});
  // PORT: JointCommandStrategy.clear_execution_timer
  void clear_execution_timer();
  // PORT: JointCommandStrategy._cancel_action (from stop_robot)
  void cancel_action();

  // PORT: _record_joint_feedback / get_joint_pose_at
  void record_joint_feedback(std::int64_t stamp_ns, const std::vector<double> & position);
  std::optional<std::vector<double>> get_joint_pose_at(
    std::int64_t stamp_ns, std::int64_t offset_ns = 0) const;

  std::int64_t buffer_epoch() const;

  // PORT: action_failure_summary
  std::string action_failure_summary();

  // Abstract contract (PORT: get_joint_pose / stop_robot / get_progression /
  // send_trajectrory).
  virtual std::vector<double> get_joint_pose() = 0;
  virtual void stop_robot() = 0;
  virtual double get_progression() = 0;
  virtual void send_trajectrory() = 0;

  const std::vector<std::string> & joint_names() const {return joint_names_;}
  std::size_t dof() const {return joint_names_.size();}

  // PORT: forwards the raw joint-state message before it is reduced to the cspace
  // pose. Unpinned lock joints (`lock_joints: {joint: null}`) appear in the
  // message but not in joint_names(), so this is how the node learns their value
  // (robot_model_update_lock_joints). Node-owned: set once, before spinning.
  void set_joint_state_callback(
    std::function<void(const sensor_msgs::msg::JointState &)> callback)
  {joint_state_callback_ = std::move(callback);}

protected:
  // Called from the strategy's joint-state subscription, before the cspace remap.
  void notify_joint_state(const sensor_msgs::msg::JointState & msg)
  {
    if (joint_state_callback_) {joint_state_callback_(msg);}
  }
  // PORT: _send_as_action / _get_progression / _cancel_action /
  // _ensure_action_client (the FollowJointTrajectory execution path).
  void send_as_action(
    const trajectory_msgs::msg::JointTrajectory & msg,
    const std::vector<std::vector<double>> & positions);
  double action_progression();

  rclcpp::Node * node_;
  double dt_;
  std::vector<std::string> joint_names_;
  std::map<std::string, std::string> params_;
  std::function<void(const sensor_msgs::msg::JointState &)> joint_state_callback_;

  mutable std::recursive_mutex buffer_lock_;
  std::vector<std::vector<double>> position_command_;
  std::vector<std::vector<double>> vel_command_;
  std::vector<std::vector<double>> accel_command_;
  std::size_t command_index_ = 0;
  double trajectory_progression_ = 0.0;
  RobotState robot_state_ = RobotState::kIdle;
  int buffer_epoch_ = 0;

  bool progression_feedback_ready_ = false;
  std::optional<double> exec_start_mono_;
  double exec_total_s_ = 0.0;
  std::optional<std::vector<std::vector<double>>> exec_positions_;

  std::deque<std::pair<std::int64_t, std::vector<double>>> feedback_buffer_;

  std::string action_topic_;
  using FollowJT = control_msgs::action::FollowJointTrajectory;
  struct ActionResult
  {
    bool succeeded = false;
    int status = 0;
    int error_code = 0;
    std::string error_string;
  };
  // One FollowJointTrajectory controller that owns a subset of the cspace.
  // MoveIt-style dispatch: a single planned trajectory is split by joint across
  // every route and executed on the controllers in parallel. With no
  // controller config the strategy keeps the legacy behaviour: ONE route on
  // ``action_topic`` owning every joint.
  struct ControllerRoute
  {
    std::string action;
    std::vector<std::string> joints;
    rclcpp_action::Client<FollowJT>::SharedPtr client;
    rclcpp_action::ClientGoalHandle<FollowJT>::SharedPtr goal_handle;
    std::optional<double> last_progress;
    std::optional<ActionResult> result;
  };
  std::vector<ControllerRoute> routes_;
  int action_seq_ = 0;
  double action_total_duration_s_ = 0.0;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__ROBOT__JOINT_CONTROL_STRATEGY_HPP_

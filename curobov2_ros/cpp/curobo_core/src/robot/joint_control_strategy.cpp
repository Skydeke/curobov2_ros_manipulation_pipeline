// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of the base half of curobov2_ros/robot/joint_control_strategy.py.
#include "curobo_core/robot/joint_control_strategy.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <trajectory_msgs/msg/joint_trajectory.hpp>

namespace curobo_core
{

JointCommandStrategy::JointCommandStrategy(
  rclcpp::Node * node, double dt, std::vector<std::string> joint_names,
  std::map<std::string, std::string> params)
: node_(node), dt_(dt), joint_names_(std::move(joint_names)), params_(std::move(params))
{
  auto it = params_.find("action_topic");
  if (it != params_.end()) {
    action_topic_ = it->second;
  }
}

// PORT: JointCommandStrategy.set_command
int JointCommandStrategy::set_command(
  const std::vector<std::string> & joint_names,
  const std::vector<std::vector<double>> & vel_command,
  const std::vector<std::vector<double>> & accel_command,
  const std::vector<std::vector<double>> & position_command)
{
  std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
  position_command_ = position_command;
  vel_command_ = vel_command;
  accel_command_ = accel_command;
  joint_names_ = joint_names;
  buffer_epoch_ += 1;
  return buffer_epoch_;
}

// PORT: JointCommandStrategy.get_joint_name
std::vector<std::string> JointCommandStrategy::get_joint_name()
{
  std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
  return joint_names_;
}

// PORT: JointCommandStrategy.mark_progression_feedback
void JointCommandStrategy::mark_progression_feedback(double value)
{
  std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
  trajectory_progression_ = value;
  progression_feedback_ready_ = true;
}

// PORT: JointCommandStrategy._dilated_dt
double JointCommandStrategy::dilated_dt() const
{
  double factor = 1.0;
  if (node_->has_parameter("time_dilation_factor")) {
    factor = node_->get_parameter("time_dilation_factor").as_double();
  }
  if (factor <= 0.0) {
    return dt_;
  }
  return dt_ / factor;
}

// PORT: JointCommandStrategy.mark_execution_start
void JointCommandStrategy::mark_execution_start(
  int num_points, const std::vector<std::vector<double>> & positions)
{
  std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
  exec_total_s_ = dilated_dt() * std::max(num_points - 1, 0);
  exec_positions_ = positions.empty() ? std::nullopt :
    std::make_optional(positions);
  exec_start_mono_ = std::chrono::duration<double>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}

// PORT: JointCommandStrategy._record_joint_feedback
void JointCommandStrategy::record_joint_feedback(
  std::int64_t stamp_ns, const std::vector<double> & position)
{
  std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
  // Never mix joint-vector sizes in one buffer: interpolation indexes both
  // neighbours by the first one's length.
  if (!feedback_buffer_.empty() && feedback_buffer_.back().second.size() != position.size()) {
    return;
  }
  // Out-of-order stamps (clock jump, sim-time reset) break the sorted-search
  // below; restart the buffer instead of interpolating across the discontinuity.
  if (!feedback_buffer_.empty() && stamp_ns < feedback_buffer_.back().first) {
    feedback_buffer_.clear();
  }
  feedback_buffer_.emplace_back(stamp_ns, position);
  while (feedback_buffer_.size() > 600) {
    feedback_buffer_.pop_front();
  }
}

// PORT: JointCommandStrategy.get_joint_pose_at (linear interpolation, clamped)
std::optional<std::vector<double>> JointCommandStrategy::get_joint_pose_at(
  std::int64_t stamp_ns, std::int64_t offset_ns) const
{
  const std::int64_t target = stamp_ns + offset_ns;
  std::deque<std::pair<std::int64_t, std::vector<double>>> buffer;
  {
    std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
    buffer = feedback_buffer_;
  }
  if (buffer.empty()) {
    return std::nullopt;
  }
  std::size_t i = 0;
  while (i < buffer.size() && buffer[i].first <= target) {
    ++i;
  }
  if (i == 0) {
    return buffer.front().second;
  }
  if (i >= buffer.size()) {
    return buffer.back().second;
  }
  const auto & [t0, p0] = buffer[i - 1];
  const auto & [t1, p1] = buffer[i];
  if (t1 <= t0 || p1.size() != p0.size()) {
    return p1;
  }
  const double w = static_cast<double>(target - t0) / static_cast<double>(t1 - t0);
  std::vector<double> out(p0.size());
  for (std::size_t j = 0; j < p0.size(); ++j) {
    out[j] = p0[j] + w * (p1[j] - p0[j]);
  }
  return out;
}

std::int64_t JointCommandStrategy::buffer_epoch() const
{
  std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
  return buffer_epoch_;
}

// PORT: JointCommandStrategy.action_failure_summary
std::string JointCommandStrategy::action_failure_summary()
{
  std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
  if (action_result_ && !action_result_->succeeded) {
    std::string msg = "status=" + std::to_string(action_result_->status) +
      " error_code=" + std::to_string(action_result_->error_code) + " " +
      action_result_->error_string;
    while (!msg.empty() && msg.back() == ' ') {
      msg.pop_back();
    }
    return msg;
  }
  return "";
}

// PORT: JointCommandStrategy._send_as_action - the ONE execution path.
void JointCommandStrategy::send_as_action(
  const trajectory_msgs::msg::JointTrajectory & msg,
  const std::vector<std::vector<double>> & positions)
{
  std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
  action_result_.reset();
  action_last_progress_.reset();
  action_seq_ += 1;
  const int seq = action_seq_;

  const int n = static_cast<int>(msg.points.size());
  action_total_duration_s_ = dilated_dt() * std::max(n - 1, 0);

  if (action_topic_.empty()) {
    action_result_ = ActionResult{
      false, 0, 0,
      "strategy_params.action_topic not configured — the FollowJointTrajectory "
      "action is the only execution path"};
    throw std::runtime_error(
      std::string("strategy_params.action_topic must name the "
      "FollowJointTrajectory action (e.g. /joint_trajectory_controller/"
      "follow_joint_trajectory) — the action is the only execution path"));
  }

  if (!action_client_) {
    action_client_ = rclcpp_action::create_client<FollowJT>(node_, action_topic_);
  }
  if (!action_client_->wait_for_action_server(std::chrono::seconds(5))) {
    action_result_ = ActionResult{
      false, 0, 0,
      "FollowJointTrajectory action server " + action_topic_ + " not available"};
    throw std::runtime_error(
      "FollowJointTrajectory action server " + action_topic_ +
      " is not available; refusing to fall back to topic publish");
  }

  FollowJT::Goal goal;
  goal.trajectory = msg;
  rclcpp_action::Client<FollowJT>::SendGoalOptions options;
  options.feedback_callback =
    [this, seq](rclcpp_action::ClientGoalHandle<FollowJT>::SharedPtr,
      const std::shared_ptr<const FollowJT::Feedback> feedback) {
      std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
      if (seq != action_seq_) {
        return;
      }
      const auto & t = feedback->desired.time_from_start;
      const double ts = static_cast<double>(t.sec) + static_cast<double>(t.nanosec) * 1e-9;
      if (action_total_duration_s_ > 0.0) {
        action_last_progress_ = std::min(1.0, std::max(0.0, ts / action_total_duration_s_));
      }
    };
  options.result_callback =
    [this, seq](const rclcpp_action::ClientGoalHandle<FollowJT>::WrappedResult & wrapped) {
      std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
      if (seq != action_seq_) {
        return;
      }
      const bool succeeded =
        wrapped.code == rclcpp_action::ResultCode::SUCCEEDED;
      ActionResult result;
      result.succeeded = succeeded;
      result.status = static_cast<int>(wrapped.code);
      result.error_code = wrapped.result ? wrapped.result->error_code : 0;
      result.error_string = wrapped.result ? wrapped.result->error_string : "";
      // PORT: _action_result_done
      action_result_ = result;
      action_last_progress_ = succeeded ? std::make_optional(1.0) : std::nullopt;
    };
  // PORT: _action_goal_response - keep the accepted handle so it can be canceled.
  options.goal_response_callback =
    [this, seq](rclcpp_action::ClientGoalHandle<FollowJT>::SharedPtr goal_handle) {
      std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
      if (seq != action_seq_) {
        return;
      }
      if (!goal_handle) {
        action_result_ = ActionResult{
          false, 0, 0, "action goal rejected / not accepted"};
        return;
      }
      action_goal_handle_ = goal_handle;
    };
  action_client_->async_send_goal(goal, options);

  mark_execution_start(n, positions);
}

// PORT: JointCommandStrategy.clear_execution_timer
void JointCommandStrategy::clear_execution_timer()
{
  std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
  exec_start_mono_.reset();
  exec_total_s_ = 0.0;
  exec_positions_.reset();
}

// PORT: JointCommandStrategy._cancel_action (from stop_robot)
void JointCommandStrategy::cancel_action()
{
  rclcpp_action::ClientGoalHandle<FollowJT>::SharedPtr goal_handle;
  {
    std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
    goal_handle = action_goal_handle_;
    action_goal_handle_.reset();
    action_result_ = ActionResult{false, 0, 0, "canceled"};
    action_seq_ += 1;
  }
  if (goal_handle && action_client_) {
    try {
      action_client_->async_cancel_goal(goal_handle);
    } catch (const std::exception &) {
      // best-effort: cancel must not throw out of a stop path
    }
  }
}

// PORT: JointCommandStrategy._get_progression - the action is the ONLY source.
double JointCommandStrategy::action_progression()
{
  std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
  if (action_result_) {
    return action_result_->succeeded ? 1.0 : -1.0;
  }
  if (action_last_progress_) {
    // action_last_progress_ is derived from the *desired* time_from_start in
    // the feedback, so it hits 1.0 when the trajectory is scheduled to end,
    // not when the robot has arrived. Never report completion from it: only
    // the result callback (action_result_) may claim the execution finished.
    return std::min(*action_last_progress_, 0.99);
  }
  return 0.0;
}

}  // namespace curobo_core

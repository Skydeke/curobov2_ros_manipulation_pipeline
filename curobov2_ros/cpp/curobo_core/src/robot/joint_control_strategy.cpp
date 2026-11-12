// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of the base half of curobov2_ros/robot/joint_control_strategy.py.
#include "curobo_core/robot/joint_control_strategy.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <trajectory_msgs/msg/joint_trajectory.hpp>

namespace curobo_core
{

namespace
{

// Split a whitespace/comma separated list, tolerating the Python list repr
// ("['a', 'b']") the descriptor bridge produces for YAML sequences.
std::vector<std::string> split_tokens(const std::string & text)
{
  std::vector<std::string> out;
  std::string current;
  for (const char c : text) {
    const bool separator =
      c == ' ' || c == ',' || c == '\t' || c == '\n' || c == '\r' ||
      c == '[' || c == ']' || c == '\'' || c == '"';
    if (separator) {
      if (!current.empty()) {
        out.push_back(current);
        current.clear();
      }
    } else {
      current.push_back(c);
    }
  }
  if (!current.empty()) {
    out.push_back(current);
  }
  return out;
}

// MoveIt simple_controller_manager behaviour: hand a controller ONLY the
// joints it owns, in the controller's own order.
trajectory_msgs::msg::JointTrajectory project_trajectory(
  const trajectory_msgs::msg::JointTrajectory & msg,
  const std::vector<std::string> & joints)
{
  trajectory_msgs::msg::JointTrajectory out;
  out.header = msg.header;
  std::vector<std::size_t> cols;
  for (const std::string & joint : joints) {
    const auto it = std::find(msg.joint_names.begin(), msg.joint_names.end(), joint);
    if (it == msg.joint_names.end()) {
      continue;
    }
    out.joint_names.push_back(joint);
    cols.push_back(static_cast<std::size_t>(it - msg.joint_names.begin()));
  }
  for (const auto & point : msg.points) {
    trajectory_msgs::msg::JointTrajectoryPoint np;
    np.time_from_start = point.time_from_start;
    for (const std::size_t col : cols) {
      np.positions.push_back(col < point.positions.size() ? point.positions[col] : 0.0);
      if (!point.velocities.empty()) {
        np.velocities.push_back(
          col < point.velocities.size() ? point.velocities[col] : 0.0);
      }
      if (!point.accelerations.empty()) {
        np.accelerations.push_back(
          col < point.accelerations.size() ? point.accelerations[col] : 0.0);
      }
    }
    out.points.push_back(std::move(np));
  }
  return out;
}

}  // namespace

JointCommandStrategy::JointCommandStrategy(
  rclcpp::Node * node, double dt, std::vector<std::string> joint_names,
  std::map<std::string, std::string> params)
: node_(node), dt_(dt), joint_names_(std::move(joint_names)), params_(std::move(params))
{
  auto it = params_.find("action_topic");
  if (it != params_.end()) {
    action_topic_ = it->second;
  }

  // MoveIt-style multi-controller dispatch. A robot whose cspace is split over
  // several controllers declares them here; the whole trajectory is then split
  // by joint and executed on each controller in parallel:
  //
  //   strategy_params:
  //     controller_names: "torso_controller arm_controller"
  //     torso_controller.action: /torso_controller/follow_joint_trajectory
  //     torso_controller.joints: "torso_lift_joint"
  //     arm_controller.action: /arm_controller/follow_joint_trajectory
  //     arm_controller.joints: "arm_1_joint arm_2_joint ... arm_7_joint"
  auto names_it = params_.find("controller_names");
  if (names_it != params_.end()) {
    for (const std::string & name : split_tokens(names_it->second)) {
      const auto action_it = params_.find(name + ".action");
      const auto joints_it = params_.find(name + ".joints");
      if (action_it == params_.end() || joints_it == params_.end()) {
        throw std::runtime_error(
          "strategy_params.controller_names lists '" + name + "' but '" + name +
          ".action' / '" + name + ".joints' are not both set");
      }
      ControllerRoute route;
      route.action = action_it->second;
      route.joints = split_tokens(joints_it->second);
      routes_.push_back(std::move(route));
    }
  }
  // Legacy single-controller path: one route owning every cspace joint.
  if (routes_.empty() && !action_topic_.empty()) {
    ControllerRoute route;
    route.action = action_topic_;
    route.joints = joint_names_;
    routes_.push_back(std::move(route));
  }
}

// PORT: JointCommandStrategy.set_command
int JointCommandStrategy::set_command(
  const std::vector<std::string> & joint_names,
  const std::vector<std::vector<double>> & vel_command,
  const std::vector<std::vector<double>> & accel_command,
  const std::vector<std::vector<double>> & position_command,
  double plan_dt, bool taper_terminal)
{
  std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
  position_command_ = position_command;
  vel_command_ = vel_command;
  accel_command_ = accel_command;
  joint_names_ = joint_names;
  taper_terminal_command_ = taper_terminal;
  // A non-positive value means "no override": the plan was sampled at the
  // strategy's own interpolation_dt. Anything else (including a stale override
  // from the previous replay) is replaced, so a plain joint command cannot
  // inherit the previous plan's faster timing.
  trajectory_dt_.store(
    (std::isfinite(plan_dt) && plan_dt > 0.0) ? plan_dt : 0.0,
    std::memory_order_relaxed);
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
    return plan_dt();
  }
  return plan_dt() / factor;
}

// Sample interval of the current command buffer: the replay's plan dt when the
// caller supplied one, otherwise this strategy's configured interpolation_dt.
double JointCommandStrategy::plan_dt() const
{
  const double override_dt = trajectory_dt_.load(std::memory_order_relaxed);
  return override_dt > 0.0 ? override_dt : dt_;
}

// PORT: time-dilation feed-forward scale.
// Re-timing stretches the stamped step to plan_dt / factor but leaves the path
// positions untouched. Velocity is dp/dt, so it must shrink/grow by the same
// factor; acceleration is d2p/dt2, so it scales by factor^2. Without this,
// slowing a trajectory leaves the controller a velocity feed-forward that is
// too fast for the timestamps it was given, which shows up as overshoot.
double JointCommandStrategy::time_dilation_scale() const
{
  const double stamp_dt = dilated_dt();
  if (plan_dt() <= 0.0 || stamp_dt <= 0.0) {
    return 1.0;
  }
  return plan_dt() / stamp_dt;
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
  // Unwrap continuous joints BEFORE storing: interpolation must see the
  // physical path, not a +pi -> -pi jump between two adjacent samples.
  const std::vector<double> tracked = track_measured(position);
  // Never mix joint-vector sizes in one buffer: interpolation indexes both
  // neighbours by the first one's length.
  if (!feedback_buffer_.empty() && feedback_buffer_.back().second.size() != tracked.size()) {
    return;
  }
  // Out-of-order stamps (clock jump, sim-time reset) break the sorted-search
  // below; restart the buffer instead of interpolating across the discontinuity.
  if (!feedback_buffer_.empty() && stamp_ns < feedback_buffer_.back().first) {
    feedback_buffer_.clear();
  }
  feedback_buffer_.emplace_back(stamp_ns, tracked);
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
  for (const auto & route : routes_) {
    if (route.result && !route.result->succeeded) {
      std::string msg = route.action + ": status=" +
        std::to_string(route.result->status) + " error_code=" +
        std::to_string(route.result->error_code) + " " + route.result->error_string;
      while (!msg.empty() && msg.back() == ' ') {
        msg.pop_back();
      }
      return msg;
    }
  }
  return "";
}

// PORT: JointCommandStrategy._send_as_action - the ONE execution path. With
// several routes the trajectory is split by joint and every controller is
// driven in parallel; the goal succeeds only if all of them succeed.
void JointCommandStrategy::send_as_action(
  const trajectory_msgs::msg::JointTrajectory & msg,
  const std::vector<std::vector<double>> & positions)
{
  // Build everything under the buffer lock, but NEVER touch the action client
  // (wait_for_action_server / async_send_goal) while holding it.
  //
  // rclcpp_action dispatches a client's feedback/result/goal-response callback
  // while holding the client's own goal-handle mutex, and those callbacks take
  // `buffer_lock_`. Calling async_send_goal under the buffer lock therefore
  // inverts the lock order:
  //   servo thread:      buffer_lock_ -> action-client mutex
  //   feedback callback: action-client mutex -> buffer_lock_
  // which deadlocks the servo lane against the controller's feedback stream.
  // The node then stops servicing every callback (the servo holds
  // RobotContext::strategy_lock_ while it is stuck), which is what "MPC went
  // bricked" looked like.
  struct PendingGoal
  {
    ControllerRoute * route;
    rclcpp_action::Client<FollowJT>::SharedPtr client;
    FollowJT::Goal goal;
    rclcpp_action::Client<FollowJT>::SendGoalOptions options;
  };
  std::vector<PendingGoal> pending;

  {
    std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
    for (auto & route : routes_) {
      route.result.reset();
      route.last_progress.reset();
      route.goal_handle.reset();
    }
    action_seq_ += 1;
    const int seq = action_seq_;

    const int n = static_cast<int>(msg.points.size());
    action_total_duration_s_ = dilated_dt() * std::max(n - 1, 0);

    if (routes_.empty()) {
      throw std::runtime_error(
        std::string("strategy_params must name either action_topic or "
        "controller_names + <controller>.action/<controller>.joints — the "
        "FollowJointTrajectory action is the only execution path"));
    }

    pending.reserve(routes_.size());
    for (auto & route : routes_) {
      if (!route.client) {
        route.client = rclcpp_action::create_client<FollowJT>(node_, route.action);
      }
      PendingGoal entry;
      entry.route = &route;
      entry.client = route.client;
      entry.goal.trajectory = project_trajectory(msg, route.joints);
      ControllerRoute * route_ptr = &route;
      entry.options.feedback_callback =
        [this, seq, route_ptr](
          rclcpp_action::ClientGoalHandle<FollowJT>::SharedPtr,
          const std::shared_ptr<const FollowJT::Feedback> feedback) {
          std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
          if (seq != action_seq_) {
            return;
          }
          const auto & t = feedback->desired.time_from_start;
          const double ts = static_cast<double>(t.sec) +
            static_cast<double>(t.nanosec) * 1e-9;
          if (action_total_duration_s_ > 0.0) {
            route_ptr->last_progress =
              std::min(1.0, std::max(0.0, ts / action_total_duration_s_));
          }
        };
      entry.options.result_callback =
        [this, seq, route_ptr](
          const rclcpp_action::ClientGoalHandle<FollowJT>::WrappedResult & wrapped) {
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
          route_ptr->result = result;
          route_ptr->last_progress = succeeded ? std::make_optional(1.0) : std::nullopt;
          // MoveIt cancels the remaining controllers when one aborts: a dead
          // peer must not let the others keep tracking the plan on their own.
          if (!succeeded) {
            for (auto & other : routes_) {
              if (&other == route_ptr || !other.goal_handle || !other.client) {
                continue;
              }
              try {
                other.client->async_cancel_goal(other.goal_handle);
              } catch (const std::exception &) {
                // best-effort
              }
            }
          }
        };
      entry.options.goal_response_callback =
        [this, seq, route_ptr](
          rclcpp_action::ClientGoalHandle<FollowJT>::SharedPtr goal_handle) {
          std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
          if (seq != action_seq_) {
            return;
          }
          if (!goal_handle) {
            route_ptr->result = ActionResult{
              false, 0, 0, "action goal rejected / not accepted"};
            return;
          }
          route_ptr->goal_handle = goal_handle;
        };
      pending.push_back(std::move(entry));
    }
  }

  for (auto & entry : pending) {
    if (!entry.client->wait_for_action_server(std::chrono::seconds(5))) {
      const std::string action = entry.route->action;
      std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
      entry.route->result = ActionResult{
        false, 0, 0,
        "FollowJointTrajectory action server " + action + " not available"};
      throw std::runtime_error(
        "FollowJointTrajectory action server " + action +
        " is not available; refusing to fall back to topic publish");
    }
  }
  for (auto & entry : pending) {
    entry.client->async_send_goal(entry.goal, entry.options);
  }

  mark_execution_start(static_cast<int>(msg.points.size()), positions);
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
  std::vector<rclcpp_action::ClientGoalHandle<FollowJT>::SharedPtr> handles;
  std::vector<rclcpp_action::Client<FollowJT>::SharedPtr> clients;
  {
    std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
    for (auto & route : routes_) {
      if (route.goal_handle && route.client) {
        handles.push_back(route.goal_handle);
        clients.push_back(route.client);
      }
      route.goal_handle.reset();
      route.result = ActionResult{false, 0, 0, "canceled"};
    }
    action_seq_ += 1;
  }
  for (std::size_t i = 0; i < handles.size(); ++i) {
    try {
      clients[i]->async_cancel_goal(handles[i]);
    } catch (const std::exception &) {
      // best-effort: cancel must not throw out of a stop path
    }
  }
}

// PORT: JointCommandStrategy._get_progression - the action is the ONLY source.
// With several routes the run is complete only when ALL have succeeded; while
// running, report the slowest controller.
double JointCommandStrategy::action_progression()
{
  std::lock_guard<std::recursive_mutex> lk(buffer_lock_);
  if (routes_.empty()) {
    return 0.0;
  }
  bool all_done = true;
  bool any_running = false;
  double min_progress = 1.0;
  for (const auto & route : routes_) {
    if (route.result) {
      if (!route.result->succeeded) {
        return -1.0;
      }
    } else {
      all_done = false;
      any_running = true;
      const double p = route.last_progress ?
        std::min(*route.last_progress, 0.99) : 0.0;
      min_progress = std::min(min_progress, p);
    }
  }
  if (all_done) {
    return 1.0;
  }
  return any_running ? min_progress : 0.0;
}

}  // namespace curobo_core

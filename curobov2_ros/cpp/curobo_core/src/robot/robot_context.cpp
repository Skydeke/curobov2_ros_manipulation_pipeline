// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/robot/robot_context.py.
#include "curobo_core/robot/robot_context.hpp"

#include <algorithm>
#include <optional>

#include <std_srvs/srv/trigger.hpp>

#include "curobo_core/robot/robot_registry.hpp"

namespace curobo_core
{

RobotContext::RobotContext(
  rclcpp::Node * node, double dt, std::vector<std::string> joint_names,
  std::string strategy_key, std::map<std::string, std::string> strategy_params,
  std::vector<double> default_home)
: node_(node), dt_(dt), joint_names_(std::move(joint_names)),
  strategy_params_(std::move(strategy_params)), default_home_(std::move(default_home))
{
  select_strategy(strategy_key);
  // PORT: self.ghost_strategy = GhostStrategy(node, self.dt, self.description)
  ghost_strategy_ = std::make_unique<GhostStrategy>(
    node_, dt_, joint_names_, strategy_params_);
  // PORT: "Control strategy initialized: {name} (robot: {description.name})"
  RCLCPP_INFO(
    node_->get_logger(), "Control strategy initialized: %s",
    current_strategy_name_.c_str());

  const std::string name = node_->get_name();
  set_strategy_srv_ = node_->create_service<curobov2_ros_interfaces::srv::SetRobotStrategy>(
    name + "/set_robot_strategy",
    std::bind(
      &RobotContext::handle_set_robot_strategy, this,
      std::placeholders::_1, std::placeholders::_2));
  get_strategy_srv_ = node_->create_service<std_srvs::srv::Trigger>(
    name + "/get_robot_strategy",
    std::bind(
      &RobotContext::handle_get_robot_strategy, this,
      std::placeholders::_1, std::placeholders::_2));
  get_strategies_srv_ =
    node_->create_service<curobov2_ros_interfaces::srv::GetRobotStrategies>(
    name + "/get_robot_strategies",
    std::bind(
      &RobotContext::handle_get_robot_strategies, this,
      std::placeholders::_1, std::placeholders::_2));
}

// PORT: RobotContext.select_strategy
void RobotContext::select_strategy(const std::string & key)
{
  std::lock_guard<std::mutex> lk(strategy_lock_);
  auto strategy = RobotRegistry::create(
    key, node_, dt_, joint_names_, strategy_params_, default_home_);
  if (strategy == nullptr) {
    RCLCPP_WARN(
      node_->get_logger(), "Unknown control strategy: '%s'", key.c_str());
    return;
  }
  robot_strategy_ = std::move(strategy);
  current_strategy_name_ = key;
  ghost_strategy_ = std::make_unique<GhostStrategy>(
    node_, dt_, joint_names_, strategy_params_);
}

// PORT: RobotContext.set_command
int RobotContext::set_command(
  const std::vector<std::string> & joint_names,
  const std::vector<std::vector<double>> & vel_command,
  const std::vector<std::vector<double>> & accel_command,
  const std::vector<std::vector<double>> & position_command)
{
  std::lock_guard<std::mutex> lk(strategy_lock_);
  // PORT: RobotContext.set_command(joint_names=None) resolves the names from the
  // active strategy INSIDE the critical section (a caller must not read
  // get_joint_name() first - that would race a strategy switch). An empty list
  // is the C++ spelling of None.
  const std::vector<std::string> names = resolve_command_names(joint_names);
  // PORT: the ghost preview is fed first, then the real strategy (its epoch is
  // what the caller gets).
  if (ghost_strategy_ != nullptr) {
    ghost_strategy_->set_command(names, vel_command, accel_command, position_command);
    ghost_strategy_->send_trajectrory();
  }
  if (robot_strategy_ == nullptr) {
    return -1;
  }
  return robot_strategy_->set_command(
    names, vel_command, accel_command, position_command);
}

// PORT: RobotContext.set_and_send_command
bool RobotContext::set_and_send_command(
  const std::vector<std::string> & joint_names,
  const std::vector<std::vector<double>> & vel_command,
  const std::vector<std::vector<double>> & accel_command,
  const std::vector<std::vector<double>> & position_command)
{
  std::lock_guard<std::mutex> lk(strategy_lock_);
  // PORT: joint_names=None resolves inside this critical section.
  const std::vector<std::string> names = resolve_command_names(joint_names);
  // PORT: the ghost preview always gets the command (and publishes it).
  if (ghost_strategy_ != nullptr) {
    ghost_strategy_->set_command(names, vel_command, accel_command, position_command);
    ghost_strategy_->send_trajectrory();
  }
  if (robot_strategy_ == nullptr) {
    return false;
  }
  robot_strategy_->set_command(names, vel_command, accel_command, position_command);
  robot_strategy_->send_trajectrory();
  return true;
}

// PORT: RobotContext's `joint_names=None` resolution (caller holds strategy_lock_).
std::vector<std::string> RobotContext::resolve_command_names(
  const std::vector<std::string> & joint_names) const
{
  if (!joint_names.empty()) {
    return joint_names;
  }
  if (robot_strategy_ != nullptr) {
    const auto names = const_cast<JointCommandStrategy *>(robot_strategy_.get())->get_joint_name();
    if (!names.empty()) {
      return names;
    }
  }
  return joint_names_;
}

// PORT: RobotContext.get_joint_pose
std::vector<double> RobotContext::get_joint_pose()
{
  std::lock_guard<std::mutex> lk(strategy_lock_);
  if (robot_strategy_ == nullptr) {
    return std::vector<double>(joint_names_.size(), 0.0);
  }
  return robot_strategy_->get_joint_pose();
}

// PORT: RobotContext.get_joint_pose_at
std::optional<std::vector<double>> RobotContext::get_joint_pose_at(std::int64_t stamp_ns)
{
  std::lock_guard<std::mutex> lk(strategy_lock_);
  if (robot_strategy_ == nullptr) {
    return std::nullopt;
  }
  return robot_strategy_->get_joint_pose_at(stamp_ns);
}

// PORT: RobotContext.get_joint_velocity
std::vector<double> RobotContext::get_joint_velocity()
{
  std::lock_guard<std::mutex> lk(strategy_lock_);
  return std::vector<double>(joint_names_.size(), 0.0);
}

// PORT: RobotContext.get_joint_name
std::vector<std::string> RobotContext::get_joint_name()
{
  std::lock_guard<std::mutex> lk(strategy_lock_);
  if (robot_strategy_ == nullptr) {
    return joint_names_;
  }
  return robot_strategy_->get_joint_name();
}

// PORT: RobotContext.stop_robot
void RobotContext::stop_robot()
{
  std::lock_guard<std::mutex> lk(strategy_lock_);
  if (robot_strategy_ != nullptr) {
    robot_strategy_->stop_robot();
  }
}

// PORT: RobotContext.get_progression
double RobotContext::get_progression()
{
  std::lock_guard<std::mutex> lk(strategy_lock_);
  if (robot_strategy_ == nullptr) {
    return 0.0;
  }
  return robot_strategy_->get_progression();
}

// PORT: RobotContext.send_trajectrory
bool RobotContext::send_trajectrory(std::optional<int> expect_epoch)
{
  std::lock_guard<std::mutex> lk(strategy_lock_);
  if (robot_strategy_ == nullptr) {
    return false;
  }
  if (expect_epoch.has_value() && robot_strategy_->buffer_epoch() != *expect_epoch) {
    RCLCPP_ERROR(
      node_->get_logger(),
      "send_trajectrory: buffer epoch mismatch (expected %d, current %d) - a newer "
      "set_command() superseded this one; refusing to execute a mismatched trajectory.",
      *expect_epoch, robot_strategy_->buffer_epoch());
    return false;
  }
  robot_strategy_->send_trajectrory();
  return true;
}

// PORT: RobotContext.set_robot_strategy_callback
void RobotContext::handle_set_robot_strategy(
  const std::shared_ptr<curobov2_ros_interfaces::srv::SetRobotStrategy::Request> request,
  std::shared_ptr<curobov2_ros_interfaces::srv::SetRobotStrategy::Response> response)
{
  const std::string & new_strategy_name = request->robot_strategy;
  const auto available = RobotRegistry::available_strategies();
  const bool known = std::find(available.begin(), available.end(), new_strategy_name) !=
    available.end();
  if (!known) {
    response->success = false;
    std::string list;
    for (const auto & key : available) {
      list += (list.empty() ? "" : ", ") + key;
    }
    response->message =
      "Unknown strategy: '" + new_strategy_name + "'. Available: " + list;
    return;
  }
  response->previous_robot_strategy = current_strategy_name_;
  if (new_strategy_name == current_strategy_name_) {
    response->success = true;
    response->message = "Already using strategy: " + new_strategy_name;
    response->current_robot_strategy = current_strategy_name_;
    return;
  }
  select_strategy(new_strategy_name);
  response->success = true;
  response->current_robot_strategy = current_strategy_name_;
  response->message = "Strategy switched from '" + response->previous_robot_strategy +
    "' to '" + current_strategy_name_ + "'";
}

// PORT: RobotContext.get_robot_strategy_callback
void RobotContext::handle_get_robot_strategy(
  const std::shared_ptr<std_srvs::srv::Trigger::Request>,
  std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
  response->success = true;
  response->message = current_strategy_name_;
}

// PORT: RobotContext.get_robot_strategies_callback
void RobotContext::handle_get_robot_strategies(
  const std::shared_ptr<curobov2_ros_interfaces::srv::GetRobotStrategies::Request>,
  std::shared_ptr<curobov2_ros_interfaces::srv::GetRobotStrategies::Response> response)
{
  response->strategy_names = RobotRegistry::available_strategies();
  response->current_strategy_name = current_strategy_name_;
  response->success = true;
}

}  // namespace curobo_core

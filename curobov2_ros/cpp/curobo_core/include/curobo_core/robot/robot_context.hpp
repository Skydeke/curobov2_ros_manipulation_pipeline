// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/robot/robot_context.py
//
// Selects/switches the joint-command control strategy at runtime and exposes the
// command/feedback API. Lock order: gpu_lock > strategy_lock > buffer_lock.
#ifndef CUROBO_CORE__ROBOT__ROBOT_CONTEXT_HPP_
#define CUROBO_CORE__ROBOT__ROBOT_CONTEXT_HPP_

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_srvs/srv/trigger.hpp>

#include "curobov2_ros_interfaces/srv/get_robot_strategies.hpp"
#include "curobov2_ros_interfaces/srv/set_robot_strategy.hpp"
#include "curobo_core/robot/ghost_strategy.hpp"
#include "curobo_core/robot/joint_control_strategy.hpp"

namespace curobo_core
{

class RobotContext
{
public:
  RobotContext(
    rclcpp::Node * node, double dt, std::vector<std::string> joint_names,
    std::string strategy_key = "emulator",
    std::map<std::string, std::string> strategy_params = {},
    std::vector<double> default_home = {});

  // PORT: RobotContext.bind_kinematics
  void bind_kinematics() {}

  // PORT: RobotContext.set_command -> buffer epoch
  int set_command(
    const std::vector<std::string> & joint_names,
    const std::vector<std::vector<double>> & vel_command,
    const std::vector<std::vector<double>> & accel_command,
    const std::vector<std::vector<double>> & position_command,
    bool feed_ghost = true);
  // Preview-only update: drive the ghost strategy without touching the active
  // control strategy. Used by the replay path, which projects the stored
  // full-joint plan to the controller's arm joints but must keep the finger
  // columns in the RViz ghost so close/attach/lift shows the commanded jaws.
  void set_ghost_command(
    const std::vector<std::string> & joint_names,
    const std::vector<std::vector<double>> & vel_command,
    const std::vector<std::vector<double>> & accel_command,
    const std::vector<std::vector<double>> & position_command);
  // PORT: RobotContext.set_and_send_command
  bool set_and_send_command(
    const std::vector<std::string> & joint_names,
    const std::vector<std::vector<double>> & vel_command,
    const std::vector<std::vector<double>> & accel_command,
    const std::vector<std::vector<double>> & position_command);

  // PORT: RobotContext.get_joint_pose / get_joint_pose_at / get_joint_velocity /
  // get_joint_name / stop_robot / get_progression / get_robot_state
  std::vector<double> get_joint_pose();
  std::optional<std::vector<double>> get_joint_pose_at(std::int64_t stamp_ns);
  std::vector<double> get_joint_velocity();
  std::vector<std::string> get_joint_name();
  void stop_robot();
  double get_progression();
  RobotState get_robot_state() const {return RobotState::kIdle;}
  // PORT: action_failure_summary passthrough (the FollowJointTrajectory Result).
  std::string action_failure_summary();

  // PORT: RobotContext.send_trajectrory (note the source spelling)
  bool send_trajectrory(std::optional<int> expect_epoch = std::nullopt);

  const std::string & current_strategy_name() const {return current_strategy_name_;}

  // PORT: forwards the raw joint-state message from the active control strategy
  // (see JointCommandStrategy::set_joint_state_callback). Re-applied on every
  // strategy switch. Node-owned: set once, before spinning.
  void set_joint_state_callback(
    std::function<void(const sensor_msgs::msg::JointState &)> callback);

private:
  // PORT: set_robot_strategy_callback / get_robot_strategy_callback
  void handle_set_robot_strategy(
    const std::shared_ptr<curobov2_ros_interfaces::srv::SetRobotStrategy::Request> request,
    std::shared_ptr<curobov2_ros_interfaces::srv::SetRobotStrategy::Response> response);
  void handle_get_robot_strategy(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response);
  void handle_get_robot_strategies(
    const std::shared_ptr<curobov2_ros_interfaces::srv::GetRobotStrategies::Request> request,
    std::shared_ptr<curobov2_ros_interfaces::srv::GetRobotStrategies::Response> response);

  void select_strategy(const std::string & key);
  // PORT: the `joint_names=None` resolution shared by set_command /
  // set_and_send_command (caller holds strategy_lock_).
  std::vector<std::string> resolve_command_names(
    const std::vector<std::string> & joint_names) const;

  rclcpp::Node * node_;
  double dt_;
  std::vector<std::string> joint_names_;
  std::map<std::string, std::string> strategy_params_;
  std::vector<double> default_home_;

  std::mutex strategy_lock_;
  std::unique_ptr<JointCommandStrategy> robot_strategy_;
  std::function<void(const sensor_msgs::msg::JointState &)> joint_state_callback_;
  // PORT: RobotContext.ghost_strategy - always present, independent of the
  // active control strategy; publishes /<node>/trajectory for the RViz display.
  std::unique_ptr<JointCommandStrategy> ghost_strategy_;
  std::string current_strategy_name_;

  rclcpp::Service<curobov2_ros_interfaces::srv::SetRobotStrategy>::SharedPtr set_strategy_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr get_strategy_srv_;
  rclcpp::Service<curobov2_ros_interfaces::srv::GetRobotStrategies>::SharedPtr
    get_strategies_srv_;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__ROBOT__ROBOT_CONTEXT_HPP_

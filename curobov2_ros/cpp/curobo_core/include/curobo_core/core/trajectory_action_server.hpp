// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/core/unified_planner_node.py
//   execute_callback / _execute_goal / _execute_replay / _replay_trajectory_from_msg
//   goal_callback / cancel_callback
//
// The SendTrajectory action server. The execute handler runs on its OWN thread
// (design section 5 / UD-5): the executor thread only starts it.
#ifndef CUROBO_CORE__CORE__TRAJECTORY_ACTION_SERVER_HPP_
#define CUROBO_CORE__CORE__TRAJECTORY_ACTION_SERVER_HPP_

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include "curobov2_ros_interfaces/action/send_trajectory.hpp"
#include "curobo_core/core/planner_services.hpp"
#include "curobo_core/robot/robot_context.hpp"

namespace curobo_core
{

class TrajectoryActionServer
{
public:
  using SendTrajectory = curobov2_ros_interfaces::action::SendTrajectory;
  using GoalHandle = rclcpp_action::ServerGoalHandle<SendTrajectory>;

  TrajectoryActionServer(
    rclcpp::Node * node, PlannerServices * planners, RobotContext * robot_context,
    std::vector<std::string> joint_names);
  ~TrajectoryActionServer();

private:
  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID & uuid,
    std::shared_ptr<const SendTrajectory::Goal> goal);
  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<GoalHandle> goal_handle);
  void handle_accepted(const std::shared_ptr<GoalHandle> goal_handle);

  // PORT: execute_callback / _execute_goal / _execute_replay
  void execute(const std::shared_ptr<GoalHandle> goal_handle);
  void handle_goal_impl(const std::shared_ptr<GoalHandle> goal_handle);

  // PORT: unified_planner_node._publish_feedback
  void publish_feedback(
    const std::shared_ptr<GoalHandle> goal_handle, const std::string & state,
    double step_progression, bool on_target);

  rclcpp::Node * node_;
  PlannerServices * planners_;
  RobotContext * robot_context_;
  std::vector<std::string> joint_names_;

  rclcpp_action::Server<SendTrajectory>::SharedPtr action_server_;
  rclcpp::CallbackGroup::SharedPtr callback_group_;

  std::mutex goal_lock_;
  bool goal_active_ = false;

  std::mutex workers_lock_;
  std::vector<std::thread> workers_;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__CORE__TRAJECTORY_ACTION_SERVER_HPP_

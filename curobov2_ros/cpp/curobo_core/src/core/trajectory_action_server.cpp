// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of the SendTrajectory action surface.
#include "curobo_core/core/trajectory_action_server.hpp"

#include "curobo_core/planners/reactive_controller.hpp"

#include <algorithm>
#include <chrono>

#include <sensor_msgs/msg/joint_state.hpp>

namespace curobo_core
{

TrajectoryActionServer::TrajectoryActionServer(
  rclcpp::Node * node, PlannerServices * planners, RobotContext * robot_context,
  std::vector<std::string> joint_names)
: node_(node), planners_(planners), robot_context_(robot_context),
  joint_names_(std::move(joint_names))
{
  // PORT: ActionServer(..., callback_group=ReentrantCallbackGroup())
  callback_group_ = node_->create_callback_group(
    rclcpp::CallbackGroupType::Reentrant);
  action_server_ = rclcpp_action::create_server<SendTrajectory>(
    node_,
    std::string(node_->get_name()) + "/execute_trajectory",
    std::bind(
      &TrajectoryActionServer::handle_goal, this, std::placeholders::_1,
      std::placeholders::_2),
    std::bind(&TrajectoryActionServer::handle_cancel, this, std::placeholders::_1),
    std::bind(&TrajectoryActionServer::handle_accepted, this, std::placeholders::_1),
    rcl_action_server_get_default_options(),
    callback_group_);
}

TrajectoryActionServer::~TrajectoryActionServer()
{
  std::lock_guard<std::mutex> lk(workers_lock_);
  for (auto & worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

// PORT: unified_planner_node.goal_callback
rclcpp_action::GoalResponse TrajectoryActionServer::handle_goal(
  const rclcpp_action::GoalUUID &, std::shared_ptr<const SendTrajectory::Goal>)
{
  std::lock_guard<std::mutex> lk(goal_lock_);
  if (goal_active_) {
    RCLCPP_ERROR(
      node_->get_logger(),
      "Rejecting execution goal: another goal is already active - cancel it first.");
    return rclcpp_action::GoalResponse::REJECT;
  }
  goal_active_ = true;
  RCLCPP_INFO(node_->get_logger(), "Received execution goal");
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

// PORT: unified_planner_node.cancel_callback
rclcpp_action::CancelResponse TrajectoryActionServer::handle_cancel(
  const std::shared_ptr<GoalHandle>)
{
  RCLCPP_INFO(node_->get_logger(), "Goal cancelled");
  if (robot_context_ != nullptr) {
    robot_context_->stop_robot();
  }
  return rclcpp_action::CancelResponse::ACCEPT;
}

// The execute handler runs on its OWN thread (design section 5 / UD-5).
void TrajectoryActionServer::handle_accepted(const std::shared_ptr<GoalHandle> goal_handle)
{
  std::lock_guard<std::mutex> lk(workers_lock_);
  workers_.emplace_back(&TrajectoryActionServer::execute, this, goal_handle);
}

// PORT: unified_planner_node._publish_feedback
void TrajectoryActionServer::publish_feedback(
  const std::shared_ptr<GoalHandle> goal_handle, const std::string & state,
  double step_progression, bool on_target)
{
  auto feedback = std::make_shared<SendTrajectory::Feedback>();
  feedback->state = state;
  feedback->step_progression = static_cast<float>(step_progression);
  feedback->on_target = on_target;
  goal_handle->publish_feedback(feedback);
}

// PORT: execute_callback
void TrajectoryActionServer::execute(const std::shared_ptr<GoalHandle> goal_handle)
{
  try {
    handle_goal_impl(goal_handle);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(node_->get_logger(), "Execution error: %s", e.what());
    auto result = std::make_shared<SendTrajectory::Result>();
    result->result.success = false;
    result->result.message = std::string("Error: ") + e.what();
    goal_handle->abort(result);
  }
  std::lock_guard<std::mutex> lk(goal_lock_);
  goal_active_ = false;
}

// PORT: _execute_goal + _execute_replay
void TrajectoryActionServer::handle_goal_impl(const std::shared_ptr<GoalHandle> goal_handle)
{
  auto result = std::make_shared<SendTrajectory::Result>();
  const auto & request = goal_handle->get_goal();

  TrajectoryPlanner * planner = planners_ ? planners_->manager().current() : nullptr;
  if (planner == nullptr) {
    result->result.success = false;
    result->result.message = "No planner selected";
    goal_handle->abort(result);
    return;
  }

  // PORT: replay needs no goalset; it runs before goal validation.
  const bool has_replay = request->replay && !request->replay_trajectory.points.empty();
  if (!has_replay && planner->is_open_loop()) {
    // PORT: an open-loop execute must be GIVEN the trajectory; re-planning is
    // not supported.
    result->result.success = false;
    result->result.message = planner->get_planner_name() +
      ": open-loop execution needs a supplied trajectory (replay_trajectory); "
      "re-planning during execute is not supported";
    RCLCPP_ERROR(node_->get_logger(), "%s", result->result.message.c_str());
    goal_handle->abort(result);
    return;
  }

  if (has_replay) {
    // PORT: _execute_replay - stream the caller-supplied trajectory verbatim.
    if (!planner->is_open_loop()) {
      result->result.success = false;
      result->result.message = planner->get_planner_name() +
        " cannot replay a supplied trajectory (only open-loop planners stream waypoints)";
      goal_handle->abort(result);
      return;
    }
    const auto & msg = request->replay_trajectory;
    const std::size_t n = msg.points.size();
    std::vector<std::vector<double>> positions;
    std::vector<std::vector<double>> velocities;
    for (const auto & point : msg.points) {
      positions.push_back(point.positions);
      velocities.push_back(point.velocities.empty() ?
        std::vector<double>(point.positions.size(), 0.0) : point.velocities);
    }
    std::vector<std::string> names = msg.joint_names;
    if (names.empty()) {
      names = joint_names_;
    }
    RCLCPP_INFO(
      node_->get_logger(), "Executing with %s (replayed %zu waypoints, no re-plan)",
      planner->get_planner_name().c_str(), n);
    // PORT: single_planner "<planner>: Trajectory execution started".
    RCLCPP_INFO(
      node_->get_logger(), "%s: Trajectory execution started",
      planner->get_planner_name().c_str());

    // PORT: single_planner "Trajectory sent to robot context for visualization".
    RCLCPP_INFO(node_->get_logger(), "Trajectory sent to robot context for visualization");
    const int epoch = robot_context_->set_command(names, velocities, {}, positions);
    if (!robot_context_->send_trajectrory(epoch)) {
      result->result.success = false;
      result->result.message = "Execution failed";
      goal_handle->abort(result);
      return;
    }

    // Feedback timeline until the trajectory plays out (or is cancelled).
    while (rclcpp::ok()) {
      if (goal_handle->is_canceling()) {
        result->result.success = false;
        result->result.message = "Execution canceled";
        goal_handle->canceled(result);
        return;
      }
      const double progression = robot_context_->get_progression();
      publish_feedback(goal_handle, "EXECUTING", progression, false);
      if (progression >= 1.0) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    result->result.success = true;
    result->result.message = "Execution completed";
    goal_handle->succeed(result);
    return;
  }

  // Reactive (closed-loop) execution: run the controller's servo loop. The
  // loop ends only on cancel (or the safety cap); reaching the target is a
  // signal (on_target), never a stop condition.
  auto * reactive = dynamic_cast<ReactiveController *>(planner);
  if (reactive != nullptr) {
    ReactiveGoal goal;
    for (const auto & goalset : request->goal.goalsets) {
      std::vector<std::array<double, 7>> poses;
      for (const auto & pose : goalset.poses) {
        poses.push_back(
          {pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.w, pose.orientation.x, pose.orientation.y,
            pose.orientation.z});
      }
      goal.goalsets.push_back(std::move(poses));
    }
    goal.start_pose_names = request->goal.start_pose.name;
    goal.start_pose_positions = request->goal.start_pose.position;

    const double convergence = node_->has_parameter("convergence_threshold") ?
      node_->get_parameter("convergence_threshold").as_double() : 0.01;
    const int max_iterations = node_->has_parameter("max_mpc_iterations") ?
      node_->get_parameter("max_mpc_iterations").as_int() : 1000;

    const PlannerResult plan_result =
      reactive->plan_reactive(goal, convergence, max_iterations, robot_context_);
    if (!plan_result.success) {
      result->result.success = false;
      result->result.message = plan_result.message;
      goal_handle->abort(result);
      return;
    }

    ReactiveExecutionContext ctx;
    ctx.robot_context = robot_context_;
    ctx.cancel_requested = [goal_handle]() {return goal_handle->is_canceling();};
    ctx.feedback = [this, goal_handle](
      const std::string & state, double progression, bool on_target,
      const std::vector<double> & command, double position_error) {
        auto feedback = std::make_shared<SendTrajectory::Feedback>();
        feedback->state = state;
        feedback->step_progression = static_cast<float>(progression);
        feedback->on_target = on_target;
        feedback->position_error = position_error;
        feedback->joint_command.position = command;
        feedback->joint_command.name = joint_names_;
        goal_handle->publish_feedback(feedback);
      };

    const bool ok = reactive->execute(ctx);
    if (goal_handle->is_canceling()) {
      result->result.success = false;
      result->result.message = "Execution canceled";
      goal_handle->canceled(result);
      return;
    }
    if (!ok) {
      result->result.success = false;
      result->result.message = "Execution failed";
      goal_handle->abort(result);
      return;
    }
    result->result.success = true;
    result->result.message = "Execution completed";
    goal_handle->succeed(result);
    return;
  }

  // Reactive execution is a later milestone; refuse clearly rather than fake it.
  result->result.success = false;
  result->result.message = "No planner selected";
  goal_handle->abort(result);
}

}  // namespace curobo_core

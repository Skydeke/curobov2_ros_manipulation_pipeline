// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of the SendTrajectory action surface.
#include "curobo_core/core/trajectory_action_server.hpp"

#include "curobo_core/planners/reactive_controller.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>

#include <sensor_msgs/msg/joint_state.hpp>

namespace curobo_core
{

namespace
{

constexpr double kTwoPi = 6.28318530717958647692;
//: MoveIt's waitForRobotToStop() treats the robot as stopped after this many
//: consecutive state samples show no motion (trajectory_execution_manager.cpp).
constexpr int kNoMotionSamples = 3;
constexpr auto kStatePollInterval = std::chrono::milliseconds(20);

double shortest_angle(double value, double target)
{
  return std::remainder(value - target, kTwoPi);
}

//: Map the executed joint names onto the strategy's feedback vector. Empty
//: when the feedback names do not cover the executed joints; callers then
//: skip the check rather than guessing an order.
std::vector<std::size_t> feedback_columns(
  RobotContext * robot_context, const std::vector<std::string> & command_names)
{
  const std::vector<std::string> feedback_names = robot_context->get_joint_name();
  std::vector<std::size_t> columns;
  columns.reserve(command_names.size());
  for (const auto & name : command_names) {
    const auto it = std::find(feedback_names.begin(), feedback_names.end(), name);
    if (it == feedback_names.end()) {
      return {};
    }
    columns.push_back(static_cast<std::size_t>(it - feedback_names.begin()));
  }
  return columns;
}

// MoveIt TrajectoryExecutionManager::validate() equivalent: refuse a replay
// whose first waypoint is not where the robot currently is. This is what
// prevents the next segment from starting while the previous one is still
// moving (the controller on its own does not enforce a start tolerance here).
bool validate_start_state(
  rclcpp::Node * node, RobotContext * robot_context,
  const std::vector<std::string> & command_names,
  const std::vector<double> & start_positions, double tolerance)
{
  if (tolerance <= 0.0 || start_positions.empty()) {
    return true;
  }
  const std::vector<std::size_t> columns =
    feedback_columns(robot_context, command_names);
  if (columns.empty()) {
    RCLCPP_WARN(
      node->get_logger(),
      "trajectory start check skipped: feedback does not name every executed joint");
    return true;
  }
  const std::vector<double> current = robot_context->get_joint_pose();
  double worst = 0.0;
  std::size_t worst_index = 0;
  for (std::size_t i = 0; i < columns.size() && i < start_positions.size(); ++i) {
    const std::size_t column = columns[i];
    if (column >= current.size()) {
      continue;
    }
    const double error = std::abs(shortest_angle(current[column], start_positions[i]));
    if (error > worst) {
      worst = error;
      worst_index = i;
    }
  }
  if (worst > tolerance) {
    const std::string joint =
      worst_index < command_names.size() ? command_names[worst_index] : "?";
    RCLCPP_ERROR(
      node->get_logger(),
      "Invalid trajectory: start point deviates from current robot state by "
      "%.4f rad at joint '%s' (allowed %.4f); refusing to execute",
      worst, joint.c_str(), tolerance);
    return false;
  }
  return true;
}

// MoveIt TrajectoryExecutionManager::waitForRobotToStop() equivalent: after
// the controller reports the path complete, wait until the MEASURED joints
// show no motion for kNoMotionSamples consecutive samples (or the timeout
// expires). MoveIt then relies on the next validate() to reject a segment
// whose start does not match the settled state.
void wait_for_robot_to_stop(
  rclcpp::Node * node, RobotContext * robot_context,
  const std::vector<std::string> & command_names,
  double motion_tolerance, double timeout_s)
{
  if (timeout_s <= 0.0 || motion_tolerance <= 0.0) {
    return;
  }
  const std::vector<std::size_t> columns =
    feedback_columns(robot_context, command_names);
  if (columns.empty()) {
    return;
  }
  std::vector<double> previous = robot_context->get_joint_pose();
  int no_motion = 0;
  const auto deadline = std::chrono::steady_clock::now() +
    std::chrono::duration<double>(timeout_s);
  while (no_motion < kNoMotionSamples &&
    std::chrono::steady_clock::now() < deadline)
  {
    std::this_thread::sleep_for(kStatePollInterval);
    const std::vector<double> current = robot_context->get_joint_pose();
    bool moved = false;
    for (const std::size_t column : columns) {
      if (column < current.size() && column < previous.size() &&
        std::abs(shortest_angle(current[column], previous[column])) >
        motion_tolerance)
      {
        moved = true;
        break;
      }
    }
    no_motion = moved ? 0 : no_motion + 1;
    previous = current;
  }
  if (no_motion < kNoMotionSamples) {
    RCLCPP_WARN(
      node->get_logger(),
      "robot still moving %.2f s after the controller reported completion; "
      "the next trajectory's start check will reject it if the pose is too far off",
      timeout_s);
  }
}

}  // namespace

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

// PORT: single_planner._active_joint_projection
void TrajectoryActionServer::project_active_joints(
  std::vector<std::string> & names,
  std::vector<std::vector<double>> & positions,
  std::vector<std::vector<double>> & velocities) const
{
  if (names.empty() || positions.empty() || positions.front().empty()) {
    return;
  }
  const std::size_t dof = positions.front().size();
  const std::size_t named = names.size();

  auto select = [&positions, &velocities](
    const std::vector<std::size_t> & cols, std::size_t new_named,
    std::vector<std::string> & out_names) {
      auto pick = [&cols](const std::vector<std::vector<double>> & rows) {
          std::vector<std::vector<double>> out;
          out.reserve(rows.size());
          for (const auto & row : rows) {
            std::vector<double> projected;
            projected.reserve(cols.size());
            for (const std::size_t col : cols) {
              projected.push_back(col < row.size() ? row[col] : 0.0);
            }
            out.push_back(std::move(projected));
          }
          return out;
        };
      positions = pick(positions);
      if (!velocities.empty()) {
        velocities = pick(velocities);
      }
      out_names.resize(new_named);
    };

  if (named < dof) {
    // Rows wider than the name list: cuRobo appended locked/fixed joint columns
    // (e.g. finger_joint) via get_full_dof_from_solution(), with the named
    // joints leading in order - a plain prefix keeps the arm joints and drops
    // the appended tail.
    std::vector<std::size_t> cols(named);
    for (std::size_t i = 0; i < named; ++i) {
      cols[i] = i;
    }
    select(cols, named, names);
    return;
  }

  // Rows are no wider than the name list: keep the node's ACTIVE joints by
  // NAME (the controller owns exactly those), dropping any locked columns.
  if (named == joint_names_.size()) {
    return;  // already active-sized and needs no projection
  }
  std::vector<std::size_t> cols;
  cols.reserve(joint_names_.size());
  for (const std::string & want : joint_names_) {
    auto it = std::find(names.begin(), names.end(), want);
    if (it == names.end()) {
      return;  // cannot map by name - leave the trajectory untouched
    }
    cols.push_back(static_cast<std::size_t>(it - names.begin()));
  }
  select(cols, cols.size(), names);
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

ReactiveGoal make_reactive_goal(
  const curobov2_ros_interfaces::action::SendTrajectory::Goal & request)
{
  ReactiveGoal goal;
  for (const auto & goalset : request.goal.goalsets) {
    std::vector<std::array<double, 7>> poses;
    for (const auto & pose : goalset.poses) {
      poses.push_back(
        {pose.position.x, pose.position.y, pose.position.z,
          pose.orientation.w, pose.orientation.x, pose.orientation.y,
          pose.orientation.z});
    }
    goal.goalsets.push_back(std::move(poses));
  }
  // Multi-tool goals are carried by tool_frames + tool_poses (the bimanual
  // OpenArm path). Legacy clients only fill `poses`.
  if (!request.goal.goalsets.empty()) {
    const auto & first = request.goal.goalsets.front();
    goal.tool_frames.assign(first.tool_frames.begin(), first.tool_frames.end());
    for (const auto & pose : first.tool_poses) {
      goal.tool_poses.push_back(
        {pose.position.x, pose.position.y, pose.position.z,
          pose.orientation.w, pose.orientation.x, pose.orientation.y,
          pose.orientation.z});
    }
  }
  goal.start_pose_names = request.goal.start_pose.name;
  goal.start_pose_positions = request.goal.start_pose.position;
  return goal;
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
    // Keep the full-joint replay for the RViz ghost: the task's stored plan
    // carries the commanded jaw value (open/closed) at this point in the
    // chain. project_active_joints() below strips those columns for the
    // controller, so feed the ghost first from the unprojected copy.
    const std::vector<std::string> ghost_names = names;
    const std::vector<std::vector<double>> ghost_positions = positions;
    const std::vector<std::vector<double>> ghost_velocities = velocities;
    // PORT: single_planner._stage_trajectory - project the replayed full-joint
    // plan onto the active (controller) joints before streaming it.
    project_active_joints(names, positions, velocities);
    // MoveIt's pre-execution validate(): the segment must start where the
    // robot actually is. This is the check that refuses a chained segment
    // when the previous one has not settled, instead of letting the
    // controller snap/jump to a stale start point.
    const double start_tolerance =
      node_->has_parameter("allowed_start_tolerance") ?
      node_->get_parameter("allowed_start_tolerance").as_double() : 0.01;
    if (!positions.empty() &&
      !validate_start_state(
        node_, robot_context_, names, positions.front(), start_tolerance))
    {
      result->result.success = false;
      result->result.message = planner->get_planner_name() +
        ": trajectory start state does not match the robot";
      goal_handle->abort(result);
      return;
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
    robot_context_->set_ghost_command(
      ghost_names, ghost_velocities, {}, ghost_positions);
    const int epoch = robot_context_->set_command(
      names, velocities, {}, positions, /*feed_ghost=*/false);
    if (!robot_context_->send_trajectrory(epoch)) {
      result->result.success = false;
      result->result.message = "Execution failed";
      goal_handle->abort(result);
      return;
    }

    // Feedback timeline until the trajectory plays out (or is cancelled). A
    // non-success FollowJointTrajectory Result reports progression -1.0
    // (rejected goal, aborted path, ...) and must END the wait instead of
    // spinning until the caller's budget expires. A slow-but-in-progress
    // execution is NOT interrupted: Gazebo's real-time factor drops while a
    // grasped object is carried, so wall-clock time is not a reliable budget
    // here — the caller's own action timeout is.
    while (rclcpp::ok()) {
      if (goal_handle->is_canceling()) {
        result->result.success = false;
        result->result.message = "Execution canceled";
        goal_handle->canceled(result);
        return;
      }
      const double progression = robot_context_->get_progression();
      publish_feedback(goal_handle, "EXECUTING", std::max(0.0, progression), false);
      if (progression >= 1.0) {
        break;
      }
      if (progression < 0.0) {
        const std::string reason = robot_context_->action_failure_summary();
        result->result.success = false;
        result->result.message = planner->get_planner_name() +
          ": trajectory execution failed: " +
          (reason.empty() ? "controller reported a non-success result" : reason);
        RCLCPP_ERROR(node_->get_logger(), "%s", result->result.message.c_str());
        robot_context_->stop_robot();
        goal_handle->abort(result);
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // MoveIt's post-execution waitForRobotToStop(). The controller result
    // marks the stamped path complete, not a physically stopped arm. Wait for
    // kNoMotionSamples consecutive feedback samples below the same tolerance
    // MoveIt uses for its start validation. If the robot does not stop in
    // time we still report the controller result; the next segment's start
    // check is what refuses it.
    const bool wait_for_completion =
      node_->has_parameter("wait_for_trajectory_completion") ?
      node_->get_parameter("wait_for_trajectory_completion").as_bool() : true;
    const double stop_timeout =
      node_->has_parameter("post_execution_settle_timeout_s") ?
      node_->get_parameter("post_execution_settle_timeout_s").as_double() : 1.0;
    if (wait_for_completion) {
      wait_for_robot_to_stop(
        node_, robot_context_, names, start_tolerance, stop_timeout);
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
    const ReactiveGoal goal = make_reactive_goal(*request);

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

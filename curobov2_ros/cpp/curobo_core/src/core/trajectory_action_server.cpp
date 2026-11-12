// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of the SendTrajectory action surface.
#include "curobo_core/core/trajectory_action_server.hpp"

#include "curobo_core/planners/reactive_controller.hpp"
#include "curobo_core/robot/continuous_joints.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>

#include <sensor_msgs/msg/joint_state.hpp>

namespace curobo_core
{

namespace
{

double duration_seconds(const builtin_interfaces::msg::Duration & duration)
{
  return static_cast<double>(duration.sec) +
         static_cast<double>(duration.nanosec) * 1e-9;
}

double median(std::vector<double> values)
{
  if (values.empty()) {
    return std::nan("");
  }
  const std::size_t mid = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + mid, values.end());
  double result = values[mid];
  if (values.size() % 2 == 0) {
    std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
    result = 0.5 * (result + values[mid - 1]);
  }
  return result;
}

}  // namespace

// Least-squares estimate of the uniform interval dt from
//   p[i] - p[i-1] ~ 0.5 * (v[i] + v[i-1]) * dt
// (trapezoidal integration, exact for constant acceleration). One candidate
// per step and joint, then the median so a few near-zero-velocity samples or a
// curvature change cannot bias the result.
std::optional<double> estimate_replay_dt(
  const std::vector<std::vector<double>> & positions,
  const std::vector<std::vector<double>> & velocities)
{
  if (positions.size() < 2 || velocities.size() != positions.size()) {
    return std::nullopt;
  }
  std::vector<double> candidates;
  candidates.reserve(positions.size() - 1);
  for (std::size_t i = 1; i < positions.size(); ++i) {
    if (positions[i].size() != positions[i - 1].size() ||
      velocities[i].size() != velocities[i - 1].size())
    {
      continue;
    }
    double numerator = 0.0;
    double denominator = 0.0;
    const std::size_t n = std::min(positions[i].size(), velocities[i].size());
    for (std::size_t j = 0; j < n; ++j) {
      const double v_average =
        0.5 * (velocities[i][j] + velocities[i - 1][j]);
      // Ignore the standstill samples at the ends: their dp is float noise.
      if (std::abs(v_average) < 1.0e-3) {
        continue;
      }
      numerator += (positions[i][j] - positions[i - 1][j]) * v_average;
      denominator += v_average * v_average;
    }
    if (denominator > 1.0e-9) {
      candidates.push_back(numerator / denominator);
    }
  }
  const double estimate = median(std::move(candidates));
  if (!std::isfinite(estimate) || estimate <= 1.0e-5 || estimate > 10.0) {
    return std::nullopt;
  }
  return estimate;
}

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
    std::vector<std::vector<double>> ghost_positions = positions;
    const std::vector<std::vector<double>> ghost_velocities = velocities;
    // PORT: single_planner._stage_trajectory - project the replayed full-joint
    // plan onto the active (controller) joints before streaming it.
    project_active_joints(names, positions, velocities);
    // Anchor the WHOLE replay to the measured 2*pi branch. A per-point unwrap
    // changes the validated path and leaves velocity feed-forward inconsistent;
    // a constant whole-trajectory shift does not (FK is 2*pi-periodic). A
    // replay that does not start at the measured pose, or that contains a
    // genuine > pi step on a continuous joint, is rejected instead of repaired.
    const std::vector<double> measured =
      robot_context_ != nullptr ? robot_context_->get_joint_pose() : std::vector<double>{};
    const std::vector<std::string> measured_names =
      robot_context_ != nullptr ? robot_context_->get_joint_name() : joint_names_;
    if (!continuous_joint_names_.empty()) {
      // MoveIt's start-state acceptance rule: a replay whose first point is not
      // the measured physical pose (within this tolerance) is refused.
      const double start_tolerance =
        node_->has_parameter("start_state_max_bounds_error") ?
        node_->get_parameter("start_state_max_bounds_error").as_double() : 0.1;
      const auto ghost_branch = align_trajectory_to_measured(
        ghost_names, ghost_positions, measured_names, measured,
        continuous_joint_names_, start_tolerance);
      const auto controller_branch = align_trajectory_to_measured(
        names, positions, measured_names, measured, continuous_joint_names_,
        start_tolerance);
      const auto & branch = ghost_branch.ok ? controller_branch : ghost_branch;
      if (!branch.ok) {
        result->result.success = false;
        result->result.message =
          std::string("Refusing to replay trajectory: ") + branch.error;
        RCLCPP_ERROR(
          node_->get_logger(), "%s", result->result.message.c_str());
        goal_handle->abort(result);
        return;
      }
    }
    // The plan's own sample interval. A native cuRobo plan is sampled on the
    // TIME-OPTIMIZED grid (~13 ms for a short joint move), not at the node's
    // 25 ms interpolation_dt. Its velocity feed-forward is consistent with the
    // former; re-stamping it at 25 ms asks the controller for ~2x the planned
    // speed and it overshoots the endpoint (most visible on a continuous
    // joint). Honour the replay's own timestamps when the client supplied
    // them (the RViz/Joints tab builds them from result.dt), otherwise
    // recover the interval from positions + velocities (the task-constructor
    // replay path carries no stamps).
    double replay_dt = 0.0;
    if (n >= 2) {
      bool have_times = true;
      double previous = -1.0;
      double first = 0.0;
      double last = 0.0;
      for (std::size_t i = 0; i < msg.points.size(); ++i) {
        const double t = duration_seconds(msg.points[i].time_from_start);
        if (i == 0) {
          first = t;
        }
        if (t < previous - 1e-9) {
          have_times = false;
          break;
        }
        previous = t;
        last = t;
      }
      if (have_times && last > first) {
        const double average = (last - first) / static_cast<double>(n - 1);
        // Only a uniform grid can be represented by a single interval.
        bool uniform = average > 0.0;
        previous = first;
        for (std::size_t i = 1; uniform && i < msg.points.size(); ++i) {
          const double t = duration_seconds(msg.points[i].time_from_start);
          uniform = std::abs((t - previous) - average) <= 1e-6 + 1e-3 * average;
          previous = t;
        }
        if (uniform) {
          replay_dt = average;
        }
      }
    }
    const auto estimated = estimate_replay_dt(positions, velocities);
    if (estimated.has_value()) {
      if (!(replay_dt > 0.0)) {
        replay_dt = *estimated;
        RCLCPP_INFO(
          node_->get_logger(),
          "Replay carries no timestamps; inferred sample interval %.6f s from "
          "its velocity feed-forward", replay_dt);
      } else {
        const double tolerance = 0.02 * std::max(replay_dt, *estimated);
        if (std::abs(*estimated - replay_dt) > tolerance) {
          RCLCPP_WARN(
            node_->get_logger(),
            "Replay timestamps imply %.6f s but its velocity feed-forward "
            "implies %.6f s; using the velocity-consistent interval",
            replay_dt, *estimated);
          replay_dt = *estimated;
        }
      }
    }
    if (replay_dt > 0.0) {
      RCLCPP_INFO(
        node_->get_logger(), "Replaying trajectory at its plan sample interval %.6f s",
        replay_dt);
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
      ghost_names, ghost_velocities, {}, ghost_positions, replay_dt);
    const int epoch = robot_context_->set_command(
      names, velocities, {}, positions, /*feed_ghost=*/false, replay_dt);
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

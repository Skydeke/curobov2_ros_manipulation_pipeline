// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/planners/reactive_controller.py.
#include "curobo_core/planners/reactive_controller.hpp"

#include "curobo_runtime/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <limits>

namespace curobo_core
{

namespace
{
// PORT: reactive_controller.lane_reactive.call(..., prio=PRIO_REACTIVE) - run a
// servo job on the reactive lane unless this thread already is its worker.
template<class F>
auto on_reactive_lane(curobo_runtime::GpuLane * lane, F && fn) -> decltype(fn())
{
  if (lane == nullptr || lane->owns_current_thread()) {
    return fn();
  }
  curobo_runtime::SubmitOptions opts;
  opts.prio = curobo_runtime::PRIO_REACTIVE;
  return lane->call(opts, std::forward<F>(fn));
}
}  // namespace

namespace
{
// PORT: reactive_controller._row - a flat [n] vector to the last row of a tensor
// shaped [rows, n] (or the [n] itself when it has a single row).
std::vector<float> last_row(const curobo_runtime::TensorF32 & tensor)
{
  if (tensor.shape.empty()) {
    return {};
  }
  const std::int64_t cols = tensor.shape.back();
  if (cols <= 0) {
    return {};
  }
  const std::int64_t rows = tensor.size() / cols;
  if (rows <= 0) {
    return {};
  }
  const std::size_t offset = static_cast<std::size_t>((rows - 1) * cols);
  return std::vector<float>(
    tensor.data.begin() + offset, tensor.data.begin() + offset + cols);
}
}  // namespace

ReactiveController::ReactiveController(
  curobo_runtime::IBridge * bridge, curobo_runtime::RobotHandle handle,
  std::vector<std::string> joint_names, rclcpp::Node * node)
: TrajectoryPlanner(bridge, handle, std::move(joint_names)), node_(node)
{
}

double ReactiveController::now() const
{
  // PORT: reactive_controller._now - node ROS clock seconds (never wall time).
  return node_->get_clock()->now().nanoseconds() * 1e-9;
}

// PORT: TrajectoryPlanner.plan (reactive controllers do not produce a trajectory)
PlannerResult ReactiveController::plan(const curobo_runtime::PlanRequest &)
{
  PlannerResult result;
  result.success = false;
  result.planner_type = get_planner_name();
  result.message = get_planner_name() +
    " is a reactive controller: no trajectory is produced (use the execute action)";
  return result;
}

// PORT: ReactiveController.plan(start_state, goal_request, config, robot_context)
// PORT: ReactiveController.plan - lane_reactive.call(plan_reactive_impl, ...).
PlannerResult ReactiveController::plan_reactive(
  const ReactiveGoal & goal, double convergence_threshold, int max_iterations,
  RobotContext * robot_context)
{
  return on_reactive_lane(
    lane_, [&]() {
      return plan_reactive_impl(goal, convergence_threshold, max_iterations, robot_context);
    });
}

PlannerResult ReactiveController::plan_reactive_impl(
  const ReactiveGoal & goal, double convergence_threshold, int max_iterations,
  RobotContext * robot_context)
{
  PlannerResult result;
  result.planner_type = get_planner_name();

  // PORT: ReactiveController.plan -> lane.call(self.ensure_solver) first.
  ensure_solver();

  if (!solver_ready()) {
    result.success = false;
    result.message = get_planner_name() + " solver not initialized.";
    return result;
  }

  convergence_threshold_ = convergence_threshold;
  max_iterations_ = max_iterations;

  // The start configuration the reactive goal is anchored on: the caller's
  // start_pose when present, else the robot's current pose.
  std::vector<float> start_q;
  if (!goal.start_pose_positions.empty()) {
    start_q.assign(goal.start_pose_positions.begin(), goal.start_pose_positions.end());
  } else if (robot_context != nullptr) {
    const auto pose = robot_context->get_joint_pose();
    start_q.assign(pose.begin(), pose.end());
  }

  try {
    // PORT: ReactiveController._setup_exclusive - setup() cold-starts the
    // solver, which CAPTURES CUDA graphs. Capture is process-global, so it must
    // run exclusive (other GPU jobs drain, then pause) or a concurrent
    // camera/ESDF kernel invalidates the capture.
    bool setup_ok = false;
    {
      auto guard = curobo_runtime::Runtime::instance().barrier().exclusive();
      setup_ok = setup_solver(start_q, goal);
      graph_capture_pending_ = true;
    }
    if (!setup_ok) {
      result.success = false;
      result.message = "Failed to set reactive goal";
      return result;
    }
    // Discard a live goal left over from a previous session.
    take_live_goal();
    is_goal_active_ = true;

    if (robot_context != nullptr && !goal.start_pose_positions.empty()) {
      init_robot_at_start(robot_context, start_q);
    }

    RCLCPP_INFO(
      node_->get_logger(), "%s goal set: convergence=%gm, max_iter=%d",
      get_planner_name().c_str(), convergence_threshold_, max_iterations_);
    result.success = true;
    result.message = get_planner_name() + " goal set";
    return result;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(node_->get_logger(), "%s setup error: %s", get_planner_name().c_str(), e.what());
    result.success = false;
    result.message = std::string("Reactive setup error: ") + e.what();
    return result;
  }
}

// PORT: ReactiveController._set_live_goal / _take_live_goal
void ReactiveController::set_live_goal(const std::array<double, 7> & raw)
{
  std::lock_guard<std::mutex> lk(live_goal_lock_);
  latest_goal_ = raw;
  latest_goal_fresh_ = true;
}

std::optional<std::array<double, 7>> ReactiveController::take_live_goal()
{
  std::lock_guard<std::mutex> lk(live_goal_lock_);
  if (!latest_goal_fresh_) {
    return std::nullopt;
  }
  std::optional<std::array<double, 7>> raw = latest_goal_;
  latest_goal_.reset();
  latest_goal_fresh_ = false;
  return raw;
}

// PORT: ReactiveController.cancel
void ReactiveController::cancel()
{
  is_goal_active_ = false;
  RCLCPP_INFO(node_->get_logger(), "%s execution cancelled", get_planner_name().c_str());
}

curobo_runtime::MpcStepOutcome ReactiveController::step_paced_impl(
  const curobo_runtime::ReactiveState & state)
{
  return step_impl(state, false);
}

// PORT: ReactiveController.read_state
curobo_runtime::ReactiveState ReactiveController::read_state(
  RobotContext * robot_context) const
{
  curobo_runtime::ReactiveState state;
  const auto pose = robot_context->get_joint_pose();
  state.position.assign(pose.begin(), pose.end());
  state.velocity.assign(state.position.size(), 0.0f);
  state.acceleration.assign(state.position.size(), 0.0f);
  return state;
}

// PORT: ReactiveController._state_from_action
curobo_runtime::ReactiveState ReactiveController::state_from_action(
  const curobo_runtime::ReactiveAction & action) const
{
  curobo_runtime::ReactiveState state;
  state.position = last_row(action.position);
  state.velocity = last_row(action.velocity);
  state.acceleration = last_row(action.acceleration);
  return state;
}

// PORT: ReactiveController._close_state_loop
void ReactiveController::close_state_loop(
  RobotContext * robot_context, curobo_runtime::ReactiveState & state) const
{
  // Real position; keep the solver's own PREDICTED velocity/acceleration.
  const auto pose = robot_context->get_joint_pose();
  state.position.assign(pose.begin(), pose.end());
}

// PORT: ReactiveController._send_command
void ReactiveController::send_command(
  RobotContext * robot_context, const curobo_runtime::ReactiveAction & action)
{
  const bool window = action.position.shape.size() == 2 && action.position.shape[0] > 1;
  const std::int64_t rows = action.position.shape.empty() ? 0 : action.position.shape[0];
  const std::int64_t cols = action.position.shape.empty() ? 0 : action.position.shape.back();
  std::vector<std::vector<double>> position;
  std::vector<std::vector<double>> velocity;
  std::vector<std::vector<double>> acceleration;
  for (std::int64_t r = 0; r < static_cast<std::int64_t>(rows); ++r) {
    std::vector<double> p(cols, 0.0), v(cols, 0.0), a(cols, 0.0);
    for (std::int64_t c = 0; c < cols; ++c) {
      const std::size_t idx = static_cast<std::size_t>(r * cols + c);
      if (idx < action.position.data.size()) {
        p[c] = action.position.data[idx];
      }
      if (idx < action.velocity.data.size()) {
        v[c] = action.velocity.data[idx];
      }
      if (idx < action.acceleration.data.size()) {
        a[c] = action.acceleration.data[idx];
      }
    }
    position.push_back(std::move(p));
    velocity.push_back(std::move(v));
    acceleration.push_back(std::move(a));
  }
  (void)window;
  robot_context->set_and_send_command({}, velocity, acceleration, position);
}

// PORT: ReactiveController._init_robot_at_start
void ReactiveController::init_robot_at_start(
  RobotContext * robot_context, const std::vector<float> & start_q)
{
  std::vector<double> start(start_q.begin(), start_q.end());
  // PORT: set_command(None, [[0.0] * n], [[0.0] * n], [start_position]) - the
  // zero rows are dof-sized, not a single element.
  const std::vector<double> zeros(start.size(), 0.0);
  robot_context->set_command({}, {zeros}, {zeros}, {start});
  start_initialized_ = true;
}

// PORT: ReactiveController._exec_csv_enabled
bool ReactiveController::exec_csv_enabled()
{
  if (!node_->has_parameter("exec_debug_csv")) {
    node_->declare_parameter("exec_debug_csv", false);
  }
  return node_->get_parameter("exec_debug_csv").as_bool();
}

// PORT: ReactiveController._exec_csv_write (exec_servo columns)
void ReactiveController::write_exec_csv(int tstep, double step_seconds)
{
  if (exec_csv_ == nullptr) {
    return;
  }
  const double t_s = std::chrono::duration<double>(
    std::chrono::steady_clock::now().time_since_epoch()).count() - exec_csv_t0_;
  std::ostringstream t, err, step;
  t << std::fixed << std::setprecision(3) << t_s;
  err << std::fixed << std::setprecision(6) << get_position_error();
  step << std::fixed << std::setprecision(4) << step_seconds;
  exec_csv_->writerow({t.str(), std::to_string(tstep), err.str(),
      is_on_target() ? "True" : "False", step.str()});
}

void ReactiveController::compute_position_error(
  const curobo_runtime::MpcStepOutcome & outcome)
{
  if (!outcome.has_ee || !target_set_) {
    last_position_error_ = std::numeric_limits<double>::infinity();
    return;
  }
  const double dx = outcome.ee_position[0] - target_[0];
  const double dy = outcome.ee_position[1] - target_[1];
  const double dz = outcome.ee_position[2] - target_[2];
  last_position_error_ = std::sqrt(dx * dx + dy * dy + dz * dz);
}

// PORT: ReactiveController._publish_feedback
namespace
{
std::vector<double> feedback_command(const curobo_runtime::ReactiveAction & action)
{
  std::vector<float> row = last_row(action.position);
  return std::vector<double>(row.begin(), row.end());
}
}  // namespace

// PORT: ReactiveController.execute
// PORT: ReactiveController.execute - the whole servo run on lane_reactive.
bool ReactiveController::execute(const ReactiveExecutionContext & ctx)
{
  return on_reactive_lane(lane_, [&]() {return execute_impl(ctx);});
}

bool ReactiveController::execute_impl(const ReactiveExecutionContext & ctx)
{
  if (!is_goal_active_ || !solver_ready()) {
    RCLCPP_ERROR(
      node_->get_logger(), "%s not initialized. Call plan() first.",
      get_planner_name().c_str());
    return false;
  }
  if (command_interval_ > 0.0) {
    return execute_paced(ctx, command_interval_);
  }
  return execute_immediate(ctx);
}

// PORT: ReactiveController._execute_immediate
bool ReactiveController::execute_immediate(const ReactiveExecutionContext & ctx)
{
  try {
    int tstep = 0;
    RCLCPP_INFO(node_->get_logger(), "Starting %s servo loop", get_planner_name().c_str());
    // PORT: ReactiveController._exec_csv_init (prefix "exec_servo").
    const bool exec_csv = exec_csv_enabled();
    if (exec_csv) {
      exec_csv_ = open_diag_csv(node_, "exec_servo");
      if (exec_csv_) {
        exec_csv_->write_header_once({"t_s", "tstep", "error_m", "on_target", "step_ms"});
      }
      exec_csv_t0_ = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    curobo_runtime::ReactiveState current_state = read_state(ctx.robot_context);

    while (is_goal_active_) {
      if (ctx.cancel_requested && ctx.cancel_requested()) {
        RCLCPP_INFO(node_->get_logger(), "%s cancel requested", get_planner_name().c_str());
        break;
      }
      if (!ctx.cancel_requested && tstep >= max_iterations_) {
        break;
      }

      // PORT: refresh the perception-based collision world periodically, like
      // the Python's _execute_immediate / _execute_paced.
      if (perception_refresh_period_ > 0 && perception_refresh_ &&
        tstep % perception_refresh_period_ == 0)
      {
        try {
          perception_refresh_();
        } catch (const std::exception &) {
          // A perception refresh must never kill the servo loop.
        }
      }

      const auto raw = take_live_goal();
      if (raw.has_value()) {
        try {
          // PORT: retargeting runs the solver's IK, which can capture a graph.
          auto guard = curobo_runtime::Runtime::instance().barrier().exclusive();
          apply_live_goal(*raw, current_state);
        } catch (const std::exception & e) {
          RCLCPP_ERROR_THROTTLE(
            node_->get_logger(), *node_->get_clock(), 1000,
            "%s: live goal rejected (%s) - keeping previous goal",
            get_planner_name().c_str(), e.what());
        }
      }

      const double step_t0 = now();
      curobo_runtime::MpcStepOutcome outcome;
      {
        auto & barrier = curobo_runtime::Runtime::instance().barrier();
        if (graph_capture_pending_) {
          graph_capture_pending_ = false;
          auto guard = barrier.exclusive();
          outcome = step_impl(current_state, false);
        } else {
          auto guard = barrier.shared();
          outcome = step_impl(current_state, false);
        }
      }
      if (tstep > 5) {
        step_times_.push_back(now() - step_t0);
      }
      compute_position_error(outcome);
      send_command(ctx.robot_context, outcome.action);

      curobo_runtime::ReactiveState predicted = state_from_action(outcome.action);
      close_state_loop(ctx.robot_context, predicted);
      current_state = predicted;

      if (ctx.feedback && tstep % 5 == 0) {
        ctx.feedback(
          is_on_target() ? "ON_TARGET" : "TRACKING", step_progression(),
          is_on_target(), feedback_command(outcome.action), get_position_error());
      }
      write_exec_csv(tstep, now() - step_t0);
      tstep += 1;
    }

    ctx.robot_context->stop_robot();
    exec_csv_.reset();
    log_step_times(tstep);
    return true;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(node_->get_logger(), "%s execution error: %s", get_planner_name().c_str(), e.what());
    ctx.robot_context->stop_robot();
    return false;
  }
}

// PORT: ReactiveController._execute_paced
bool ReactiveController::execute_paced(const ReactiveExecutionContext & ctx, double interval)
{
  try {
    int tstep = 0;
    RCLCPP_INFO(
      node_->get_logger(), "Starting %s servo loop (paced, interval=%gs, single-plan windows)",
      get_planner_name().c_str(), interval);
    // PORT: ReactiveController._exec_csv_init (prefix "exec_servo").
    const bool exec_csv = exec_csv_enabled();
    if (exec_csv) {
      exec_csv_ = open_diag_csv(node_, "exec_servo");
      if (exec_csv_) {
        exec_csv_->write_header_once({"t_s", "tstep", "error_m", "on_target", "step_ms"});
      }
      exec_csv_t0_ = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    curobo_runtime::ReactiveState current_state = read_state(ctx.robot_context);

    while (is_goal_active_) {
      if (ctx.cancel_requested && ctx.cancel_requested()) {
        RCLCPP_INFO(node_->get_logger(), "%s cancel requested", get_planner_name().c_str());
        break;
      }
      if (!ctx.cancel_requested && tstep >= max_iterations_) {
        break;
      }

      // PORT: refresh the perception-based collision world periodically, like
      // the Python's _execute_immediate / _execute_paced.
      if (perception_refresh_period_ > 0 && perception_refresh_ &&
        tstep % perception_refresh_period_ == 0)
      {
        try {
          perception_refresh_();
        } catch (const std::exception &) {
          // A perception refresh must never kill the servo loop.
        }
      }

      const auto raw = take_live_goal();
      if (raw.has_value()) {
        try {
          // PORT: retargeting runs the solver's IK, which can capture a graph.
          auto guard = curobo_runtime::Runtime::instance().barrier().exclusive();
          apply_live_goal(*raw, current_state);
        } catch (const std::exception & e) {
          RCLCPP_ERROR_THROTTLE(
            node_->get_logger(), *node_->get_clock(), 1000,
            "%s: live goal rejected (%s) - keeping previous goal",
            get_planner_name().c_str(), e.what());
        }
      }

      const double step_t0 = now();
      curobo_runtime::MpcStepOutcome outcome;
      {
        auto & barrier = curobo_runtime::Runtime::instance().barrier();
        if (graph_capture_pending_) {
          graph_capture_pending_ = false;
          auto guard = barrier.exclusive();
          outcome = step_paced_impl(current_state);
        } else {
          auto guard = barrier.shared();
          outcome = step_paced_impl(current_state);
        }
      }
      if (tstep > 5) {
        step_times_.push_back(now() - step_t0);
      }
      compute_position_error(outcome);
      send_command(ctx.robot_context, outcome.action);
      if (ctx.feedback) {
        ctx.feedback(
          is_on_target() ? "ON_TARGET" : "TRACKING", step_progression(),
          is_on_target(), feedback_command(outcome.action), get_position_error());
      }

      wait_for_execution(ctx.robot_context, interval);

      curobo_runtime::ReactiveState predicted = state_from_action(outcome.action);
      close_state_loop(ctx.robot_context, predicted);
      current_state = predicted;
      write_exec_csv(tstep, now() - step_t0);
      tstep += 1;
    }

    ctx.robot_context->stop_robot();
    exec_csv_.reset();
    log_step_times(tstep);
    return true;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(node_->get_logger(), "%s execution error: %s", get_planner_name().c_str(), e.what());
    ctx.robot_context->stop_robot();
    return false;
  }
}

// PORT: the Python "stopped: N steps, avg time=..ms/step" log, extended with
// p50/p99 for the M5 step-latency acceptance measurement.
void ReactiveController::log_step_times(int tstep)
{
  if (step_times_.empty()) {
    RCLCPP_INFO(
      node_->get_logger(), "%s stopped: %d steps", get_planner_name().c_str(), tstep);
    return;
  }
  double sum = 0.0;
  for (double t : step_times_) {
    sum += t;
  }
  std::vector<double> sorted = step_times_;
  std::sort(sorted.begin(), sorted.end());
  const auto pct = [&sorted](double p) {
      const std::size_t idx = std::min(
        sorted.size() - 1, static_cast<std::size_t>(p * static_cast<double>(sorted.size())));
      return sorted[idx] * 1000.0;
    };
  RCLCPP_INFO(
    node_->get_logger(),
    "%s stopped: %d steps, avg time=%.1fms/step (p50=%.1fms p99=%.1fms, n=%zu)",
    get_planner_name().c_str(), tstep, 1000.0 * sum / static_cast<double>(step_times_.size()),
    pct(0.50), pct(0.99), step_times_.size());
}

// PORT: ReactiveController._wait_for_execution (event-driven poll on the node
// executor, not a busy loop).
void ReactiveController::wait_for_execution(RobotContext * robot_context, double interval)
{
  // The Python creates a timer in a PERSISTENT callback group
  // (self._wait_cb_group) whose closure keeps a threading.Event alive, and
  // destroys the timer in a finally; mirror that.
  //
  // The state MUST be a shared_ptr captured BY VALUE: the executor can still
  // invoke the timer callback after this function returns (cancel() does not
  // guarantee the in-flight callback is done), so referencing stack locals
  // (the old `[&]` capture) was a use-after-free that corrupted the host heap
  // in the paced servo soak.
  struct WaitState
  {
    std::mutex mutex;
    std::condition_variable cv;
    bool done = false;
  };
  auto state = std::make_shared<WaitState>();
  const double timeout_s = std::max(interval * 2.0 + 0.2, 0.5);

  if (wait_cb_group_ == nullptr) {
    // Own group so these poll callbacks never serialize with the default group
    // (a long open-loop plan must not stall the solve-and-shoot cadence).
    wait_cb_group_ = node_->create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive);
  }
  auto timer = node_->create_wall_timer(
    std::chrono::duration<double>(0.02),
    [state, this, robot_context]() {
      std::lock_guard<std::mutex> lk(state->mutex);
      if (!is_goal_active_ || robot_context->get_progression() >= 1.0) {
        state->done = true;
        state->cv.notify_all();
      }
    }, wait_cb_group_);

  {
    std::unique_lock<std::mutex> lk(state->mutex);
    state->cv.wait_for(
      lk, std::chrono::duration<double>(timeout_s), [&] {return state->done;});
  }
  timer->cancel();
}

}  // namespace curobo_core

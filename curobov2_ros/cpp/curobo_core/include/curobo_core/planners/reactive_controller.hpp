// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/planners/reactive_controller.py
//
// Closed-loop sibling of the open-loop planners: a thin ROS wrapper around a
// cuRobo reactive solver (ModelPredictiveControl). Owns the control loop
// (immediate free-running and paced solve-and-shoot), the live-goal hand-off and
// the action feedback; the concrete controller implements the few cuRobo-specific
// hooks (build/setup/step/apply_live_goal) through the bridge.
#ifndef CUROBO_CORE__PLANNERS__REACTIVE_CONTROLLER_HPP_
#define CUROBO_CORE__PLANNERS__REACTIVE_CONTROLLER_HPP_

#include <array>
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "curobo_core/planners/trajectory_planner.hpp"
#include "curobo_core/core/diagnostics.hpp"
#include "curobo_core/robot/robot_context.hpp"

namespace curobo_core
{

// The reactive goal (PORT: the SendTrajectory goal's goalsets). Reactive
// controllers require exactly one goalset entry with a single pose.
struct ReactiveGoal
{
  std::vector<std::vector<std::array<double, 7>>> goalsets;   // [set][pose][7]
  std::vector<std::string> start_pose_names;
  std::vector<double> start_pose_positions;
};

// PORT: the robot_context + action-handle surface the servo loop needs. The
// action server supplies the callbacks; the loop body stays in the controller.
struct ReactiveExecutionContext
{
  RobotContext * robot_context = nullptr;
  std::function<bool()> cancel_requested;                     // goal_handle.is_cancel_requested
  std::function<void(
      const std::string & state, double step_progression, bool on_target,
      const std::vector<double> & joint_command, double position_error)> feedback;
};

class ReactiveController : public TrajectoryPlanner
{
public:
  ReactiveController(
    curobo_runtime::IBridge * bridge, curobo_runtime::RobotHandle handle,
    std::vector<std::string> joint_names, rclcpp::Node * node);

  ExecutionMode execution_mode() const override {return ExecutionMode::kClosedLoop;}

  // PORT: TrajectoryPlanner.plan - reactive controllers do not produce a
  // trajectory; planning a reactive goal goes through plan_reactive.
  PlannerResult plan(const curobo_runtime::PlanRequest & request) override;

  // PORT: ReactiveController.plan(start_state, goal_request, config, robot_context).
  PlannerResult plan_reactive(
    const ReactiveGoal & goal, double convergence_threshold, int max_iterations,
    RobotContext * robot_context);

  // PORT: ReactiveController.execute (dispatch paced vs immediate).
  bool execute(const ReactiveExecutionContext & ctx);
  // PORT: ReactiveController.cancel
  void cancel();

  // PORT: the node's refresh_perception_world - the servo loop refreshes the
  // ESDF world every `perception_refresh_period` steps so it reacts to new
  // camera data.
  void set_perception_refresh(std::function<void()> refresh)
  {perception_refresh_ = std::move(refresh);}

  bool is_goal_active() const {return is_goal_active_;}
  double get_position_error() const {return last_position_error_;}
  // PORT: ReactiveController._step_times (per-step seconds, first 5 skipped).
  // Exposed for the M5 latency benchmark (p50/p99 step time).
  const std::vector<double> & step_times() const {return step_times_;}
  bool is_on_target() const {return last_position_error_ < convergence_threshold_;}

  // PORT: ReactiveController.set_live_goal / _take_live_goal
  void set_live_goal(const std::array<double, 7> & raw);
  std::optional<std::array<double, 7>> take_live_goal();

  // PORT: ReactiveController.ensure_solver (build the solver lazily) and
  // update_world (push the shared scene into the solver's collision world).
  virtual void ensure_solver() {}
  virtual void update_world(const std::string & scene_json) {(void)scene_json;}

  // PORT: ReactiveController._now (node clock seconds)
  double now() const;

protected:
  // cuRobo-specific hooks (subclasses implement through the bridge).
  virtual bool solver_ready() const = 0;
  virtual bool setup_solver(
    const std::vector<float> & start_q, const ReactiveGoal & goal) = 0;
  virtual curobo_runtime::MpcStepOutcome step_impl(
    const curobo_runtime::ReactiveState & state, bool paced) = 0;
  virtual bool apply_live_goal(
    const std::array<double, 7> & raw, const curobo_runtime::ReactiveState & state) = 0;

  // PORT: ReactiveController.step_paced (default -> step_impl(paced=false)).
  virtual curobo_runtime::MpcStepOutcome step_paced_impl(
    const curobo_runtime::ReactiveState & state);

  double command_interval() const {return command_interval_;}

  // PORT: reactive_controller._set_target (the target xyz drives _fk_position_error)
  void set_target(const std::array<double, 7> & raw)
  {
    target_ = {raw[0], raw[1], raw[2]};
    target_set_ = true;
  }

  // PORT: reactive_controller._publish_feedback step_progression
  double step_progression() const
  {
    if (!(last_position_error_ == last_position_error_)) {   // NaN -> 0.0
      return 0.0;
    }
    if (std::isinf(last_position_error_)) {
      return 0.0;
    }
    return 1.0 - std::min(last_position_error_ / 0.1, 1.0);
  }

  rclcpp::Node * node_;
  bool is_goal_active_ = false;
  double convergence_threshold_ = 0.01;
  int max_iterations_ = 1000;
  int perception_refresh_period_ = 2;
  double command_interval_ = 0.0;
  double last_position_error_ = std::numeric_limits<double>::infinity();

private:
  bool execute_immediate(const ReactiveExecutionContext & ctx);
  bool execute_paced(const ReactiveExecutionContext & ctx, double interval);
  void wait_for_execution(RobotContext * robot_context, double interval);
  void log_step_times(int tstep);

  // PORT: ReactiveController._read_state / _state_from_action / _close_state_loop
  curobo_runtime::ReactiveState read_state(RobotContext * robot_context) const;
  curobo_runtime::ReactiveState state_from_action(
    const curobo_runtime::ReactiveAction & action) const;
  void close_state_loop(
    RobotContext * robot_context, curobo_runtime::ReactiveState & state) const;
  // PORT: ReactiveController._send_command
  void send_command(
    RobotContext * robot_context, const curobo_runtime::ReactiveAction & action);
  // PORT: ReactiveController._init_robot_at_start
  void init_robot_at_start(
    RobotContext * robot_context, const std::vector<float> & start_q);

  void compute_position_error(const curobo_runtime::MpcStepOutcome & outcome);
  // PORT: ReactiveController._exec_csv_write (one row per step).
  void write_exec_csv(int tstep, double step_seconds);

  // PORT: trajectory_planner._exec_csv_enabled - the reactive exec CSV
  // (exec_servo_<ts>.csv, one row per step) is gated by `exec_debug_csv`.
  bool exec_csv_enabled();

  std::mutex live_goal_lock_;
  std::optional<std::array<double, 7>> latest_goal_;
  bool latest_goal_fresh_ = false;
  bool start_initialized_ = false;
  std::function<void()> perception_refresh_;
  // PORT: ReactiveController._wait_cb_group - persistent (created once) so the
  // execution-wait timer never needs a per-call callback group.
  rclcpp::CallbackGroup::SharedPtr wait_cb_group_;
  // PORT: ReactiveController._exec_csv / _exec_csv_t0.
  std::shared_ptr<DiagCsv> exec_csv_;
  double exec_csv_t0_ = 0.0;
  std::vector<double> step_times_;
  bool target_set_ = false;
  // PORT: node.take_graph_capture_pending - the step after a build/reset may
  // CAPTURE a CUDA graph and must run under the exclusive barrier.
  bool graph_capture_pending_ = false;
  std::array<double, 3> target_{0.0, 0.0, 0.0};
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__PLANNERS__REACTIVE_CONTROLLER_HPP_

// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/planners/mpc_planner.py
//
// Closed-loop MPC built on cuRobo ModelPredictiveControl (v2). The solver config
// assembly (the MPPI/L-BFGS dicts and the cuRobo task-config paths) lives in
// curobo_bridge (mpc_build); this class owns the ROS-facing behaviour: goal
// validation, the Cartesian->joint-tracking fallback, the live-retarget reseed
// and the RViz debug publishers.
#ifndef CUROBO_CORE__PLANNERS__MPC_PLANNER_HPP_
#define CUROBO_CORE__PLANNERS__MPC_PLANNER_HPP_

#include <array>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <curobov2_ros_interfaces/msg/mpc_costs.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include "curobo_core/core/diagnostics.hpp"
#include "curobo_core/planners/reactive_controller.hpp"

namespace curobo_core
{

class MPCController : public ReactiveController
{
public:
  MPCController(
    curobo_runtime::IBridge * bridge, curobo_runtime::RobotHandle handle,
    std::vector<std::string> joint_names, rclcpp::Node * node);

  // PORT: MPCController.get_planner_name
  std::string get_planner_name() const override {return "Model Predictive Control (MPC)";}

  // PORT: MPCController.get_config_parameters
  std::vector<std::string> get_config_parameters() const
  {
    return {"convergence_threshold", "max_mpc_iterations"};
  }

  // PORT: ReactiveController.ensure_solver (MPC builds the reactive solver)
  void ensure_solver();

  // PORT: MPCController.update_world -> scene_collision_checker.load_collision_model
  void update_world(const std::string & scene_json);

protected:
  bool solver_ready() const override {return built_;}
  bool setup_solver(
    const std::vector<float> & start_q, const ReactiveGoal & goal) override;
  curobo_runtime::MpcStepOutcome step_impl(
    const curobo_runtime::ReactiveState & state, bool paced) override;
  curobo_runtime::MpcStepOutcome step_paced_impl(
    const curobo_runtime::ReactiveState & state) override;
  bool apply_live_goal(
    const std::array<double, 7> & raw, const curobo_runtime::ReactiveState & state) override;

private:
  // PORT: MPCController._apply_goal
  bool apply_goal(const std::array<double, 7> & raw, const std::vector<float> & seed_q);
  // PORT: MPCController._reseed_toward_goal
  void reseed_toward_goal(
    const std::array<double, 7> & raw, const curobo_runtime::ReactiveState & state);
  // PORT: MPCController._publish_goal_marker
  void publish_goal_marker(const std::array<double, 7> & raw, bool applied);
  // PORT: MPCController._publish_predicted_path
  void publish_predicted_path(const curobo_runtime::MpcStepOutcome & outcome);
  // PORT: MPCController._publish_costs (mpc_debug diagnostics)
  void publish_costs(const curobo_runtime::MpcStepOutcome & outcome);
  // PORT: MPCController._csv_write (mpc_diag diagnostics, mpc_debug only)
  void write_csv(const curobo_runtime::MpcStepOutcome & outcome, double solve_ms);
  // PORT: MPCController._debug_enabled
  bool debug_enabled();

  bool built_ = false;
  std::string base_link_;
  double step_dt_ = 0.03;
  bool use_mppi_acceleration_ = true;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr path_pub_;
  rclcpp::Publisher<curobov2_ros_interfaces::msg::MpcCosts>::SharedPtr cost_pub_;
  // PORT: MPCController._csv / _csv_t0 / _csv_t_prev / _csv_last_vlast.
  std::shared_ptr<curobo_core::DiagCsv> csv_;
  double csv_t0_ = 0.0;
  double csv_t_prev_ = 0.0;
  std::vector<double> csv_last_vlast_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr goal_marker_pub_;
  rclcpp::Time last_path_pub_at_;
  bool has_published_path_ = false;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__PLANNERS__MPC_PLANNER_HPP_

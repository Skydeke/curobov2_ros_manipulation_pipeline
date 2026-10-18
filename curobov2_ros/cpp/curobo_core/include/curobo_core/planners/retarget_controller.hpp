// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/planners/retarget_controller.py
//
// Motion Retargeting - reactive teleop over cuRobo MotionRetargeter: closed-loop
// IK on a live tool-pose stream, sharing the ReactiveController control loop.
#ifndef CUROBO_CORE__PLANNERS__RETARGET_CONTROLLER_HPP_
#define CUROBO_CORE__PLANNERS__RETARGET_CONTROLLER_HPP_

#include <array>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "curobo_core/planners/reactive_controller.hpp"

namespace curobo_core
{

class RetargetController : public ReactiveController
{
public:
  RetargetController(
    curobo_runtime::IBridge * bridge, curobo_runtime::RobotHandle handle,
    std::vector<std::string> joint_names, rclcpp::Node * node);

  // PORT: RetargetController.get_planner_name
  std::string get_planner_name() const override {return "Motion Retargeting (Teleop)";}

  // PORT: RetargetController.get_config_parameters
  std::vector<std::string> get_config_parameters() const
  {
    return {"convergence_threshold", "retarget_position_weight",
      "retarget_orientation_weight"};
  }

  // PORT: ReactiveController.ensure_solver (retarget builds the retargeter)
  void ensure_solver();

  // PORT: RetargetController.update_world
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
  bool built_ = false;
  std::array<double, 7> goal_{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__PLANNERS__RETARGET_CONTROLLER_HPP_

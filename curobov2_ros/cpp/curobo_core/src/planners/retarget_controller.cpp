// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/planners/retarget_controller.py.
#include "curobo_core/planners/retarget_controller.hpp"

#include "curobo_runtime/runtime.hpp"

#include <sstream>

namespace curobo_core
{

RetargetController::RetargetController(
  curobo_runtime::IBridge * bridge, curobo_runtime::RobotHandle handle,
  std::vector<std::string> joint_names, rclcpp::Node * node)
: ReactiveController(bridge, handle, std::move(joint_names), node)
{
}

// PORT: ReactiveController.ensure_solver / RetargetController.build_solver
void RetargetController::ensure_solver()
{
  if (built_) {
    return;
  }
  const auto param = [&](const char * name, double fallback) {
      return node_ != nullptr && node_->has_parameter(name) ?
             node_->get_parameter(name).as_double() : fallback;
    };
  const bool use_mpc = node_ != nullptr && node_->has_parameter("retarget_use_mpc") ?
    node_->get_parameter("retarget_use_mpc").as_bool() : false;

  std::ostringstream cfg;
  cfg << "{\"position_weight\":" << param("retarget_position_weight", 1.0)
      << ",\"orientation_weight\":" << param("retarget_orientation_weight", 1.0)
      << ",\"use_mpc\":" << (use_mpc ? "true" : "false")
      << ",\"tool_frame\":\"" << (node_ != nullptr && node_->has_parameter("retarget_tool_frame") ?
    node_->get_parameter("retarget_tool_frame").as_string() : "") << "\"}";
  {
    auto guard = curobo_runtime::Runtime::instance().barrier().exclusive();
    bridge_->retarget_build(handle_, cfg.str());
  }
  built_ = true;
  RCLCPP_INFO(
    node_ != nullptr ? node_->get_logger() : rclcpp::get_logger("retarget"),
    "Retargeter built: global IK 64 seeds, use_mpc=%d", static_cast<int>(use_mpc));
}

// PORT: RetargetController.update_world
void RetargetController::update_world(const std::string & scene_json)
{
  bridge_->retarget_load_scene(handle_, scene_json);
}

// PORT: RetargetController.setup
bool RetargetController::setup_solver(
  const std::vector<float> &, const ReactiveGoal & goal)
{
  if (!(goal.goalsets.size() == 1 && goal.goalsets[0].size() == 1)) {
    std::ostringstream sizes;
    for (const auto & g : goal.goalsets) {
      sizes << g.size() << ",";
    }
    RCLCPP_ERROR(
      node_->get_logger(),
      "Retarget requires exactly one goalset entry with a single pose (got %zu sets, "
      "poses_per_set=[%s])",
      goal.goalsets.size(), sizes.str().c_str());
    return false;
  }
  goal_ = goal.goalsets[0][0];
  set_target(goal_);
  // PORT: reset() clears the warm-start so the first solve_frame uses global IK.
  bridge_->retarget_reset(handle_);
  return true;
}

// PORT: RetargetController.step
curobo_runtime::MpcStepOutcome RetargetController::step_impl(
  const curobo_runtime::ReactiveState &, bool)
{
  return bridge_->retarget_step(handle_, goal_);
}

curobo_runtime::MpcStepOutcome RetargetController::step_paced_impl(
  const curobo_runtime::ReactiveState & state)
{
  return step_impl(state, false);
}

// PORT: RetargetController.apply_live_goal
bool RetargetController::apply_live_goal(
  const std::array<double, 7> & raw, const curobo_runtime::ReactiveState &)
{
  goal_ = raw;
  set_target(goal_);
  return true;
}

}  // namespace curobo_core

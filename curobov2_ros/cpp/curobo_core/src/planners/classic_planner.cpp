// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of the open-loop planners' plan() paths.
#include "curobo_core/planners/classic_planner.hpp"
#include "curobo_core/planners/joint_space_planner.hpp"

#include "curobo_core/planners/zero_motion.hpp"

#include <rclcpp/rclcpp.hpp>

namespace curobo_core
{

// PORT: classic_planner.ClassicPlanner._plan_trajectory
PlannerResult ClassicPlanner::plan(const curobo_runtime::PlanRequest & request)
{
  PlannerResult result;
  result.planner_type = "classic";

  // PORT: classic_planner.ClassicPlanner._plan_trajectory - goalsets must hold
  // at least one pose; a joint-space segment (no poses, only
  // target_joint_positions) belongs to the joint-space planner. Say which cause
  // it is rather than letting the solver raise a bare KeyError.
  if (request.goalset.shape.empty() || request.goalset.shape[0] == 0) {
    RCLCPP_WARN(
      rclcpp::get_logger("classic_planner"),
      "Classic Motion Generation: goalsets must contain exactly 1 segment with >= 1 "
      "pose for classic planning, got empty request (%s)",
      request.joint_target.empty() ?
      "no segments / no poses or joint targets" :
      "a joint-space segment - select the joint-space planner (set_planner 5)");
    result.success = false;
    result.message =
      "Classic Motion Generation: goalsets must contain exactly 1 segment with "
      ">= 1 pose for classic planning (a joint-space segment needs the "
      "joint-space planner)";
    return result;
  }

  // PORT: classic_planner "planning to candidate set (N poses) resolved within
  // a single solve" / "Planning with max_attempts=N".
  const std::int64_t num_goalset =
    request.goalset.shape.empty() ? 0 : request.goalset.shape[0];
  if (num_goalset > 1) {
    RCLCPP_INFO(
      rclcpp::get_logger("classic_planner"),
      "Classic Motion Generation: planning to candidate set (%ld poses) resolved "
      "within a single solve", static_cast<long>(num_goalset));
  }
  RCLCPP_INFO(
    rclcpp::get_logger("classic_planner"), "Planning with max_attempts=%d",
    request.max_attempts);
  try {
    result.trajectory = bridge_->plan_pose(handle_, request);
  } catch (const std::exception & e) {
    result.success = false;
    result.message = std::string("Planning error: ") + e.what();
    return result;
  }
  if (!result.trajectory.success) {
    result.success = false;
    // PORT: single_planner "Planning failed: no solution found (plan_pose returned None)"
    result.message = "Planning failed: no solution found (plan_pose returned None)";
    return result;
  }
  result.success = true;
  result.message = "Trajectory planned successfully";
  return result;
}

// PORT: joint_space_planner.JointSpacePlanner._plan_trajectory
PlannerResult JointSpacePlanner::plan(const curobo_runtime::PlanRequest & request)
{
  PlannerResult result;
  result.planner_type = "joint_space";

  // PORT: the degenerate-start guard (is_zero_delta): no motion needed, skip the
  // solver entirely (avoids a degenerate CUDA-graph solve).
  if (!request.joint_target.empty() && is_zero_delta(request.start, request.joint_target)) {
    result.success = true;
    result.message = "Trajectory planned successfully";
    result.trajectory.success = true;
    result.trajectory.feasible = true;
    result.trajectory.joint_names = joint_names_;
    result.trajectory.positions.shape = {1, static_cast<std::int64_t>(request.start.size())};
    result.trajectory.positions.data = request.start;
    return result;
  }

  try {
    result.trajectory = bridge_->plan_cspace(handle_, request);
  } catch (const std::exception & e) {
    result.success = false;
    result.message = std::string("Planning error: ") + e.what();
    return result;
  }
  if (!result.trajectory.success) {
    result.success = false;
    result.message = "Planning failed: no solution found (plan_cspace returned None)";
    return result;
  }
  result.success = true;
  result.message = "Trajectory planned successfully";
  return result;
}

}  // namespace curobo_core

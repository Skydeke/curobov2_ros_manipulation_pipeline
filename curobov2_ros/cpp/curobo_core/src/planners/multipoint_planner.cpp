// SPDX-License-Identifier: Apache-2.0
// SetPlanner.MULTIPOINT (4) / "Cartesian Waypoint": one solve through an
// ordered tool-pose path.
#include "curobo_core/planners/multipoint_planner.hpp"

#include <rclcpp/rclcpp.hpp>

namespace curobo_core
{

bool MultipointPlanner::has_ordered_waypoints(
  const curobo_runtime::PlanRequest & request)
{
  return !request.waypoints.data.empty() ||
         !request.tool_waypoints.data.empty();
}

PlannerResult MultipointPlanner::plan(const curobo_runtime::PlanRequest & request)
{
  PlannerResult result;
  result.planner_type = "multipoint";

  if (!has_ordered_waypoints(request)) {
    RCLCPP_WARN(
      rclcpp::get_logger("multipoint_planner"),
      "Cartesian Waypoint: request has no ordered waypoint path");
    result.success = false;
    result.message =
      "Cartesian Waypoint Motion Generation: request has no ordered waypoint "
      "path (set Goalset.waypoints or Goalset.tool_waypoints)";
    return result;
  }
  if (!request.waypoints.data.empty() && !request.tool_waypoints.data.empty()) {
    result.success = false;
    result.message =
      "Cartesian Waypoint Motion Generation: waypoints and tool_waypoints are "
      "mutually exclusive";
    return result;
  }
  if (!request.joint_target.empty()) {
    result.success = false;
    result.message =
      "Cartesian Waypoint Motion Generation: a waypoint path cannot mix with "
      "target_joint_positions";
    return result;
  }

  // Same bridge call as ClassicPlanner::_plan_trajectory; the bridge's
  // curobo_python_runtime.planning.plan_pose dispatches to plan_pose_waypoints because
  // the ordered fields are present.
  result = ClassicPlanner::plan(request);
  result.planner_type = "multipoint";
  return result;
}

}  // namespace curobo_core

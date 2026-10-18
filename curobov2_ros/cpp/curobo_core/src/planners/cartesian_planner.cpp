// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/planners/cartesian_planner.py (min_fraction machinery
// removed - see the header and docs/port_progress.md round 31/32).
#include "curobo_core/planners/cartesian_planner.hpp"

namespace curobo_core
{

// PORT: cartesian_planner.CartesianPlanner._is_cartesian_request
bool CartesianPlanner::is_cartesian_request(const curobo_runtime::PlanRequest & request)
{
  // PORT: _is_cartesian_request - ONE goalset holding >= 1 pose, no joint
  // target. The comparison is on the GOALSET count (the request's first
  // segment's poses are what `goalset` carries); requiring exactly one POSE
  // wrongly refused MTC CartesianPath legs, which send several candidate goal
  // poses.
  return request.goalset_count == 1 && !request.goalset.data.empty() &&
         request.joint_target.empty();
}

// PORT: cartesian_planner.CartesianPlanner._plan_trajectory
PlannerResult CartesianPlanner::plan(const curobo_runtime::PlanRequest & request)
{
  if (!is_cartesian_request(request)) {
    PlannerResult result;
    result.planner_type = "cartesian";
    result.success = false;
    result.message = "Planning failed: no solution found (plan_pose returned None)";
    return result;
  }
  PlannerResult result = ClassicPlanner::plan(request);
  if (!result.success) {
    return result;
  }
  // PORT: cartesian_planner does not set _considered_rows (unlike
  // ClassicPlanner/JointSpacePlanner), so the node reports none for a cartesian
  // plan - clear the rows inherited from ClassicPlanner::plan.
  result.trajectory.considered.clear();
  return result;
}

}  // namespace curobo_core

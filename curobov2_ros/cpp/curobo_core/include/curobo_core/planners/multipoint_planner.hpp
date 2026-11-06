// SPDX-License-Identifier: Apache-2.0
// Ordered Cartesian waypoint planner (SetPlanner.MULTIPOINT = 4).
//
// Plans ONE trajectory through an ordered list of tool poses. The path comes
// from Goalset.waypoints (single tool) or Goalset.tool_waypoints
// (waypoint-major, one pose per tool per waypoint), not from the candidate
// `poses` goalset: every waypoint is visited in order. The solve itself is the
// Python/native bridge's plan_pose waypoint path, reached through the same
// bridge_->plan_pose call the classic planner uses.
#ifndef CUROBO_CORE__PLANNERS__MULTIPOINT_PLANNER_HPP_
#define CUROBO_CORE__PLANNERS__MULTIPOINT_PLANNER_HPP_

#include "curobo_core/planners/classic_planner.hpp"

namespace curobo_core
{

class MultipointPlanner : public ClassicPlanner
{
public:
  using ClassicPlanner::ClassicPlanner;

  std::string get_planner_name() const override
  {
    return "Cartesian Waypoint Motion Generation";
  }

  // The ordered path fields are the planner's contract: refuse a request that
  // has only legacy candidate poses / joint targets instead of silently
  // planning a single goal.
  static bool has_ordered_waypoints(const curobo_runtime::PlanRequest & request);

  PlannerResult plan(const curobo_runtime::PlanRequest & request) override;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__PLANNERS__MULTIPOINT_PLANNER_HPP_

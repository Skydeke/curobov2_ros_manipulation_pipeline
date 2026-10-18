// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/planners/classic_planner.py
#ifndef CUROBO_CORE__PLANNERS__CLASSIC_PLANNER_HPP_
#define CUROBO_CORE__PLANNERS__CLASSIC_PLANNER_HPP_

#include "curobo_core/planners/trajectory_planner.hpp"

namespace curobo_core
{

class ClassicPlanner : public TrajectoryPlanner
{
public:
  using TrajectoryPlanner::TrajectoryPlanner;

  // PORT: classic_planner.ClassicPlanner.get_planner_name
  std::string get_planner_name() const override {return "Classic Motion Generation";}
  ExecutionMode execution_mode() const override {return ExecutionMode::kOpenLoop;}

  // PORT: classic_planner.ClassicPlanner._plan_trajectory (plan_pose)
  PlannerResult plan(const curobo_runtime::PlanRequest & request) override;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__PLANNERS__CLASSIC_PLANNER_HPP_

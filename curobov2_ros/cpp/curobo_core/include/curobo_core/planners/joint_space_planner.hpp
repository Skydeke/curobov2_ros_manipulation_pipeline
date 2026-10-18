// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/planners/joint_space_planner.py
#ifndef CUROBO_CORE__PLANNERS__JOINT_SPACE_PLANNER_HPP_
#define CUROBO_CORE__PLANNERS__JOINT_SPACE_PLANNER_HPP_

#include "curobo_core/planners/trajectory_planner.hpp"

namespace curobo_core
{

class JointSpacePlanner : public TrajectoryPlanner
{
public:
  using TrajectoryPlanner::TrajectoryPlanner;

  // PORT: joint_space_planner.JointSpacePlanner.get_planner_name
  std::string get_planner_name() const override {return "Joint Space Motion Generation";}
  ExecutionMode execution_mode() const override {return ExecutionMode::kOpenLoop;}

  // PORT: joint_space_planner.JointSpacePlanner._plan_trajectory (plan_cspace,
  // with the zero-delta guard).
  PlannerResult plan(const curobo_runtime::PlanRequest & request) override;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__PLANNERS__JOINT_SPACE_PLANNER_HPP_

// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/planners/cartesian_planner.py
//
// The cuRobo port of MoveIt Task Constructor's solvers::CartesianPath: accepts
// only a single-pose-goalset (Cartesian) request and plans it through the
// shared classic path.
//
// min_fraction is GONE (user decision, round 32). The reference's achieved-
// fraction gate could never run: _tool_positions -> kin.get_active_js /
// compute_kinematics reject cuRobo v2's [batch, 1, horizon, dof] interpolated
// plan, and _achieved_fraction's designed failure mode was "cannot measure ->
// accept the solve" (fraction 1.0). So the knob was inert on both servers, and
// the whole machinery - the measurement, the rejection branch, the
// cartesian_min_fraction request field, step_size/precision's only consumer -
// has been removed rather than carried as dead weight. See
// docs/port_progress.md round 31/32.
#ifndef CUROBO_CORE__PLANNERS__CARTESIAN_PLANNER_HPP_
#define CUROBO_CORE__PLANNERS__CARTESIAN_PLANNER_HPP_

#include "curobo_core/planners/classic_planner.hpp"

namespace curobo_core
{

class CartesianPlanner : public ClassicPlanner
{
public:
  using ClassicPlanner::ClassicPlanner;

  // PORT: cartesian_planner.CartesianPlanner.get_planner_name
  std::string get_planner_name() const override {return "Cartesian Motion Generation";}

  // PORT: _is_cartesian_request
  static bool is_cartesian_request(const curobo_runtime::PlanRequest & request);

  // PORT: _plan_trajectory
  PlannerResult plan(const curobo_runtime::PlanRequest & request) override;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__PLANNERS__CARTESIAN_PLANNER_HPP_

// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/planners/trajectory_planner.py
//
// The planner interface: ExecutionMode, PlannerResult and the TrajectoryPlanner
// abstract base. No GPU (design section 3).
#ifndef CUROBO_CORE__PLANNERS__TRAJECTORY_PLANNER_HPP_
#define CUROBO_CORE__PLANNERS__TRAJECTORY_PLANNER_HPP_

#include <optional>
#include <string>

#include "curobo_runtime/bridge.hpp"

namespace curobo_core
{

// PORT: trajectory_planner.ExecutionMode
enum class ExecutionMode { kOpenLoop, kClosedLoop };

inline const char * execution_mode_value(ExecutionMode mode)
{
  return mode == ExecutionMode::kOpenLoop ? "open_loop" : "closed_loop";
}

// PORT: trajectory_planner.PlannerResult
struct PlannerResult
{
  bool success = false;
  std::string message;
  curobo_runtime::PlanOutcome trajectory;   // solver-shaped result
  std::string planner_type;
};

// PORT: trajectory_planner.TrajectoryPlanner
class TrajectoryPlanner
{
public:
  TrajectoryPlanner(curobo_runtime::IBridge * bridge, curobo_runtime::RobotHandle handle,
    std::vector<std::string> joint_names)
  : bridge_(bridge), handle_(handle), joint_names_(std::move(joint_names)) {}

  virtual ~TrajectoryPlanner() = default;

  virtual std::string get_planner_name() const = 0;
  virtual ExecutionMode execution_mode() const = 0;

  // PORT: TrajectoryPlanner.is_open_loop / is_closed_loop
  bool is_open_loop() const {return execution_mode() == ExecutionMode::kOpenLoop;}
  bool is_closed_loop() const {return execution_mode() == ExecutionMode::kClosedLoop;}

  // PORT: TrajectoryPlanner.plan
  virtual PlannerResult plan(const curobo_runtime::PlanRequest & request) = 0;

protected:
  curobo_runtime::IBridge * bridge_;
  curobo_runtime::RobotHandle handle_;
  std::vector<std::string> joint_names_;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__PLANNERS__TRAJECTORY_PLANNER_HPP_

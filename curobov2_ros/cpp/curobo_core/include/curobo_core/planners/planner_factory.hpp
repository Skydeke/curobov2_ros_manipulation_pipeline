// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/planners/planner_factory.py
//
// PlannerFactory (catalog + creation) and PlannerManager (caching, dynamic
// switching). Pure C++ (no GPU).
#ifndef CUROBO_CORE__PLANNERS__PLANNER_FACTORY_HPP_
#define CUROBO_CORE__PLANNERS__PLANNER_FACTORY_HPP_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "curobo_core/planners/trajectory_planner.hpp"
#include "curobo_runtime/gpu_lanes.hpp"

namespace curobo_core
{

class PlannerFactory
{
public:
  struct CatalogEntry
  {
    const char * key;
    int id;
    const char * display;
  };

  // PORT: PlannerFactory._PLANNER_CATALOG
  static const std::vector<CatalogEntry> & catalog();

  // PORT: PlannerFactory.create_planner (key or alias -> a planner instance).
  // Returns nullptr for a planner this milestone has not ported (mpc/retarget).
  static std::unique_ptr<TrajectoryPlanner> create(
    const std::string & planner_type, curobo_runtime::IBridge * bridge,
    curobo_runtime::RobotHandle handle, std::vector<std::string> joint_names,
    rclcpp::Node * node = nullptr, curobo_runtime::GpuLane * reactive_lane = nullptr);

  // PORT: PlannerFactory.get_available_planners
  static std::vector<std::string> available_planners();
};

// PORT: planner_factory.PlannerManager
class PlannerManager
{
public:
  PlannerManager(
    curobo_runtime::IBridge * bridge, curobo_runtime::RobotHandle handle,
    std::vector<std::string> joint_names, rclcpp::Node * node = nullptr);

  // PORT: PlannerManager.get_planner
  TrajectoryPlanner * get_planner(const std::string & planner_type);
  // PORT: PlannerManager.set_current_planner
  void set_current_planner(const std::string & planner_type);
  TrajectoryPlanner * current() {return current_;}
  std::string current_type() const {return current_type_;}

  // PORT: node.lane_reactive - handed to the reactive planners at build.
  void set_reactive_lane(curobo_runtime::GpuLane * lane) {reactive_lane_ = lane;}

private:
  curobo_runtime::IBridge * bridge_;
  curobo_runtime::RobotHandle handle_;
  std::vector<std::string> joint_names_;
  rclcpp::Node * node_ = nullptr;
  std::map<std::string, std::unique_ptr<TrajectoryPlanner>> planners_;
  TrajectoryPlanner * current_ = nullptr;
  std::string current_type_;
  curobo_runtime::GpuLane * reactive_lane_ = nullptr;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__PLANNERS__PLANNER_FACTORY_HPP_

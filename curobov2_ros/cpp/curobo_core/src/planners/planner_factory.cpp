// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/planners/planner_factory.py.
#include "curobo_core/planners/planner_factory.hpp"

#include <algorithm>

#include "curobo_core/planners/cartesian_planner.hpp"
#include "curobo_core/planners/classic_planner.hpp"
#include "curobo_core/planners/joint_space_planner.hpp"
#include "curobo_core/planners/mpc_planner.hpp"
#include "curobo_core/planners/retarget_controller.hpp"

namespace curobo_core
{

// PORT: PlannerFactory._PLANNER_CATALOG
const std::vector<PlannerFactory::CatalogEntry> & PlannerFactory::catalog()
{
  static const std::vector<CatalogEntry> kCatalog = {
    {"classic", 0, "Classic"},
    {"mpc", 1, "MPC"},
    {"cartesian", 3, "Cartesian"},
    {"joint_space", 5, "Joint Space"},
    {"retarget", 6, "Motion Retargeting"},
  };
  return kCatalog;
}

namespace
{
// PORT: PlannerFactory._PLANNER_REGISTRY aliases
std::string canonical_key(const std::string & planner_type)
{
  std::string key = planner_type;
  std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
  });
  key.erase(std::remove_if(key.begin(), key.end(), [](unsigned char c) {return std::isspace(c);}),
    key.end());
  if (key == "motion_gen") {return "classic";}
  if (key == "cartes" || key == "cartesian_path") {return "cartesian";}
  if (key == "model_predictive_control") {return "mpc";}
  if (key == "motion_retargeting" || key == "teleop") {return "retarget";}
  return key;
}
}  // namespace

// PORT: PlannerFactory.create_planner
std::unique_ptr<TrajectoryPlanner> PlannerFactory::create(
  const std::string & planner_type, curobo_runtime::IBridge * bridge,
  curobo_runtime::RobotHandle handle, std::vector<std::string> joint_names,
  rclcpp::Node * node, curobo_runtime::GpuLane * reactive_lane)
{
  const std::string key = canonical_key(planner_type);
  if (key == "classic") {
    return std::make_unique<ClassicPlanner>(bridge, handle, std::move(joint_names));
  }
  if (key == "cartesian") {
    // PORT: CartesianPlanner subclasses ClassicPlanner (the reference's inert
    // min_fraction gate is removed - see the class header).
    return std::make_unique<CartesianPlanner>(bridge, handle, std::move(joint_names));
  }
  if (key == "joint_space") {
    return std::make_unique<JointSpacePlanner>(bridge, handle, std::move(joint_names));
  }
  if (key == "mpc") {
    auto planner = std::make_unique<MPCController>(
      bridge, handle, std::move(joint_names), node);
    // PORT: reactive_controller.lane_reactive.
    planner->set_lane(reactive_lane);
    return planner;
  }
  if (key == "retarget") {
    auto planner = std::make_unique<RetargetController>(
      bridge, handle, std::move(joint_names), node);
    planner->set_lane(reactive_lane);
    return planner;
  }
  return nullptr;
}

// PORT: PlannerFactory.get_available_planners
std::vector<std::string> PlannerFactory::available_planners()
{
  std::vector<std::string> keys;
  for (const auto & entry : catalog()) {
    keys.push_back(entry.key);
  }
  return keys;
}

PlannerManager::PlannerManager(
  curobo_runtime::IBridge * bridge, curobo_runtime::RobotHandle handle,
  std::vector<std::string> joint_names, rclcpp::Node * node)
: bridge_(bridge), handle_(handle), joint_names_(std::move(joint_names)), node_(node)
{
  set_current_planner("classic");
}

// PORT: PlannerManager.get_planner
TrajectoryPlanner * PlannerManager::get_planner(const std::string & planner_type)
{
  const std::string key = canonical_key(planner_type);
  auto it = planners_.find(key);
  if (it != planners_.end()) {
    return it->second.get();
  }
  auto planner = PlannerFactory::create(
    key, bridge_, handle_, joint_names_, node_, reactive_lane_);
  TrajectoryPlanner * raw = planner.get();
  planners_[key] = std::move(planner);
  return raw;
}

// PORT: PlannerManager.set_current_planner
void PlannerManager::set_current_planner(const std::string & planner_type)
{
  const std::string key = canonical_key(planner_type);
  if (key == current_type_) {
    return;
  }
  TrajectoryPlanner * planner = get_planner(key);
  current_ = planner;
  current_type_ = key;
}

}  // namespace curobo_core

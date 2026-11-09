// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros service surface the benchmark drives:
//   config_wrapper_motion.set_motion_gen_config  -> /update_motion_gen_config
//   robot_model_manager.set_lock_joints           -> /set_joint_locks
//
// These rebuild the shared MotionPlanner (and, for joint locks, the robot model)
// from the node's current parameters.
#ifndef CUROBO_CORE__CORE__CONFIG_SERVICES_HPP_
#define CUROBO_CORE__CORE__CONFIG_SERVICES_HPP_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>

#include "curobov2_ros_interfaces/srv/get_workspace.hpp"
#include "curobov2_ros_interfaces/srv/set_joint_locks.hpp"
#include "curobo_runtime/bridge.hpp"

namespace curobo_core
{

class ConfigServices
{
public:
  ConfigServices(
    rclcpp::Node * node, curobo_runtime::IBridge * bridge,
    curobo_runtime::RobotHandle handle, std::vector<std::string> * joint_names);

  // PORT: ConfigWrapperMotion.set_motion_gen_config (build/rebuild the planner).
  void rebuild_planner();

  // PORT: obstacle_manager.collision_cache (read by the benchmark).
  std::map<std::string, int> collision_cache() const;

  // Frame the Mapper is integrated in (map/odom when TF provides one, else the
  // robot base); reported by get_workspace so the panel knows the box frame.
  void set_integration_frame_provider(std::function<std::string()> cb)
  {integration_frame_provider_ = std::move(cb);}
  // Called when a rebuild's world push FAILED: the obstacle manager must treat
  // the solvers as stale so the next plan rebuilds and re-pushes the perception
  // (voxel) world instead of planning against an empty one.
  void set_world_invalidate(std::function<void()> cb)
  {world_invalidate_ = std::move(cb);}

  // Whether a Mapper actually exists (a camera/laser feeds it). Reported by
  // get_workspace so a UI knows an edit changed collision geometry or only
  // the drawn box.
  bool perception_active() const;

private:
  void handle_update_motion_gen_config(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response);
  void handle_set_joint_locks(
    const std::shared_ptr<curobov2_ros_interfaces::srv::SetJointLocks::Request> request,
    std::shared_ptr<curobov2_ros_interfaces::srv::SetJointLocks::Response> response);
  void handle_get_workspace(
    const std::shared_ptr<curobov2_ros_interfaces::srv::GetWorkspace::Request> request,
    std::shared_ptr<curobov2_ros_interfaces::srv::GetWorkspace::Response> response);

  rclcpp::Node * node_;
  curobo_runtime::IBridge * bridge_;
  curobo_runtime::RobotHandle handle_;
  std::vector<std::string> * joint_names_;
  std::map<std::string, double> locks_;
  int voxel_layers_ = 1;   // PORT: collision_cache["voxel"]["layers"]
  std::function<std::string()> integration_frame_provider_;
  std::function<void()> world_invalidate_;

  rclcpp::Service<curobov2_ros_interfaces::srv::GetWorkspace>::SharedPtr get_workspace_srv_;
  rclcpp::Service<curobov2_ros_interfaces::srv::SetJointLocks>::SharedPtr set_joint_locks_srv_;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__CORE__CONFIG_SERVICES_HPP_

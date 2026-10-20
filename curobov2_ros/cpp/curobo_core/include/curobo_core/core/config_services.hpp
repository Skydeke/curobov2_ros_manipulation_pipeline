// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros service surface the benchmark drives:
//   config_wrapper_motion.set_motion_gen_config  -> /update_motion_gen_config
//   obstacle_manager.set_collision_cache          -> /set_collision_cache
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
#include "curobov2_ros_interfaces/srv/set_collision_cache.hpp"
#include "curobov2_ros_interfaces/srv/set_joint_locks.hpp"
#include "curobov2_ros_interfaces/srv/set_workspace.hpp"
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

  // set_workspace rebuilds the Mapper grid (the node owns that config) and
  // redraws the <node>/mapper_workspace wireframe (RosServiceManager owns that
  // publisher); both are wired in by the node.
  void set_mapper_reconfigure(std::function<void()> cb) {mapper_reconfigure_ = std::move(cb);}
  void set_workspace_republish(std::function<void()> cb) {workspace_republish_ = std::move(cb);}
  // Frame the Mapper is integrated in (map/odom when TF provides one, else the
  // robot base); reported by get/set_workspace so the panel knows the box frame.
  void set_integration_frame_provider(std::function<std::string()> cb)
  {integration_frame_provider_ = std::move(cb);}
  // Called when a rebuild's world push FAILED: the obstacle manager must treat
  // the solvers as stale so the next plan rebuilds and re-pushes the perception
  // (voxel) world instead of planning against an empty one.
  void set_world_invalidate(std::function<void()> cb)
  {world_invalidate_ = std::move(cb);}

  // Whether a Mapper actually exists (a camera/laser feeds it). Reported by
  // get/set_workspace so a UI knows an edit changed collision geometry or only
  // the drawn box.
  bool perception_active() const;

private:
  void handle_set_collision_cache(
    const std::shared_ptr<curobov2_ros_interfaces::srv::SetCollisionCache::Request> request,
    std::shared_ptr<curobov2_ros_interfaces::srv::SetCollisionCache::Response> response);
  void handle_update_motion_gen_config(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response);
  void handle_set_joint_locks(
    const std::shared_ptr<curobov2_ros_interfaces::srv::SetJointLocks::Request> request,
    std::shared_ptr<curobov2_ros_interfaces::srv::SetJointLocks::Response> response);
  void handle_get_workspace(
    const std::shared_ptr<curobov2_ros_interfaces::srv::GetWorkspace::Request> request,
    std::shared_ptr<curobov2_ros_interfaces::srv::GetWorkspace::Response> response);
  void handle_set_workspace(
    const std::shared_ptr<curobov2_ros_interfaces::srv::SetWorkspace::Request> request,
    std::shared_ptr<curobov2_ros_interfaces::srv::SetWorkspace::Response> response);

  rclcpp::Node * node_;
  curobo_runtime::IBridge * bridge_;
  curobo_runtime::RobotHandle handle_;
  std::vector<std::string> * joint_names_;
  std::map<std::string, double> locks_;
  int voxel_layers_ = 1;   // PORT: collision_cache["voxel"]["layers"]
  std::function<void()> mapper_reconfigure_;
  std::function<void()> workspace_republish_;
  std::function<std::string()> integration_frame_provider_;
  std::function<void()> world_invalidate_;

  rclcpp::Service<curobov2_ros_interfaces::srv::SetCollisionCache>::SharedPtr cache_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr motion_gen_srv_;
  rclcpp::Service<curobov2_ros_interfaces::srv::SetJointLocks>::SharedPtr locks_srv_;
  rclcpp::Service<curobov2_ros_interfaces::srv::GetWorkspace>::SharedPtr get_workspace_srv_;
  rclcpp::Service<curobov2_ros_interfaces::srv::SetWorkspace>::SharedPtr set_workspace_srv_;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__CORE__CONFIG_SERVICES_HPP_

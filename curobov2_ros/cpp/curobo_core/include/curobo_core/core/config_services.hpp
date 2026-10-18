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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>

#include "curobov2_ros_interfaces/srv/set_collision_cache.hpp"
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

  rclcpp::Node * node_;
  curobo_runtime::IBridge * bridge_;
  curobo_runtime::RobotHandle handle_;
  std::vector<std::string> * joint_names_;
  std::map<std::string, double> locks_;
  int voxel_layers_ = 1;   // PORT: collision_cache["voxel"]["layers"]

  rclcpp::Service<curobov2_ros_interfaces::srv::SetCollisionCache>::SharedPtr cache_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr motion_gen_srv_;
  rclcpp::Service<curobov2_ros_interfaces::srv::SetJointLocks>::SharedPtr locks_srv_;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__CORE__CONFIG_SERVICES_HPP_

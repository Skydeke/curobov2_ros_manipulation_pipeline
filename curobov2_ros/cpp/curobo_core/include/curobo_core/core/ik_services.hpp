// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/core/ik_services.py
//
// 1:1 C++ port of IKServices: /warmup_ik, /ik and /ik_batch over the shared
// InverseKinematics owned by the bridge. Method names, error strings and
// control flow follow the Python source. The solve is serialised on one lock
// (the Python _solve_lock).
#ifndef CUROBO_CORE__CORE__IK_SERVICES_HPP_
#define CUROBO_CORE__CORE__IK_SERVICES_HPP_

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "curobov2_ros_interfaces/srv/ik.hpp"
#include "curobov2_ros_interfaces/srv/ik_batch.hpp"
#include "curobov2_ros_interfaces/srv/warmup_ik.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "curobo_core/convert.hpp"
#include "curobo_runtime/bridge.hpp"

namespace curobo_core
{

class IKServices
{
public:
  IKServices(
    rclcpp::Node * node, curobo_runtime::IBridge * bridge,
    curobo_runtime::RobotHandle handle, std::string base_link, int num_ik_seeds = 32);

  // PORT: ik_services.IKServices.warmup
  std::string warmup(int batch_size, int num_seeds = -1);

  // PORT: ik_services.IKServices.solve_poses (programmatic batch IK, reused by
  // ReachabilityServices).
  bool solve_poses(
    const std::vector<Pose7> & poses, std::vector<std::vector<float>> & positions,
    std::vector<std::uint8_t> & flags, std::vector<std::string> & joint_names);

  // PORT: ik_services.IKServices.update_world / rebuild (delegate to the bridge
  // at M3; the scene push is a bridge concern).
  void update_world() {}
  void rebuild() {}

  bool ready() const {return ready_;}
  int batch_size() const {return ik_batch_size_;}
  int num_seeds() const {return ik_num_seeds_;}

  // PORT: ik_services.IKServices._resolve_num_seeds
  int resolve_num_seeds() const {return num_ik_seeds_;}

  // PORT: config_wrapper_ik.ConfigWrapperIK.needs_rebuild
  bool needs_rebuild(int batch_size, int num_seeds) const;

private:
  bool solve(
    const std::vector<Pose7> & poses, curobo_runtime::IkOutcome & outcome);

  void handle_warmup_ik(
    const std::shared_ptr<curobov2_ros_interfaces::srv::WarmupIK::Request> request,
    std::shared_ptr<curobov2_ros_interfaces::srv::WarmupIK::Response> response);
  void handle_ik(
    const std::shared_ptr<curobov2_ros_interfaces::srv::Ik::Request> request,
    std::shared_ptr<curobov2_ros_interfaces::srv::Ik::Response> response);
  void handle_ik_batch(
    const std::shared_ptr<curobov2_ros_interfaces::srv::IkBatch::Request> request,
    std::shared_ptr<curobov2_ros_interfaces::srv::IkBatch::Response> response);
  // PORT: config_wrapper_ik.ConfigWrapperIK.set_ik_config
  void handle_update_ik_config(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response);

  rclcpp::Node * node_;
  curobo_runtime::IBridge * bridge_;
  curobo_runtime::RobotHandle handle_;
  std::string base_link_;
  int num_ik_seeds_;
  bool ready_ = false;
  int ik_batch_size_ = 0;
  int ik_num_seeds_ = 0;
  std::mutex solve_lock_;

  rclcpp::Service<curobov2_ros_interfaces::srv::WarmupIK>::SharedPtr warmup_srv_;
  rclcpp::Service<curobov2_ros_interfaces::srv::Ik>::SharedPtr ik_srv_;
  rclcpp::Service<curobov2_ros_interfaces::srv::IkBatch>::SharedPtr ik_batch_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr update_ik_config_srv_;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__CORE__IK_SERVICES_HPP_

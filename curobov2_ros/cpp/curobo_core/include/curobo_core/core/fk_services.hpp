// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/core/fk_services.py
//
// 1:1 C++ port of FKServices: the /warmup_fk, /fk and /fk_batch services over
// the shared GPU kinematics held by the bridge. Method names, log/error strings
// and control flow follow the Python source.
//
// Construction note: the Python ctor takes `config_wrapper` and reads
// `kin_model`, `robot_model_manager.robot_cfg` and `obstacle_manager.collision_cache`
// from it. Those collaborators are not ported yet, so this ctor takes the
// shared bridge handle and the collision-cache sizes directly; when
// ConfigWrapper lands the signature becomes (node, config_wrapper). The bodies
// are unchanged.
#ifndef CUROBO_CORE__CORE__FK_SERVICES_HPP_
#define CUROBO_CORE__CORE__FK_SERVICES_HPP_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "curobov2_ros_interfaces/srv/fk.hpp"
#include "curobov2_ros_interfaces/srv/fk_batch.hpp"
#include "curobov2_ros_interfaces/srv/warmup_fk.hpp"
#include "curobo_runtime/bridge.hpp"

namespace curobo_core
{

class FKServices
{
public:
  FKServices(
    rclcpp::Node * node, curobo_runtime::IBridge * bridge,
    curobo_runtime::RobotHandle handle,
    std::vector<std::string> joint_names,
    int cache_cuboid = 32, int cache_mesh = 4);

  // PORT: fk_services.FKServices.warmup
  std::string warmup(int batch_size);

  // PORT: fk_services.FKServices._compute_poses (exposed for the service paths
  // and tests): joint rows -> tool poses in bridge order.
  bool compute_poses(
    const std::vector<std::vector<double>> & qs,
    std::vector<curobo_runtime::TensorF32> & poses) const;

  // PORT: fk_services.FKServices._validate
  std::vector<std::uint8_t> validate(const std::vector<std::vector<double>> & qs) const;

  // PORT: fk_services.FKServices._positions_for_model
  std::vector<double> positions_for_model(
    const std::vector<double> & positions, const std::vector<std::string> & names) const;

  bool initialized() const {return initialized_;}
  int batch() const {return fk_batch_;}
  const std::vector<std::string> & joint_names() const {return joint_names_;}

  // PORT: fk_services.FKServices.update_world / rebuild (no model to rebuild at
  // M2 - the bridge owns the shared kinematics).
  void update_world() {}
  void rebuild() {}

private:
  void init(int batch_size);   // PORT: _init

  void handle_warmup_fk(
    const std::shared_ptr<curobov2_ros_interfaces::srv::WarmupFK::Request> request,
    std::shared_ptr<curobov2_ros_interfaces::srv::WarmupFK::Response> response);
  void handle_fk(
    const std::shared_ptr<curobov2_ros_interfaces::srv::Fk::Request> request,
    std::shared_ptr<curobov2_ros_interfaces::srv::Fk::Response> response);
  void handle_fk_batch(
    const std::shared_ptr<curobov2_ros_interfaces::srv::FkBatch::Request> request,
    std::shared_ptr<curobov2_ros_interfaces::srv::FkBatch::Response> response);

  rclcpp::Node * node_;
  curobo_runtime::IBridge * bridge_;
  curobo_runtime::RobotHandle handle_;
  std::vector<std::string> joint_names_;
  int cache_cuboid_;
  int cache_mesh_;
  bool initialized_ = false;
  int fk_batch_ = 0;

  rclcpp::Service<curobov2_ros_interfaces::srv::WarmupFK>::SharedPtr warmup_srv_;
  rclcpp::Service<curobov2_ros_interfaces::srv::Fk>::SharedPtr fk_srv_;
  rclcpp::Service<curobov2_ros_interfaces::srv::FkBatch>::SharedPtr fk_batch_srv_;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__CORE__FK_SERVICES_HPP_

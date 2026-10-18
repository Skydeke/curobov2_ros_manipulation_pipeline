// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/core/attachment_services.py
//
// 1:1 C++ port of AttachmentServices: /attach_object and /detach_object over the
// bridge's batched attach/detach. Message strings and control flow follow the
// Python source.
//
// Construction note: as with FKServices/IKServices, the Python ctor's
// `config_wrapper` collaborator is replaced here by the shared bridge handle plus
// the scene bookkeeping (ObstacleManager) and a `current_pose` provider (the
// node's _resolve_start_state fallback) until those collaborators are ported.
#ifndef CUROBO_CORE__CORE__ATTACHMENT_SERVICES_HPP_
#define CUROBO_CORE__CORE__ATTACHMENT_SERVICES_HPP_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>

#include "curobov2_ros_interfaces/srv/attach_object.hpp"
#include "curobo_core/core/obstacle_manager.hpp"
#include "curobo_runtime/bridge.hpp"

namespace curobo_core
{

class AttachmentServices
{
public:
  // PORT: ATTACH_LINK (attachment_services.ATTACH_LINK)
  static constexpr const char * ATTACH_LINK = "attached_object";

  AttachmentServices(
    rclcpp::Node * node, curobo_runtime::IBridge * bridge,
    curobo_runtime::RobotHandle handle, ObstacleManager * obstacle_manager,
    std::function<std::vector<float>()> current_pose);

  // PORT: attachment_services.attach
  void attach(const std::string & object_name, const std::vector<float> & q);
  // PORT: attachment_services.detach
  std::string detach();

  const std::string & attached_name() const {return attached_name_;}

  // PORT: attachment_services._joint_state_from_message (positions only; the
  // reorder-into-cspace step needs the kinematics joint names, which the bridge
  // owns, so names are passed through here).
  std::vector<float> joint_state_from_message(
    const std::vector<double> & positions, const std::vector<std::string> & names) const;

private:
  void handle_attach_object(
    const std::shared_ptr<curobov2_ros_interfaces::srv::AttachObject::Request> request,
    std::shared_ptr<curobov2_ros_interfaces::srv::AttachObject::Response> response);
  void handle_detach_object(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response);

  rclcpp::Node * node_;
  curobo_runtime::IBridge * bridge_;
  curobo_runtime::RobotHandle handle_;
  ObstacleManager * obstacle_manager_;
  std::function<std::vector<float>()> current_pose_;
  std::string attached_name_;

  rclcpp::Service<curobov2_ros_interfaces::srv::AttachObject>::SharedPtr attach_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr detach_srv_;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__CORE__ATTACHMENT_SERVICES_HPP_

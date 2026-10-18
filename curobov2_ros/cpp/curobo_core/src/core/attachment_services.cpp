// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/core/attachment_services.py.
#include "curobo_core/core/attachment_services.hpp"
#include "curobo_runtime/runtime.hpp"

#include <map>
#include <stdexcept>

#include <std_srvs/srv/trigger.hpp>

namespace curobo_core
{

AttachmentServices::AttachmentServices(
  rclcpp::Node * node, curobo_runtime::IBridge * bridge,
  curobo_runtime::RobotHandle handle, ObstacleManager * obstacle_manager,
  std::function<std::vector<float>()> current_pose)
: node_(node), bridge_(bridge), handle_(handle), obstacle_manager_(obstacle_manager),
  current_pose_(std::move(current_pose))
{
  const std::string name = node_->get_name();
  attach_srv_ = node_->create_service<curobov2_ros_interfaces::srv::AttachObject>(
    name + "/attach_object",
    std::bind(
      &AttachmentServices::handle_attach_object, this,
      std::placeholders::_1, std::placeholders::_2));
  detach_srv_ = node_->create_service<std_srvs::srv::Trigger>(
    name + "/detach_object",
    std::bind(
      &AttachmentServices::handle_detach_object, this,
      std::placeholders::_1, std::placeholders::_2));
}

// PORT: attachment_services.attach (the cuRobo work is one batched bridge call)
void AttachmentServices::attach(const std::string & object_name, const std::vector<float> & q)
{
  // PORT: attachment_services.attach's `with self.node.gpu_lock` - the sphere
  // fit, the obstacle exclusion, the voxel clear + ESDF refresh and the world
  // push are all GPU work, so they must not overlap a CUDA-graph capture.
  auto gpu_guard = curobo_runtime::Runtime::instance().barrier().exclusive();
  bridge_->attach_object(handle_, object_name, q);
  attached_name_ = object_name;
  if (obstacle_manager_ != nullptr) {
    // PORT: attachment_services.attach - exclude_obstacle() (the static copy is
    // dropped from the voxel-map rasterization, still registered in the solver
    // scenes so the name-based disable applies) + the epilogue push: the
    // reference pushes unconditionally (inside _clear_attached_voxels on the
    // ESDF path, in the fallback branch otherwise), so mark the ESDF gate here
    // regardless of whether an ESDF was produced.
    obstacle_manager_->set_attached(object_name);
    obstacle_manager_->mark_world_pushed();
  }
  RCLCPP_INFO(
    node_->get_logger(), "Attached '%s' to link '%s'", object_name.c_str(), ATTACH_LINK);
}

// PORT: attachment_services.detach
std::string AttachmentServices::detach()
{
  // PORT: detach's own `with self.node.gpu_lock` around
  // am.detach(...) + include_obstacle() + update_all_solvers_world().
  auto gpu_guard = curobo_runtime::Runtime::instance().barrier().exclusive();
  const std::string released = bridge_->detach_object(handle_);
  attached_name_.clear();
  if (obstacle_manager_ != nullptr) {
    obstacle_manager_->clear_attached();
    obstacle_manager_->mark_world_pushed();
  }
  RCLCPP_INFO(node_->get_logger(), "Detached '%s' from '%s'", released.c_str(), ATTACH_LINK);
  return released;
}

// PORT: attachment_services._joint_state_from_message
std::vector<float> AttachmentServices::joint_state_from_message(
  const std::vector<double> & positions, const std::vector<std::string> & names) const
{
  std::vector<float> out;
  out.reserve(positions.size());
  for (double v : positions) {
    out.push_back(static_cast<float>(v));
  }
  // PORT: a NAMED message is re-ordered into the planner's cspace order (a
  // nameless one is assumed already in that order). Without the reorder a
  // caller whose joint order differs from the robot YAML's would fit the
  // payload at a permuted configuration - i.e. attach the geometry in the
  // wrong place. Best-effort: any failure keeps the caller's order.
  if (!names.empty() && names.size() == positions.size()) {
    try {
      const auto limits = bridge_->joint_limits(handle_);
      const std::vector<std::string> & order = limits.joint_names;
      if (!order.empty()) {
        std::map<std::string, float> by_name;
        for (std::size_t i = 0; i < names.size(); ++i) {
          by_name[names[i]] = out[i];
        }
        bool complete = true;
        for (const auto & n : order) {
          if (by_name.find(n) == by_name.end()) {
            complete = false;
            break;
          }
        }
        if (complete) {
          std::vector<float> reordered;
          reordered.reserve(order.size());
          for (const auto & n : order) {
            reordered.push_back(by_name[n]);
          }
          return reordered;
        }
      }
    } catch (const std::exception &) {
      // best effort - fall through with the caller's order
    }
  }
  return out;
}

// PORT: attachment_services._attach_object_callback
void AttachmentServices::handle_attach_object(
  const std::shared_ptr<curobov2_ros_interfaces::srv::AttachObject::Request> request,
  std::shared_ptr<curobov2_ros_interfaces::srv::AttachObject::Response> response)
{
  try {
    std::vector<float> fit_state = joint_state_from_message(
      request->joint_state.position, request->joint_state.name);
    if (fit_state.empty()) {
      // PORT: the fallback `node._resolve_start_state(None)` -> the robot's
      // current joint configuration.
      fit_state = current_pose_ ? current_pose_() : std::vector<float>{};
    }
    attach(request->object_name, fit_state);
    response->success = true;
    response->message = "Attached '" + request->object_name + "'";
  } catch (const std::exception & e) {
    response->success = false;
    response->message = std::string("Attach error: ") + e.what();
    RCLCPP_ERROR(node_->get_logger(), "%s", response->message.c_str());
  }
}

// PORT: attachment_services._detach_object_callback
void AttachmentServices::handle_detach_object(
  const std::shared_ptr<std_srvs::srv::Trigger::Request>,
  std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
  try {
    const std::string released = detach();
    response->success = true;
    response->message = released.empty() ? "Nothing attached" : ("Detached '" + released + "'");
  } catch (const std::exception & e) {
    response->success = false;
    response->message = std::string("Detach error: ") + e.what();
  }
}

}  // namespace curobo_core

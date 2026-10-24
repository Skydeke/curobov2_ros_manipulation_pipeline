// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/core/robot_segmentation.py.
#include "curobo_core/core/robot_segmentation.hpp"
#include "curobo_core/cameras/camera_cfg.hpp"

#include <fstream>
#include <sstream>

namespace curobo_core
{

namespace
{
std::string json_escape(const std::string & value)
{
  std::string out;
  for (char c : value) {
    if (c == '"' || c == '\\') {
      out += '\\';
    }
    out += c;
  }
  return out;
}
}  // namespace

RobotSegmentation::RobotSegmentation(
  rclcpp::Node * node, curobo_runtime::IBridge * bridge,
  curobo_runtime::RobotHandle handle,
  const std::vector<PerceptionCameraCfg> & camera_cfgs,
  std::function<std::vector<double>(std::int64_t)> joint_pose_at,
  rclcpp::CallbackGroup::SharedPtr callback_group)
: node_(node), bridge_(bridge), handle_(handle)
{
  base_frame_ = integration_frame(node_);
  if (!node_->has_parameter("robot_segmentation_mask_margin")) {
    node_->declare_parameter("robot_segmentation_mask_margin", 0.0);
  }
  if (!node_->has_parameter("robot_segmentation_distance_threshold")) {
    node_->declare_parameter("robot_segmentation_distance_threshold", 0.05);
  }
  mask_margin_ = node_->get_parameter("robot_segmentation_mask_margin").as_double();
  distance_threshold_ =
    node_->get_parameter("robot_segmentation_distance_threshold").as_double();

  for (const auto & cfg : camera_cfgs) {
    if (!cfg.for_segmentation() || cfg.depth_topic.empty()) {
      continue;
    }
    auto strategy = std::make_shared<RobotSegmentationCameraStrategy>(
      node_, cfg.name, cfg.depth_topic, cfg.camera_info_topic, cfg.frame_id,
      cfg.intrinsics, cfg.extrinsics, bridge_, handle_, joint_pose_at,
      [this]() {return masks_json();}, callback_group, cfg.frame_rate_hz);
    streams_.push_back(strategy);
    RCLCPP_INFO(
      node_->get_logger(),
      "RobotSegmentation stream for camera '%s' (index %d): %s -> %s", cfg.name.c_str(),
      cfg.camera_index, cfg.depth_topic.c_str(), strategy->output_topic().c_str());
  }
  // These reach the bridge through the robot-model config, but log them here too:
  // they are the knobs that decide how much of the depth image is deleted as
  // "robot", and a silently-ignored threshold is hard to notice otherwise.
  RCLCPP_INFO(
    node_->get_logger(),
    "Robot segmentation masking: distance_threshold=%.4f m, mask_margin=%.4f m (%zu stream%s)",
    distance_threshold_, mask_margin_, streams_.size(), streams_.size() == 1 ? "" : "s");
}

void RobotSegmentation::destroy()
{
  for (auto & stream : streams_) {
    stream->destroy();
  }
  streams_.clear();
}

// PORT: RobotSegmentation.masks_json (shared masks dict, by reference in Python)
std::string RobotSegmentation::masks_json() const
{
  std::ostringstream out;
  out << "[";
  bool first = true;
  for (const auto & [name, mask] : masks_) {
    out << (first ? "" : ",") << mask;
    first = false;
  }
  out << "]";
  return out.str();
}

bool RobotSegmentation::mesh_file_exists(const std::string & path) const
{
  std::ifstream file(path);
  return file.good();
}

// PORT: RobotSegmentation.set_mask_callback
void RobotSegmentation::handle_set_mask(
  const std::shared_ptr<curobov2_ros_interfaces::srv::SetMask::Request> request,
  std::shared_ptr<curobov2_ros_interfaces::srv::SetMask::Response> response)
{
  const int type = static_cast<int>(request->type);
  const std::array<double, 3> dims{
    request->dimensions.x, request->dimensions.y, request->dimensions.z};
  if (type == 4) {
    if (!mesh_file_exists(request->mesh_file_path)) {
      response->success = false;
      response->message = "Mesh file not found: " + request->mesh_file_path;
      return;
    }
  } else if (dims[0] <= 0.0 || dims[1] <= 0.0 || dims[2] <= 0.0) {
    response->success = false;
    response->message = "Mask dimensions must be positive";
    return;
  }

  std::ostringstream mask;
  mask << "{\"name\":\"" << json_escape(request->name) << "\",\"type\":" << type
       << ",\"frame_id\":\"" << json_escape(request->frame_id) << "\""
       << ",\"mesh_file_path\":\"" << json_escape(request->mesh_file_path) << "\""
       << ",\"pos\":[" << request->pose.position.x << "," << request->pose.position.y << ","
       << request->pose.position.z << "]"
       << ",\"quat\":[" << request->pose.orientation.x << "," << request->pose.orientation.y
       << "," << request->pose.orientation.z << "," << request->pose.orientation.w << "]"
       << ",\"dims\":[" << dims[0] << "," << dims[1] << "," << dims[2] << "]}";
  masks_[request->name] = mask.str();

  response->success = true;
  const std::string frame = request->frame_id.empty() ? base_frame_ : request->frame_id;
  response->message = "Mask '" + request->name + "' set (type=" + std::to_string(type) +
    ", frame='" + frame + "')";
}

// PORT: RobotSegmentation.remove_mask_callback
void RobotSegmentation::handle_remove_mask(
  const std::shared_ptr<curobov2_ros_interfaces::srv::RemoveObject::Request> request,
  std::shared_ptr<curobov2_ros_interfaces::srv::RemoveObject::Response> response)
{
  response->success = true;
  if (masks_.erase(request->name) == 0u) {
    response->message = "Mask '" + request->name + "' not present (nothing to remove)";
    return;
  }
  response->message = "Mask '" + request->name + "' removed";
}

}  // namespace curobo_core

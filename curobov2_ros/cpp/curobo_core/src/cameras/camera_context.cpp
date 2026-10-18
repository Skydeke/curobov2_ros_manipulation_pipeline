// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/cameras/camera_context.py.
#include "curobo_core/cameras/camera_context.hpp"

namespace curobo_core
{

CameraContext::CameraContext(rclcpp::Node * node) : node_(node) {}

// PORT: camera_context.add_camera
void CameraContext::add_camera(
  const std::string & name, std::shared_ptr<CameraStrategy> camera, double frame_rate_hz)
{
  cameras_[name] = std::move(camera);
  camera_frame_rates_[name] = frame_rate_hz;
  RCLCPP_INFO(node_->get_logger(), "Added camera strategy '%s'", name.c_str());
}

// PORT: camera_context.remove_camera
void CameraContext::remove_camera(const std::string & name)
{
  if (cameras_.erase(name) == 0u) {
    RCLCPP_WARN(node_->get_logger(), "Camera '%s' not found", name.c_str());
    return;
  }
  camera_frame_rates_.erase(name);
  RCLCPP_INFO(node_->get_logger(), "Removed camera strategy '%s'", name.c_str());
}

// PORT: camera_context.set_camera_update_callback
void CameraContext::set_camera_update_callback(
  std::function<void(const sensor_msgs::msg::Image &)> callback)
{
  for (auto & [name, camera] : cameras_) {
    camera->set_update_callback(callback);
  }
}

// PORT: camera_context.get_total_frame_rate_hz
double CameraContext::get_total_frame_rate_hz() const
{
  double total = 0.0;
  for (const auto & [name, rate] : camera_frame_rates_) {
    total += rate;
  }
  return total;
}

// PORT: camera_context.get_camera_names
std::vector<std::string> CameraContext::get_camera_names() const
{
  std::vector<std::string> names;
  names.reserve(cameras_.size());
  for (const auto & [name, camera] : cameras_) {
    names.push_back(name);
  }
  return names;
}

CameraStrategy * CameraContext::get(const std::string & name) const
{
  auto it = cameras_.find(name);
  return it == cameras_.end() ? nullptr : it->second.get();
}

}  // namespace curobo_core

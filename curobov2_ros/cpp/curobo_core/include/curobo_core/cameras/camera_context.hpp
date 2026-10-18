// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/cameras/camera_context.py
//
// Strategy-pattern registry of camera strategies keyed by name, plus the
// per-camera frame-rate aggregate used to normalise TSDF decay.
#ifndef CUROBO_CORE__CAMERAS__CAMERA_CONTEXT_HPP_
#define CUROBO_CORE__CAMERAS__CAMERA_CONTEXT_HPP_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "curobo_core/cameras/camera_strategy.hpp"

namespace curobo_core
{

class CameraContext
{
public:
  explicit CameraContext(rclcpp::Node * node);

  // PORT: camera_context.add_camera (registers an already-built strategy).
  void add_camera(const std::string & name, std::shared_ptr<CameraStrategy> camera,
    double frame_rate_hz);
  // PORT: camera_context.remove_camera
  void remove_camera(const std::string & name);
  // PORT: camera_context.set_camera_update_callback
  void set_camera_update_callback(std::function<void(const sensor_msgs::msg::Image &)> callback);
  // PORT: camera_context.get_total_frame_rate_hz
  double get_total_frame_rate_hz() const;
  // PORT: camera_context.get_camera_names
  std::vector<std::string> get_camera_names() const;

  CameraStrategy * get(const std::string & name) const;

private:
  rclcpp::Node * node_;
  std::map<std::string, std::shared_ptr<CameraStrategy>> cameras_;
  std::map<std::string, double> camera_frame_rates_;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__CAMERAS__CAMERA_CONTEXT_HPP_

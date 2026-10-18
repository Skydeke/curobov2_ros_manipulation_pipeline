// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/cameras/camera_strategy.py
//
// Abstract base for camera strategies: owns the TF buffer/listener and parses
// the configured intrinsics/extrinsics. The Python stores torch tensors / a
// cuRobo Pose; C++ keeps plain host data (a 3x3 K and a [x,y,z,qw,qx,qy,qz]
// pose) — the bridge builds the cuRobo objects.
#ifndef CUROBO_CORE__CAMERAS__CAMERA_STRATEGY_HPP_
#define CUROBO_CORE__CAMERAS__CAMERA_STRATEGY_HPP_

#include <array>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace curobo_core
{

// cuRobo pose order [x, y, z, qw, qx, qy, qz].
using Pose7d = std::array<double, 7>;

class CameraStrategy
{
public:
  CameraStrategy(
    rclcpp::Node * node, const std::string & camera_name, const std::string & topic,
    const std::string & camera_info, const std::string & frame_id,
    const std::optional<std::vector<double>> & intrinsics = std::nullopt,
    const std::optional<std::vector<double>> & extrinsics = std::nullopt);
  virtual ~CameraStrategy() = default;

  // PORT: camera_strategy.set_update_callback / _update_callback
  void set_update_callback(std::function<void(const sensor_msgs::msg::Image &)> callback)
  {
    update_callback_ = std::move(callback);
  }

  const std::string & name() const {return name_;}
  // PORT: the intrinsics may arrive ASYNCHRONOUSLY from the camera_info topic
  // (the Python's wait_for_message fallback), so every read locks.
  bool has_intrinsics() const
  {
    std::lock_guard<std::mutex> lk(intrinsics_mutex_);
    return intrinsics_.has_value();
  }
  // 3x3 row-major K (copy; safe against a concurrent camera_info update).
  std::optional<std::array<double, 9>> intrinsics_copy() const
  {
    std::lock_guard<std::mutex> lk(intrinsics_mutex_);
    return intrinsics_;
  }
  bool has_static_pose() const {return static_pose_.has_value();}
  const Pose7d & static_pose() const {return static_pose_.value();}
  const std::string & frame_id() const {return frame_id_;}

protected:
  // PORT: _parse_intrinsics (9-element row-major K or fx,fy,cx,cy).
  std::optional<std::array<double, 9>> parse_intrinsics(
    const std::optional<std::vector<double>> & intrinsics);
  // PORT: _parse_extrinsics (7-element pose or 4x4 row-major matrix).
  std::optional<Pose7d> parse_extrinsics(
    const std::optional<std::vector<double>> & extrinsics);
  // PORT: camera_depth_map_strategy's "waiting for camera info" fallback.
  void on_camera_info(const sensor_msgs::msg::CameraInfo & msg);

  rclcpp::Node * node_;
  std::string name_;
  std::string topic_;
  std::string camera_info_topic_;
  std::string frame_id_;
  std::optional<std::array<double, 9>> intrinsics_;
  mutable std::mutex intrinsics_mutex_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  std::optional<Pose7d> static_pose_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::function<void(const sensor_msgs::msg::Image &)> update_callback_;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__CAMERAS__CAMERA_STRATEGY_HPP_

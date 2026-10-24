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
#include <chrono>
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
    const std::optional<std::vector<double>> & extrinsics = std::nullopt,
    double max_update_rate_hz = 0.0);
  virtual ~CameraStrategy() = default;

  // PORT: MoveIt's occupancy-map monitor `max_update_rate`. Returns false for a
  // frame that arrived sooner than 1/max_update_rate_hz after the last ACCEPTED
  // one, so the caller drops it before doing any per-frame work (decode, TF
  // lookup, FK, segmentation, Mapper integration). `max_update_rate_hz <= 0`
  // accepts every frame (the historical behaviour).
  //
  // Without this the configured `camera_frame_rate_hz` only sized the Mapper's
  // decay factor: the strategies processed every frame the driver published
  // (measured 7 Hz from a 3 Hz-configured d435), and each frame costs a TF
  // lookup plus a GPU segmentation pass and a TSDF/ESDF integration.
  bool allow_frame();

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
  // The frame the depth pixels are expressed in. An explicit `camera_frame`
  // wins (a Gazebo RGBD camera emits x-forward pixels under its LINK frame,
  // which is right for the point cloud but wrong for a z-forward projection, so
  // such setups must name the optical frame); otherwise the driver's own
  // camera_info header frame is used. Depth is projected with a z-forward
  // (optical) convention, so the resolved frame must be the camera's optical
  // frame.
  std::string frame_for_lookup() const
  {
    std::lock_guard<std::mutex> lk(intrinsics_mutex_);
    if (!frame_id_.empty()) {
      return frame_id_;
    }
    return camera_info_frame_;
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
  // Frame reported by the camera_info header (empty until one arrives). Guarded
  // by intrinsics_mutex_ (read from the depth worker, written from the
  // camera_info callback).
  std::string camera_info_frame_;
  std::function<void(const sensor_msgs::msg::Image &)> update_callback_;
  // Max update rate (Hz) and the wall-clock stamp of the last accepted frame.
  // steady_clock, not the message stamp: the rate limit is about how much work
  // this node does, and sim time can jump.
  double max_update_rate_hz_ = 0.0;
  std::optional<std::chrono::steady_clock::time_point> last_frame_at_;
  std::mutex rate_mutex_;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__CAMERAS__CAMERA_STRATEGY_HPP_

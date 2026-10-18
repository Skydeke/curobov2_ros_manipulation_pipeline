// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/cameras/camera_strategy.py.
#include "curobo_core/cameras/camera_strategy.hpp"

#include <cmath>

#include "curobo_core/core/frame_utils.hpp"

namespace curobo_core
{

CameraStrategy::CameraStrategy(
  rclcpp::Node * node, const std::string & camera_name, const std::string & topic,
  const std::string & camera_info, const std::string & frame_id,
  const std::optional<std::vector<double>> & intrinsics,
  const std::optional<std::vector<double>> & extrinsics)
: node_(node), name_(camera_name), topic_(topic), camera_info_topic_(camera_info),
  frame_id_(frame_id)
{
  // PORT: each strategy owns its own TF buffer/listener.
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node_->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, node_, false);
  intrinsics_ = parse_intrinsics(intrinsics);
  static_pose_ = parse_extrinsics(extrinsics);

  // PORT: "No intrinsics in config, waiting for camera info on {topic}...".
  // The Python blocks on wait_for_message; the node must not block (UD-5), so
  // subscribe and fill the intrinsics on the first CameraInfo.
  if (!intrinsics_.has_value() && !camera_info_topic_.empty()) {
    RCLCPP_INFO(node_->get_logger(),
      "No intrinsics in config, waiting for camera info on %s...", camera_info_topic_.c_str());
    camera_info_sub_ = node_->create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic_, 10,
      [this](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {on_camera_info(*msg);});
  }
}

// PORT: the CameraInfo -> 3x3 K fallback (camera_depth_map_strategy lines 74-92).
void CameraStrategy::on_camera_info(const sensor_msgs::msg::CameraInfo & msg)
{
  {
    std::lock_guard<std::mutex> lk(intrinsics_mutex_);
    if (intrinsics_.has_value()) {
      return;
    }
    const auto & K = msg.k;
    intrinsics_ = std::array<double, 9>{
      K[0], K[1], K[2], K[3], K[4], K[5], K[6], K[7], K[8]};
  }
  RCLCPP_INFO(
    node_->get_logger(), "Camera intrinsics from topic: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f",
    msg.k[0], msg.k[4], msg.k[2], msg.k[5]);
  // One message is enough; drop the subscription.
  camera_info_sub_.reset();
}

// PORT: camera_strategy._parse_intrinsics
std::optional<std::array<double, 9>> CameraStrategy::parse_intrinsics(
  const std::optional<std::vector<double>> & intrinsics)
{
  if (!intrinsics.has_value()) {
    return std::nullopt;
  }
  const auto & v = *intrinsics;
  std::array<double, 9> k{};
  if (v.size() == 9) {
    // Full row-major K.
    for (std::size_t i = 0; i < 9; ++i) {
      k[i] = v[i];
    }
  } else if (v.size() == 4) {
    // PORT: fx,fy,cx,cy -> [[fx,0,cx],[0,fy,cy],[0,0,1]]
    k = {v[0], 0.0, v[2], 0.0, v[1], v[3], 0.0, 0.0, 1.0};
  } else {
    RCLCPP_ERROR(
      node_->get_logger(),
      "Intrinsics list must have 9 elements [fx,0,cx,0,fy,cy,0,0,1], got %zu", v.size());
    return std::nullopt;
  }
  return k;
}

// PORT: camera_strategy._parse_extrinsics
std::optional<Pose7d> CameraStrategy::parse_extrinsics(
  const std::optional<std::vector<double>> & extrinsics)
{
  if (!extrinsics.has_value()) {
    return std::nullopt;
  }
  const auto & v = *extrinsics;
  if (v.size() == 7) {
    // PORT: 7-element pose [x,y,z,qw,qx,qy,qz] (already cuRobo order).
    return Pose7d{v[0], v[1], v[2], v[3], v[4], v[5], v[6]};
  }
  if (v.size() == 16) {
    // PORT: 4x4 row-major matrix -> position + quaternion (via Rotation).
    Eigen::Matrix3d rot;
    rot << v[0], v[1], v[2], v[4], v[5], v[6], v[8], v[9], v[10];
    const double trace = rot.trace();
    std::array<double, 4> q{};
    if (trace > 0.0) {
      const double s = std::sqrt(trace + 1.0) * 2.0;
      q = {0.25 * s, (rot(2, 1) - rot(1, 2)) / s, (rot(0, 2) - rot(2, 0)) / s,
        (rot(1, 0) - rot(0, 1)) / s};
    } else if (rot(0, 0) > rot(1, 1) && rot(0, 0) > rot(2, 2)) {
      const double s = std::sqrt(1.0 + rot(0, 0) - rot(1, 1) - rot(2, 2)) * 2.0;
      q = {(rot(2, 1) - rot(1, 2)) / s, 0.25 * s, (rot(0, 1) + rot(1, 0)) / s,
        (rot(0, 2) + rot(2, 0)) / s};
    } else if (rot(1, 1) > rot(2, 2)) {
      const double s = std::sqrt(1.0 + rot(1, 1) - rot(0, 0) - rot(2, 2)) * 2.0;
      q = {(rot(0, 2) - rot(2, 0)) / s, (rot(0, 1) + rot(1, 0)) / s, 0.25 * s,
        (rot(1, 2) + rot(2, 1)) / s};
    } else {
      const double s = std::sqrt(1.0 + rot(2, 2) - rot(0, 0) - rot(1, 1)) * 2.0;
      q = {(rot(1, 0) - rot(0, 1)) / s, (rot(0, 2) + rot(2, 0)) / s,
        (rot(1, 2) + rot(2, 1)) / s, 0.25 * s};
    }
    q = normalize_quat_wxyz(q);
    return Pose7d{v[3], v[7], v[11], q[0], q[1], q[2], q[3]};
  }
  RCLCPP_ERROR(
    node_->get_logger(),
    "Extrinsics must be list of 7 elements [x,y,z,qw,qx,qy,qz] or 4x4 matrix, got %zu",
    v.size());
  return std::nullopt;
}

}  // namespace curobo_core

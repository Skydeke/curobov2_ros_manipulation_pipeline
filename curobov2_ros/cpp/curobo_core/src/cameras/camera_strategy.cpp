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
  const std::optional<std::vector<double>> & extrinsics,
  double max_update_rate_hz)
: node_(node), name_(camera_name), topic_(topic), camera_info_topic_(camera_info),
  frame_id_(frame_id), max_update_rate_hz_(max_update_rate_hz)
{
  // PORT: each strategy owns its own TF buffer/listener.
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node_->get_clock());
  // The buffer is filled by the node's executor thread, not the thread calling
  // canTransform/lookupTransform. Without this flag tf2 refuses to honour a
  // lookup timeout ("it will always timeout") and get_transform() throws on
  // every frame -- which silently drops every depth frame.
  tf_buffer_->setUsingDedicatedThread(true);
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, node_, false);
  intrinsics_ = parse_intrinsics(intrinsics);
  static_pose_ = parse_extrinsics(extrinsics);

  // PORT: "No intrinsics in config, waiting for camera info on {topic}...".
  // The Python blocks on wait_for_message; the node must not block (UD-5), so
  // subscribe and fill the intrinsics on the first CameraInfo. The same
  // message carries header.frame_id, which is now the authoritative source for
  // the camera frame used by the depth projection, so subscribe whenever we
  // still need either the intrinsics or the frame.
  if ((!intrinsics_.has_value() || frame_id_.empty()) && !camera_info_topic_.empty()) {
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
  bool have_intrinsics = false;
  bool have_frame = false;
  {
    std::lock_guard<std::mutex> lk(intrinsics_mutex_);
    // The frame travels with the depth pixels, so take it from the driver's own
    // camera_info header rather than a hardcoded `camera_frame` param.
    if (!msg.header.frame_id.empty()) {
      camera_info_frame_ = msg.header.frame_id;
    }
    if (!intrinsics_.has_value()) {
      const auto & K = msg.k;
      intrinsics_ = std::array<double, 9>{
        K[0], K[1], K[2], K[3], K[4], K[5], K[6], K[7], K[8]};
    }
    have_intrinsics = intrinsics_.has_value();
    have_frame = !camera_info_frame_.empty();
  }
  RCLCPP_INFO(
    node_->get_logger(),
    "Camera info: frame='%s' fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f",
    msg.header.frame_id.c_str(), msg.k[0], msg.k[4], msg.k[2], msg.k[5]);
  // One message carries both; drop the subscription once we have them.
  if (have_intrinsics && have_frame) {
    camera_info_sub_.reset();
  }
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

// PORT: MoveIt's occupancy-map monitor `max_update_rate`. The gate is on
// ACCEPTED frames, so a stream faster than the configured rate is decimated to
// it rather than stopped: a frame is dropped only when it arrives less than
// 1/max_update_rate_hz after the previous accepted one.
bool CameraStrategy::allow_frame()
{
  if (max_update_rate_hz_ <= 0.0) {
    return true;
  }
  const auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lk(rate_mutex_);
  if (last_frame_at_.has_value()) {
    const double dt_s = std::chrono::duration<double>(now - *last_frame_at_).count();
    if (dt_s < 1.0 / max_update_rate_hz_) {
      return false;
    }
  }
  last_frame_at_ = now;
  return true;
}

}  // namespace curobo_core

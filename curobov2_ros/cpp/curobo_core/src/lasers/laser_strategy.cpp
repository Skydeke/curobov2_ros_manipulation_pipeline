// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/lasers/laser_strategy.py.
#include "curobo_core/lasers/laser_strategy.hpp"

#include <cmath>

#include <tf2/exceptions.h>
#include <tf2/time.h>

#include "curobo_core/core/frame_utils.hpp"

namespace curobo_core
{

LaserStrategy::LaserStrategy(
  rclcpp::Node * node, const std::string & laser_name, const std::string & topic,
  const std::string & frame_id, const std::optional<std::vector<double>> & extrinsics)
: node_(node), name_(laser_name), topic_(topic), frame_id_(frame_id)
{
  base_frame_ = node_->has_parameter("base_link") ?
    node_->get_parameter("base_link").as_string() : "base_0";
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node_->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, node_, false);
  static_pose_ = parse_extrinsics(extrinsics);
  if (static_pose_.has_value()) {
    RCLCPP_INFO(node_->get_logger(), "[%s] Using static extrinsics from config file", name_.c_str());
  }
}

// PORT: laser_strategy._parse_extrinsics
std::optional<std::array<double, 7>> LaserStrategy::parse_extrinsics(
  const std::optional<std::vector<double>> & extrinsics)
{
  if (!extrinsics.has_value()) {
    return std::nullopt;
  }
  const auto & v = *extrinsics;
  if (v.size() == 7) {
    return std::array<double, 7>{v[0], v[1], v[2], v[3], v[4], v[5], v[6]};
  }
  if (v.size() == 16) {
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
    return std::array<double, 7>{v[3], v[7], v[11], q[0], q[1], q[2], q[3]};
  }
  RCLCPP_ERROR(
    node_->get_logger(), "[%s] extrinsics must be 7-element list or 4x4 matrix, got %zu",
    name_.c_str(), v.size());
  return std::nullopt;
}

// PORT: laser_strategy._resolve_pose_7
std::pair<bool, std::optional<std::array<double, 7>>> LaserStrategy::resolve_pose_7(
  bool has_stamp, const builtin_interfaces::msg::Time & stamp)
{
  if (static_pose_.has_value()) {
    return {true, std::nullopt};   // context holds the static pose
  }
  if (frame_id_.empty()) {
    RCLCPP_WARN_THROTTLE(
      node_->get_logger(), *node_->get_clock(), 2000,
      "[%s] No laser_frame set and no static extrinsics: cannot resolve the sensor "
      "pose. Dropping the frame (no integration).", name_.c_str());
    return {false, std::nullopt};
  }
  geometry_msgs::msg::TransformStamped transform;
  try {
    if (has_stamp) {
      transform = tf_buffer_->lookupTransform(
        base_frame_, frame_id_,
        tf2::TimePoint(std::chrono::nanoseconds(
          static_cast<std::int64_t>(stamp.sec) * 1000000000 + stamp.nanosec)));
    } else {
      transform = tf_buffer_->lookupTransform(base_frame_, frame_id_, tf2::TimePointZero);
    }
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN_THROTTLE(
      node_->get_logger(), *node_->get_clock(), 2000,
      "[%s] Cannot transform %s -> %s: %s. Dropping the frame (no integration).",
      name_.c_str(), base_frame_.c_str(), frame_id_.c_str(), ex.what());
    return {false, std::nullopt};
  }
  const auto & t = transform.transform.translation;
  const auto & r = transform.transform.rotation;
  const auto q = normalize_quat_wxyz({r.w, r.x, r.y, r.z});
  return {true, std::array<double, 7>{t.x, t.y, t.z, q[0], q[1], q[2], q[3]}};
}

}  // namespace curobo_core

// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/core/frame_utils.py.
#include "curobo_core/core/frame_utils.hpp"

#include <cmath>
#include <stdexcept>

#include <tf2/exceptions.h>
#include <tf2/time.h>

namespace curobo_core
{

// PORT: frame_utils._normalize_quat_wxyz
std::array<double, 4> normalize_quat_wxyz(const std::array<double, 4> & q)
{
  const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (n == 0.0) {
    return {1.0, 0.0, 0.0, 0.0};
  }
  return {q[0] / n, q[1] / n, q[2] / n, q[3] / n};
}

// PORT: frame_utils._quat_to_matrix
Eigen::Matrix3d quat_to_matrix(const std::array<double, 4> & q_in)
{
  const auto q = normalize_quat_wxyz(q_in);
  const double w = q[0];
  const double x = q[1];
  const double y = q[2];
  const double z = q[3];
  Eigen::Matrix3d m;
  m << 1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y);
  return m;
}

// PORT: frame_utils._quat_multiply
std::array<double, 4> quat_multiply(
  const std::array<double, 4> & a, const std::array<double, 4> & b)
{
  const double a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const double b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  return {
    a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
    a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
    a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
    a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0};
}

// PORT: frame_utils.apply_transform
geometry_msgs::msg::Pose apply_transform(
  const geometry_msgs::msg::Pose & pose, const geometry_msgs::msg::TransformStamped & transform)
{
  const auto & t = transform.transform.translation;
  const auto & r = transform.transform.rotation;
  const auto q = normalize_quat_wxyz({r.w, r.x, r.y, r.z});
  const Eigen::Matrix3d rot = quat_to_matrix(q);

  Eigen::Vector3d p(pose.position.x, pose.position.y, pose.position.z);
  const Eigen::Vector3d out_p = Eigen::Vector3d(t.x, t.y, t.z) + rot * p;

  const std::array<double, 4> pq = normalize_quat_wxyz(
    {pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z});
  const auto out_q = quat_multiply(q, pq);

  geometry_msgs::msg::Pose out;
  out.position.x = out_p.x();
  out.position.y = out_p.y();
  out.position.z = out_p.z();
  out.orientation.w = out_q[0];
  out.orientation.x = out_q[1];
  out.orientation.y = out_q[2];
  out.orientation.z = out_q[3];
  return out;
}

// PORT: frame_utils.get_transform
geometry_msgs::msg::TransformStamped get_transform(
  const tf2_ros::Buffer & tf_buffer, const std::string & target_frame,
  const std::string & source_frame, double timeout_sec)
{
  try {
    return tf_buffer.lookupTransform(
      target_frame, source_frame, tf2::TimePointZero,
      tf2::durationFromSec(timeout_sec));
  } catch (const tf2::TransformException & e) {
    throw std::runtime_error(
      "cannot transform '" + source_frame + "' -> '" + target_frame + "': " + e.what());
  }
}

// PORT: frame_utils.resolve_to_base
geometry_msgs::msg::Pose resolve_to_base(
  const geometry_msgs::msg::Pose & pose, const std::string & frame_id,
  const std::string & base_link, const tf2_ros::Buffer * tf_buffer, double timeout_sec)
{
  if (frame_id.empty() || frame_id == base_link) {
    return pose;
  }
  if (tf_buffer == nullptr) {
    throw std::runtime_error(
      "cannot transform '" + frame_id + "' -> '" + base_link + "': no TF buffer");
  }
  return apply_transform(pose, get_transform(*tf_buffer, base_link, frame_id, timeout_sec));
}

// PORT: frame_utils.resolve_poses_to_base
std::vector<geometry_msgs::msg::Pose> resolve_poses_to_base(
  const std::vector<geometry_msgs::msg::Pose> & poses, const std::string & frame_id,
  const std::string & base_link, const tf2_ros::Buffer * tf_buffer, double timeout_sec)
{
  if (frame_id.empty() || frame_id == base_link) {
    return poses;
  }
  if (tf_buffer == nullptr) {
    throw std::runtime_error(
      "cannot transform '" + frame_id + "' -> '" + base_link + "': no TF buffer");
  }
  const auto transform = get_transform(*tf_buffer, base_link, frame_id, timeout_sec);
  std::vector<geometry_msgs::msg::Pose> out;
  out.reserve(poses.size());
  for (const auto & pose : poses) {
    out.push_back(apply_transform(pose, transform));
  }
  return out;
}

}  // namespace curobo_core

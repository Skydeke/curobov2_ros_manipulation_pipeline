// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/core/frame_utils.py
//
// Server-side TF frame resolution for pose-carrying requests; pure
// geometry_msgs/Pose in/out, quaternion math in (w,x,y,z). Identity when the
// frame is "" or the base link.
#ifndef CUROBO_CORE__CORE__FRAME_UTILS_HPP_
#define CUROBO_CORE__CORE__FRAME_UTILS_HPP_

#include <array>
#include <vector>

#include <Eigen/Core>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/buffer.h>

namespace curobo_core
{

// PORT: frame_utils._normalize_quat_wxyz
std::array<double, 4> normalize_quat_wxyz(const std::array<double, 4> & q);

// PORT: frame_utils._quat_to_matrix
Eigen::Matrix3d quat_to_matrix(const std::array<double, 4> & q);

// PORT: frame_utils._quat_multiply (Hamilton product, w first)
std::array<double, 4> quat_multiply(
  const std::array<double, 4> & a, const std::array<double, 4> & b);

// PORT: frame_utils.apply_transform (out = transform * pose)
geometry_msgs::msg::Pose apply_transform(
  const geometry_msgs::msg::Pose & pose, const geometry_msgs::msg::TransformStamped & transform);

// PORT: frame_utils.get_transform (throws std::runtime_error on TF failure)
geometry_msgs::msg::TransformStamped get_transform(
  const tf2_ros::Buffer & tf_buffer, const std::string & target_frame,
  const std::string & source_frame, double timeout_sec = 0.5);

// PORT: frame_utils.resolve_to_base / resolve_poses_to_base
geometry_msgs::msg::Pose resolve_to_base(
  const geometry_msgs::msg::Pose & pose, const std::string & frame_id,
  const std::string & base_link, const tf2_ros::Buffer * tf_buffer, double timeout_sec = 0.5);
std::vector<geometry_msgs::msg::Pose> resolve_poses_to_base(
  const std::vector<geometry_msgs::msg::Pose> & poses, const std::string & frame_id,
  const std::string & base_link, const tf2_ros::Buffer * tf_buffer, double timeout_sec = 0.5);

}  // namespace curobo_core

#endif  // CUROBO_CORE__CORE__FRAME_UTILS_HPP_

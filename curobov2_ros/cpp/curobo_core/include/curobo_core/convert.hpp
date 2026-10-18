// SPDX-License-Identifier: Apache-2.0
// PORT: (design section 6) the ONE place poses are converted between the ROS
// wire order (geometry_msgs/Pose: x, y, z, w) and the cuRobo bridge order
// ([x, y, z, qw, qx, qy, qz]). Every other conversion is a bug.
//
// Unit tests: test/test_convert.cpp.
#ifndef CUROBO_CORE__CONVERT_HPP_
#define CUROBO_CORE__CONVERT_HPP_

#include <array>
#include <cstddef>
#include <vector>

#include <Eigen/Core>
#include <geometry_msgs/msg/pose.hpp>

namespace curobo_core
{

// cuRobo pose order: [x, y, z, qw, qx, qy, qz].
using Pose7 = Eigen::Matrix<double, 7, 1>;

// PORT: bridge_api.md data conventions - poses are [x,y,z,qw,qx,qy,qz].
geometry_msgs::msg::Pose to_ros_pose(const Pose7 & pose);

Pose7 from_ros_pose(const geometry_msgs::msg::Pose & msg);

// The bridge's float32 array form (C-contiguous [N,7]).
std::vector<std::array<float, 7>> to_pose_array(const std::vector<Pose7> & poses);
std::vector<Pose7> from_pose_array(const std::vector<std::array<float, 7>> & rows);

}  // namespace curobo_core

#endif  // CUROBO_CORE__CONVERT_HPP_

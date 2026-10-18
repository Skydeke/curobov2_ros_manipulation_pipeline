// SPDX-License-Identifier: Apache-2.0
#include "curobo_core/convert.hpp"

namespace curobo_core
{

// PORT: convert.cpp - geometry_msgs/Pose (x, y, z, w) <-> [x, y, z, qw, qx, qy, qz]
geometry_msgs::msg::Pose to_ros_pose(const Pose7 & pose)
{
  geometry_msgs::msg::Pose msg;
  msg.position.x = pose[0];
  msg.position.y = pose[1];
  msg.position.z = pose[2];
  msg.orientation.w = pose[3];
  msg.orientation.x = pose[4];
  msg.orientation.y = pose[5];
  msg.orientation.z = pose[6];
  return msg;
}

Pose7 from_ros_pose(const geometry_msgs::msg::Pose & msg)
{
  Pose7 pose;
  pose << msg.position.x, msg.position.y, msg.position.z,
    msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z;
  return pose;
}

std::vector<std::array<float, 7>> to_pose_array(const std::vector<Pose7> & poses)
{
  std::vector<std::array<float, 7>> rows;
  rows.reserve(poses.size());
  for (const auto & pose : poses) {
    std::array<float, 7> row{};
    for (std::size_t i = 0; i < 7; ++i) {
      row[i] = static_cast<float>(pose[static_cast<Eigen::Index>(i)]);
    }
    rows.push_back(row);
  }
  return rows;
}

std::vector<Pose7> from_pose_array(const std::vector<std::array<float, 7>> & rows)
{
  std::vector<Pose7> poses;
  poses.reserve(rows.size());
  for (const auto & row : rows) {
    Pose7 pose;
    for (std::size_t i = 0; i < 7; ++i) {
      pose[static_cast<Eigen::Index>(i)] = row[i];
    }
    poses.push_back(pose);
  }
  return poses;
}

}  // namespace curobo_core

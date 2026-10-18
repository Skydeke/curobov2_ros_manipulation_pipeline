// SPDX-License-Identifier: Apache-2.0
// Unit tests for convert.cpp (design section 6: one conversion site, tested).
// No ROS graph, no GPU.
#include <gtest/gtest.h>

#include <array>
#include <vector>

#include "curobo_core/convert.hpp"

using curobo_core::from_pose_array;
using curobo_core::from_ros_pose;
using curobo_core::Pose7;
using curobo_core::to_pose_array;
using curobo_core::to_ros_pose;

TEST(Convert, RoboOrderToRosPutsQuaternionWLast)
{
  Pose7 pose;
  pose << 1.0, 2.0, 3.0, /*qw*/ 0.1, /*qx*/ 0.2, /*qy*/ 0.3, /*qz*/ 0.4;
  const auto msg = to_ros_pose(pose);
  EXPECT_DOUBLE_EQ(msg.position.x, 1.0);
  EXPECT_DOUBLE_EQ(msg.position.y, 2.0);
  EXPECT_DOUBLE_EQ(msg.position.z, 3.0);
  EXPECT_DOUBLE_EQ(msg.orientation.w, 0.1);
  EXPECT_DOUBLE_EQ(msg.orientation.x, 0.2);
  EXPECT_DOUBLE_EQ(msg.orientation.y, 0.3);
  EXPECT_DOUBLE_EQ(msg.orientation.z, 0.4);
}

TEST(Convert, RosPoseToRoboOrderReadsWFromOrientation)
{
  geometry_msgs::msg::Pose msg;
  msg.position.x = -1.5;
  msg.position.y = 0.25;
  msg.position.z = 9.0;
  msg.orientation.w = 0.5;
  msg.orientation.x = -0.5;
  msg.orientation.y = 0.5;
  msg.orientation.z = -0.5;
  const Pose7 pose = from_ros_pose(msg);
  EXPECT_DOUBLE_EQ(pose[0], -1.5);
  EXPECT_DOUBLE_EQ(pose[1], 0.25);
  EXPECT_DOUBLE_EQ(pose[2], 9.0);
  EXPECT_DOUBLE_EQ(pose[3], 0.5);   // qw
  EXPECT_DOUBLE_EQ(pose[4], -0.5);  // qx
  EXPECT_DOUBLE_EQ(pose[5], 0.5);   // qy
  EXPECT_DOUBLE_EQ(pose[6], -0.5);  // qz
}

TEST(Convert, RosRoundTripIsIdentity)
{
  Pose7 pose;
  pose << 0.0255, 0.1521, 0.8555, 0.1288, -0.8049, -0.3629, -0.4516;
  const Pose7 back = from_ros_pose(to_ros_pose(pose));
  EXPECT_TRUE(back.isApprox(pose, 0.0));
}

TEST(Convert, ArrayRoundTripKeepsOrder)
{
  std::vector<Pose7> poses(2);
  poses[0] << 1, 2, 3, 0.1288, -0.8049, -0.3629, -0.4516;
  poses[1] << -1, -2, -3, 1, 0, 0, 0;
  const auto rows = to_pose_array(poses);
  ASSERT_EQ(rows.size(), 2u);
  EXPECT_FLOAT_EQ(rows[0][3], 0.1288f);   // qw first among the quaternion
  const auto back = from_pose_array(rows);
  ASSERT_EQ(back.size(), 2u);
  EXPECT_TRUE(back[0].isApprox(poses[0], 1e-6));
  EXPECT_TRUE(back[1].isApprox(poses[1], 1e-6));
}

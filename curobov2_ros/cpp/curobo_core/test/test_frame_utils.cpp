// SPDX-License-Identifier: Apache-2.0
// Unit tests for frame_utils (pure quaternion math + pose transform). No ROS
// graph, no GPU.
#include <gtest/gtest.h>

#include <cmath>

#include "curobo_core/core/frame_utils.hpp"

using curobo_core::apply_transform;
using curobo_core::normalize_quat_wxyz;
using curobo_core::quat_multiply;
using curobo_core::quat_to_matrix;

TEST(FrameUtils, NormalizeQuatWxyz)
{
  // PORT: _normalize_quat_wxyz - an all-zero quaternion is identity.
  const auto q = normalize_quat_wxyz({0.0, 0.0, 0.0, 0.0});
  EXPECT_DOUBLE_EQ(q[0], 1.0);
  EXPECT_DOUBLE_EQ(q[1], 0.0);
  const auto n = normalize_quat_wxyz({2.0, 0.0, 0.0, 0.0});
  EXPECT_DOUBLE_EQ(n[0], 1.0);
}

TEST(FrameUtils, QuatToMatrixQuarterTurnAboutZ)
{
  // PORT: _quat_to_matrix - 90 deg about z: x -> y.
  const double s = std::sqrt(0.5);
  const Eigen::Matrix3d m = quat_to_matrix({s, 0.0, 0.0, s});
  EXPECT_NEAR(m(0, 0), 0.0, 1e-12);
  EXPECT_NEAR(m(0, 1), -1.0, 1e-12);
  EXPECT_NEAR(m(1, 0), 1.0, 1e-12);
  EXPECT_NEAR(m(1, 1), 0.0, 1e-12);
  EXPECT_NEAR(m(2, 2), 1.0, 1e-12);
}

TEST(FrameUtils, QuatMultiplyIdentity)
{
  // PORT: _quat_multiply
  const auto q = quat_multiply({1.0, 0.0, 0.0, 0.0}, {0.1, 0.2, 0.3, 0.4});
  EXPECT_DOUBLE_EQ(q[0], 0.1);
  EXPECT_DOUBLE_EQ(q[1], 0.2);
  EXPECT_DOUBLE_EQ(q[2], 0.3);
  EXPECT_DOUBLE_EQ(q[3], 0.4);
}

TEST(FrameUtils, ApplyTransformTranslationAndRotation)
{
  // PORT: apply_transform - a translate-and-rotate transform composed with a pose.
  geometry_msgs::msg::Pose pose;
  pose.position.x = 1.0;
  pose.position.y = 0.0;
  pose.position.z = 0.0;
  pose.orientation.w = 1.0;

  geometry_msgs::msg::TransformStamped t;
  t.transform.translation.x = 1.0;
  t.transform.translation.y = 2.0;
  t.transform.translation.z = 3.0;
  const double s = std::sqrt(0.5);
  t.transform.rotation.w = s;   // 90 deg about z: (1,0,0) -> (0,1,0)
  t.transform.rotation.z = s;

  const auto out = apply_transform(pose, t);
  EXPECT_NEAR(out.position.x, 1.0, 1e-12);   // 1 + R*(1,0,0) = 1 + (0,1,0)
  EXPECT_NEAR(out.position.y, 3.0, 1e-12);
  EXPECT_NEAR(out.position.z, 3.0, 1e-12);
  EXPECT_NEAR(out.orientation.w, s, 1e-12);
  EXPECT_NEAR(out.orientation.z, s, 1e-12);
}

TEST(FrameUtils, ResolveToBaseEmptyFrameIsIdentity)
{
  // PORT: resolve_to_base - "" (or the base link) needs no transform.
  geometry_msgs::msg::Pose pose;
  pose.position.x = 0.5;
  const auto out = curobo_core::resolve_to_base(pose, "", "base_0", nullptr);
  EXPECT_DOUBLE_EQ(out.position.x, 0.5);
  const auto out2 = curobo_core::resolve_to_base(pose, "base_0", "base_0", nullptr);
  EXPECT_DOUBLE_EQ(out2.position.x, 0.5);
  // A real frame with no buffer errors exactly like the Python path.
  EXPECT_THROW(
    curobo_core::resolve_to_base(pose, "camera", "base_0", nullptr),
    std::runtime_error);
}

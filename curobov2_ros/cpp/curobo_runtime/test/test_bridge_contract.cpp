// SPDX-License-Identifier: Apache-2.0
// Bridge contract test (design section 14.2): the M2 bridge surface is exercised
// through FakeBridge so the C++ marshalling and layouts are pinned without a
// GPU or an interpreter. Every assertion is on documented FakeBridge values.
#include <gtest/gtest.h>

#include <array>
#include <string>
#include <vector>

#include "curobo_runtime/fake_bridge.hpp"

using curobo_runtime::FakeBridge;
using curobo_runtime::IkOutcome;
using curobo_runtime::TensorF32;

namespace
{
TensorF32 make_q(const std::vector<std::vector<float>> & rows)
{
  TensorF32 t;
  t.shape = {static_cast<std::int64_t>(rows.size()),
    static_cast<std::int64_t>(rows.empty() ? 0 : rows[0].size())};
  for (const auto & row : rows) {
    t.data.insert(t.data.end(), row.begin(), row.end());
  }
  return t;
}
}  // namespace

TEST(BridgeContract, RobotModelBuildRecordsCfg)
{
  FakeBridge bridge;
  bridge.robot_model_build(1, R"({"robot_config_file":"/tmp/franka.yml"})");
  EXPECT_EQ(bridge.model_cfg(), R"({"robot_config_file":"/tmp/franka.yml"})");
}

TEST(BridgeContract, FkShapeAndLayout)
{
  FakeBridge bridge;
  const auto q = make_q({{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f}});
  const TensorF32 poses = bridge.fk(1, q);

  ASSERT_EQ(poses.shape, (std::vector<std::int64_t>{1, 1, 7}));
  ASSERT_EQ(poses.data.size(), 7u);
  // [x,y,z,qw,qx,qy,qz]
  EXPECT_FLOAT_EQ(poses.data[0], 0.1f);
  EXPECT_FLOAT_EQ(poses.data[1], 0.2f);
  EXPECT_FLOAT_EQ(poses.data[2], 0.3f);
  EXPECT_FLOAT_EQ(poses.data[3], 1.0f);   // qw identity
  EXPECT_FLOAT_EQ(poses.data[4], 0.4f);   // qx
  EXPECT_FLOAT_EQ(poses.data[5], 0.5f);   // qy
  EXPECT_FLOAT_EQ(poses.data[6], 0.6f);   // qz
}

TEST(BridgeContract, IkValidityFollowsReachability)
{
  FakeBridge bridge;
  bridge.set_dof(7);
  // reachable pose (unit-sphere test double) and the recorded unreachable one.
  const auto poses = make_q({{0.0255f, 0.1521f, 0.8555f, 0.1288f, -0.8049f, -0.3629f, -0.4516f},
      {10.0f, 10.0f, 10.0f, 0.0f, 0.0f, 0.0f, 1.0f}});
  const IkOutcome outcome = bridge.ik(1, poses);

  ASSERT_EQ(outcome.valid.size(), 2u);
  EXPECT_EQ(outcome.valid[0], 1);
  EXPECT_EQ(outcome.valid[1], 0);
  EXPECT_TRUE(outcome.success);
  ASSERT_EQ(outcome.positions.shape, (std::vector<std::int64_t>{2, 1, 7}));
  EXPECT_EQ(outcome.joint_names.size(), 7u);
}

TEST(BridgeContract, ValidateIsAllTrue)
{
  FakeBridge bridge;
  const auto q = make_q({{0, 0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 1, 1}});
  const auto valid = bridge.validate(1, q);
  ASSERT_EQ(valid.size(), 2u);
  EXPECT_EQ(valid[0], 1);
  EXPECT_EQ(valid[1], 1);
}

TEST(BridgeContract, CollisionDistanceIsOrderPreserving)
{
  FakeBridge bridge;
  const auto q = make_q({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
  const auto distances = bridge.collision_distance(1, q);
  EXPECT_EQ(distances, (std::vector<float>{1, 2, 3, 4, 5, 6}));
}

TEST(BridgeContract, SceneSetVersionsAndClearFlag)
{
  FakeBridge bridge;
  EXPECT_EQ(bridge.scene_set(1, R"([{"name":"cube"}])", true), 1);
  EXPECT_TRUE(bridge.last_clear_first());
  EXPECT_EQ(bridge.scene_set(1, R"([{"name":"cube2"}])", false), 2);
  EXPECT_FALSE(bridge.last_clear_first());
  EXPECT_EQ(bridge.scene_version(), 2);
  EXPECT_EQ(bridge.scene_get(1), R"([{"name":"cube2"}])");
}

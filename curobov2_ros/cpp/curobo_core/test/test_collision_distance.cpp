// SPDX-License-Identifier: Apache-2.0
// In-process tests for the GetCollisionDistance service wrapper, over FakeBridge
// (no GPU). FakeBridge::collision_distance returns one value per joint, so the
// response length and ordering are asserted directly.
#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "curobo_core/core/collision_distance.hpp"
#include "curobo_runtime/fake_bridge.hpp"
#include "curobov2_ros_interfaces/srv/get_collision_distance.hpp"

using namespace std::chrono_literals;
using curobo_core::CollisionDistance;
using curobo_runtime::FakeBridge;

namespace
{
class CollisionDistanceTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }
  }
};
}  // namespace

TEST_F(CollisionDistanceTest, ResponseCarriesOneEntryPerSphere)
{
  auto node = std::make_shared<rclcpp::Node>("unified_planner");
  FakeBridge bridge;
  std::vector<float> pose{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  CollisionDistance service(node.get(), &bridge, 1, [&pose]() {return pose;});

  auto client = node->create_client<curobov2_ros_interfaces::srv::GetCollisionDistance>(
    "unified_planner/get_collision_distance");
  ASSERT_TRUE(client->wait_for_service(5s));
  auto future = client->async_send_request(
    std::make_shared<curobov2_ros_interfaces::srv::GetCollisionDistance::Request>());
  rclcpp::spin_until_future_complete(node, future, 10s);
  auto response = future.get();

  EXPECT_EQ(response->nb_sphere, 7);
  ASSERT_EQ(response->data.size(), 7u);
  EXPECT_FLOAT_EQ(response->data[0], 1.0f);
  EXPECT_FLOAT_EQ(response->data[6], 7.0f);

  // The direct helper returns the same distances.
  EXPECT_EQ(service.computeSphereDistance(), pose);
}

TEST_F(CollisionDistanceTest, EmptyPoseReportsZeroSpheres)
{
  auto node = std::make_shared<rclcpp::Node>("unified_planner");
  FakeBridge bridge;
  CollisionDistance service(node.get(), &bridge, 1, []() {return std::vector<float>{};});
  EXPECT_EQ(service.computeSphereDistance(), (std::vector<float>{}));
}

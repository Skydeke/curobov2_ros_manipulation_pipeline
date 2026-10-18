// SPDX-License-Identifier: Apache-2.0
// In-process tests for the C++ IKServices: /warmup_ik + /ik + /ik_batch wiring
// and the ported validity handling, over FakeBridge (no GPU, no interpreter).
#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "curobo_core/core/ik_services.hpp"
#include "curobo_runtime/fake_bridge.hpp"
#include "curobov2_ros_interfaces/srv/ik.hpp"
#include "curobov2_ros_interfaces/srv/ik_batch.hpp"

using namespace std::chrono_literals;
using curobo_core::IKServices;
using curobo_runtime::FakeBridge;

namespace
{
class IKServicesTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }
  }
};

template<class FutureT>
void spin_until(rclcpp::Node::SharedPtr node, FutureT & future)
{
  rclcpp::spin_until_future_complete(node, future, 10s);
}

geometry_msgs::msg::Pose pose_of(double x, double y, double z)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z;
  pose.orientation.w = 1.0;
  return pose;
}
}  // namespace

TEST_F(IKServicesTest, WarmupMessagesAndRebuildOnChangedBatch)
{
  auto node = std::make_shared<rclcpp::Node>("unified_planner");
  FakeBridge bridge;
  IKServices services(node.get(), &bridge, 1, "base_link", 32);

  EXPECT_EQ(services.warmup(1), "IK solver ready (batch_size=1)");
  EXPECT_EQ(services.warmup(1), "IK solver already ready (batch_size=1)");
  EXPECT_FALSE(services.needs_rebuild(1, 32));
  EXPECT_EQ(services.warmup(2), "IK solver ready (batch_size=2)");
}

TEST_F(IKServicesTest, IkValidityFollowsTheFakeBridgeReachabilityRule)
{
  auto node = std::make_shared<rclcpp::Node>("unified_planner");
  FakeBridge bridge;
  bridge.set_dof(7);
  IKServices services(node.get(), &bridge, 1, "base_link", 32);
  ASSERT_EQ(services.warmup(1), "IK solver ready (batch_size=1)");

  auto client = node->create_client<curobov2_ros_interfaces::srv::Ik>("unified_planner/ik");
  ASSERT_TRUE(client->wait_for_service(5s));

  // reachable pose (FakeBridge: position inside the unit sphere -> valid)
  {
    auto request = std::make_shared<curobov2_ros_interfaces::srv::Ik::Request>();
    request->pose = pose_of(0.0255, 0.1521, 0.8555);
    auto future = client->async_send_request(request);
    spin_until(node, future);
    auto response = future.get();
    ASSERT_TRUE(response->success);
    EXPECT_TRUE(response->joint_states_valid.data);
    EXPECT_EQ(response->joint_states.position.size(), 7u);
  }
  // the recorded unreachable pose (10,10,10) -> invalid
  {
    auto request = std::make_shared<curobov2_ros_interfaces::srv::Ik::Request>();
    request->pose = pose_of(10.0, 10.0, 10.0);
    auto future = client->async_send_request(request);
    spin_until(node, future);
    auto response = future.get();
    ASSERT_TRUE(response->success);
    EXPECT_FALSE(response->joint_states_valid.data);
  }
}

TEST_F(IKServicesTest, UninitializedIsRefusedWithTheRecordedString)
{
  auto node = std::make_shared<rclcpp::Node>("unified_planner");
  FakeBridge bridge;
  IKServices services(node.get(), &bridge, 1, "base_link", 32);

  auto client = node->create_client<curobov2_ros_interfaces::srv::Ik>("unified_planner/ik");
  ASSERT_TRUE(client->wait_for_service(5s));
  auto request = std::make_shared<curobov2_ros_interfaces::srv::Ik::Request>();
  request->pose = pose_of(0.0, 0.0, 0.5);
  auto future = client->async_send_request(request);
  spin_until(node, future);
  auto response = future.get();
  EXPECT_FALSE(response->success);
  EXPECT_EQ(response->error_msg.data, "IK not initialized. Call warmup_ik first.");
}

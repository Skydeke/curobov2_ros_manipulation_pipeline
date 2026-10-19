// SPDX-License-Identifier: Apache-2.0
// In-process tests for the C++ IKServices: /warmup_ik + /ik + /ik_batch wiring
// and the ported validity handling, over FakeBridge (no GPU, no interpreter).
#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
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

// REGRESSION: the port dropped the node params for the standalone IK solver and
// built it with the ctor default (32 seeds, collisions on, graphs on). Mapping
// runs a 500-pose batch with num_ik_seeds=4; 32 seeds is 8x the VRAM and OOMed
// the 8 GiB GPU. The resolved params must reach the bridge build.
TEST_F(IKServicesTest, WarmupForwardsNumSeedsAndFlagsResolvedFromNodeParams)
{
  // The CUROBO_USE_CUDA_GRAPH env var wins over the ROS param (Python
  // precedence); keep this test deterministic regardless of the environment.
  ::unsetenv("CUROBO_USE_CUDA_GRAPH");

  auto node = std::make_shared<rclcpp::Node>("unified_planner");
  node->declare_parameter<int>("num_ik_seeds", 4);
  node->declare_parameter<bool>("ik_collision_check", false);
  node->declare_parameter<bool>("ik_use_lm_seed", false);
  node->declare_parameter<bool>("use_cuda_graph", false);

  FakeBridge bridge;
  IKServices services(node.get(), &bridge, 1, "base_link", 32);

  EXPECT_EQ(services.resolve_num_seeds(), 4);
  EXPECT_FALSE(services.resolve_collision_check());
  EXPECT_FALSE(services.resolve_use_lm_seed());
  EXPECT_FALSE(services.resolve_use_cuda_graph());

  EXPECT_EQ(services.warmup(500), "IK solver ready (batch_size=500)");
  EXPECT_EQ(services.num_seeds(), 4);          // ctor fallback (32) is overridden

  EXPECT_EQ(bridge.ik_builds(), 1);
  EXPECT_EQ(bridge.ik_batch(), 500);
  EXPECT_EQ(bridge.ik_num_seeds(), 4);
  EXPECT_FALSE(bridge.ik_self_collision_check());
  EXPECT_FALSE(bridge.ik_use_lm_seed());
  EXPECT_FALSE(bridge.ik_use_cuda_graph());
}

// A `ros2 param set num_ik_seeds` flip must be picked up by the next warmup /
// update_ik_config rebuild (Python needs_rebuild compares the resolved params).
TEST_F(IKServicesTest, ResolvedSeedChangeTriggersARebuild)
{
  ::unsetenv("CUROBO_USE_CUDA_GRAPH");

  auto node = std::make_shared<rclcpp::Node>("unified_planner");
  node->declare_parameter<int>("num_ik_seeds", 4);
  FakeBridge bridge;
  IKServices services(node.get(), &bridge, 1, "base_link", 32);

  EXPECT_EQ(services.warmup(1), "IK solver ready (batch_size=1)");
  EXPECT_EQ(bridge.ik_num_seeds(), 4);
  EXPECT_EQ(services.warmup(1), "IK solver already ready (batch_size=1)");

  node->set_parameter(rclcpp::Parameter("num_ik_seeds", 8));
  EXPECT_TRUE(services.needs_rebuild(1, services.resolve_num_seeds()));
  EXPECT_EQ(services.warmup(1), "IK solver ready (batch_size=1)");
  EXPECT_EQ(bridge.ik_num_seeds(), 8);
  EXPECT_EQ(bridge.ik_builds(), 2);
}

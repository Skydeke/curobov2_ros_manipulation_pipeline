// SPDX-License-Identifier: Apache-2.0
// Smoke test for the M2 node: it registers the scene/FK/IK/collision/attach
// services and answers them, using FakeBridge (no GPU, no interpreter).
#include <gtest/gtest.h>

#include <chrono>
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include "curobo_core/core/unified_planner_node.hpp"
#include "curobo_runtime/fake_bridge.hpp"
#include "curobo_runtime/runtime.hpp"
#include "curobov2_ros_interfaces/srv/fk.hpp"
#include "curobov2_ros_interfaces/srv/fk_batch.hpp"
#include "curobov2_ros_interfaces/srv/get_joint_info.hpp"
#include "curobov2_ros_interfaces/srv/get_scene_objects.hpp"
#include "curobov2_ros_interfaces/srv/warmup_fk.hpp"

using namespace std::chrono_literals;
using curobo_core::UnifiedPlannerNode;
using curobo_runtime::FakeBridge;

TEST(UnifiedPlannerNode, RegistersAndAnswersTheM2Services)
{
  rclcpp::init(0, nullptr);
  curobo_runtime::Runtime::instance().set_bridge_for_test(std::make_unique<FakeBridge>());

  rclcpp::NodeOptions options;
  options.parameter_overrides({rclcpp::Parameter("robot_config_file", "/tmp/franka.yml")});
  auto node = std::make_shared<UnifiedPlannerNode>(options);

  // joint names come from the bridge's joint_limits (7 for the fake).
  ASSERT_EQ(node->joint_names().size(), 7u);

  {
    auto client = node->create_client<curobov2_ros_interfaces::srv::GetJointInfo>(
      "unified_planner/get_joint_info");
    ASSERT_TRUE(client->wait_for_service(5s));
    auto future = client->async_send_request(
      std::make_shared<curobov2_ros_interfaces::srv::GetJointInfo::Request>());
    rclcpp::spin_until_future_complete(node, future, 10s);
    auto response = future.get();
    EXPECT_TRUE(response->success);
    EXPECT_EQ(response->message, "7 joints in cspace");
    EXPECT_EQ(response->joint_names.size(), 7u);
    ASSERT_EQ(response->tool_frames.size(), 1u);
    EXPECT_EQ(response->tool_frames[0], "tool0");
  }
  {
    auto client = node->create_client<curobov2_ros_interfaces::srv::FkBatch>(
      "unified_planner/fk_batch");
    ASSERT_TRUE(client->wait_for_service(5s));
    auto request = std::make_shared<curobov2_ros_interfaces::srv::FkBatch::Request>();
    sensor_msgs::msg::JointState state;
    state.name = node->joint_names();
    state.position.assign(node->joint_names().size(), 0.0);
    request->joint_states.push_back(state);
    request->link_names = {"tool0", "base_link"};
    auto future = client->async_send_request(request);
    rclcpp::spin_until_future_complete(node, future, 10s);
    auto response = future.get();
    EXPECT_TRUE(response->success);
    ASSERT_EQ(response->link_names.size(), 2u);
    EXPECT_EQ(response->link_poses.size(), 2u);
  }
  {
    auto client = node->create_client<curobov2_ros_interfaces::srv::WarmupFK>(
      "unified_planner/warmup_fk");
    ASSERT_TRUE(client->wait_for_service(5s));
    auto request = std::make_shared<curobov2_ros_interfaces::srv::WarmupFK::Request>();
    request->batch_size = 1;
    auto future = client->async_send_request(request);
    rclcpp::spin_until_future_complete(node, future, 10s);
    auto response = future.get();
    EXPECT_TRUE(response->success);
    // The node warms FK up at startup (warmup_fk default true), matching the
    // Python server, so an explicit /warmup_fk reports "already ready".
    EXPECT_EQ(response->message, "FK model already ready (batch_size=1)");
  }
  {
    auto client = node->create_client<curobov2_ros_interfaces::srv::GetSceneObjects>(
      "unified_planner/get_scene_objects");
    ASSERT_TRUE(client->wait_for_service(5s));
    auto future = client->async_send_request(
      std::make_shared<curobov2_ros_interfaces::srv::GetSceneObjects::Request>());
    rclcpp::spin_until_future_complete(node, future, 10s);
    auto response = future.get();
    EXPECT_TRUE(response->success);
    EXPECT_EQ(response->message, "0 object(s)");
  }

  node.reset();
  rclcpp::shutdown();
}

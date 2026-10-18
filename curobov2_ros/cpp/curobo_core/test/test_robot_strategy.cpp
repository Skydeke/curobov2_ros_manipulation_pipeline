// SPDX-License-Identifier: Apache-2.0
// In-process tests for the emulator strategy + RobotContext (no GPU).
#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include "curobo_core/robot/joint_speed_strategy.hpp"
#include "curobo_core/robot/robot_context.hpp"
#include "curobo_core/robot/robot_registry.hpp"
#include "curobov2_ros_interfaces/srv/get_robot_strategies.hpp"
#include "curobov2_ros_interfaces/srv/set_robot_strategy.hpp"

using namespace std::chrono_literals;
using curobo_core::RobotContext;

namespace
{
std::vector<std::string> joints()
{
  return {"panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7"};
}
}  // namespace

TEST(RobotStrategy, EmulatorPlaysTheCommandBuffer)
{
  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
  }
  auto node = std::make_shared<rclcpp::Node>("unified_planner");
  RobotContext context(node.get(), 0.001, joints(), "emulator", {}, std::vector<double>(7, 0.1));

  // Seed the emulator at the supplied home pose.
  EXPECT_DOUBLE_EQ(context.get_joint_pose()[0], 0.1);

  std::vector<std::vector<double>> positions{{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
    {0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2}};
  const int epoch = context.set_command(joints(), positions, positions, positions);
  EXPECT_GT(epoch, 0);
  ASSERT_TRUE(context.send_trajectrory(epoch));
  std::this_thread::sleep_for(50ms);
  EXPECT_DOUBLE_EQ(context.get_joint_pose()[0], 0.2);
  EXPECT_EQ(context.get_joint_name(), joints());
}

TEST(RobotStrategy, EpochMismatchRefusesToExecute)
{
  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
  }
  auto node = std::make_shared<rclcpp::Node>("unified_planner");
  RobotContext context(node.get(), 0.001, joints(), "emulator", {}, std::vector<double>(7, 0.0));

  std::vector<std::vector<double>> one{{0.1, 0, 0, 0, 0, 0, 0}};
  const int epoch = context.set_command(joints(), one, one, one);
  // A newer command supersedes it.
  context.set_command(joints(), one, one, one);
  EXPECT_FALSE(context.send_trajectrory(epoch));
}

TEST(RobotStrategy, StrategySwitchServiceReportsTheRecordedWording)
{
  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
  }
  auto node = std::make_shared<rclcpp::Node>("unified_planner");
  RobotContext context(node.get(), 0.001, joints(), "emulator", {}, std::vector<double>(7, 0.0));

  auto set_client = node->create_client<curobov2_ros_interfaces::srv::SetRobotStrategy>(
    "unified_planner/set_robot_strategy");
  auto get_client = node->create_client<curobov2_ros_interfaces::srv::GetRobotStrategies>(
    "unified_planner/get_robot_strategies");
  ASSERT_TRUE(set_client->wait_for_service(5s));
  ASSERT_TRUE(get_client->wait_for_service(5s));

  // PORT: already on this strategy.
  {
    auto request = std::make_shared<curobov2_ros_interfaces::srv::SetRobotStrategy::Request>();
    request->robot_strategy = "emulator";
    auto future = set_client->async_send_request(request);
    rclcpp::spin_until_future_complete(node, future, 10s);
    auto response = future.get();
    EXPECT_TRUE(response->success);
    EXPECT_EQ(response->message, "Already using strategy: emulator");
  }
  // PORT: unknown strategy -> failure naming the available list.
  {
    auto request = std::make_shared<curobov2_ros_interfaces::srv::SetRobotStrategy::Request>();
    request->robot_strategy = "bogus";
    auto future = set_client->async_send_request(request);
    rclcpp::spin_until_future_complete(node, future, 10s);
    auto response = future.get();
    EXPECT_FALSE(response->success);
    EXPECT_EQ(
      response->message,
      "Unknown strategy: 'bogus'. Available: emulator, joint_pose, joint_speed");
  }
  {
    auto future = get_client->async_send_request(
      std::make_shared<curobov2_ros_interfaces::srv::GetRobotStrategies::Request>());
    rclcpp::spin_until_future_complete(node, future, 10s);
    auto response = future.get();
    EXPECT_TRUE(response->success);
    EXPECT_EQ(response->current_strategy_name, "emulator");
    ASSERT_EQ(response->strategy_names.size(), 3u);
  }
}

// PORT: robot_registry.available_strategies / create_strategy (M5 strategies)
TEST(RobotStrategy, RegistryCreatesRealRobotStrategies)
{
  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
  }
  auto node = std::make_shared<rclcpp::Node>("unified_planner");
  const auto keys = curobo_core::RobotRegistry::available_strategies();
  EXPECT_EQ(keys, (std::vector<std::string>{"emulator", "joint_pose", "joint_speed"}));

  auto joint_speed = curobo_core::RobotRegistry::create(
    "joint_speed", node.get(), 0.02, joints(),
    {{"action_topic", "/j"}, {"max_joint_accel_dps", "45"}, {"accel_clamp_dt", "0.02"}});
  ASSERT_NE(joint_speed, nullptr);
  EXPECT_EQ(joint_speed->get_joint_name(), joints());

  auto joint_pose = curobo_core::RobotRegistry::create(
    "joint_pose", node.get(), 0.02, joints(), {{"action_topic", "/j"}});
  ASSERT_NE(joint_pose, nullptr);

  // ghost exists in Python but is not registered -> None.
  EXPECT_EQ(curobo_core::RobotRegistry::create("ghost", node.get(), 0.02, joints()),
    nullptr);
}

// PORT: joint_speed_strategy._clamp_velocities
TEST(JointSpeed, ClampsAccelerationAgainstMeasuredVelocity)
{
  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
  }
  auto node = std::make_shared<rclcpp::Node>("unified_planner");
  curobo_core::JointSpeedStrategy strategy(
    node.get(), 0.02, joints(),
    {{"action_topic", "/j"}, {"max_joint_accel_dps", "45"}, {"accel_clamp_dt", "0.02"}});

  // Measured velocity = 0 (feedback), commanded jump = 10 rad/s on joint 1.
  sensor_msgs::msg::JointState state;
  state.name = joints();
  state.position.assign(7, 0.0);
  state.velocity.assign(7, 0.0);
  strategy.callback_joint_pose(state);

  const auto clamped = strategy.clamp_velocities({{10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}});
  const double max_dv = 45.0 * 3.14159265358979323846 / 180.0 * 0.02;
  ASSERT_EQ(clamped.size(), 1u);
  EXPECT_NEAR(clamped[0][0], max_dv, 1e-9);
  EXPECT_DOUBLE_EQ(clamped[0][1], 0.0);
}

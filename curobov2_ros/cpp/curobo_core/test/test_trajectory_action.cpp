// SPDX-License-Identifier: Apache-2.0
// In-process tests for the SendTrajectory action (no GPU): the open-loop
// refusal and a verbatim replay streamed through the emulator.
//
// spin_until_future_complete is used here deliberately: it is fine in a TEST.
// The node never uses it (UD-5).
#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "curobo_core/core/planner_services.hpp"
#include "curobo_core/core/trajectory_action_server.hpp"
#include "curobo_core/robot/robot_context.hpp"
#include "curobo_runtime/fake_bridge.hpp"
#include "curobov2_ros_interfaces/action/send_trajectory.hpp"

using namespace std::chrono_literals;
using curobo_core::PlannerServices;
using curobo_core::RobotContext;
using curobo_core::TrajectoryActionServer;
using SendTrajectory = curobov2_ros_interfaces::action::SendTrajectory;

namespace
{
std::vector<std::string> joints()
{
  return {"panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7"};
}

class ActionTest : public ::testing::Test
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
void spin(rclcpp::Node::SharedPtr node, FutureT & future)
{
  rclcpp::spin_until_future_complete(node, future, 10s);
}
}  // namespace

TEST_F(ActionTest, OpenLoopGoalWithoutTrajectoryIsRefusedWithTheSourceWording)
{
  auto node = std::make_shared<rclcpp::Node>("unified_planner");
  curobo_runtime::FakeBridge bridge;
  PlannerServices planners(node.get(), &bridge, 1, joints());
  RobotContext context(node.get(), 0.001, joints(), "emulator", {}, std::vector<double>(7, 0.0));
  TrajectoryActionServer server(node.get(), &planners, &context, joints());

  auto client = rclcpp_action::create_client<SendTrajectory>(
    node, "unified_planner/execute_trajectory");
  ASSERT_TRUE(client->wait_for_action_server(5s));

  auto goal = std::make_shared<SendTrajectory::Goal>();
  goal->replay = false;
  auto goal_future = client->async_send_goal(*goal);
  spin(node, goal_future);
  auto handle = goal_future.get();
  ASSERT_NE(handle, nullptr);

  auto result_future = client->async_get_result(handle);
  spin(node, result_future);
  auto wrapped = result_future.get();
  EXPECT_EQ(wrapped.code, rclcpp_action::ResultCode::ABORTED);
  EXPECT_EQ(
    wrapped.result->result.message,
    "Classic Motion Generation: open-loop execution needs a supplied trajectory "
    "(replay_trajectory); re-planning during execute is not supported");
}

TEST_F(ActionTest, ReplayStreamsTheSuppliedTrajectoryThroughTheRobotContext)
{
  auto node = std::make_shared<rclcpp::Node>("unified_planner");
  curobo_runtime::FakeBridge bridge;
  PlannerServices planners(node.get(), &bridge, 1, joints());
  RobotContext context(node.get(), 0.001, joints(), "emulator", {}, std::vector<double>(7, 0.0));
  TrajectoryActionServer server(node.get(), &planners, &context, joints());

  auto client = rclcpp_action::create_client<SendTrajectory>(
    node, "unified_planner/execute_trajectory");
  ASSERT_TRUE(client->wait_for_action_server(5s));

  auto goal = std::make_shared<SendTrajectory::Goal>();
  goal->replay = true;
  goal->replay_trajectory.joint_names = joints();
  for (double v : {0.1, 0.2}) {
    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions.assign(7, v);
    point.velocities.assign(7, 0.0);
    goal->replay_trajectory.points.push_back(point);
  }

  auto goal_future = client->async_send_goal(*goal);
  spin(node, goal_future);
  auto handle = goal_future.get();
  ASSERT_NE(handle, nullptr);

  auto result_future = client->async_get_result(handle);
  spin(node, result_future);
  auto wrapped = result_future.get();
  EXPECT_EQ(wrapped.code, rclcpp_action::ResultCode::SUCCEEDED);
  EXPECT_EQ(wrapped.result->result.message, "Execution completed");
  // The emulator drove the supplied waypoints.
  EXPECT_DOUBLE_EQ(context.get_joint_pose()[0], 0.2);
}

// cuRobo's interpolated plan carries EVERY model joint: a replay of a
// full-joint plan (active joints + appended locked columns) must be projected
// onto the active joints, or the controller rejects the goal outright
// ("Joints on incoming trajectory don't match the controller joints").
TEST_F(ActionTest, ReplayProjectsTheLockedJointTailOntoTheActiveJoints)
{
  auto node = std::make_shared<rclcpp::Node>("unified_planner");
  curobo_runtime::FakeBridge bridge;
  PlannerServices planners(node.get(), &bridge, 1, joints());
  RobotContext context(node.get(), 0.001, joints(), "emulator", {}, std::vector<double>(7, 0.0));
  TrajectoryActionServer server(node.get(), &planners, &context, joints());

  auto client = rclcpp_action::create_client<SendTrajectory>(
    node, "unified_planner/execute_trajectory");
  ASSERT_TRUE(client->wait_for_action_server(5s));

  std::vector<std::string> full = joints();
  full.push_back("panda_finger_joint1");
  full.push_back("panda_finger_joint2");

  auto goal = std::make_shared<SendTrajectory::Goal>();
  goal->replay = true;
  goal->replay_trajectory.joint_names = full;
  for (double v : {0.1, 0.2}) {
    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions.assign(9, v);
    point.positions[7] = 99.0;  // locked columns must never reach the robot
    point.positions[8] = 99.0;
    point.velocities.assign(9, 0.0);
    goal->replay_trajectory.points.push_back(point);
  }

  auto goal_future = client->async_send_goal(*goal);
  spin(node, goal_future);
  auto handle = goal_future.get();
  ASSERT_NE(handle, nullptr);

  auto result_future = client->async_get_result(handle);
  spin(node, result_future);
  auto wrapped = result_future.get();
  EXPECT_EQ(wrapped.code, rclcpp_action::ResultCode::SUCCEEDED);
  // Only the 7 active joints were streamed; the locked tail was dropped.
  EXPECT_EQ(context.get_joint_pose().size(), 7u);
  EXPECT_DOUBLE_EQ(context.get_joint_pose()[0], 0.2);
}

// SPDX-License-Identifier: Apache-2.0
//
// M5: the reactive control loop (ReactiveController + MPCController) with the
// FakeBridge double and a real RobotContext. No GPU.
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "curobo_core/planners/mpc_planner.hpp"
#include "curobo_core/planners/retarget_controller.hpp"
#include "curobo_core/robot/robot_context.hpp"
#include "curobo_runtime/fake_bridge.hpp"

using curobo_core::MPCController;
using curobo_core::RetargetController;
using curobo_core::ReactiveExecutionContext;
using curobo_core::ReactiveGoal;
using curobo_core::RobotContext;

namespace
{
std::vector<std::string> joint_names()
{
  return {"panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7"};
}
}  // namespace

// PORT: ReactiveController._execute_immediate (the free-running servo loop)
TEST(ReactiveController, ImmediateLoopRunsStepsAndPublishesFeedback)
{
  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
  }
  auto node = std::make_shared<rclcpp::Node>("unified_planner");
  node->declare_parameter<double>("mpc_command_interval", 0.0);
  curobo_runtime::FakeBridge bridge;
  RobotContext robot_context(
    node.get(), 0.025, joint_names(), "emulator", {},
    std::vector<double>(7, 0.0));
  MPCController mpc(&bridge, 7, joint_names(), node.get());

  ReactiveGoal goal;
  goal.goalsets = {{{0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0}}};
  goal.start_pose_positions = std::vector<double>(7, 0.0);

  const auto plan_result = mpc.plan_reactive(goal, 0.01, 1000, &robot_context);
  ASSERT_TRUE(plan_result.success) << plan_result.message;
  EXPECT_EQ(bridge.mpc_builds(), 1);
  EXPECT_TRUE(mpc.is_goal_active());

  int checks = 0;
  int feedback = 0;
  ReactiveExecutionContext ctx;
  ctx.robot_context = &robot_context;
  ctx.cancel_requested = [&checks]() {return ++checks > 4;};
  ctx.feedback = [&feedback](
    const std::string &, double, bool, const std::vector<double> &, double) {feedback++;};

  EXPECT_TRUE(mpc.execute(ctx));
  EXPECT_EQ(bridge.mpc_steps(), 4);            // one step per accepted iteration
  EXPECT_GE(feedback, 1);                      // feedback on tstep 0
  EXPECT_GT(mpc.get_position_error(), 0.0f);   // still far from the goal
}

// PORT: ReactiveController.plan - exactly one goalset entry with a single pose
TEST(ReactiveController, RejectsMultiPoseGoalset)
{
  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
  }
  auto node = std::make_shared<rclcpp::Node>("unified_planner");
  curobo_runtime::FakeBridge bridge;
  RobotContext robot_context(
    node.get(), 0.025, joint_names(), "emulator", {},
    std::vector<double>(7, 0.0));
  MPCController mpc(&bridge, 7, joint_names(), node.get());

  ReactiveGoal goal;
  goal.goalsets = {{
      {0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0},
      {0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0}}};
  goal.start_pose_positions = std::vector<double>(7, 0.0);

  const auto plan_result = mpc.plan_reactive(goal, 0.01, 1000, &robot_context);
  EXPECT_FALSE(plan_result.success);
  EXPECT_FALSE(mpc.is_goal_active());
}

// PORT: ReactiveController.execute -> _execute_paced when a command interval is set
TEST(ReactiveController, PacedLoopUsesStepPaced)
{
  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
  }
  auto node = std::make_shared<rclcpp::Node>("unified_planner");
  node->declare_parameter<double>("mpc_command_interval", 0.01);
  curobo_runtime::FakeBridge bridge;
  RobotContext robot_context(
    node.get(), 0.025, joint_names(), "emulator", {},
    std::vector<double>(7, 0.0));
  MPCController mpc(&bridge, 7, joint_names(), node.get());

  ReactiveGoal goal;
  goal.goalsets = {{{0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0}}};
  goal.start_pose_positions = std::vector<double>(7, 0.0);
  ASSERT_TRUE(mpc.plan_reactive(goal, 0.01, 1000, &robot_context).success);

  int checks = 0;
  ReactiveExecutionContext ctx;
  ctx.robot_context = &robot_context;
  ctx.cancel_requested = [&checks]() {return ++checks > 1;};
  ctx.feedback = [](const std::string &, double, bool, const std::vector<double> &, double) {};

  EXPECT_TRUE(mpc.execute(ctx));
  EXPECT_EQ(bridge.mpc_paced_steps(), 1);
}

// PORT: ReactiveController.cancel
TEST(ReactiveController, CancelClearsGoalActive)
{
  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
  }
  auto node = std::make_shared<rclcpp::Node>("unified_planner");
  curobo_runtime::FakeBridge bridge;
  RobotContext robot_context(
    node.get(), 0.025, joint_names(), "emulator", {},
    std::vector<double>(7, 0.0));
  MPCController mpc(&bridge, 7, joint_names(), node.get());

  ReactiveGoal goal;
  goal.goalsets = {{{0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0}}};
  goal.start_pose_positions = std::vector<double>(7, 0.0);
  ASSERT_TRUE(mpc.plan_reactive(goal, 0.01, 1000, &robot_context).success);
  ASSERT_TRUE(mpc.is_goal_active());

  mpc.cancel();
  EXPECT_FALSE(mpc.is_goal_active());
}

// PORT: RetargetController.step (solve_frame servo loop)
TEST(ReactiveController, RetargetStepsThroughSolveFrame)
{
  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
  }
  auto node = std::make_shared<rclcpp::Node>("unified_planner");
  curobo_runtime::FakeBridge bridge;
  RobotContext robot_context(
    node.get(), 0.025, joint_names(), "emulator", {},
    std::vector<double>(7, 0.0));
  RetargetController retarget(&bridge, 7, joint_names(), node.get());

  ReactiveGoal goal;
  goal.goalsets = {{{0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0}}};
  goal.start_pose_positions = std::vector<double>(7, 0.0);
  ASSERT_TRUE(retarget.plan_reactive(goal, 0.01, 1000, &robot_context).success);

  int checks = 0;
  ReactiveExecutionContext ctx;
  ctx.robot_context = &robot_context;
  ctx.cancel_requested = [&checks]() {return ++checks > 3;};
  ctx.feedback = [](const std::string &, double, bool, const std::vector<double> &, double) {};

  EXPECT_TRUE(retarget.execute(ctx));
  EXPECT_EQ(bridge.retarget_steps(), 3);
}

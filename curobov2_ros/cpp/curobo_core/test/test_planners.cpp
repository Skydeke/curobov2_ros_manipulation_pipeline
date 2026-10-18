// SPDX-License-Identifier: Apache-2.0
// Unit tests for the planner catalog/manager, the zero-delta guard and the
// planner names/modes. No GPU (FakeBridge), no ROS graph.
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "curobo_core/planners/cartesian_planner.hpp"
#include "curobo_core/planners/classic_planner.hpp"
#include "curobo_core/planners/joint_space_planner.hpp"
#include "curobo_core/planners/planner_factory.hpp"
#include "curobo_core/planners/zero_motion.hpp"
#include "curobo_runtime/fake_bridge.hpp"

using curobo_core::ExecutionMode;
using curobo_core::is_zero_delta;
using curobo_core::PlannerFactory;
using curobo_core::PlannerManager;

TEST(ZeroMotion, DetectsEqualAndDifferentJointVectors)
{
  EXPECT_TRUE(is_zero_delta({0.0f, 1.0f, 2.0f}, {0.0f, 1.0f, 2.0f}));
  EXPECT_FALSE(is_zero_delta({0.0f, 1.0f, 2.0f}, {0.0f, 1.0f, 2.1f}));
  // Length disagreement is NOT a zero delta (a genuine partial goal).
  EXPECT_FALSE(is_zero_delta({0.0f, 1.0f}, {0.0f, 1.0f, 2.0f}));
  EXPECT_FALSE(is_zero_delta({}, {}));
}

TEST(PlannerFactory, CatalogAndCreation)
{
  // PORT: PlannerFactory._PLANNER_CATALOG
  ASSERT_EQ(PlannerFactory::catalog().size(), 5u);
  EXPECT_EQ(std::string(PlannerFactory::catalog()[0].key), "classic");
  EXPECT_EQ(PlannerFactory::catalog()[0].id, 0);
  EXPECT_EQ(std::string(PlannerFactory::catalog()[3].key), "joint_space");
  EXPECT_EQ(PlannerFactory::catalog()[3].id, 5);

  curobo_runtime::FakeBridge bridge;
  auto classic = PlannerFactory::create("classic", &bridge, 1, {"j1", "j2"});
  ASSERT_NE(classic, nullptr);
  EXPECT_EQ(classic->get_planner_name(), "Classic Motion Generation");
  EXPECT_TRUE(classic->is_open_loop());
  EXPECT_EQ(classic->execution_mode(), ExecutionMode::kOpenLoop);

  auto joint_space = PlannerFactory::create("joint_space", &bridge, 1, {"j1", "j2"});
  ASSERT_NE(joint_space, nullptr);
  EXPECT_EQ(joint_space->get_planner_name(), "Joint Space Motion Generation");

  // Alias (PORT: _PLANNER_REGISTRY).
  auto alias = PlannerFactory::create("motion_gen", &bridge, 1, {});
  ASSERT_NE(alias, nullptr);
  EXPECT_EQ(alias->get_planner_name(), "Classic Motion Generation");

  // PORT: closed-loop planners are ported at M5 (MPC now; retarget next).
  auto mpc = PlannerFactory::create("mpc", &bridge, 1, {});
  ASSERT_NE(mpc, nullptr);
  EXPECT_EQ(mpc->get_planner_name(), "Model Predictive Control (MPC)");
  EXPECT_TRUE(mpc->is_closed_loop());
  EXPECT_EQ(mpc->execution_mode(), ExecutionMode::kClosedLoop);
  auto retarget = PlannerFactory::create("retarget", &bridge, 1, {});
  ASSERT_NE(retarget, nullptr);
  EXPECT_EQ(retarget->get_planner_name(), "Motion Retargeting (Teleop)");
  EXPECT_TRUE(retarget->is_closed_loop());
}

TEST(PlannerManager, SwitchesAndCaches)
{
  curobo_runtime::FakeBridge bridge;
  PlannerManager manager(&bridge, 1, {"j1", "j2"});
  ASSERT_NE(manager.current(), nullptr);
  EXPECT_EQ(manager.current()->get_planner_name(), "Classic Motion Generation");
  EXPECT_EQ(manager.current_type(), "classic");

  manager.set_current_planner("joint_space");
  EXPECT_EQ(manager.current()->get_planner_name(), "Joint Space Motion Generation");
  EXPECT_EQ(manager.current_type(), "joint_space");

  // Switching back returns the cached instance (same pointer).
  auto * first_classic = manager.get_planner("classic");
  manager.set_current_planner("classic");
  EXPECT_EQ(manager.current(), first_classic);
}

// (segment_distance / _achieved_fraction coverage removed with the reference's
// inert min_fraction gate - round 32.)

TEST(CartesianPlanner, NameRequestGateAndDefaults)
{
  curobo_runtime::FakeBridge bridge;
  auto planner = PlannerFactory::create("cartesian", &bridge, 1, {"j1", "j2", "j3"});
  ASSERT_NE(planner, nullptr);
  EXPECT_EQ(planner->get_planner_name(), "Cartesian Motion Generation");

  // PORT: _is_cartesian_request - ONE goalset holding >= 1 pose, no joint
  // target. The gate is on the GOALSET count (a segmented path with N goalsets
  // is refused); an MTC CartesianPath leg sends ONE goalset with several
  // candidate goal poses.
  curobo_runtime::PlanRequest ok;
  ok.goalset.shape = {1, 7};
  ok.goalset.data.assign(7, 0.0f);
  ok.start = {0.0f, 0.0f, 0.0f};
  EXPECT_TRUE(curobo_core::CartesianPlanner::is_cartesian_request(ok));

  // One goalset with SEVERAL candidate poses is STILL a Cartesian request
  // (regression guard: the count compared must be goalsets, not poses).
  curobo_runtime::PlanRequest poses = ok;
  poses.goalset.shape = {3, 7};
  poses.goalset.data.assign(21, 0.0f);
  EXPECT_TRUE(curobo_core::CartesianPlanner::is_cartesian_request(poses));

  curobo_runtime::PlanRequest joint = ok;
  joint.joint_target = {0.0f, 0.0f, 0.0f};
  EXPECT_FALSE(curobo_core::CartesianPlanner::is_cartesian_request(joint));

  // TWO goalsets (a segmented path) is not a Cartesian request.
  curobo_runtime::PlanRequest two = ok;
  two.goalset_count = 2;
  EXPECT_FALSE(curobo_core::CartesianPlanner::is_cartesian_request(two));

  // A non-Cartesian request is refused with the classic "no solution" wording.
  auto result = planner->plan(joint);
  EXPECT_FALSE(result.success);
  EXPECT_EQ(result.message, "Planning failed: no solution found (plan_pose returned None)");
}

// SPDX-License-Identifier: Apache-2.0
// MULTIPOINT (4) / Cartesian Waypoint: catalog entry, ordered-path validation
// and the request it forwards to the bridge. No GPU, no ROS graph.
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "curobo_core/planners/multipoint_planner.hpp"
#include "curobo_core/planners/planner_factory.hpp"
#include "curobo_runtime/fake_bridge.hpp"

namespace
{

class CapturingBridge : public curobo_runtime::FakeBridge
{
public:
  curobo_runtime::PlanOutcome plan_pose(
    curobo_runtime::RobotHandle, const curobo_runtime::PlanRequest & request) override
  {
    last_request = request;
    plan_pose_calls++;
    curobo_runtime::PlanOutcome out;
    out.success = true;
    out.feasible = true;
    out.status = "success";
    out.positions.shape = {2, static_cast<std::int64_t>(request.start.size())};
    out.positions.data = request.start;
    out.positions.data.insert(
      out.positions.data.end(), request.start.begin(), request.start.end());
    return out;
  }

  curobo_runtime::PlanRequest last_request;
  int plan_pose_calls = 0;
};

curobo_runtime::PlanRequest waypoint_request()
{
  curobo_runtime::PlanRequest request;
  request.start = {0.0f, 0.0f, 0.0f};
  request.waypoints.shape = {2, 7};
  request.waypoints.data = {
    0.1f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    0.2f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
  request.waypoint_count = 2;
  request.path_position_tolerance = 0.04;
  request.path_orientation_tolerance = 0.2;
  request.final_position_tolerance = 0.004;
  request.final_orientation_tolerance = 0.03;
  request.track_orientation = false;
  request.outer_iterations = 7;
  return request;
}

}  // namespace

TEST(MultipointPlanner, CatalogEntryAndFactory)
{
  ASSERT_EQ(curobo_core::PlannerFactory::catalog().size(), 6u);
  bool found = false;
  for (const auto & entry : curobo_core::PlannerFactory::catalog()) {
    if (entry.id == 4) {
      found = true;
      EXPECT_EQ(std::string(entry.key), "multipoint");
      EXPECT_EQ(std::string(entry.display), "Cartesian Waypoint");
    }
  }
  EXPECT_TRUE(found);

  curobo_runtime::FakeBridge bridge;
  auto planner = curobo_core::PlannerFactory::create(
    "multipoint", &bridge, 1, {"j1", "j2", "j3"});
  ASSERT_NE(planner, nullptr);
  EXPECT_EQ(
    planner->get_planner_name(), "Cartesian Waypoint Motion Generation");
  EXPECT_TRUE(planner->is_open_loop());

  auto alias = curobo_core::PlannerFactory::create(
    "cartesian_waypoint", &bridge, 1, {"j1"});
  ASSERT_NE(alias, nullptr);
  EXPECT_EQ(
    alias->get_planner_name(), "Cartesian Waypoint Motion Generation");
}

TEST(MultipointPlanner, ForwardsTheOrderedPathToTheBridge)
{
  CapturingBridge bridge;
  curobo_core::MultipointPlanner planner(&bridge, 1, {"j1", "j2", "j3"});

  const auto request = waypoint_request();
  EXPECT_TRUE(curobo_core::MultipointPlanner::has_ordered_waypoints(request));

  const auto result = planner.plan(request);
  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.planner_type, "multipoint");
  EXPECT_EQ(bridge.plan_pose_calls, 1);
  EXPECT_EQ(bridge.last_request.waypoint_count, 2);
  EXPECT_EQ(bridge.last_request.waypoints.shape, (std::vector<std::int64_t>{2, 7}));
  EXPECT_DOUBLE_EQ(bridge.last_request.path_position_tolerance, 0.04);
  EXPECT_DOUBLE_EQ(bridge.last_request.path_orientation_tolerance, 0.2);
  EXPECT_DOUBLE_EQ(bridge.last_request.final_position_tolerance, 0.004);
  EXPECT_DOUBLE_EQ(bridge.last_request.final_orientation_tolerance, 0.03);
  EXPECT_FALSE(bridge.last_request.track_orientation);
  EXPECT_EQ(bridge.last_request.outer_iterations, 7);
}

TEST(MultipointPlanner, RefusesRequestsWithoutAnOrderedPath)
{
  curobo_runtime::FakeBridge bridge;
  curobo_core::MultipointPlanner planner(&bridge, 1, {"j1", "j2", "j3"});

  curobo_runtime::PlanRequest legacy;
  legacy.start = {0.0f, 0.0f, 0.0f};
  legacy.goalset.shape = {1, 7};
  legacy.goalset.data.assign(7, 0.0f);
  EXPECT_FALSE(curobo_core::MultipointPlanner::has_ordered_waypoints(legacy));
  auto result = planner.plan(legacy);
  EXPECT_FALSE(result.success);
  EXPECT_NE(result.message.find("no ordered waypoint path"), std::string::npos);

  curobo_runtime::PlanRequest both = waypoint_request();
  both.tool_waypoints.shape = {1, 7};
  both.tool_waypoints.data.assign(7, 0.0f);
  result = planner.plan(both);
  EXPECT_FALSE(result.success);
  EXPECT_NE(result.message.find("mutually exclusive"), std::string::npos);

  curobo_runtime::PlanRequest joint = waypoint_request();
  joint.joint_target = {0.0f, 0.0f, 0.0f};
  result = planner.plan(joint);
  EXPECT_FALSE(result.success);
  EXPECT_NE(result.message.find("target_joint_positions"), std::string::npos);
}

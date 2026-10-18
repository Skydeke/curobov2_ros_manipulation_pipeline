// SPDX-License-Identifier: Apache-2.0
// M5 parity: the collision-feedback formatting matches the Python reference
// (curobov2_ros/core/unified_planner_node.py:_format_state_collisions). The
// expected strings are the exact layout observed in the Python reference logs
// (see tools/harness/m0/launch_tests/test_test_trajectory_franka.log).
#include <gtest/gtest.h>

#include <string>

#include "curobo_core/core/collision_diagnostics.hpp"

namespace
{

curobo_runtime::WorldContact world(
  const std::string & link, const std::string & obstacle, double depth)
{
  curobo_runtime::WorldContact c;
  c.link = link;
  c.obstacle = obstacle;
  c.depth = depth;
  return c;
}

curobo_runtime::SelfContact self(
  const std::string & a, const std::string & b, double depth)
{
  curobo_runtime::SelfContact c;
  c.link_a = a;
  c.link_b = b;
  c.depth = depth;
  return c;
}

curobo_runtime::CspaceViolation cspace(
  const std::string & joint, double value, double lower, double upper)
{
  curobo_runtime::CspaceViolation c;
  c.joint = joint;
  c.value = value;
  c.lower = lower;
  c.upper = upper;
  return c;
}

}  // namespace

// PORT: _format_state_collisions - empty attribution -> the "clear" line.
TEST(CollisionDiagnostics, ClearStateIsReportedClear)
{
  curobo_runtime::CollisionAttribution sc;
  EXPECT_EQ(
    curobo_core::format_state_collisions("start", sc),
    "  start: clear (no world/self contacts, within joint limits)");
}

// PORT: _format_state_collisions - one row per world contact, link column
// padded to the widest link, deepest contact first.
TEST(CollisionDiagnostics, WorldContactsAlignedDeepestFirst)
{
  curobo_runtime::CollisionAttribution sc;
  // Inserted shallow-first to prove the sort.
  sc.world_contacts.push_back(world("panda_link5", "box", 0.010));   // 10.0 mm
  sc.world_contacts.push_back(world("panda_link0", "box", 0.100));   // 100.0 mm

  const std::string expected =
    "  start: 2 contacts\n"
    "    panda_link0  ->  box  100.0 mm past surface\n"
    "    panda_link5  ->  box  10.0 mm past surface";
  EXPECT_EQ(curobo_core::format_state_collisions("start", sc), expected);
}

// PORT: _format_state_collisions - self contacts use "<->"/"penetration" and
// joint-limit rows use "limit"/"outside [lo, hi] (value v)" and sort last.
TEST(CollisionDiagnostics, SelfContactsAndCspaceViolations)
{
  curobo_runtime::CollisionAttribution sc;
  sc.self_contacts.push_back(self("panda_link1", "panda_link3", 0.005));
  sc.cspace_violations.push_back(cspace("joint_2", 3.5, -3.0, 3.0));

  const std::string expected =
    "  start: 2 contacts\n"
    "    panda_link1  <->  panda_link3  5.0 mm penetration\n"
    "    joint_2      limit  outside [-3.000, 3.000] (value 3.500)";
  EXPECT_EQ(curobo_core::format_state_collisions("start", sc), expected);
}

// PORT: singular noun for a single contact.
TEST(CollisionDiagnostics, SingleContactUsesSingularNoun)
{
  curobo_runtime::CollisionAttribution sc;
  sc.world_contacts.push_back(world("panda_link7", "mesh:rubber_duck", 0.2674));

  const std::string expected =
    "  goal: 1 contact\n"
    "    panda_link7  ->  mesh:rubber_duck  267.4 mm past surface";
  EXPECT_EQ(curobo_core::format_state_collisions("goal", sc), expected);
}

// PORT: _format_collision_feedback - the exact block the Python reference logs
// on a failed plan with no joint-space goal.
TEST(CollisionDiagnostics, FeedbackBlockMatchesReferenceLog)
{
  curobo_runtime::CollisionAttribution start;
  curobo_runtime::CollisionAttribution goal;

  const std::string expected =
    "collision feedback:\n"
    "  start: clear (no world/self contacts, within joint limits)\n"
    "  goal: clear (no world/self contacts, within joint limits)";
  EXPECT_EQ(curobo_core::format_collision_feedback(start, goal), expected);
}

// PORT: the clobber note ("mesh:x or depth-voxel-layer") is printed verbatim.
TEST(CollisionDiagnostics, ClobberNotePassesThrough)
{
  curobo_runtime::CollisionAttribution sc;
  sc.world_contacts.push_back(
    world("panda_link0", "mesh:rubber_duck or depth-voxel-layer", 0.2093));

  const std::string expected =
    "  start: 1 contact\n"
    "    panda_link0  ->  mesh:rubber_duck or depth-voxel-layer  209.3 mm past surface";
  EXPECT_EQ(curobo_core::format_state_collisions("start", sc), expected);
}

// PORT: the reference log's column width comes from the widest link name
// ("panda_rightfinger", 17 chars) - reproduces the exact reference row layout.
TEST(CollisionDiagnostics, WidthComesFromWidestLinkName)
{
  curobo_runtime::CollisionAttribution sc;
  sc.world_contacts.push_back(world("panda_link0", "box", 0.100));
  sc.world_contacts.push_back(world("panda_rightfinger", "box", 0.010));

  const std::string expected =
    "  start: 2 contacts\n"
    "    panda_link0        ->  box  100.0 mm past surface\n"
    "    panda_rightfinger  ->  box  10.0 mm past surface";
  EXPECT_EQ(curobo_core::format_state_collisions("start", sc), expected);
}

// PORT: round(x, 6) on every value entering the StateCollisions message.
TEST(CollisionDiagnostics, Round6MatchesPython)
{
  EXPECT_DOUBLE_EQ(curobo_core::round6(0.1234567), 0.123457);
  EXPECT_DOUBLE_EQ(curobo_core::round6(0.1234561), 0.123456);
  EXPECT_DOUBLE_EQ(curobo_core::round6(209.29999999999998 / 1000.0), 0.2093);
  EXPECT_DOUBLE_EQ(curobo_core::round6(-0.0000001), 0.0);
}

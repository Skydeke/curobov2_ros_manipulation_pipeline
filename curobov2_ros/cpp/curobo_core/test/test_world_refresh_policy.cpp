// SPDX-License-Identifier: Apache-2.0
// C++ port of curobov2_ros/test/test_world_refresh_policy.py (design 14.2).
// Same test names, same inputs, same expected values.
#include <gtest/gtest.h>

#include "curobo_core/core/unified_planner_node.hpp"

using curobo_core::should_refresh_world;

// PORT: test_world_refresh_policy.py:test_force_with_unchanged_epoch_still_skips
TEST(WorldRefreshPolicy, ForceWithUnchangedEpochStillSkips)
{
  EXPECT_FALSE(should_refresh_world(true, 5, 5, 100.0, 99.9, 2.0));
}

// PORT: test_world_refresh_policy.py:test_force_with_new_epoch_refreshes
TEST(WorldRefreshPolicy, ForceWithNewEpochRefreshes)
{
  EXPECT_TRUE(should_refresh_world(true, 6, 5, 100.0, 99.9, 2.0));
}

// PORT: test_world_refresh_policy.py:test_unchanged_epoch_never_refreshes
TEST(WorldRefreshPolicy, UnchangedEpochNeverRefreshes)
{
  EXPECT_FALSE(should_refresh_world(false, 5, 5, 1000.0, 0.0, 2.0));
}

// PORT: test_world_refresh_policy.py:test_new_epoch_past_throttle_refreshes
TEST(WorldRefreshPolicy, NewEpochPastThrottleRefreshes)
{
  EXPECT_TRUE(should_refresh_world(false, 6, 5, 100.0, 97.0, 2.0));
}

// PORT: test_world_refresh_policy.py:test_new_epoch_within_throttle_skips
TEST(WorldRefreshPolicy, NewEpochWithinThrottleSkips)
{
  EXPECT_FALSE(should_refresh_world(false, 6, 5, 100.0, 99.9, 2.0));
}

// PORT: test_world_refresh_policy.py:test_never_pushed_always_refreshes
TEST(WorldRefreshPolicy, NeverPushedAlwaysRefreshes)
{
  EXPECT_TRUE(should_refresh_world(false, 0, -1, 100.0, 0.0, 2.0));
}

// PORT: test_world_refresh_policy.py:test_throttle_boundary_refreshes
TEST(WorldRefreshPolicy, ThrottleBoundaryRefreshes)
{
  EXPECT_TRUE(should_refresh_world(false, 6, 5, 102.0, 100.0, 2.0));
}

// PORT: test_world_refresh_policy.py:test_zero_interval_recovers_today_behavior
TEST(WorldRefreshPolicy, ZeroIntervalRecoversTodayBehavior)
{
  EXPECT_TRUE(should_refresh_world(false, 6, 5, 100.0, 100.0, 0.0));
}

// PORT: test_world_refresh_policy.py:test_reactive_forces_through_throttle
TEST(WorldRefreshPolicy, ReactiveForcesThroughThrottle)
{
  EXPECT_TRUE(should_refresh_world(true, 6, 5, 100.0, 100.0, 60.0));
}

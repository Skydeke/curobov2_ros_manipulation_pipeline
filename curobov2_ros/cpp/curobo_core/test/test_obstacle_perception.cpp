// SPDX-License-Identifier: Apache-2.0
// ObstacleManager perception half: ESDF refresh, dense voxel grid, sparse
// occupied indices. No GPU (FakeBridge returns a fixed 4x4x4 grid).
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "curobo_core/core/obstacle_manager.hpp"
#include "curobo_runtime/fake_bridge.hpp"

using curobo_core::ObstacleManager;

TEST(ObstaclePerception, RefreshEsdfReturnsTrueAndBumpsTheEpoch)
{
  curobo_runtime::FakeBridge bridge;
  ObstacleManager manager(&bridge, 1);
  EXPECT_EQ(manager.perception_epoch(), 0);
  EXPECT_TRUE(manager.refresh_esdf());
  EXPECT_TRUE(manager.refresh_esdf());
  EXPECT_EQ(manager.perception_epoch(), 2);
}

TEST(ObstaclePerception, DenseVoxelGridShapeAndOccupiedIndices)
{
  curobo_runtime::FakeBridge bridge;
  ObstacleManager manager(&bridge, 1);
  const auto grid = manager.get_voxel_grid();
  ASSERT_EQ(grid.shape, (std::vector<std::int64_t>{4, 4, 4}));
  EXPECT_EQ(grid.data.size(), 64u);
  EXPECT_DOUBLE_EQ(grid.resolution, 0.05);

  // Fake grid: cell (i,j,k) occupied iff (i+j+k) is even -> 32 of 64.
  const auto indices = ObstacleManager::occupied_indices(grid);
  EXPECT_EQ(indices.size(), 32u);
  // Row-major [X][Y][Z]: (0,0,0) -> 0 (even -> occupied), (0,0,1) -> 1 (odd).
  EXPECT_EQ(indices.front(), 0);
  EXPECT_EQ(indices[1], 2);
}

// Analytic scene primitives must never be OR-ed into the published voxel grid:
// the solver collides with them analytically, and their voxel approximation is
// the axis-aligned/thin box the voxel display used to show. This pins the C++
// layer's pass-through (the bridge is the only producer, and it is
// perception-only).
TEST(ObstaclePerception, AnalyticPrimitivesDoNotEnterTheVoxelGrid)
{
  curobo_runtime::FakeBridge bridge;
  curobo_runtime::VoxelGrid perception;
  perception.shape = {2, 2, 2};
  perception.resolution = 0.05;
  perception.origin = {0.0, 0.0, 0.0};
  perception.data = {1, 0, 0, 0, 0, 0, 0, 0};
  bridge.set_voxel_grid(perception);

  ObstacleManager manager(&bridge, 1);
  curobo_core::SceneObjectSpec cube;
  cube.type = curobo_core::CUBOID;
  cube.name = "table";
  cube.dims = {1.0, 1.0, 0.5};
  std::string message;
  ASSERT_TRUE(manager.add_object(cube, message));

  const auto grid = manager.get_voxel_grid();
  EXPECT_EQ(grid.data, perception.data);
  const auto indices = ObstacleManager::occupied_indices(grid);
  ASSERT_EQ(indices.size(), 1u);
  EXPECT_EQ(indices.front(), 0);
}

TEST(ObstaclePerception, ClearCountsComeFromTheBridge)
{
  curobo_runtime::FakeBridge bridge;
  ObstacleManager manager(&bridge, 1);
  EXPECT_EQ(manager.clear_dynamic_voxels(), 1);
  EXPECT_EQ(manager.clear_voxel_region({0, 0, 0}, {1, 1, 1}), 1);
}

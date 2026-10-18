// SPDX-License-Identifier: Apache-2.0
//
// M4: the dense voxel grid round-trips through the C++ layer unchanged. The
// ANALYTIC RASTERIZATION now lives in the bridge (curobo_bridge.voxel_grid ->
// _compute_dense_voxel_grid, the oracle's own GPU method); its parity with the
// recorded oracle is asserted by the m2_object_franka record/replay suite
// (get_voxel_grid's data hash). This test pins the C++ side: the bridge grid is
// passed through and ObstacleManager::occupied_indices reproduces the oracle's
// 1000 cells (expected_voxel_grid.hpp; see its header for the hash).
#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "curobo_core/core/obstacle_manager.hpp"
#include "curobo_runtime/fake_bridge.hpp"

#include "expected_voxel_grid.hpp"

namespace
{

// Minimal base64 decoder (RFC 4648, no whitespace inside the payload).
std::vector<std::uint8_t> base64_decode(const std::string & text)
{
  static const std::string alphabet =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::vector<std::uint8_t> out;
  int bits = 0;
  int value = 0;
  for (char c : text) {
    const std::size_t idx = alphabet.find(c);
    if (idx == std::string::npos) {
      continue;
    }
    value = (value << 6) | static_cast<int>(idx);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(static_cast<std::uint8_t>((value >> bits) & 0xFF));
    }
  }
  return out;
}

std::vector<std::int32_t> decode_indices(const std::string & b64)
{
  const auto bytes = base64_decode(b64);
  std::vector<std::int32_t> indices;
  indices.reserve(bytes.size() / 4);
  for (std::size_t i = 0; i + 4 <= bytes.size(); i += 4) {
    const std::uint32_t raw = static_cast<std::uint32_t>(bytes[i]) |
      (static_cast<std::uint32_t>(bytes[i + 1]) << 8) |
      (static_cast<std::uint32_t>(bytes[i + 2]) << 16) |
      (static_cast<std::uint32_t>(bytes[i + 3]) << 24);
    indices.push_back(static_cast<std::int32_t>(raw));
  }
  return indices;
}

}  // namespace

// PORT: obstacle_manager.get_voxel_grid (_compute_dense_voxel_grid)
TEST(VoxelGrid, RecordedObjectSceneParity)
{
  curobo_runtime::FakeBridge bridge;

  // The recorded server ran with mapper_extent_xyz [2,2,2], grid_center 0,
  // voxel_size 0.05 -> a 40^3 grid at origin -1.0 with no camera perception.
  const std::vector<std::int32_t> expected =
    decode_indices(curobo_core_test::kRecordedVoxelIndicesB64);

  // The bridge returns the oracle's grid; the C++ layer must pass it through.
  curobo_runtime::VoxelGrid from_bridge;
  from_bridge.shape = {
    curobo_core_test::kRecordedSize[0], curobo_core_test::kRecordedSize[1],
    curobo_core_test::kRecordedSize[2]};
  from_bridge.resolution = curobo_core_test::kRecordedResolution;
  from_bridge.origin = {
    curobo_core_test::kRecordedOrigin[0], curobo_core_test::kRecordedOrigin[1],
    curobo_core_test::kRecordedOrigin[2]};
  from_bridge.data.assign(
    static_cast<std::size_t>(curobo_core_test::kRecordedSize[0]) *
    curobo_core_test::kRecordedSize[1] * curobo_core_test::kRecordedSize[2], 0);
  for (std::int32_t index : expected) {
    from_bridge.data[static_cast<std::size_t>(index)] = 1;
  }
  bridge.set_voxel_grid(from_bridge);

  curobo_core::ObstacleManager manager(&bridge, 7);
  const auto grid = manager.get_voxel_grid();
  ASSERT_EQ(grid.shape.size(), 3u);
  EXPECT_EQ(grid.shape[0], curobo_core_test::kRecordedSize[0]);
  EXPECT_EQ(grid.shape[1], curobo_core_test::kRecordedSize[1]);
  EXPECT_EQ(grid.shape[2], curobo_core_test::kRecordedSize[2]);
  EXPECT_DOUBLE_EQ(grid.resolution, curobo_core_test::kRecordedResolution);

  const std::vector<std::int32_t> actual = curobo_core::ObstacleManager::occupied_indices(grid);
  ASSERT_EQ(expected.size(), 1000u);
  EXPECT_EQ(actual, expected);
}

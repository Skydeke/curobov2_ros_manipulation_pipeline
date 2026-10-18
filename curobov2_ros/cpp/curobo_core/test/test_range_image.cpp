// SPDX-License-Identifier: Apache-2.0
// LiDAR range-image projection (pure). No GPU.
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

#include "curobo_core/lasers/range_image.hpp"

using curobo_core::finite_range_mask;
using curobo_core::make_range_image;

TEST(RangeImage, FiniteRangeMaskFiltersByDistance)
{
  std::vector<std::array<double, 3>> xyz{{1, 0, 0}, {0, 0, 0}, {100, 0, 0},
    {std::nan(""), 0, 0}};
  std::vector<std::uint8_t> mask;
  std::vector<float> ranges;
  finite_range_mask(xyz, 0.1, 10.0, mask, ranges);
  EXPECT_EQ(mask[0], 1);     // range 1 in [0.1, 10]
  EXPECT_EQ(mask[1], 0);     // range 0 below 0.1
  EXPECT_EQ(mask[2], 0);     // range 100 above 10
  EXPECT_EQ(mask[3], 0);     // NaN
  EXPECT_FLOAT_EQ(ranges[0], 1.0f);
}

TEST(RangeImage, SingleRowAzimuthBinsThePointAndNeighbours)
{
  // W=4: (1,0,0) has azimuth 0 -> u = (pi)*(4/2pi) = 2.0 -> bins 2 and 3.
  std::vector<std::array<double, 3>> xyz{{1, 0, 0}};
  const auto image = make_range_image(xyz, 1, 4, 0.1, 10.0, -1.5, 1.5);
  ASSERT_EQ(image.range.size(), 4u);
  EXPECT_FLOAT_EQ(image.range[2], 1.0f);
  EXPECT_FLOAT_EQ(image.range[3], 1.0f);
  EXPECT_FLOAT_EQ(image.range[0], 0.0f);
  EXPECT_FLOAT_EQ(image.range[1], 0.0f);
  EXPECT_EQ(image.projected_pixels, 2);
  EXPECT_EQ(image.rgb.size(), 12u);
}

TEST(RangeImage, NearestPointWinsABin)
{
  // Two points project into the same bin; the nearer range wins.
  std::vector<std::array<double, 3>> xyz{{1, 0, 0}, {2, 0, 0}};
  const auto image = make_range_image(xyz, 1, 4, 0.1, 10.0, -1.5, 1.5);
  EXPECT_FLOAT_EQ(image.range[2], 1.0f);   // nearest, not 2.0
}

TEST(RangeImage, OutOfElevationIsDropped)
{
  // Point straight up (elevation pi/2) with a narrow elevation window.
  std::vector<std::array<double, 3>> xyz{{0, 0, 1}};
  const auto image = make_range_image(xyz, 2, 8, 0.1, 10.0, -0.5, 0.5);
  EXPECT_EQ(image.projected_pixels, 0);
  for (float v : image.range) {
    EXPECT_FLOAT_EQ(v, 0.0f);
  }
}

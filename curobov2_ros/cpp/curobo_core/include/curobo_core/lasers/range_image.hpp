// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/lasers/laser_pointcloud_strategy.py
//   finite_range_mask / make_range_image
//
// Geometry-only LiDAR range-image projection (no color/intensity). Pure math;
// no GPU.
#ifndef CUROBO_CORE__LASERS__RANGE_IMAGE_HPP_
#define CUROBO_CORE__LASERS__RANGE_IMAGE_HPP_

#include <array>
#include <cstdint>
#include <vector>

namespace curobo_core
{

struct RangeImage
{
  int height = 0;
  int width = 0;
  std::vector<float> range;           // (H, W), 0.0 = no measurement
  std::vector<std::uint8_t> rgb;      // (H, W, 3), zeros
  int projected_pixels = 0;
};

// PORT: finite_range_mask
void finite_range_mask(
  const std::vector<std::array<double, 3>> & xyz, double range_min, double range_max,
  std::vector<std::uint8_t> & mask, std::vector<float> & ranges);

// PORT: make_range_image
RangeImage make_range_image(
  const std::vector<std::array<double, 3>> & xyz, int image_height, int image_width,
  double range_min, double range_max, double elevation_min_rad, double elevation_max_rad);

}  // namespace curobo_core

#endif  // CUROBO_CORE__LASERS__RANGE_IMAGE_HPP_

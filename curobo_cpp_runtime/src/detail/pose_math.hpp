// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <vector>

namespace curobo_cpp_runtime
{
namespace detail
{

// Pose layout is [x, y, z, qw, qx, qy, qz].
using Pose7f = std::array<float, 7>;

Pose7f pose_from_values(const float * values);

Pose7f pose_multiply(const Pose7f & a, const Pose7f & b);

Pose7f pose_inverse(const Pose7f & p);

void fill_pose_arrays(
  const std::vector<Pose7f> & poses,
  std::vector<float> & positions, std::vector<float> & quaternions);

}  // namespace detail
}  // namespace curobo_cpp_runtime

// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/planners/zero_motion.py
//
// Server-side guard for degenerate joint-space requests (start == goal). Pure
// small-data logic; no GPU.
#ifndef CUROBO_CORE__PLANNERS__ZERO_MOTION_HPP_
#define CUROBO_CORE__PLANNERS__ZERO_MOTION_HPP_

#include <cmath>
#include <vector>

namespace curobo_core
{

// PORT: zero_motion.ZERO_DELTA_TOL
//
// Radians. A tolerance below float32 resolution can never fire for "already
// there": float32 quantizes the joint limits at ~5e-7 (ulp(2*pi)), and the
// continuous-joint snapper clamps a wrapped target to `hi - 1e-3`, which leaves
// a few ulps (~1e-6) between the snapped goal and the start. A 1e-9 tolerance
// missed those, so a stationary request went to cuRobo as a ~1e-6 rad motion -
// the degenerate CUDA-graph solve this guard exists to prevent. 1e-4 rad
// (0.006 deg) is still far below any motion worth planning.
inline constexpr double ZERO_DELTA_TOL = 1e-4;

// PORT: zero_motion.is_zero_delta
inline bool is_zero_delta(
  const std::vector<float> & start, const std::vector<float> & goal,
  double tol = ZERO_DELTA_TOL)
{
  if (start.empty() || start.size() != goal.size()) {
    return false;
  }
  for (std::size_t i = 0; i < start.size(); ++i) {
    if (std::fabs(static_cast<double>(start[i]) - static_cast<double>(goal[i])) > tol) {
      return false;
    }
  }
  return true;
}

}  // namespace curobo_core

#endif  // CUROBO_CORE__PLANNERS__ZERO_MOTION_HPP_

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
inline constexpr double ZERO_DELTA_TOL = 1e-9;

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

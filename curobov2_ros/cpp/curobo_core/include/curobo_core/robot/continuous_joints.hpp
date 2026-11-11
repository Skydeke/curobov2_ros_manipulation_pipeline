// SPDX-License-Identifier: Apache-2.0
// Continuous joints are circles, not bounded revolute joints.
//
// cuRobo plans in R^n and its URDF parser turns every `continuous` joint into a
// revolute joint with a fake [-2*pi, 2*pi] limit. That window is wide enough to
// hold one full turn, so it is a usable coordinate chart -- but only if every
// consumer treats it as a chart:
//
//   * a target is a pose, so pick its 2*pi-lift by SHORTEST PATH from the start;
//   * shift the whole segment by ONE common multiple of 2*pi so it fits the
//     chart (never clamp a point into the chart);
//   * shift the whole planned trajectory back by that same constant, and never
//     rewrite individual points after collision checking.
//
// This header is the single implementation of that arithmetic. It has no ROS or
// cuRobo dependency so the planner, execution, robot state and reactive paths
// all use exactly the same rules.
#ifndef CUROBO_CORE__ROBOT__CONTINUOUS_JOINTS_HPP_
#define CUROBO_CORE__ROBOT__CONTINUOUS_JOINTS_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace curobo_core
{

inline constexpr double kPiContinuous = 3.14159265358979323846;
inline constexpr double kTwoPiContinuous = 2.0 * kPiContinuous;

// Stay this far inside the model window. cuRobo rejects a goal pinned exactly
// to a limit, and the parser's 4*pi window is ~0.0032 rad narrower than two
// full turns, so a shortest-path lift can land just outside it. The margin is
// far below the smallest physical motion worth planning (1e-3 rad = 0.057 deg).
inline constexpr double kContinuousWindowMargin = 1e-3;

// Shortest signed representative in (-pi, pi]. The tie at exactly pi resolves
// to +pi deterministically, so two equivalent plans never oscillate between
// directions.
inline double wrap_pi(double x)
{
  if (!std::isfinite(x)) {
    return x;
  }
  double r = std::fmod(x, kTwoPiContinuous);
  if (r > kPiContinuous) {
    r -= kTwoPiContinuous;
  } else if (r <= -kPiContinuous) {
    r += kTwoPiContinuous;
  }
  return r;
}

// A joint whose reported position range spans a full turn is continuous. This
// is only the fallback for models that do not carry an explicit joint type; a
// descriptor's `continuous_joints` override is authoritative.
inline bool is_continuous_range(double lo, double hi)
{
  return std::isfinite(lo) && std::isfinite(hi) && (hi - lo) >= (kTwoPiContinuous - 1e-6);
}

inline bool is_continuous_joint(
  const std::vector<std::string> & continuous_names, const std::string & name)
{
  return std::find(continuous_names.begin(), continuous_names.end(), name) !=
         continuous_names.end();
}

// The 2*pi-equivalent of `value` nearest `reference`.
inline double periodic_lift(double value, double reference)
{
  if (!std::isfinite(value) || !std::isfinite(reference)) {
    return value;
  }
  return value + kTwoPiContinuous * std::round((reference - value) / kTwoPiContinuous);
}

// Choose the common whole-turn shift that puts BOTH the start lift and the
// target lift inside the model window. Returns false only when the window is
// narrower than one turn, which is a model/configuration error (fail closed).
inline bool choose_window_shift(
  double start_lift, double target_lift, double lo, double hi, double & shift)
{
  const double window_lo = lo + kContinuousWindowMargin;
  const double window_hi = hi - kContinuousWindowMargin;
  if (!std::isfinite(window_lo) || !std::isfinite(window_hi) ||
    (window_hi - window_lo) < (kTwoPiContinuous - 1e-9))
  {
    return false;
  }
  const double mid = 0.5 * (start_lift + target_lift);
  const double center = 0.5 * (window_lo + window_hi);
  const long long k_center = static_cast<long long>(std::llround((center - mid) / kTwoPiContinuous));
  const long long candidates[] = {
    k_center, k_center - 1, k_center + 1, k_center - 2, k_center + 2};
  for (const long long k : candidates) {
    const double s = start_lift + static_cast<double>(k) * kTwoPiContinuous;
    const double t = target_lift + static_cast<double>(k) * kTwoPiContinuous;
    if (s >= window_lo && s <= window_hi && t >= window_lo && t <= window_hi) {
      shift = static_cast<double>(k) * kTwoPiContinuous;
      return true;
    }
  }
  return false;
}

// Phase-unwrapping recursion for a measured continuous joint. The physical
// motion between two joint-state samples is assumed to be < pi; with normal
// joint-state rates that holds, and a larger jump is treated as a sample
// discontinuity, not silently extrapolated.
class ContinuousJointTracker
{
public:
  void configure(
    const std::vector<std::string> & joint_names,
    const std::vector<std::string> & continuous_names)
  {
    mask_.assign(joint_names.size(), false);
    for (std::size_t j = 0; j < joint_names.size(); ++j) {
      mask_[j] = is_continuous_joint(continuous_names, joint_names[j]);
    }
    last_raw_.assign(joint_names.size(), 0.0);
    unwrapped_.assign(joint_names.size(), 0.0);
    initialized_ = false;
  }

  bool configured() const {return !mask_.empty();}

  const std::vector<double> & unwrap(const std::vector<double> & raw)
  {
    if (mask_.empty() || raw.size() != mask_.size()) {
      unwrapped_ = raw;
      return unwrapped_;
    }
    if (!initialized_) {
      last_raw_ = raw;
      unwrapped_ = raw;
      initialized_ = true;
      return unwrapped_;
    }
    for (std::size_t j = 0; j < raw.size(); ++j) {
      if (mask_[j]) {
        const double delta = wrap_pi(raw[j] - last_raw_[j]);
        unwrapped_[j] += delta;
      } else {
        unwrapped_[j] = raw[j];
      }
      last_raw_[j] = raw[j];
    }
    return unwrapped_;
  }

  const std::vector<double> & current() const {return unwrapped_;}

private:
  std::vector<bool> mask_;
  std::vector<double> last_raw_;
  std::vector<double> unwrapped_;
  bool initialized_ = false;
};

// Result of anchoring a supplied trajectory to the measured 2*pi branch.
struct BranchAlignmentResult
{
  bool ok = true;
  std::string error;
};

// Shift `positions` as a WHOLE per continuous joint so its first point is on
// the measured branch. This preserves the path exactly (a constant 2*pi shift)
// and never rewrites a point to a different branch. A first point that is not
// the measured physical pose is rejected, as is a trajectory that contains a
// genuine > pi step on a continuous joint (that needs intermediate waypoints,
// not a silent unwrap).
inline BranchAlignmentResult align_trajectory_to_measured(
  const std::vector<std::string> & names,
  std::vector<std::vector<double>> & positions,
  const std::vector<std::string> & measured_names,
  const std::vector<double> & measured,
  const std::vector<std::string> & continuous_names,
  double start_tolerance = 0.1)
{
  BranchAlignmentResult result;
  if (positions.empty() || names.empty()) {
    return result;
  }
  const std::size_t columns = std::min(names.size(), positions.front().size());
  for (std::size_t j = 0; j < columns; ++j) {
    if (!is_continuous_joint(continuous_names, names[j])) {
      continue;
    }
    // Measured value for this joint by NAME; fall back to the same column only
    // when the measured vector has exactly this shape and no names.
    double q_measured = 0.0;
    bool have_measured = false;
    if (!measured_names.empty()) {
      const auto it = std::find(measured_names.begin(), measured_names.end(), names[j]);
      if (it != measured_names.end()) {
        const std::size_t idx =
          static_cast<std::size_t>(std::distance(measured_names.begin(), it));
        if (idx < measured.size()) {
          q_measured = measured[idx];
          have_measured = true;
        }
      }
    } else if (measured.size() == positions.front().size()) {
      q_measured = measured[j];
      have_measured = true;
    }
    if (!have_measured || !std::isfinite(q_measured)) {
      result.ok = false;
      result.error = "no measured state for continuous joint '" + names[j] + "'";
      return result;
    }

    const double p0 = positions.front()[j];
    const double k = std::round((q_measured - p0) / kTwoPiContinuous);
    const double p0_lift = p0 + k * kTwoPiContinuous;
    if (std::fabs(wrap_pi(p0_lift - q_measured)) > start_tolerance) {
      result.ok = false;
      result.error = "trajectory first point for continuous joint '" + names[j] +
        "' is not the measured pose (needs " + std::to_string(wrap_pi(p0_lift - q_measured)) +
        " rad correction)";
      return result;
    }
    const double shift = k * kTwoPiContinuous;
    for (auto & row : positions) {
      if (j < row.size()) {
        row[j] += shift;
      }
    }
    for (std::size_t i = 1; i < positions.size(); ++i) {
      if (j >= positions[i].size() || j >= positions[i - 1].size()) {
        continue;
      }
      if (std::fabs(positions[i][j] - positions[i - 1][j]) > kPiContinuous) {
        result.ok = false;
        result.error = "trajectory contains a > pi step on continuous joint '" +
          names[j] + "'; supply intermediate waypoints";
        return result;
      }
    }
  }
  return result;
}

}  // namespace curobo_core

#endif  // CUROBO_CORE__ROBOT__CONTINUOUS_JOINTS_HPP_

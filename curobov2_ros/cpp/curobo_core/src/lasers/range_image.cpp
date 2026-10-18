// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of make_range_image / finite_range_mask.
#include "curobo_core/lasers/range_image.hpp"

#include <algorithm>
#include <cmath>

namespace curobo_core
{

namespace
{
constexpr double kPi = 3.14159265358979323846;

bool is_finite(double v) {return std::isfinite(v);}

double wrap_mod(double value, double modulus)
{
  const double r = std::fmod(value, modulus);
  return r < 0.0 ? r + modulus : r;
}
}  // namespace

// PORT: finite_range_mask
void finite_range_mask(
  const std::vector<std::array<double, 3>> & xyz, double range_min, double range_max,
  std::vector<std::uint8_t> & mask, std::vector<float> & ranges)
{
  mask.assign(xyz.size(), 0);
  ranges.assign(xyz.size(), 0.0f);
  for (std::size_t i = 0; i < xyz.size(); ++i) {
    const double norm = std::sqrt(
      xyz[i][0] * xyz[i][0] + xyz[i][1] * xyz[i][1] + xyz[i][2] * xyz[i][2]);
    ranges[i] = static_cast<float>(norm);
    const bool finite = is_finite(xyz[i][0]) && is_finite(xyz[i][1]) && is_finite(xyz[i][2]) &&
      is_finite(norm);
    mask[i] = (finite && norm >= range_min && norm <= range_max) ? 1 : 0;
  }
}

// PORT: make_range_image
RangeImage make_range_image(
  const std::vector<std::array<double, 3>> & xyz, int image_height, int image_width,
  double range_min, double range_max, double elevation_min_rad, double elevation_max_rad)
{
  const int h = image_height;
  const int w = image_width;
  RangeImage out;
  out.height = h;
  out.width = w;
  out.range.assign(static_cast<std::size_t>(h) * w, 0.0f);
  out.rgb.assign(static_cast<std::size_t>(h) * w * 3, 0);

  std::vector<std::uint8_t> mask;
  std::vector<float> ranges;
  finite_range_mask(xyz, range_min, range_max, mask, ranges);

  std::vector<double> elevation(xyz.size(), 0.0);
  for (std::size_t i = 0; i < xyz.size(); ++i) {
    const double xy_norm = std::sqrt(xyz[i][0] * xyz[i][0] + xyz[i][1] * xyz[i][1]);
    elevation[i] = std::atan2(xyz[i][2], xy_norm);
    if (mask[i] != 0 &&
      (elevation[i] < elevation_min_rad || elevation[i] > elevation_max_rad))
    {
      mask[i] = 0;
    }
  }

  // Gather masked points.
  std::vector<std::size_t> src;
  for (std::size_t i = 0; i < mask.size(); ++i) {
    if (mask[i] != 0) {
      src.push_back(i);
    }
  }
  if (src.empty()) {
    return out;
  }

  // Candidate (flat, range, source) entries.
  struct Candidate
  {
    std::int64_t flat;
    float range;
    std::size_t source;
  };
  std::vector<Candidate> candidates;
  candidates.reserve(src.size() * (h == 1 ? 2 : 4));
  for (std::size_t k = 0; k < src.size(); ++k) {
    const std::size_t i = src[k];
    const double azimuth = std::atan2(xyz[i][1], xyz[i][0]);
    const double u = wrap_mod((azimuth + kPi) * (static_cast<double>(w) / (2.0 * kPi)), w);
    const std::int64_t u0 = static_cast<std::int64_t>(
      wrap_mod(std::floor(u), static_cast<double>(w)));
    const std::int64_t u1 = (u0 + 1) % w;
    if (h == 1) {
      candidates.push_back({u0, ranges[i], i});
      candidates.push_back({u1, ranges[i], i});
    } else {
      const double v = (elevation_max_rad - elevation[i]) *
        ((h - 1.0) / (elevation_max_rad - elevation_min_rad));
      std::int64_t v0 = static_cast<std::int64_t>(std::floor(v));
      v0 = std::max<std::int64_t>(0, std::min<std::int64_t>(v0, h - 1));
      const std::int64_t v1 = std::max<std::int64_t>(0, std::min<std::int64_t>(v0 + 1, h - 1));
      candidates.push_back({v0 * w + u0, ranges[i], i});
      candidates.push_back({v0 * w + u1, ranges[i], i});
      candidates.push_back({v1 * w + u0, ranges[i], i});
      candidates.push_back({v1 * w + u1, ranges[i], i});
    }
  }

  // PORT: np.lexsort((candidate_ranges, candidate_flat)) -> primary flat,
  // secondary range ascending; the first entry per flat is the nearest.
  std::sort(
    candidates.begin(), candidates.end(),
    [](const Candidate & a, const Candidate & b) {
      if (a.flat != b.flat) {
        return a.flat < b.flat;
      }
      return a.range < b.range;
    });

  std::int64_t last_flat = -1;
  for (const auto & candidate : candidates) {
    if (candidate.flat == last_flat) {
      continue;
    }
    last_flat = candidate.flat;
    if (candidate.flat >= 0 && candidate.flat < static_cast<std::int64_t>(h) * w) {
      out.range[static_cast<std::size_t>(candidate.flat)] = ranges[candidate.source];
      out.projected_pixels += 1;
    }
  }
  return out;
}

}  // namespace curobo_core

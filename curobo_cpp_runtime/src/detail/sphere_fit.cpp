// SPDX-License-Identifier: Apache-2.0
#include "detail/sphere_fit.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <utility>

namespace curobo_cpp_runtime
{
namespace detail
{

int estimate_sphere_count(float dx, float dy, float dz)
{
  const double volume_cm3 =
    static_cast<double>(dx) * static_cast<double>(dy) * static_cast<double>(dz) *
    1.0e6;
  const int n = static_cast<int>((1.0 / 15.0) * volume_cm3);
  return std::max(std::min(100, n), 3);
}

std::vector<MeshTriangle> box_mesh_triangles(float dx, float dy, float dz)
{
  const float hx = dx * 0.5F, hy = dy * 0.5F, hz = dz * 0.5F;
  const std::array<std::array<float, 3>, 8> v = {{
    {-hx, -hy, -hz}, {hx, -hy, -hz}, {hx, hy, -hz}, {-hx, hy, -hz},
    {-hx, -hy, hz}, {hx, -hy, hz}, {hx, hy, hz}, {-hx, hy, hz}}};
  const int faces[12][3] = {
    {0, 1, 2}, {0, 2, 3}, {4, 6, 5}, {4, 7, 6}, {0, 4, 5}, {0, 5, 1},
    {3, 2, 6}, {3, 6, 7}, {0, 3, 7}, {0, 7, 4}, {1, 5, 6}, {1, 6, 2}};
  std::vector<MeshTriangle> tris;
  tris.reserve(12);
  for (const auto & f : faces) {
    MeshTriangle t;
    t.origin = v[static_cast<std::size_t>(f[0])];
    const auto & b = v[static_cast<std::size_t>(f[1])];
    const auto & c = v[static_cast<std::size_t>(f[2])];
    t.edge1 = {b[0] - t.origin[0], b[1] - t.origin[1], b[2] - t.origin[2]};
    t.edge2 = {c[0] - t.origin[0], c[1] - t.origin[1], c[2] - t.origin[2]};
    const std::array<float, 3> cross = {
      t.edge1[1] * t.edge2[2] - t.edge1[2] * t.edge2[1],
      t.edge1[2] * t.edge2[0] - t.edge1[0] * t.edge2[2],
      t.edge1[0] * t.edge2[1] - t.edge1[1] * t.edge2[0]};
    t.area = 0.5F * std::sqrt(
      cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
    tris.push_back(t);
  }
  return tris;
}

std::vector<std::array<float, 3>> remove_close(
  const std::vector<std::array<float, 3>> & points, float radius)
{
  const std::size_t n = points.size();
  std::vector<int> degree(n, 0);
  std::vector<std::pair<std::size_t, std::size_t>> pairs;
  const float r2 = radius * radius;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      const float dx = points[i][0] - points[j][0];
      const float dy = points[i][1] - points[j][1];
      const float dz = points[i][2] - points[j][2];
      if (dx * dx + dy * dy + dz * dz <= r2) {
        pairs.emplace_back(i, j);
        degree[i] += 1;
        degree[j] += 1;
      }
    }
  }
  // Ties drop the first index, matching numpy's argmax on [degree_i, degree_j].
  std::vector<bool> keep(n, true);
  for (const auto & pair : pairs) {
    keep[degree[pair.first] >= degree[pair.second] ? pair.first : pair.second] =
      false;
  }
  std::vector<std::array<float, 3>> kept;
  kept.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (keep[i]) {
      kept.push_back(points[i]);
    }
  }
  return kept;
}

std::vector<curobo::robot::CollisionSphere> fit_surface_spheres(
  int type, const std::array<float, 3> & dims, int capacity)
{
  constexpr float kSurfaceRadius = 0.002F;
  constexpr float kPi = 3.14159265358979F;
  std::vector<curobo::robot::CollisionSphere> spheres;
  const float dx = std::max(1e-4F, dims[0]);
  const float dy = std::max(1e-4F, dims[1]);
  const float dz = std::max(1e-4F, dims[2]);
  const int slots = std::max(1, capacity);
  // A sphere carries its radius in `dims`, so its bounding box is (2r)^3.
  const float bx = (type == 1) ? 2.0F * dx : dx;
  const float by = (type == 1) ? 2.0F * dy : dy;
  const float bz = (type == 1) ? 2.0F * dz : dz;
  const int n = std::max(
    1, std::min(estimate_sphere_count(bx, by, bz), slots));

  std::mt19937 rng(0x5EEDu);
  std::uniform_real_distribution<float> uniform(0.0F, 1.0F);
  std::vector<std::array<float, 3>> samples;
  const int draws = n * 3;
  samples.reserve(static_cast<std::size_t>(draws));
  float radius = 0.0F;

  if (type == 0) {
    const auto tris = box_mesh_triangles(dx, dy, dz);
    float area = 0.0F;
    std::vector<float> cumulative;
    cumulative.reserve(tris.size());
    for (const auto & t : tris) {
      area += t.area;
      cumulative.push_back(area);
    }
    radius = std::sqrt(area / (3.0F * static_cast<float>(n)));
    for (int k = 0; k < draws; ++k) {
      const float pick = uniform(rng) * area;
      std::size_t index = 0;
      while (index + 1 < cumulative.size() && cumulative[index] < pick) {
        ++index;
      }
      const auto & t = tris[index];
      float a = uniform(rng);
      float b = uniform(rng);
      if (a + b > 1.0F) {
        a = 1.0F - a;
        b = 1.0F - b;
      }
      samples.push_back({
        t.origin[0] + a * t.edge1[0] + b * t.edge2[0],
        t.origin[1] + a * t.edge1[1] + b * t.edge2[1],
        t.origin[2] + a * t.edge1[2] + b * t.edge2[2]});
    }
  } else {
    const bool is_sphere = (type == 1);
    const float R = 0.5F * dx;
    const float hx = 0.5F * dx, hy = 0.5F * dy, hz = 0.5F * dz;
    const float area = is_sphere
      ? 4.0F * kPi * R * R
      : 2.0F * (dx * dy + dy * dz + dz * dx);
    radius = std::sqrt(area / (3.0F * static_cast<float>(n)));
    for (int k = 0; k < draws; ++k) {
      const float phi = std::acos(1.0F - 2.0F * uniform(rng));
      const float theta = 2.0F * kPi * uniform(rng);
      const float sx = std::sin(phi) * std::cos(theta);
      const float sy = std::sin(phi) * std::sin(theta);
      const float sz = std::cos(phi);
      std::array<float, 3> point;
      if (is_sphere) {
        point = {R * sx, R * sy, R * sz};
      } else {
        point = {hx * sx, hy * sy, hz * sz};
      }
      samples.push_back(point);
    }
  }

  const auto kept = remove_close(samples, radius);
  for (std::size_t i = 0; i < kept.size() && static_cast<int>(i) < n; ++i) {
    curobo::robot::CollisionSphere s;
    s.center = kept[i];
    s.radius = kSurfaceRadius;
    spheres.push_back(s);
  }
  return spheres;
}

}  // namespace detail
}  // namespace curobo_cpp_runtime

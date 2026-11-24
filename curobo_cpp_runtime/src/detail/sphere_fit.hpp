// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <vector>

#include "curobo_robot_runtime/attachment.h"

namespace curobo_cpp_runtime
{
namespace detail
{

int estimate_sphere_count(float dx, float dy, float dz);

struct MeshTriangle
{
  std::array<float, 3> origin;
  std::array<float, 3> edge1;
  std::array<float, 3> edge2;
  float area{0.0F};
};

std::vector<MeshTriangle> box_mesh_triangles(float dx, float dy, float dz);

std::vector<std::array<float, 3>> remove_close(
  const std::vector<std::array<float, 3>> & points, float radius);

// type 0 = cuboid box mesh, type 1 = sphere shell; dims are full extents.
std::vector<curobo::robot::CollisionSphere> fit_surface_spheres(
  int type, const std::array<float, 3> & dims, int capacity);

}  // namespace detail
}  // namespace curobo_cpp_runtime

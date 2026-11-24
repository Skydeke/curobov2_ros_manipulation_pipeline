// SPDX-License-Identifier: Apache-2.0
#include "detail/pose_math.hpp"

namespace curobo_cpp_runtime
{
namespace detail
{

Pose7f pose_from_values(const float * values)
{
  return {values[0], values[1], values[2], values[3], values[4], values[5], values[6]};
}

Pose7f pose_multiply(const Pose7f & a, const Pose7f & b)
{
  const float aw = a[3], ax = a[4], ay = a[5], az = a[6];
  const float bw = b[3], bx = b[4], by = b[5], bz = b[6];
  const float w = aw * bw - ax * bx - ay * by - az * bz;
  const float x = aw * bx + ax * bw + ay * bz - az * by;
  const float y = aw * by - ax * bz + ay * bw + az * bx;
  const float z = aw * bz + ax * by - ay * bx + az * bw;
  const float tx = 2.0F * (ay * b[2] - az * b[1]);
  const float ty = 2.0F * (az * b[0] - ax * b[2]);
  const float tz = 2.0F * (ax * b[1] - ay * b[0]);
  const float rx = b[0] + aw * tx + (ay * tz - az * ty);
  const float ry = b[1] + aw * ty + (az * tx - ax * tz);
  const float rz = b[2] + aw * tz + (ax * ty - ay * tx);
  return {a[0] + rx, a[1] + ry, a[2] + rz, w, x, y, z};
}

Pose7f pose_inverse(const Pose7f & p)
{
  const float w = p[3], x = -p[4], y = -p[5], z = -p[6];
  const float tx = 2.0F * (y * p[2] - z * p[1]);
  const float ty = 2.0F * (z * p[0] - x * p[2]);
  const float tz = 2.0F * (x * p[1] - y * p[0]);
  const float rx = -(p[0] + w * tx + (y * tz - z * ty));
  const float ry = -(p[1] + w * ty + (z * tx - x * tz));
  const float rz = -(p[2] + w * tz + (x * ty - y * tx));
  return {rx, ry, rz, w, x, y, z};
}

void fill_pose_arrays(
  const std::vector<Pose7f> & poses,
  std::vector<float> & positions, std::vector<float> & quaternions)
{
  positions.clear();
  quaternions.clear();
  positions.reserve(poses.size() * 3);
  quaternions.reserve(poses.size() * 4);
  for (const auto & pose : poses) {
    positions.push_back(pose[0]);
    positions.push_back(pose[1]);
    positions.push_back(pose[2]);
    quaternions.push_back(pose[3]);
    quaternions.push_back(pose[4]);
    quaternions.push_back(pose[5]);
    quaternions.push_back(pose[6]);
  }
}

}  // namespace detail
}  // namespace curobo_cpp_runtime

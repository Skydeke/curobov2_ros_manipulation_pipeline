// SPDX-License-Identifier: Apache-2.0
//
// The analytic scene MarkerArray must contain exactly one CUBE/SPHERE/CYLINDER
// per primitive (never a MESH/voxel approximation), with the primitive's real
// dimensions. This is the only visualization of the analytic obstacles now that
// they are no longer rasterized into the voxel grid.
#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include "curobo_core/core/obstacle_manager.hpp"
#include "curobo_core/core/ros_service_manager.hpp"

namespace
{
using curobo_core::CAPSULE;
using curobo_core::CUBOID;
using curobo_core::CYLINDER;
using curobo_core::MESH;
using curobo_core::Pose7;
using curobo_core::SceneView;
using curobo_core::SPHERE;

Pose7 pose(double x, double y, double z, double qw = 1.0, double qx = 0.0,
  double qy = 0.0, double qz = 0.0)
{
  Pose7 out = Pose7::Zero();
  out << x, y, z, qw, qx, qy, qz;
  return out;
}
}  // namespace

TEST(SceneObstacleMarkers, PublishesExactPrimitiveTypesAndDimensions)
{
  SceneView view;
  view.names = {"box", "ball", "capsule", "cylinder", "mesh"};
  view.types = {CUBOID, SPHERE, CAPSULE, CYLINDER, MESH};
  view.poses = {
    pose(0, 0, 0),
    pose(0, 0, 0),
    pose(0, 0, 1.0),                 // base pose; the marker is offset by L/2
    pose(0, 0, 0),
    pose(0, 0, 0)};
  view.dimensions = {
    {1.0, 2.0, 3.0},
    {0.1, 0.1, 0.1},
    {0.05, 0.4, 0.05},
    {0.2, 0.5, 0.2},
    {1.0, 1.0, 1.0}};
  view.colors.assign(5, {0.0, 0.0, 0.0, 1.0});
  view.mesh_file_paths.assign(5, "");
  view.attached = {false, true, false, false, false};

  builtin_interfaces::msg::Time stamp;
  stamp.sec = 7;
  const auto array = curobo_core::RosServiceManager::scene_obstacle_markers(
    view, "base_link", stamp);

  // DELETEALL + one marker per non-mesh primitive (the mesh must be skipped).
  ASSERT_EQ(array.markers.size(), 5u);
  EXPECT_EQ(array.markers[0].action, visualization_msgs::msg::Marker::DELETEALL);

  const auto & box = array.markers[1];
  EXPECT_EQ(box.type, visualization_msgs::msg::Marker::CUBE);
  EXPECT_EQ(box.ns, "scene_obstacles");
  EXPECT_EQ(box.header.frame_id, "base_link");
  EXPECT_EQ(box.header.stamp.sec, 7);
  EXPECT_DOUBLE_EQ(box.scale.x, 1.0);
  EXPECT_DOUBLE_EQ(box.scale.y, 2.0);
  EXPECT_DOUBLE_EQ(box.scale.z, 3.0);

  const auto & ball = array.markers[2];
  EXPECT_EQ(ball.type, visualization_msgs::msg::Marker::SPHERE);
  EXPECT_DOUBLE_EQ(ball.scale.x, 0.2);   // diameter = 2 * radius
  EXPECT_DOUBLE_EQ(ball.scale.y, 0.2);
  EXPECT_DOUBLE_EQ(ball.scale.z, 0.2);
  EXPECT_NEAR(ball.color.a, 0.35, 1e-6);  // attached -> grey/translucent

  const auto & capsule = array.markers[3];
  EXPECT_EQ(capsule.type, visualization_msgs::msg::Marker::CYLINDER);
  EXPECT_DOUBLE_EQ(capsule.scale.x, 0.1);
  EXPECT_DOUBLE_EQ(capsule.scale.z, 0.4);
  // cuRobo's capsule spans local z in [0, dims[1]]; the cylinder is centred.
  EXPECT_NEAR(capsule.pose.position.z, 1.2, 1e-9);

  const auto & cylinder = array.markers[4];
  EXPECT_EQ(cylinder.type, visualization_msgs::msg::Marker::CYLINDER);
  EXPECT_DOUBLE_EQ(cylinder.scale.x, 0.4);
  EXPECT_DOUBLE_EQ(cylinder.scale.z, 0.5);
  EXPECT_DOUBLE_EQ(cylinder.pose.position.z, 0.0);   // cylinder is pose-centred
}

TEST(SceneObstacleMarkers, CapsuleOffsetFollowsTheSceneOrientation)
{
  SceneView view;
  view.names = {"capsule"};
  view.types = {CAPSULE};
  // 90 deg about +y: local +z maps to world +x.
  const double h = std::sqrt(0.5);
  view.poses = {pose(1.0, 2.0, 3.0, h, 0.0, h, 0.0)};
  view.dimensions = {{0.05, 0.4, 0.05}};
  view.colors = {{0.0, 0.0, 0.0, 1.0}};
  view.mesh_file_paths = {""};
  view.attached = {false};

  builtin_interfaces::msg::Time stamp;
  const auto array = curobo_core::RosServiceManager::scene_obstacle_markers(
    view, "base_link", stamp);
  ASSERT_EQ(array.markers.size(), 2u);
  const auto & marker = array.markers[1];
  EXPECT_NEAR(marker.pose.position.x, 1.2, 1e-9);
  EXPECT_NEAR(marker.pose.position.y, 2.0, 1e-9);
  EXPECT_NEAR(marker.pose.position.z, 3.0, 1e-9);
}

TEST(SceneObstacleMarkers, EmptySceneStillClearsPreviousMarkers)
{
  SceneView view;
  builtin_interfaces::msg::Time stamp;
  const auto array = curobo_core::RosServiceManager::scene_obstacle_markers(
    view, "base_link", stamp);
  ASSERT_EQ(array.markers.size(), 1u);
  EXPECT_EQ(array.markers[0].action, visualization_msgs::msg::Marker::DELETEALL);
}

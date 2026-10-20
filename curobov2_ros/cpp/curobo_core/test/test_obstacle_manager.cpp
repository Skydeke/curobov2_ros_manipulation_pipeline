// SPDX-License-Identifier: Apache-2.0
// Tests for the ObstacleManager scene bookkeeping, with the exact messages the
// recorded object scenario captured from the Python server
// (recordings/m2_object_franka). No GPU, no interpreter (FakeBridge).
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "curobo_core/core/obstacle_manager.hpp"
#include "curobo_runtime/fake_bridge.hpp"

using curobo_core::CAPSULE;
using curobo_core::CUBOID;
using curobo_core::MESH;
using curobo_core::ObstacleManager;
using curobo_core::SceneObjectSpec;
using curobo_core::SPHERE;

namespace
{
SceneObjectSpec cube(const std::string & name, double size)
{
  SceneObjectSpec spec;
  spec.type = CUBOID;
  spec.name = name;
  spec.dims = {size, size, size};
  return spec;
}
}  // namespace

TEST(ObstacleManager, AddObjectMessagesMatchRecordedResponses)
{
  curobo_runtime::FakeBridge bridge;
  ObstacleManager manager(&bridge, 1);
  std::string message;

  ASSERT_TRUE(manager.add_object(cube("cube", 0.5), message));
  EXPECT_EQ(message, "Object 'cube' added (1 cuboids, 0 meshes)");

  ASSERT_TRUE(manager.add_object(cube("cube2", 0.04), message));
  EXPECT_EQ(message, "Object 'cube2' added (2 cuboids, 0 meshes)");

  // get_obstacles: one name per line, trailing newline.
  EXPECT_EQ(manager.get_obstacles(), "cube\ncube2\n");

  // Every change pushed the analytic scene once (bridge scene_set).
  EXPECT_EQ(bridge.scene_version(), 2);
  EXPECT_EQ(manager.epoch(), 2);
}

TEST(ObstacleManager, DuplicateNameIsRejectedWithTheRecordedString)
{
  curobo_runtime::FakeBridge bridge;
  ObstacleManager manager(&bridge, 1);
  std::string message;
  ASSERT_TRUE(manager.add_object(cube("cube", 0.5), message));
  EXPECT_FALSE(manager.add_object(cube("cube", 0.5), message));
  EXPECT_EQ(message, "Object with name \"cube\" already exists");
  EXPECT_EQ(bridge.scene_version(), 1);   // rejected request pushes nothing
}

TEST(ObstacleManager, NonPositiveDimensionsAreRejected)
{
  curobo_runtime::FakeBridge bridge;
  ObstacleManager manager(&bridge, 1);
  std::string message;
  SceneObjectSpec bad = cube("bad", 0.0);
  EXPECT_FALSE(manager.add_object(bad, message));
  EXPECT_EQ(message, "Object dimensions must be positive");
}

TEST(ObstacleManager, UnknownTypeIsRejected)
{
  curobo_runtime::FakeBridge bridge;
  ObstacleManager manager(&bridge, 1);
  std::string message;
  SceneObjectSpec bad;
  bad.type = 9;
  bad.name = "weird";
  bad.dims = {1, 1, 1};
  EXPECT_FALSE(manager.add_object(bad, message));
  EXPECT_EQ(message, "Object type \"9\" not recognized");
}

TEST(ObstacleManager, RemoveObjectReportsTheBucket)
{
  curobo_runtime::FakeBridge bridge;
  ObstacleManager manager(&bridge, 1);
  std::string message;
  ASSERT_TRUE(manager.add_object(cube("cube", 0.5), message));
  ASSERT_TRUE(manager.add_object(cube("cube2", 0.04), message));

  ASSERT_TRUE(manager.remove_object("cube", message));
  EXPECT_EQ(message, "Object 'cube' removed from cuboids");
  EXPECT_EQ(manager.get_obstacles(), "cube2\n");

  EXPECT_FALSE(manager.remove_object("cube", message));
  EXPECT_EQ(message, "Object 'cube' not found");
}

TEST(ObstacleManager, RemoveAllReportsTheTotal)
{
  curobo_runtime::FakeBridge bridge;
  ObstacleManager manager(&bridge, 1);
  std::string message;
  ASSERT_TRUE(manager.add_object(cube("cube", 0.5), message));
  ASSERT_TRUE(manager.add_object(cube("cube2", 0.04), message));
  ASSERT_TRUE(manager.remove_object("cube", message));   // leaves 1

  ASSERT_TRUE(manager.remove_all_objects(message));
  EXPECT_EQ(message, "All 1 obstacles removed");
  EXPECT_TRUE(manager.empty());
}

TEST(ObstacleManager, SetObjectsValidatesAllBeforeMutating)
{
  curobo_runtime::FakeBridge bridge;
  ObstacleManager manager(&bridge, 1);
  std::string message;
  ASSERT_TRUE(manager.add_object(cube("keep", 0.5), message));

  std::vector<SceneObjectSpec> specs{cube("a", 0.1), cube("a", 0.1)};
  EXPECT_FALSE(manager.set_objects(specs, true, message));
  EXPECT_EQ(message, "Invalid object 'a': Object with name \"a\" already exists");
  EXPECT_FALSE(manager.empty());   // unchanged
  EXPECT_EQ(manager.get_obstacles(), "keep\n");

  std::vector<SceneObjectSpec> good{cube("a", 0.1), cube("b", 0.2)};
  ASSERT_TRUE(manager.set_objects(good, true, message));
  EXPECT_EQ(message, "Scene set to 2 objects (2 cuboids, 0 meshes)");
  EXPECT_EQ(manager.get_obstacles(), "a\nb\n");
}

TEST(ObstacleManager, SceneViewReportsTypesAndAttachState)
{
  curobo_runtime::FakeBridge bridge;
  ObstacleManager manager(&bridge, 1);
  std::string message;

  SceneObjectSpec sphere;
  sphere.type = SPHERE;
  sphere.name = "ball";
  // AddObject rejects any non-positive dimension, so a sphere is sent [r,r,r]
  // (and get_scene_objects reports it back the same way).
  sphere.dims = {0.05, 0.05, 0.05};
  ASSERT_TRUE(manager.add_object(sphere, message));
  SceneObjectSpec capsule;
  capsule.type = CAPSULE;
  capsule.name = "rod";
  capsule.dims = {0.02, 0.3, 0.02};
  ASSERT_TRUE(manager.add_object(capsule, message));

  manager.set_attached("ball");
  const auto view = manager.get_scene_objects();
  ASSERT_EQ(view.names.size(), 2u);
  EXPECT_EQ(view.types[0], SPHERE);
  EXPECT_EQ(view.types[1], CAPSULE);
  // sphere reports [r,r,r] so a round trip is accepted.
  EXPECT_DOUBLE_EQ(view.dimensions[0][1], 0.05);
  EXPECT_TRUE(view.attached[0]);
  EXPECT_FALSE(view.attached[1]);
  // remove clears the attach state.
  ASSERT_TRUE(manager.remove_object("ball", message));
  EXPECT_TRUE(manager.attached_name().empty());
}

TEST(ObstacleManager, InlineMeshRequiresBothVertexAndTriangleData)
{
  curobo_runtime::FakeBridge bridge;
  ObstacleManager manager(&bridge, 1);
  std::string message;
  SceneObjectSpec mesh;
  mesh.type = MESH;
  mesh.name = "duck";
  mesh.dims = {1, 1, 1};
  mesh.vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
  EXPECT_FALSE(manager.add_object(mesh, message));
  EXPECT_EQ(message, "Inline MESH requires both vertices and triangles");
}

// A scene mutation must NOT mark the perception world as pushed. Analytic
// pushes (e.g. the MoveIt side syncing a collision object) carry only the
// primitives; the camera voxels are re-attached by the next ESDF refresh. If the
// mutation closed the freshness gate, the next plan would collide against a
// scene whose voxel layer is stale or absent and solve straight through it.
TEST(ObstacleManager, SceneMutationStillRequiresPerceptionRefresh)
{
  curobo_runtime::FakeBridge bridge;
  ObstacleManager manager(&bridge, 1);
  std::string message;

  // Simulate a prior successful ESDF rebuild + push (gate now closed).
  ASSERT_TRUE(manager.refresh_esdf());
  manager.mark_world_pushed();
  EXPECT_FALSE(manager.should_refresh_world(true));

  // An analytic mutation must reopen the gate for the ESDF refresh.
  ASSERT_TRUE(manager.add_object(cube("cube", 0.5), message));
  EXPECT_TRUE(manager.should_refresh_world(true));
}

// A failed world push (the solver never got the voxel layer) must not be treated
// as fresh: the next plan has to retry.
TEST(ObstacleManager, InvalidateWorldForcesTheNextRefresh)
{
  curobo_runtime::FakeBridge bridge;
  ObstacleManager manager(&bridge, 1);
  ASSERT_TRUE(manager.refresh_esdf());
  manager.mark_world_pushed();
  EXPECT_FALSE(manager.should_refresh_world(true));

  manager.invalidate_world();
  EXPECT_TRUE(manager.should_refresh_world(true));
}

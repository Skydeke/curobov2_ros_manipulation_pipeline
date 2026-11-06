// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "curobo_cpp_runtime/cpp_bridge.hpp"
#include "curobo_cpp_runtime/runtime_selector.hpp"

using curobo_cpp_runtime::CppBridge;
using curobo_cpp_runtime::RuntimeMode;
using curobo_cpp_runtime::RuntimeSelection;
using curobo_cpp_runtime::RuntimeSelector;

namespace
{

template<typename Function>
std::string throws_with_method(const char * method, Function function)
{
  try {
    function();
  } catch (const std::runtime_error & error) {
    const std::string message = error.what();
    EXPECT_NE(message.find(method), std::string::npos) << message;
    return message;
  }
  ADD_FAILURE() << method << " did not throw";
  return {};
}

}  // namespace

TEST(CppBridgeContract, IdentityAndNoInterpreter)
{
  CppBridge bridge("/tmp/curobo-nonexistent-artifact", "openarm", "", 0);
  EXPECT_EQ(bridge.artifact_directory(), "/tmp/curobo-nonexistent-artifact");
  EXPECT_EQ(bridge.robot_name(), "openarm");
  EXPECT_EQ(bridge.interpreter_init_count(), 0);
  EXPECT_FALSE(bridge.initialized());
}

TEST(CppBridgeContract, InitFailsClearlyOnMissingArtifact)
{
  CppBridge bridge("/tmp/curobo-nonexistent-artifact", "openarm", "", 0);
  try {
    bridge.init("");
    FAIL() << "init should have thrown";
  } catch (const std::runtime_error & error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("init"), std::string::npos) << message;
    EXPECT_NE(message.find("artifact"), std::string::npos) << message;
  }
  EXPECT_FALSE(bridge.initialized());
}

TEST(CppBridgeContract, UnsupportedMethodsNameThemselves)
{
  CppBridge bridge("/tmp/curobo-nonexistent-artifact", "openarm");

  curobo_runtime::TensorF32 q;
  q.shape = {1, 7};
  q.data.assign(7, 0.0F);

  throws_with_method("collision_distance",
    [&]() {bridge.collision_distance(1, q);});
  throws_with_method("collision_attribution",
    [&]() {bridge.collision_attribution(1, q);});
  throws_with_method("collision_cache_bucket_counts",
    [&]() {bridge.collision_cache_bucket_counts(1);});
  throws_with_method("render_plan_image",
    [&]() {
      bridge.render_plan_image(q, q, q, {"tool"}, 0.025);
    });
  throws_with_method("attach_object",
    [&]() {bridge.attach_object(1, "box", std::vector<float>(7, 0.0F));});
  throws_with_method("detach_object",
    [&]() {bridge.detach_object(1);});
  throws_with_method("collision_spheres_with_attached",
    [&]() {bridge.collision_spheres_with_attached(1, std::vector<float>(7, 0.0F));});
  throws_with_method("configure_mapper",
    [&]() {bridge.configure_mapper(1, "{}");});
  throws_with_method("warmup_mapper",
    [&]() {bridge.warmup_mapper(1);});
  throws_with_method("esdf_refresh",
    [&]() {bridge.esdf_refresh(1);});
  throws_with_method("voxel_grid",
    [&]() {bridge.voxel_grid(1);});
  throws_with_method("clear_voxel_map",
    [&]() {bridge.clear_voxel_map(1);});
  throws_with_method("depth_integrate",
    [&]() {
      bridge.depth_integrate(1, q, q, {0, 0, 0, 1, 0, 0, 0});
    });
  throws_with_method("lidar_integrate",
    [&]() {bridge.lidar_integrate(1, q, q, q, q);});
  throws_with_method("robot_segment",
    [&]() {
      bridge.robot_segment(1, q, q, {0, 0, 0, 1, 0, 0, 0},
        std::vector<float>(7, 0.0F), "[]");
    });
  throws_with_method("robot_model_rebuild_locks",
    [&]() {bridge.robot_model_rebuild_locks(1, "{}");});
  throws_with_method("robot_model_update_lock_joints",
    [&]() {bridge.robot_model_update_lock_joints(1, "{}");});
  throws_with_method("retarget_build",
    [&]() {bridge.retarget_build(1, "{}");});
  throws_with_method("retarget_reset",
    [&]() {bridge.retarget_reset(1);});
  throws_with_method("retarget_step",
    [&]() {bridge.retarget_step(1, {0, 0, 0, 1, 0, 0, 0});});
  throws_with_method("retarget_load_scene",
    [&]() {bridge.retarget_load_scene(1, "[]");});
}

TEST(CppBridgeContract, GlobalFlagsAndLanesAreLocal)
{
  CppBridge bridge("/tmp/curobo-nonexistent-artifact", "openarm");
  bridge.bind_lane("plan");
  bridge.bind_lane("plan");
  bridge.bind_lane("kin");
  const auto lanes = bridge.bound_lanes();
  ASSERT_EQ(lanes.size(), 2u);
  EXPECT_EQ(lanes[0], "plan");
  EXPECT_EQ(lanes[1], "kin");
  bridge.set_global_flags(false, true);
}

TEST(RuntimeSelectorContract, PythonModeConstructsPyBridge)
{
  RuntimeSelection selection;
  selection.selected_mode = RuntimeMode::Python;
  selection.backend = "python";
  const auto bridge = RuntimeSelector::create_bridge(selection);
  ASSERT_TRUE(bridge != nullptr);
  // Constructing PyBridge must not start CPython; init() does that later.
  EXPECT_EQ(bridge->interpreter_init_count(), 0);
}

TEST(RuntimeSelectorContract, ExplicitCppCannotFallBack)
{
  RuntimeSelection selection;
  selection.selected_mode = RuntimeMode::Cpp;
  selection.backend = "cpp";
  selection.artifact_valid = false;
  selection.artifact.error = "artifact directory does not exist";
  EXPECT_THROW(
    RuntimeSelector::create_bridge(selection), std::runtime_error);
}

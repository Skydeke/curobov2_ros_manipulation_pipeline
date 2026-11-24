// SPDX-License-Identifier: Apache-2.0
#include "cpp_bridge_impl.hpp"

namespace curobo_cpp_runtime
{

std::vector<float> CppBridge::collision_distance(
  curobo_runtime::RobotHandle, const curobo_runtime::TensorF32 & q)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->native_scene_valid) {
    impl_->require_python("collision_distance");
    return impl_->python->collision_distance(impl_->python_handle, q);
  }
  unsupported("collision_distance");
}

curobo_runtime::CollisionAttribution CppBridge::collision_attribution(
  curobo_runtime::RobotHandle, const curobo_runtime::TensorF32 & q)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->native_scene_valid) {
    impl_->require_python("collision_attribution");
    return impl_->python->collision_attribution(impl_->python_handle, q);
  }
  unsupported("collision_attribution");
}

curobo_runtime::CacheBucketCounts CppBridge::collision_cache_bucket_counts(
  curobo_runtime::RobotHandle)
{
  unsupported("collision_cache_bucket_counts");
}

curobo_runtime::MpcCostBreakdown CppBridge::mpc_cost_breakdown(
  curobo_runtime::RobotHandle)
{
  curobo_runtime::MpcCostBreakdown breakdown;
  return breakdown;
}

void CppBridge::torch_synchronize()
{
  cuda_check("torch_synchronize", cudaDeviceSynchronize());
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->python) {
    impl_->python->torch_synchronize();
  }
}

curobo_runtime::PlanImageOutcome CppBridge::render_plan_image(
  const curobo_runtime::TensorF32 &, const curobo_runtime::TensorF32 &,
  const curobo_runtime::TensorF32 &, const std::vector<std::string> &, double)
{
  unsupported("render_plan_image");
}

void CppBridge::retarget_build(
  curobo_runtime::RobotHandle, const std::string &)
{
  unsupported("retarget_build");
}

void CppBridge::retarget_reset(curobo_runtime::RobotHandle)
{
  unsupported("retarget_reset");
}

curobo_runtime::MpcStepOutcome CppBridge::retarget_step(
  curobo_runtime::RobotHandle, const std::array<double, 7> &)
{
  unsupported("retarget_step");
}

void CppBridge::retarget_load_scene(
  curobo_runtime::RobotHandle, const std::string &)
{
  unsupported("retarget_load_scene");
}
}  // namespace curobo_cpp_runtime

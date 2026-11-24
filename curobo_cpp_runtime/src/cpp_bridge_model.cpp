// SPDX-License-Identifier: Apache-2.0
#include "cpp_bridge_impl.hpp"

namespace curobo_cpp_runtime
{

curobo_runtime::TensorF32 CppBridge::fk(
  curobo_runtime::RobotHandle, const curobo_runtime::TensorF32 & q)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("fk");
  return impl_->fk_tensor(q, "fk");
}

curobo_runtime::LinkPosesOutcome CppBridge::link_poses(
  curobo_runtime::RobotHandle, const curobo_runtime::TensorF32 & q,
  const std::vector<std::string> & link_names)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("link_poses");
  if (q.shape.size() != 2 || static_cast<int>(q.shape[1]) != impl_->dof()) {
    fail("link_poses", "q must have shape [N,dof]");
  }
  const int batch = static_cast<int>(q.shape[0]);
  const int tools = impl_->num_tools();
  if (link_names.empty()) {
    return {};
  }
  std::vector<int> indices;
  std::vector<std::string> used_names;
  for (const auto & name : link_names) {
    const auto it = std::find(impl_->tool_frames.begin(), impl_->tool_frames.end(), name);
    if (it != impl_->tool_frames.end()) {
      indices.push_back(static_cast<int>(std::distance(impl_->tool_frames.begin(), it)));
      used_names.push_back(name);
    }
  }
  if (indices.empty()) {
    return {};
  }

  DeviceBuffer input;
  DeviceBuffer positions_device;
  DeviceBuffer quaternions_device;
  DeviceBuffer jacobians_device;
  input.upload(q.data);
  positions_device.allocate(static_cast<std::size_t>(batch) * tools * 3 * sizeof(float));
  quaternions_device.allocate(static_cast<std::size_t>(batch) * tools * 4 * sizeof(float));
  jacobians_device.allocate(
    static_cast<std::size_t>(batch) * tools * 6 * static_cast<std::size_t>(impl_->dof()) *
    sizeof(float));
  impl_->kinematics->forward(
    static_cast<const float *>(input.get()),
    static_cast<float *>(positions_device.get()),
    static_cast<float *>(quaternions_device.get()),
    static_cast<float *>(jacobians_device.get()), batch, 1, nullptr);
  cuda_check("link_poses", cudaDeviceSynchronize());
  std::vector<float> positions(static_cast<std::size_t>(batch) * tools * 3);
  std::vector<float> quaternions(static_cast<std::size_t>(batch) * tools * 4);
  positions_device.download(positions);
  quaternions_device.download(quaternions);

  LinkPosesOutcome outcome;
  outcome.link_names = used_names;
  outcome.poses.shape = {batch, static_cast<std::int64_t>(indices.size()), 7};
  outcome.poses.data.resize(
    static_cast<std::size_t>(batch) * indices.size() * 7);
  for (int i = 0; i < batch; ++i) {
    for (std::size_t slot = 0; slot < indices.size(); ++slot) {
      const std::size_t source = static_cast<std::size_t>(i) * tools +
        static_cast<std::size_t>(indices[slot]);
      const std::size_t destination =
        (static_cast<std::size_t>(i) * indices.size() + slot) * 7;
      outcome.poses.data[destination + 0] = positions[source * 3 + 0];
      outcome.poses.data[destination + 1] = positions[source * 3 + 1];
      outcome.poses.data[destination + 2] = positions[source * 3 + 2];
      outcome.poses.data[destination + 3] = quaternions[source * 4 + 0];
      outcome.poses.data[destination + 4] = quaternions[source * 4 + 1];
      outcome.poses.data[destination + 5] = quaternions[source * 4 + 2];
      outcome.poses.data[destination + 6] = quaternions[source * 4 + 3];
    }
  }
  return outcome;
}



curobo_runtime::IkOutcome CppBridge::ik(
  curobo_runtime::RobotHandle, const curobo_runtime::TensorF32 & poses,
  const std::string & tool_frame)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->native_scene_valid) {
    impl_->require_python("ik");
    return impl_->python->ik(impl_->python_handle, poses, tool_frame);
  }
  const std::string active = tool_frame.empty() ? impl_->tool_frames.front() : tool_frame;
  const auto full = impl_->fill_native_ik_goals(poses, {active}, true);
  return impl_->solve_ik(full, {}, nullptr, 1, "ik");
}

curobo_runtime::IkOutcome CppBridge::ik_multi(
  curobo_runtime::RobotHandle, const curobo_runtime::TensorF32 & poses,
  const std::vector<std::string> & tool_frames)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->native_scene_valid) {
    impl_->require_python("ik_multi");
    return impl_->python->ik_multi(impl_->python_handle, poses, tool_frames);
  }
  if (tool_frames.empty()) {
    fail("ik_multi", "no tool frames were provided");
  }
  // `ik_multi` describes ONE simultaneous multi-tool goal: rows are tools,
  // not batches.
  const auto full = impl_->fill_native_ik_goals(poses, tool_frames, false);
  return impl_->solve_ik(full, {}, nullptr, 1, "ik_multi");
}

void CppBridge::ik_config(
  curobo_runtime::RobotHandle, int batch_size, int num_seeds,
  bool self_collision_check, bool use_lm_seed, bool use_cuda_graph)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("ik_config");
  impl_->ik_settings.batch_size = std::max(1, batch_size);
  impl_->ik_settings.num_seeds = std::max(1, num_seeds);
  impl_->ik_settings.self_collision_check = self_collision_check;
  impl_->ik_settings.use_lm_seed = use_lm_seed;
  impl_->ik_settings.use_cuda_graph = use_cuda_graph;
  impl_->ik.reset();
  if (impl_->python) {
    impl_->python->ik_config(
      impl_->python_handle, batch_size, num_seeds, self_collision_check,
      use_lm_seed, use_cuda_graph);
  }
}

std::vector<std::uint8_t> CppBridge::validate(
  curobo_runtime::RobotHandle, const curobo_runtime::TensorF32 & q)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->native_scene_valid) {
    impl_->require_python("validate");
    return impl_->python->validate(impl_->python_handle, q);
  }
  impl_->ensure_planner("validate");
  if (q.shape.size() != 2 || static_cast<int>(q.shape[1]) != impl_->dof()) {
    fail("validate", "q must have shape [N,dof]");
  }
  const int batch = static_cast<int>(q.shape[0]);
  const int dof_count = impl_->dof();
  const int capacity = impl_->manifest.interpolation_buffer_size > 0 ?
    impl_->manifest.interpolation_buffer_size :
    impl_->planner->info().trajectory_steps;
  if (capacity <= 0) {
    fail("validate", "artifact has no interpolation buffer size");
  }
  const int lanes = std::max(1, impl_->planner_settings.num_trajopt_seeds);
  std::vector<std::uint8_t> valid(static_cast<std::size_t>(batch), 1);
  int invalid_samples = 0;
  int32_t invalid_flags = 0;

  for (int offset = 0; offset < batch; offset += lanes) {
    const int count = std::min(lanes, batch - offset);
    std::vector<float> position(
      static_cast<std::size_t>(lanes) * capacity * dof_count, 0.0F);
    std::vector<float> velocity(position.size(), 0.0F);
    std::vector<float> acceleration(position.size(), 0.0F);
    std::vector<float> jerk(position.size(), 0.0F);
    for (int lane = 0; lane < lanes; ++lane) {
      const int source = offset + std::min(lane, count - 1);
      for (int step = 0; step < capacity; ++step) {
        const std::size_t destination =
          (static_cast<std::size_t>(lane) * capacity + static_cast<std::size_t>(step)) *
          dof_count;
        for (int joint = 0; joint < dof_count; ++joint) {
          position[destination + joint] =
            q.data[static_cast<std::size_t>(source) * dof_count + joint];
        }
      }
    }
    DeviceBuffer position_device;
    DeviceBuffer velocity_device;
    DeviceBuffer acceleration_device;
    DeviceBuffer jerk_device;
    position_device.upload(position);
    velocity_device.upload(velocity);
    acceleration_device.upload(acceleration);
    jerk_device.upload(jerk);
    curobo::robot::TrajectoryValidationRequest request;
    request.position = static_cast<const float *>(position_device.get());
    request.velocity = static_cast<const float *>(velocity_device.get());
    request.acceleration = static_cast<const float *>(acceleration_device.get());
    request.jerk = static_cast<const float *>(jerk_device.get());
    request.num_samples = capacity;
    const int32_t * violations = impl_->planner->validate(request);
    cuda_check("validate", cudaDeviceSynchronize());
    std::vector<int32_t> host(static_cast<std::size_t>(lanes));
    cuda_check(
      "validate",
      cudaMemcpy(host.data(), violations, host.size() * sizeof(int32_t),
        cudaMemcpyDeviceToHost));
    for (int lane = 0; lane < count; ++lane) {
      const int32_t flags = host[static_cast<std::size_t>(lane)];
      valid[static_cast<std::size_t>(offset + lane)] = flags == 0 ? 1 : 0;
      if (flags != 0) {
        invalid_samples += 1;
        invalid_flags |= flags;
      }
    }
  }
  // One summary line per validate() call instead of one per rejected sample:
  // a single fan-out validation can reject hundreds of lanes, which otherwise
  // floods the log.
  if (invalid_samples > 0) {
    std::fprintf(
      stderr,
      "[validate] %d/%d sample(s) invalid (flags=%d q=%d v=%d a=%d jerk=%d "
      "world=%d self=%d density=%d)\n",
      invalid_samples, batch, invalid_flags,
      invalid_flags & 1, invalid_flags & 2, invalid_flags & 4,
      invalid_flags & 8, invalid_flags & 16, invalid_flags & 32,
      invalid_flags & 64);
  }
  return valid;
}

curobo_runtime::JointLimitsOutcome CppBridge::joint_limits(curobo_runtime::RobotHandle)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("joint_limits");
  if (static_cast<int>(impl_->joint_lower.size()) != impl_->dof() ||
    static_cast<int>(impl_->joint_upper.size()) != impl_->dof())
  {
    fail("joint_limits",
      "artifact manifest has no complete joint_limits.lower/upper arrays");
  }
  curobo_runtime::JointLimitsOutcome outcome;
  outcome.joint_names = impl_->joint_names;
  outcome.positions.shape = {2, impl_->dof()};
  outcome.positions.data.resize(static_cast<std::size_t>(2 * impl_->dof()));
  constexpr double kTwoPi = 6.283185307179586;
  outcome.continuous.resize(static_cast<std::size_t>(impl_->dof()), 0);
  for (int i = 0; i < impl_->dof(); ++i) {
    const double lower = impl_->joint_lower[static_cast<std::size_t>(i)];
    const double upper = impl_->joint_upper[static_cast<std::size_t>(i)];
    outcome.positions.data[static_cast<std::size_t>(i)] =
      static_cast<float>(lower);
    outcome.positions.data[static_cast<std::size_t>(impl_->dof() + i)] =
      static_cast<float>(upper);
    // The artifact does not record the URDF joint type; a full-turn range is
    // the only available fallback. The ROS layer overlays the descriptor's
    // explicit continuous_joints override on top of this mask.
    outcome.continuous[static_cast<std::size_t>(i)] =
      (std::isfinite(lower) && std::isfinite(upper) &&
      (upper - lower) >= (kTwoPi - 1e-6)) ? 1 : 0;
  }
  outcome.locked_joint_names = impl_->locked_joint_names;
  outcome.locked_positions = impl_->locked_positions;
  outcome.unpinned_lock_joint_names = impl_->unpinned_lock_joint_names;
  return outcome;
}

std::vector<std::string> CppBridge::tool_frames(curobo_runtime::RobotHandle)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("tool_frames");
  return impl_->tool_frames;
}
}  // namespace curobo_cpp_runtime

// SPDX-License-Identifier: Apache-2.0
#include "cpp_bridge_impl.hpp"

namespace curobo_cpp_runtime
{

std::array<float, 7> CppBridge::Impl::fk_pose(
  const std::vector<float> & q, const char * method)
{
  if (kinematics == nullptr) {
    fail(method, "kinematics is not loaded");
  }
  if (static_cast<int>(q.size()) != dof()) {
    fail(method, "start pose has " + std::to_string(q.size()) + " values, expected " +
      std::to_string(dof()));
  }
  const int tools = num_tools();
  DeviceBuffer input;
  DeviceBuffer positions;
  DeviceBuffer quaternions;
  DeviceBuffer jacobians;
  input.upload(q);
  positions.allocate(static_cast<std::size_t>(tools) * 3 * sizeof(float));
  quaternions.allocate(static_cast<std::size_t>(tools) * 4 * sizeof(float));
  jacobians.allocate(
    static_cast<std::size_t>(tools) * 6 * static_cast<std::size_t>(dof()) *
    sizeof(float));
  kinematics->forward(
    static_cast<const float *>(input.get()),
    static_cast<float *>(positions.get()), static_cast<float *>(quaternions.get()),
    static_cast<float *>(jacobians.get()), 1, 1, nullptr);
  cuda_check(method, cudaDeviceSynchronize());
  std::vector<float> p(static_cast<std::size_t>(tools) * 3);
  std::vector<float> r(static_cast<std::size_t>(tools) * 4);
  positions.download(p);
  quaternions.download(r);
  std::array<float, 7> pose{};
  pose[0] = p[0];
  pose[1] = p[1];
  pose[2] = p[2];
  pose[3] = r[0];
  pose[4] = r[1];
  pose[5] = r[2];
  pose[6] = r[3];
  return pose;
}

std::array<float, 7> CppBridge::Impl::tool_pose(
  const std::vector<float> & q, int tool_index, const char * method)
{
  if (kinematics == nullptr) {
    fail(method, "kinematics is not loaded");
  }
  const int tools = num_tools();
  if (tool_index < 0 || tool_index >= tools) {
    fail(method, "tool index is outside the artifact tools");
  }
  const int dof_count = dof();
  std::vector<float> clamped = q;
  for (int i = 0; i < std::min(dof_count, static_cast<int>(clamped.size())); ++i) {
    // A continuous joint's limits are a chart, not a stop: clamping an
    // unwrapped value (e.g. 6.5) would compute FK of a different pose.
    if (static_cast<std::size_t>(i) < joint_continuous.size() &&
      joint_continuous[static_cast<std::size_t>(i)] != 0)
    {
      continue;
    }
    clamped[static_cast<std::size_t>(i)] = std::clamp(
      clamped[static_cast<std::size_t>(i)],
      static_cast<float>(joint_lower[static_cast<std::size_t>(i)]),
      static_cast<float>(joint_upper[static_cast<std::size_t>(i)]));
  }
  clamped.resize(static_cast<std::size_t>(dof_count), 0.0F);
  DeviceBuffer input;
  DeviceBuffer positions;
  DeviceBuffer quaternions;
  DeviceBuffer jacobians;
  input.upload(clamped);
  positions.allocate(static_cast<std::size_t>(tools) * 3 * sizeof(float));
  quaternions.allocate(static_cast<std::size_t>(tools) * 4 * sizeof(float));
  jacobians.allocate(
    static_cast<std::size_t>(tools) * 6 * static_cast<std::size_t>(dof_count) *
    sizeof(float));
  kinematics->forward(
    static_cast<const float *>(input.get()),
    static_cast<float *>(positions.get()), static_cast<float *>(quaternions.get()),
    static_cast<float *>(jacobians.get()), 1, 1, nullptr);
  cuda_check(method, cudaDeviceSynchronize());
  std::vector<float> p(static_cast<std::size_t>(tools) * 3);
  std::vector<float> r(static_cast<std::size_t>(tools) * 4);
  positions.download(p);
  quaternions.download(r);
  const std::size_t pi = static_cast<std::size_t>(tool_index) * 3;
  const std::size_t ri = static_cast<std::size_t>(tool_index) * 4;
  return {p[pi], p[pi + 1], p[pi + 2], r[ri], r[ri + 1], r[ri + 2], r[ri + 3]};
}

curobo_runtime::TensorF32 CppBridge::Impl::fk_tensor(
  const curobo_runtime::TensorF32 & q, const char * method)
{
  if (q.shape.size() != 2 || static_cast<int>(q.shape[1]) != dof()) {
    fail(method, "q must have shape [N, dof]");
  }
  const int batch = static_cast<int>(q.shape[0]);
  const int tools = num_tools();
  DeviceBuffer input;
  DeviceBuffer positions;
  DeviceBuffer quaternions;
  DeviceBuffer jacobians;
  input.upload(q.data);
  positions.allocate(
    static_cast<std::size_t>(batch) * tools * 3 * sizeof(float));
  quaternions.allocate(
    static_cast<std::size_t>(batch) * tools * 4 * sizeof(float));
  jacobians.allocate(
    static_cast<std::size_t>(batch) * tools * 6 *
    static_cast<std::size_t>(dof()) * sizeof(float));
  kinematics->forward(
    static_cast<const float *>(input.get()),
    static_cast<float *>(positions.get()), static_cast<float *>(quaternions.get()),
    static_cast<float *>(jacobians.get()), batch, 1, nullptr);
  cuda_check(method, cudaDeviceSynchronize());
  std::vector<float> p(static_cast<std::size_t>(batch) * tools * 3);
  std::vector<float> r(static_cast<std::size_t>(batch) * tools * 4);
  positions.download(p);
  quaternions.download(r);

  TensorF32 output;
  output.shape = {batch, 1, 7};
  output.data.resize(static_cast<std::size_t>(batch) * 7);
  for (int i = 0; i < batch; ++i) {
    const std::size_t p_offset = static_cast<std::size_t>(i) * tools * 3;
    const std::size_t r_offset = static_cast<std::size_t>(i) * tools * 4;
    output.data[static_cast<std::size_t>(i) * 7 + 0] = p[p_offset + 0];
    output.data[static_cast<std::size_t>(i) * 7 + 1] = p[p_offset + 1];
    output.data[static_cast<std::size_t>(i) * 7 + 2] = p[p_offset + 2];
    output.data[static_cast<std::size_t>(i) * 7 + 3] = r[r_offset + 0];
    output.data[static_cast<std::size_t>(i) * 7 + 4] = r[r_offset + 1];
    output.data[static_cast<std::size_t>(i) * 7 + 5] = r[r_offset + 2];
    output.data[static_cast<std::size_t>(i) * 7 + 6] = r[r_offset + 3];
  }
  return output;
}

curobo_runtime::IkOutcome CppBridge::Impl::solve_ik(
  const curobo_runtime::TensorF32 & poses, const std::vector<std::string> & frames,
  const std::vector<float> * seed_q, int num_goalset, const char * method)
{
  ensure_ik(method);
  const int tools = num_tools();
  const int dof_count = dof();
  if (tools <= 0) {
    fail(method, "artifact has no tool frames");
  }

  int batch = 0;
  if (poses.shape.size() == 2) {
    if (static_cast<int>(poses.shape[1]) != 7) {
      fail(method, "poses must have shape [N,7]");
    }
    batch = static_cast<int>(poses.shape[0]);
    if (tools > 1) {
      fail(method,
        "cpp_runtime IK targets every artifact tool frame simultaneously; this artifact has " +
        std::to_string(tools) + " tool frames but the request supplied one. Use ik_multi.");
    }
  } else if (poses.shape.size() == 3) {
    if (static_cast<int>(poses.shape[2]) != 7) {
      fail(method, "poses must have shape [N,T,7]");
    }
    batch = static_cast<int>(poses.shape[0]);
    if (static_cast<int>(poses.shape[1]) != tools) {
      fail(method, "poses second dimension must match the artifact tool frames (" +
        std::to_string(tools) + ")");
    }
  } else {
    fail(method, "poses must have shape [N,7] or [N,T,7]");
  }
  if (batch <= 0) {
    fail(method, "empty IK batch");
  }

  // Map artifact tool order to the request's frames. All artifact frames must
  // be addressed; native IK always produces one goal per configured tool.
  std::vector<int> tool_order(static_cast<std::size_t>(tools));
  for (int i = 0; i < tools; ++i) {
    tool_order[static_cast<std::size_t>(i)] = i;
  }
  if (!frames.empty()) {
    if (static_cast<int>(frames.size()) != tools) {
      fail(method, "IK requires one target per artifact tool frame (" +
        std::to_string(tools) + "), got " + std::to_string(frames.size()));
    }
    std::vector<bool> used(static_cast<std::size_t>(tools), false);
    for (int slot = 0; slot < tools; ++slot) {
      const auto it = std::find(tool_frames.begin(), tool_frames.end(), frames[static_cast<std::size_t>(slot)]);
      if (it == tool_frames.end()) {
        fail(method, "unknown tool frame '" + frames[static_cast<std::size_t>(slot)] + "'");
      }
      const int index = static_cast<int>(std::distance(tool_frames.begin(), it));
      if (used[static_cast<std::size_t>(index)]) {
        fail(method, "duplicate tool frame '" + frames[static_cast<std::size_t>(slot)] + "'");
      }
      used[static_cast<std::size_t>(index)] = true;
      tool_order[static_cast<std::size_t>(slot)] = index;
    }
  }

  std::vector<float> goal_positions(
    static_cast<std::size_t>(batch) * tools * num_goalset * 3);
  std::vector<float> goal_quaternions(
    static_cast<std::size_t>(batch) * tools * num_goalset * 4);
  for (int i = 0; i < batch; ++i) {
    for (int slot = 0; slot < tools; ++slot) {
      const int source_tool = poses.shape.size() == 2 ? 0 : slot;
      const std::size_t source =
        (static_cast<std::size_t>(i) * (poses.shape.size() == 2 ? 1 : tools) +
        static_cast<std::size_t>(source_tool)) * 7;
      const int destination_tool = tool_order[static_cast<std::size_t>(slot)];
      const std::size_t destination =
        (static_cast<std::size_t>(i) * tools + static_cast<std::size_t>(destination_tool)) *
        static_cast<std::size_t>(num_goalset);
      for (int g = 0; g < num_goalset; ++g) {
        // IkSolveRequest keeps positions and quaternions in two separate
        // contiguous tensors: [batch, tool, goalset, 3] and [..., 4]. They are
        // NOT interleaved 7-float rows.
        const std::size_t value = destination + static_cast<std::size_t>(g);
        goal_positions[value * 3 + 0] = poses.data[source + 0];
        goal_positions[value * 3 + 1] = poses.data[source + 1];
        goal_positions[value * 3 + 2] = poses.data[source + 2];
        goal_quaternions[value * 4 + 0] = poses.data[source + 3];
        goal_quaternions[value * 4 + 1] = poses.data[source + 4];
        goal_quaternions[value * 4 + 2] = poses.data[source + 5];
        goal_quaternions[value * 4 + 3] = poses.data[source + 6];
      }
    }
  }

  DeviceBuffer goal_pos_device;
  DeviceBuffer goal_quat_device;
  DeviceBuffer seed_device;
  DeviceBuffer solution_device;
  DeviceBuffer success_device;
  DeviceBuffer pos_err_device;
  DeviceBuffer orient_err_device;
  DeviceBuffer collision_device;
  DeviceBuffer goalset_index_device;

  goal_pos_device.upload(goal_positions);
  goal_quat_device.upload(goal_quaternions);
  if (seed_q != nullptr) {
    if (static_cast<int>(seed_q->size()) != batch * dof_count) {
      fail(method, "IK seeds must have shape [N,dof]");
    }
    seed_device.upload(*seed_q);
  }
  solution_device.allocate(
    static_cast<std::size_t>(batch) * dof_count * sizeof(float));
  success_device.allocate(static_cast<std::size_t>(batch) * sizeof(uint8_t));
  pos_err_device.allocate(static_cast<std::size_t>(batch) * sizeof(float));
  orient_err_device.allocate(static_cast<std::size_t>(batch) * sizeof(float));
  collision_device.allocate(static_cast<std::size_t>(batch) * sizeof(float));
  if (num_goalset > 1) {
    goalset_index_device.allocate(
      static_cast<std::size_t>(batch) * tools * sizeof(int32_t));
  }

  curobo::robot::IkSolveRequest request;
  request.goal_positions = static_cast<const float *>(goal_pos_device.get());
  request.goal_quaternions = static_cast<const float *>(goal_quat_device.get());
  request.seed_positions = seed_q == nullptr ? nullptr :
    static_cast<const float *>(seed_device.get());
  request.solution = static_cast<float *>(solution_device.get());
  request.success = static_cast<uint8_t *>(success_device.get());
  request.position_error = static_cast<float *>(pos_err_device.get());
  request.orientation_error = static_cast<float *>(orient_err_device.get());
  request.collision_cost = static_cast<float *>(collision_device.get());
  request.goalset_index = num_goalset > 1 ?
    static_cast<int32_t *>(goalset_index_device.get()) : nullptr;
  request.batch_size = batch;
  request.num_goalset = num_goalset;
  request.num_solutions = 1;
  ik->solve(request);
  cuda_check(method, cudaDeviceSynchronize());

  std::vector<float> solutions(static_cast<std::size_t>(batch) * dof_count);
  std::vector<uint8_t> success(static_cast<std::size_t>(batch));
  solution_device.download(solutions);
  success_device.download(success);

  IkOutcome outcome;
  outcome.positions.shape = {batch, 1, dof_count};
  outcome.positions.data = std::move(solutions);
  outcome.valid.resize(static_cast<std::size_t>(batch));
  outcome.joint_names = joint_names;
  for (int i = 0; i < batch; ++i) {
    outcome.valid[static_cast<std::size_t>(i)] = success[static_cast<std::size_t>(i)] != 0 ? 1 : 0;
    if (outcome.valid[static_cast<std::size_t>(i)] != 0) {
      outcome.success = true;
    }
  }
  return outcome;
}
}  // namespace curobo_cpp_runtime

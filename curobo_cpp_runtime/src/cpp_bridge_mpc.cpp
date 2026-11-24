// SPDX-License-Identifier: Apache-2.0
#include "cpp_bridge_impl.hpp"

namespace curobo_cpp_runtime
{

void CppBridge::mpc_build(
  curobo_runtime::RobotHandle, const std::string & cfg_json)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("mpc_build");
  if (!cfg_json.empty()) {
    try {
      const json::Value document = json::parse(cfg_json);
      // The node sends mpc_step_dt as "step_dt" (see mpc_planner.cpp). Older
      // callers used "trajectory_dt"; keep that as a fallback.
      const double default_dt = impl_->mpc_trajectory_dt > 0.0F ?
        impl_->mpc_trajectory_dt : impl_->planner_settings.interpolation_dt;
      impl_->mpc_trajectory_dt = static_cast<float>(
        json_double(document, "step_dt", json_double(document, "trajectory_dt", default_dt)));
      impl_->mpc_command_interval = static_cast<float>(
        json_double(document, "command_interval", impl_->mpc_command_interval));
    } catch (const std::exception &) {
    }
  }
  impl_->mpc.reset();
  impl_->mpc_accept_pending = false;
  impl_->mpc_accept_knots = 0;
  impl_->ensure_mpc("mpc_build");
}

curobo_runtime::MpcGoalOutcome CppBridge::mpc_setup(
  curobo_runtime::RobotHandle, const std::vector<float> & start_q)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("mpc_setup");
  if (static_cast<int>(start_q.size()) != impl_->dof()) {
    fail("mpc_setup", "start state size does not match the artifact dof");
  }
  impl_->mpc_start = start_q;
  curobo_runtime::MpcGoalOutcome outcome;
  outcome.applied = true;
  try {
    const auto pose = impl_->fk_pose(start_q, "mpc_setup");
    outcome.has_ee = true;
    outcome.ee_position = {pose[0], pose[1], pose[2]};
  } catch (const std::exception &) {
    outcome.has_ee = false;
  }
  return outcome;
}

curobo_runtime::MpcStepOutcome CppBridge::mpc_step(
  curobo_runtime::RobotHandle, const curobo_runtime::ReactiveState & state, bool paced)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("mpc_step");
  auto & mpc = impl_->ensure_mpc("mpc_step");
  const int dof_count = impl_->dof();
  if (static_cast<int>(state.position.size()) != dof_count) {
    fail("mpc_step", "state.position size does not match the artifact dof");
  }
  std::vector<float> velocity(dof_count, 0.0F);
  std::vector<float> acceleration(dof_count, 0.0F);
  if (static_cast<int>(state.velocity.size()) == dof_count) {
    velocity = state.velocity;
  }
  if (static_cast<int>(state.acceleration.size()) == dof_count) {
    acceleration = state.acceleration;
  }
  std::vector<float> jerk(dof_count, 0.0F);
  DeviceBuffer state_device;
  DeviceBuffer velocity_device;
  DeviceBuffer acceleration_device;
  DeviceBuffer jerk_device;
  state_device.upload(state.position);
  velocity_device.upload(velocity);
  acceleration_device.upload(acceleration);
  jerk_device.upload(jerk);
  // Commit the proposal the executor has just finished and advance the seed by
  // the whole knots it consumed. This is the same accept+shift handoff the
  // native MPC tests use; skipping it freezes warm-starting at mpc_start.
  int executed_knots = 0;
  if (impl_->mpc_accept_pending) {
    mpc.accept();
    executed_knots = impl_->mpc_accept_knots;
    impl_->mpc_accept_pending = false;
    impl_->mpc_accept_knots = 0;
  }
  impl_->last_mpc_result = mpc.step(
    static_cast<const float *>(state_device.get()),
    static_cast<const float *>(velocity_device.get()),
    static_cast<const float *>(acceleration_device.get()),
    static_cast<const float *>(jerk_device.get()), executed_knots);
  cuda_check("mpc_step", cudaDeviceSynchronize());

  const auto info = impl_->planner->info();
  const int steps = std::max(1, info.trajectory_steps);
  const float sample_dt = impl_->last_mpc_result.trajectory_dt;
  int command_index = 0;
  if (impl_->last_mpc_result.feasible) {
    int advance = 1;
    if (paced && impl_->mpc_command_interval > 0.0F && sample_dt > 0.0F) {
      advance = static_cast<int>(std::lround(impl_->mpc_command_interval / sample_dt));
    }
    command_index = std::max(1, std::min(steps - 1, advance));
  }
  if (impl_->last_mpc_result.feasible) {
    const int samples_per_knot = std::max(1, info.spline_layout.interpolation_steps);
    const int knots = std::max(
      0, std::min(
        info.num_knots,
        (command_index + samples_per_knot / 2) / samples_per_knot));
    impl_->mpc_accept_pending = true;
    impl_->mpc_accept_knots = knots;
  }

  curobo_runtime::MpcStepOutcome outcome;
  outcome.action.has_action = impl_->last_mpc_result.feasible;
  const bool segment = paced && impl_->mpc_command_interval > 0.0F;
  const int command_rows = std::max(1, std::min(steps, command_index));
  const int overlap_rows = (segment && sample_dt > 0.0F) ?
    static_cast<int>(std::lround(0.12F / sample_dt)) : 0;
  const int first_row = 0;
  const int row_count = segment ?
    std::max(1, std::min(steps, command_rows + std::max(0, overlap_rows))) : 1;
  const std::size_t offset = static_cast<std::size_t>(first_row * dof_count);
  const std::size_t count = static_cast<std::size_t>(row_count * dof_count);
  if (impl_->last_mpc_result.position != nullptr) {
    std::vector<float> values(count);
    cuda_check(
      "mpc_step",
      cudaMemcpy(
        values.data(), impl_->last_mpc_result.position + offset,
        values.size() * sizeof(float), cudaMemcpyDeviceToHost));
    outcome.action.position.shape = {row_count, dof_count};
    outcome.action.position.data = std::move(values);
  }
  if (impl_->last_mpc_result.velocity != nullptr) {
    std::vector<float> values(count);
    cuda_check(
      "mpc_step",
      cudaMemcpy(
        values.data(), impl_->last_mpc_result.velocity + offset,
        values.size() * sizeof(float), cudaMemcpyDeviceToHost));
    outcome.action.velocity.shape = {row_count, dof_count};
    outcome.action.velocity.data = std::move(values);
  }
  if (impl_->last_mpc_result.acceleration != nullptr) {
    std::vector<float> values(count);
    cuda_check(
      "mpc_step",
      cudaMemcpy(
        values.data(), impl_->last_mpc_result.acceleration + offset,
        values.size() * sizeof(float), cudaMemcpyDeviceToHost));
    outcome.action.acceleration.shape = {row_count, dof_count};
    outcome.action.acceleration.data = std::move(values);
  }
  if (segment && sample_dt > 0.0F) {
    outcome.action_dt = sample_dt;
  }
  try {
    const auto pose = impl_->fk_pose(state.position, "mpc_step");
    outcome.has_ee = true;
    outcome.ee_position = {pose[0], pose[1], pose[2]};
  } catch (const std::exception &) {
  }
  return outcome;
}

bool CppBridge::mpc_update_goal_pose(
  curobo_runtime::RobotHandle handle, const std::array<double, 7> & goal_raw, bool run_ik)
{
  curobo_runtime::TensorF32 poses;
  poses.shape = {1, 7};
  poses.data = {
    static_cast<float>(goal_raw[0]), static_cast<float>(goal_raw[1]),
    static_cast<float>(goal_raw[2]), static_cast<float>(goal_raw[3]),
    static_cast<float>(goal_raw[4]), static_cast<float>(goal_raw[5]),
    static_cast<float>(goal_raw[6])};
  return mpc_update_goal_poses(handle, {}, poses, run_ik, {});
}

bool CppBridge::mpc_update_goal_poses(
  curobo_runtime::RobotHandle, const std::vector<std::string> & tool_frames,
  const curobo_runtime::TensorF32 & poses, bool,
  const std::vector<float> & hold_q)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("mpc_update_goal_poses");
  auto & mpc = impl_->ensure_mpc("mpc_update_goal_poses");
  if (impl_->mpc_start.empty()) {
    fail("mpc_update_goal_poses", "mpc_setup must be called before applying a goal");
  }
  if (poses.shape.size() != 2 || poses.shape[1] != 7) {
    fail("mpc_update_goal_poses", "goal poses must have shape [N,7]");
  }
  const int tools = impl_->num_tools();
  if (tools <= 0) {
    fail("mpc_update_goal_poses", "artifact has no tool frames");
  }
  const int rows = static_cast<int>(poses.shape[0]);
  if (rows <= 0) {
    fail("mpc_update_goal_poses", "empty goal pose list");
  }
  if (!tool_frames.empty() && static_cast<int>(tool_frames.size()) != rows) {
    fail("mpc_update_goal_poses",
      "tool_frames and goal poses must be parallel (got " +
      std::to_string(tool_frames.size()) + " frame(s), " + std::to_string(rows) +
      " pose(s))");
  }
  if (tool_frames.empty() && rows != 1) {
    fail("mpc_update_goal_poses",
      "a " + std::to_string(rows) + "-pose goal needs one tool frame per pose");
  }
  const std::vector<float> & phase_q =
    hold_q.empty() ? impl_->mpc_start : hold_q;
  if (static_cast<int>(phase_q.size()) != impl_->dof()) {
    fail("mpc_update_goal_poses", "hold state size does not match the artifact dof");
  }
  // Every tool starts at its pose for the phase state; the named goals then
  // override their own slot. An unnamed tool therefore HOLDs where it is
  // instead of being dragged to another tool's target.
  std::vector<std::array<float, 7>> poses_by_tool =
    impl_->tool_poses_for_q(phase_q);
  std::vector<bool> assigned(static_cast<std::size_t>(tools), false);
  for (int row = 0; row < rows; ++row) {
    int tool_index = 0;
    if (!tool_frames.empty() && !tool_frames[static_cast<std::size_t>(row)].empty()) {
      const auto it = std::find(
        impl_->tool_frames.begin(), impl_->tool_frames.end(),
        tool_frames[static_cast<std::size_t>(row)]);
      if (it == impl_->tool_frames.end()) {
        fail("mpc_update_goal_poses",
          "unknown tool frame '" + tool_frames[static_cast<std::size_t>(row)] + "'");
      }
      tool_index = static_cast<int>(std::distance(impl_->tool_frames.begin(), it));
    }
    if (assigned[static_cast<std::size_t>(tool_index)]) {
      fail("mpc_update_goal_poses", "duplicate tool frame in the goal");
    }
    assigned[static_cast<std::size_t>(tool_index)] = true;
    for (std::size_t k = 0; k < 7; ++k) {
      poses_by_tool[static_cast<std::size_t>(tool_index)][k] =
        poses.data[static_cast<std::size_t>(row) * 7 + k];
    }
  }
  std::vector<float> positions;
  std::vector<float> quaternions;
  fill_pose_arrays(poses_by_tool, positions, quaternions);

  const auto info = impl_->planner->info();
  const int knots = info.num_knots;
  std::vector<float> seed(static_cast<std::size_t>(knots) * impl_->dof());
  // Interpolate the seed from the phase start to the target state. A constant
  // target seed made the optimizer start from the endpoint and reject the
  // handoff, especially at the straight-arm singularity.
  const std::vector<float> * seed_start = &phase_q;
  if (static_cast<int>(impl_->mpc_start.size()) == impl_->dof()) {
    seed_start = &impl_->mpc_start;
  }
  const float seed_denom = knots > 1 ? static_cast<float>(knots - 1) : 1.0F;
  for (int k = 0; k < knots; ++k) {
    const float alpha = static_cast<float>(k) / seed_denom;
    for (int j = 0; j < impl_->dof(); ++j) {
      const float start = (*seed_start)[static_cast<std::size_t>(j)];
      const float goal = phase_q[static_cast<std::size_t>(j)];
      seed[static_cast<std::size_t>(k * impl_->dof() + j)] =
        (1.0F - alpha) * start + alpha * goal;
    }
  }
  DeviceBuffer position_device;
  DeviceBuffer quaternion_device;
  DeviceBuffer seed_device;
  position_device.upload(positions);
  quaternion_device.upload(quaternions);
  seed_device.upload(seed);
  curobo::robot::MpcCartesianConfig config;
  config.constrain_path = false;
  mpc.reset_cartesian(
    static_cast<const float *>(position_device.get()),
    static_cast<const float *>(quaternion_device.get()),
    static_cast<const float *>(seed_device.get()), config);
  cuda_check("mpc_update_goal_poses", cudaDeviceSynchronize());
  impl_->mpc_goal_pose = poses_by_tool.front();
  impl_->mpc_has_goal_pose = true;
  impl_->mpc_has_goal_state = false;
  impl_->mpc_accept_pending = false;
  impl_->mpc_accept_knots = 0;
  return true;
}

void CppBridge::mpc_update_goal_state(
  curobo_runtime::RobotHandle, const std::vector<float> & q)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("mpc_update_goal_state");
  auto & mpc = impl_->ensure_mpc("mpc_update_goal_state");
  if (static_cast<int>(q.size()) != impl_->dof()) {
    fail("mpc_update_goal_state", "goal state size does not match the artifact dof");
  }
  if (impl_->mpc_start.empty()) {
    fail("mpc_update_goal_state", "mpc_setup must be called before applying a goal");
  }
  const auto info = impl_->planner->info();
  const int knots = info.num_knots;
  std::vector<float> seed(static_cast<std::size_t>(knots) * impl_->dof());
  for (int k = 0; k < knots; ++k) {
    for (int j = 0; j < impl_->dof(); ++j) {
      seed[static_cast<std::size_t>(k * impl_->dof() + j)] =
        impl_->mpc_start[static_cast<std::size_t>(j)];
    }
  }
  DeviceBuffer goal_device;
  DeviceBuffer seed_device;
  goal_device.upload(q);
  seed_device.upload(seed);
  mpc.reset(
    static_cast<const float *>(goal_device.get()),
    static_cast<const float *>(seed_device.get()));
  cuda_check("mpc_update_goal_state", cudaDeviceSynchronize());
  impl_->mpc_goal_state = q;
  impl_->mpc_has_goal_state = true;
  impl_->mpc_has_goal_pose = false;
  impl_->mpc_accept_pending = false;
  impl_->mpc_accept_knots = 0;
}

void CppBridge::mpc_reseed(
  curobo_runtime::RobotHandle, const std::vector<float> & q)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("mpc_reseed");
  if (static_cast<int>(q.size()) != impl_->dof()) {
    fail("mpc_reseed", "seed state size does not match the artifact dof");
  }
  impl_->mpc_start = q;
  if (impl_->mpc_has_goal_state) {
    // Reset is the native way to replace the fingerprint seed.
    auto & mpc = impl_->ensure_mpc("mpc_reseed");
    const auto info = impl_->planner->info();
    std::vector<float> seed(
      static_cast<std::size_t>(info.num_knots) * static_cast<std::size_t>(impl_->dof()));
    for (int k = 0; k < info.num_knots; ++k) {
      for (int j = 0; j < impl_->dof(); ++j) {
        seed[static_cast<std::size_t>(k * impl_->dof() + j)] =
          q[static_cast<std::size_t>(j)];
      }
    }
    DeviceBuffer goal_device;
    DeviceBuffer seed_device;
    goal_device.upload(impl_->mpc_goal_state);
    seed_device.upload(seed);
    mpc.reset(
      static_cast<const float *>(goal_device.get()),
      static_cast<const float *>(seed_device.get()));
    cuda_check("mpc_reseed", cudaDeviceSynchronize());
  }
}

curobo_runtime::IkOutcome CppBridge::mpc_solve_goal_state(
  curobo_runtime::RobotHandle handle, const std::array<double, 7> & goal_raw,
  const std::vector<float> & seed_q)
{
  TensorF32 poses;
  poses.shape = {1, 7};
  poses.data = {
    static_cast<float>(goal_raw[0]), static_cast<float>(goal_raw[1]),
    static_cast<float>(goal_raw[2]), static_cast<float>(goal_raw[3]),
    static_cast<float>(goal_raw[4]), static_cast<float>(goal_raw[5]),
    static_cast<float>(goal_raw[6])};
  // A single-tool anchor on a multi-tool robot: the other tools hold the FK
  // pose of the seed state, not a foreign tool's target.
  return mpc_solve_goal_poses(handle, {}, poses, seed_q, seed_q);
}

curobo_runtime::IkOutcome CppBridge::mpc_solve_goal_poses(
  curobo_runtime::RobotHandle, const std::vector<std::string> & tool_frames,
  const curobo_runtime::TensorF32 & poses, const std::vector<float> & hold_q,
  const std::vector<float> & seed_q)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("mpc_solve_goal_poses");
  const int tools = impl_->num_tools();
  if (tools <= 0) {
    fail("mpc_solve_goal_poses", "artifact has no tool frames");
  }
  const int dof_count = impl_->dof();
  if (poses.shape.size() != 2 || poses.shape[1] != 7) {
    fail("mpc_solve_goal_poses", "goal poses must have shape [N,7]");
  }
  const int rows = static_cast<int>(poses.shape[0]);
  if (rows <= 0) {
    fail("mpc_solve_goal_poses", "empty goal pose list");
  }
  if (!tool_frames.empty() && static_cast<int>(tool_frames.size()) != rows) {
    fail("mpc_solve_goal_poses",
      "tool_frames and goal poses must be parallel (got " +
      std::to_string(tool_frames.size()) + " frame(s), " + std::to_string(rows) +
      " pose(s))");
  }
  if (tool_frames.empty() && rows != 1) {
    fail("mpc_solve_goal_poses",
      "a " + std::to_string(rows) + "-pose goal needs one tool frame per pose");
  }
  const std::vector<float> & phase_q =
    hold_q.empty() ? impl_->mpc_start : hold_q;
  if (static_cast<int>(phase_q.size()) != dof_count) {
    fail("mpc_solve_goal_poses", "hold state size does not match the artifact dof");
  }
  std::vector<std::array<float, 7>> poses_by_tool =
    impl_->tool_poses_for_q(phase_q);
  std::vector<bool> assigned(static_cast<std::size_t>(tools), false);
  for (int row = 0; row < rows; ++row) {
    int tool_index = 0;
    if (!tool_frames.empty() && !tool_frames[static_cast<std::size_t>(row)].empty()) {
      const auto it = std::find(
        impl_->tool_frames.begin(), impl_->tool_frames.end(),
        tool_frames[static_cast<std::size_t>(row)]);
      if (it == impl_->tool_frames.end()) {
        fail("mpc_solve_goal_poses",
          "unknown tool frame '" + tool_frames[static_cast<std::size_t>(row)] + "'");
      }
      tool_index = static_cast<int>(std::distance(impl_->tool_frames.begin(), it));
    }
    if (assigned[static_cast<std::size_t>(tool_index)]) {
      fail("mpc_solve_goal_poses", "duplicate tool frame in the goal");
    }
    assigned[static_cast<std::size_t>(tool_index)] = true;
    for (std::size_t k = 0; k < 7; ++k) {
      poses_by_tool[static_cast<std::size_t>(tool_index)][k] =
        poses.data[static_cast<std::size_t>(row) * 7 + k];
    }
  }
  // solve_ik takes one fully-populated goal per artifact tool frame.
  TensorF32 full;
  full.shape = {1, tools, 7};
  full.data.resize(static_cast<std::size_t>(tools) * 7);
  for (int tool = 0; tool < tools; ++tool) {
    for (std::size_t k = 0; k < 7; ++k) {
      full.data[static_cast<std::size_t>(tool) * 7 + k] =
        poses_by_tool[static_cast<std::size_t>(tool)][k];
    }
  }
  return impl_->solve_ik(
    full, {}, seed_q.empty() ? nullptr : &seed_q, 1, "mpc_solve_goal_poses");
}

void CppBridge::mpc_load_scene(
  curobo_runtime::RobotHandle, const std::string &)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("mpc_load_scene");
  // The native MPC shares MotionPlanner's live Scene; callers still update the
  // scene through scene_set. There is no separate world copy to load.
}

std::string CppBridge::mpc_metrics(curobo_runtime::RobotHandle)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  std::ostringstream out;
  out << "{\"feasible\":" << (impl_->last_mpc_result.feasible ? "true" : "false")
      << ",\"cost\":" << impl_->last_mpc_result.cost
      << ",\"collision_cost\":" << impl_->last_mpc_result.collision_cost
      << ",\"trajectory_dt\":" << impl_->last_mpc_result.trajectory_dt
      << ",\"selected_candidate\":" << impl_->last_mpc_result.selected_candidate
      << "}";
  return out.str();
}
}  // namespace curobo_cpp_runtime

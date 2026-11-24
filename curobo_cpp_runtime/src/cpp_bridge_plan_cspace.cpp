// SPDX-License-Identifier: Apache-2.0
#include "cpp_bridge_impl.hpp"

namespace curobo_cpp_runtime
{

curobo_runtime::PlanOutcome CppBridge::Impl::make_outcome(
  const std::vector<float> & position, const std::vector<float> & velocity,
  const std::vector<float> & acceleration, int dof_count, int steps, double dt,
  const std::vector<std::string> & names)
{
  PlanOutcome outcome;
  outcome.positions.shape = {steps, dof_count};
  outcome.velocity.shape = {steps, dof_count};
  outcome.acceleration.shape = {steps, dof_count};
  outcome.positions.data = position;
  outcome.velocity.data = velocity;
  outcome.acceleration.data = acceleration;
  outcome.joint_names = names;
  outcome.total_time = steps > 0 ? static_cast<double>(steps - 1) * dt : 0.0;
  return outcome;
}

curobo_runtime::PlanOutcome CppBridge::Impl::plan_cspace_impl(
  const curobo_runtime::PlanRequest & request)
{
  ensure_planner("plan_cspace");
  const int dof_count = dof();
  if (static_cast<int>(request.start.size()) != dof_count) {
    fail("plan_cspace", "start has " + std::to_string(request.start.size()) +
      " values, expected " + std::to_string(dof_count));
  }
  if (static_cast<int>(request.joint_target.size()) != dof_count) {
    fail("plan_cspace", "joint_target has " + std::to_string(request.joint_target.size()) +
      " values, expected " + std::to_string(dof_count));
  }
  std::vector<float> start = request.start;
  std::vector<float> goal = request.joint_target;
  const int limit_count = std::min(
    dof_count, static_cast<int>(std::min(joint_lower.size(), joint_upper.size())));
  for (int i = 0; i < limit_count; ++i) {
    if (static_cast<std::size_t>(i) < joint_continuous.size() &&
      joint_continuous[static_cast<std::size_t>(i)] != 0)
    {
      continue;  // the ROS layer already placed it inside the chart window
    }
    const float lower = static_cast<float>(joint_lower[static_cast<std::size_t>(i)]);
    const float upper = static_cast<float>(joint_upper[static_cast<std::size_t>(i)]);
    start[static_cast<std::size_t>(i)] = std::clamp(start[static_cast<std::size_t>(i)], lower, upper);
    goal[static_cast<std::size_t>(i)] = std::clamp(goal[static_cast<std::size_t>(i)], lower, upper);
  }

  const auto info = planner->info();
  const int steps = info.trajectory_steps;
  const int knots = info.num_knots;

  DeviceBuffer start_device;
  DeviceBuffer goal_device;
  DeviceBuffer knots_device;
  DeviceBuffer position_device;
  DeviceBuffer velocity_device;
  DeviceBuffer acceleration_device;
  DeviceBuffer jerk_device;
  DeviceBuffer success_device;
  DeviceBuffer cost_device;
  DeviceBuffer collision_device;
  DeviceBuffer dt_device;
  DeviceBuffer rrt_device;
  DeviceBuffer rrt_iterations_device;
  DeviceBuffer selected_seed_device;

  start_device.upload(start);
  goal_device.upload(goal);
  knots_device.allocate(static_cast<std::size_t>(knots) * dof_count * sizeof(float));
  position_device.allocate(static_cast<std::size_t>(steps) * dof_count * sizeof(float));
  velocity_device.allocate(static_cast<std::size_t>(steps) * dof_count * sizeof(float));
  acceleration_device.allocate(static_cast<std::size_t>(steps) * dof_count * sizeof(float));
  jerk_device.allocate(static_cast<std::size_t>(steps) * dof_count * sizeof(float));
  success_device.allocate(sizeof(uint8_t));
  cost_device.allocate(sizeof(float));
  collision_device.allocate(sizeof(float));
  dt_device.allocate(sizeof(float));
  rrt_device.allocate(sizeof(uint8_t));
  rrt_iterations_device.allocate(sizeof(int32_t));
  selected_seed_device.allocate(sizeof(int32_t));

  curobo::robot::MotionPlanningRequest native;
  native.start_position = static_cast<const float *>(start_device.get());
  native.goal_position = static_cast<const float *>(goal_device.get());
  native.optimized_knots = static_cast<float *>(knots_device.get());
  native.trajectory_position = static_cast<float *>(position_device.get());
  native.trajectory_velocity = static_cast<float *>(velocity_device.get());
  native.trajectory_acceleration = static_cast<float *>(acceleration_device.get());
  native.trajectory_jerk = static_cast<float *>(jerk_device.get());
  native.success = static_cast<uint8_t *>(success_device.get());
  native.cost = static_cast<float *>(cost_device.get());
  native.collision_cost = static_cast<float *>(collision_device.get());
  native.trajectory_dt = static_cast<float *>(dt_device.get());
  native.used_rrt = static_cast<uint8_t *>(rrt_device.get());
  native.rrt_iterations = static_cast<int32_t *>(rrt_iterations_device.get());
  native.selected_seed = static_cast<int32_t *>(selected_seed_device.get());
  native.attempt_direct_trajectory_optimization = true;
  native.allow_rrt = request.max_attempts >= request.enable_graph_attempt;
  native.finetune_attempts = -1;
  native.finetune_dt_scale = -1.0F;
  native.outer_iterations = std::max(1, request.outer_iterations);

  const auto start_time = std::chrono::steady_clock::now();
  planner->plan(native);
  cuda_check("plan_cspace", cudaDeviceSynchronize());
  const auto end_time = std::chrono::steady_clock::now();

  uint8_t success = 0;
  float dt = 0.0F;
  float cost = 0.0F;
  int32_t selected_seed = 0;
  uint8_t used_rrt = 0;
  int32_t rrt_iterations = 0;
  success_device.download(&success, sizeof(success));
  dt_device.download(&dt, sizeof(dt));
  cost_device.download(&cost, sizeof(cost));
  selected_seed_device.download(&selected_seed, sizeof(selected_seed));
  rrt_device.download(&used_rrt, sizeof(used_rrt));
  rrt_iterations_device.download(&rrt_iterations, sizeof(rrt_iterations));

  std::vector<float> position;
  std::vector<float> velocity;
  std::vector<float> acceleration;
  position_device.download(position);
  velocity_device.download(velocity);
  acceleration_device.download(acceleration);

  PlanOutcome outcome = make_outcome(
    std::move(position), std::move(velocity), std::move(acceleration), dof_count,
    steps, dt, joint_names);
  outcome.success = success != 0;
  outcome.feasible = outcome.success;
  outcome.status = outcome.success ? "success" : "cspace planning failed";
  outcome.seed_index = selected_seed;
  outcome.solver_time_s =
    std::chrono::duration<double>(end_time - start_time).count();
  outcome.motion_time_s = (steps > 0) ? static_cast<double>(steps - 1) * dt : 0.0;
  outcome.candidates_generated = 1;
  outcome.candidates_solved = outcome.success ? 1 : 0;
  outcome.waypoints_planned = 1;
  return outcome;
}

curobo_runtime::PlanOutcome CppBridge::plan_cspace(
  curobo_runtime::RobotHandle, const curobo_runtime::PlanRequest & request)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->native_scene_valid) {
    impl_->ensure_python_planner("plan_cspace");
    return impl_->python->plan_cspace(impl_->python_handle, request);
  }
  return impl_->plan_cspace_impl(request);
}
}  // namespace curobo_cpp_runtime

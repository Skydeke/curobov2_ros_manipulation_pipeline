// SPDX-License-Identifier: Apache-2.0
#include "cpp_bridge_impl.hpp"

namespace curobo_cpp_runtime
{

curobo_runtime::PlanOutcome CppBridge::Impl::plan_pose_impl(
  const curobo_runtime::PlanRequest & request)
{
  ensure_planner("plan_pose");
  const int dof_count = dof();
  const int tools = num_tools();
  if (static_cast<int>(request.start.size()) != dof_count) {
    fail("plan_pose", "start has " + std::to_string(request.start.size()) +
      " values, expected " + std::to_string(dof_count));
  }
  std::vector<float> start = request.start;
  {
    const int count = std::min(
      static_cast<int>(start.size()),
      static_cast<int>(std::min(joint_lower.size(), joint_upper.size())));
    for (int i = 0; i < count; ++i) {
      if (static_cast<std::size_t>(i) < joint_continuous.size() &&
        joint_continuous[static_cast<std::size_t>(i)] != 0)
      {
        continue;
      }
      const float lower = static_cast<float>(joint_lower[static_cast<std::size_t>(i)]);
      const float upper = static_cast<float>(joint_upper[static_cast<std::size_t>(i)]);
      start[static_cast<std::size_t>(i)] =
        std::clamp(start[static_cast<std::size_t>(i)], lower, upper);
    }
  }
  if (!request.trajectory_constraints.empty() ||
    !request.tool_trajectory_constraints.empty())
  {
    fail("plan_pose",
      "legacy 6-axis holds are not supported by the native planner; use ordered "
      "waypoints or run with curobo_runtime:=python");
  }
  if (request.goalset_count > 1) {
    fail("plan_pose",
      "multi-segment goalsets (goalset_count > 1) must be planned one segment at a "
      "time by the caller; cpp_runtime plan_pose handles one goalset");
  }

  struct Candidate
  {
    std::vector<std::array<float, 7>> poses;
  };
  std::vector<Candidate> candidates;

  if (!request.tool_poses.data.empty()) {
    const int tool_count = request.tool_poses.shape.size() >= 2 ?
      static_cast<int>(request.tool_poses.shape[0]) : 0;
    if (tool_count <= 0) {
      fail("plan_pose", "tool_poses must have shape [T,7]");
    }
    Candidate candidate;
    candidate.poses.reserve(static_cast<std::size_t>(tools));
    for (int tool = 0; tool < tools; ++tool) {
      candidate.poses.push_back(tool_pose(start, tool, "plan_pose"));
    }
    if (!request.tool_frames.empty() &&
      static_cast<int>(request.tool_frames.size()) != tool_count)
    {
      fail("plan_pose", "tool_frames and tool_poses must be parallel");
    }
    for (int i = 0; i < tool_count; ++i) {
      const float * row = request.tool_poses.data.data() + static_cast<std::size_t>(i) * 7;
      if (!request.tool_frames.empty()) {
        const auto it = std::find(
          tool_frames.begin(), tool_frames.end(), request.tool_frames[static_cast<std::size_t>(i)]);
        if (it == tool_frames.end()) {
          fail("plan_pose", "unknown tool frame '" +
            request.tool_frames[static_cast<std::size_t>(i)] + "'");
        }
        candidate.poses[static_cast<std::size_t>(std::distance(tool_frames.begin(), it))] =
          pose_from_values(row);
      } else if (i < tools) {
        candidate.poses[static_cast<std::size_t>(i)] = pose_from_values(row);
      }
    }
    candidates.push_back(std::move(candidate));
  } else {
    if (request.goalset.data.empty()) {
      fail("plan_pose", "request has neither goalset nor tool_poses");
    }
    const int goal_count = request.goalset.shape.size() >= 2 ?
      static_cast<int>(request.goalset.shape[0]) : 0;
    if (goal_count <= 0) {
      fail("plan_pose", "goalset must have shape [G,7]");
    }
    std::vector<std::array<float, 7>> held;
    held.reserve(static_cast<std::size_t>(tools));
    for (int tool = 0; tool < tools; ++tool) {
      held.push_back(tool_pose(start, tool, "plan_pose"));
    }
    for (int g = 0; g < goal_count; ++g) {
      Candidate candidate;
      candidate.poses = held;
      candidate.poses[0] = pose_from_values(
        request.goalset.data.data() + static_cast<std::size_t>(g) * 7);
      candidates.push_back(std::move(candidate));
    }
  }

  const auto info = planner->info();
  const int steps = info.trajectory_steps;
  const int knots = info.num_knots;
  PlanOutcome best;
  bool have_best = false;
  double best_cost = std::numeric_limits<double>::infinity();
  std::string last_status = "pose planning failed";

  for (std::size_t candidate_index = 0; candidate_index < candidates.size();
    ++candidate_index)
  {
    const auto & candidate = candidates[candidate_index];
    std::vector<float> goal_positions;
    std::vector<float> goal_quaternions;
    fill_pose_arrays(candidate.poses, goal_positions, goal_quaternions);

    DeviceBuffer start_device;
    DeviceBuffer goal_pos_device;
    DeviceBuffer goal_quat_device;
    DeviceBuffer knots_device;
    DeviceBuffer position_device;
    DeviceBuffer velocity_device;
    DeviceBuffer acceleration_device;
    DeviceBuffer jerk_device;
    DeviceBuffer success_device;
    DeviceBuffer cost_device;
    DeviceBuffer collision_device;
    DeviceBuffer dt_device;
    DeviceBuffer position_error_device;
    DeviceBuffer orientation_error_device;
    DeviceBuffer used_rrt_device;
    DeviceBuffer rrt_iterations_device;
    DeviceBuffer selected_seed_device;
    DeviceBuffer planning_attempts_device;

    start_device.upload(start);
    goal_pos_device.upload(goal_positions);
    goal_quat_device.upload(goal_quaternions);
    knots_device.allocate(static_cast<std::size_t>(knots) * dof_count * sizeof(float));
    position_device.allocate(static_cast<std::size_t>(steps) * dof_count * sizeof(float));
    velocity_device.allocate(static_cast<std::size_t>(steps) * dof_count * sizeof(float));
    acceleration_device.allocate(static_cast<std::size_t>(steps) * dof_count * sizeof(float));
    jerk_device.allocate(static_cast<std::size_t>(steps) * dof_count * sizeof(float));
    success_device.allocate(sizeof(uint8_t));
    cost_device.allocate(sizeof(float));
    collision_device.allocate(sizeof(float));
    dt_device.allocate(sizeof(float));
    position_error_device.allocate(sizeof(float));
    orientation_error_device.allocate(sizeof(float));
    used_rrt_device.allocate(sizeof(uint8_t));
    rrt_iterations_device.allocate(sizeof(int32_t));
    selected_seed_device.allocate(sizeof(int32_t));
    planning_attempts_device.allocate(sizeof(int32_t));

    curobo::robot::PoseMotionPlanningRequest native;
    native.goal_positions = static_cast<const float *>(goal_pos_device.get());
    native.goal_quaternions = static_cast<const float *>(goal_quat_device.get());
    native.current_position = static_cast<const float *>(start_device.get());
    native.optimized_knots = static_cast<float *>(knots_device.get());
    native.trajectory_position = static_cast<float *>(position_device.get());
    native.trajectory_velocity = static_cast<float *>(velocity_device.get());
    native.trajectory_acceleration = static_cast<float *>(acceleration_device.get());
    native.trajectory_jerk = static_cast<float *>(jerk_device.get());
    native.success = static_cast<uint8_t *>(success_device.get());
    native.cost = static_cast<float *>(cost_device.get());
    native.collision_cost = static_cast<float *>(collision_device.get());
    native.trajectory_dt = static_cast<float *>(dt_device.get());
    native.position_error = static_cast<float *>(position_error_device.get());
    native.orientation_error = static_cast<float *>(orientation_error_device.get());
    native.used_rrt = static_cast<uint8_t *>(used_rrt_device.get());
    native.rrt_iterations = static_cast<int32_t *>(rrt_iterations_device.get());
    native.selected_seed = static_cast<int32_t *>(selected_seed_device.get());
    native.planning_attempts = static_cast<int32_t *>(planning_attempts_device.get());
    native.max_attempts = std::max(1, request.max_attempts);
    native.enable_graph_attempt = request.enable_graph_attempt;
    native.finetune_attempts = 1;
    native.finetune_dt_scale = 0.6F;
    native.constrain_tool_path = false;
    native.outer_iterations = std::max(1, request.outer_iterations);

    const auto start_time = std::chrono::steady_clock::now();
    planner->plan_pose(native);
    cuda_check("plan_pose", cudaDeviceSynchronize());
    const auto end_time = std::chrono::steady_clock::now();

    uint8_t success = 0;
    float dt = 0.0F;
    float cost = 0.0F;
    int32_t selected_seed = 0;
    float position_error = 0.0F;
    success_device.download(&success, sizeof(success));
    dt_device.download(&dt, sizeof(dt));
    cost_device.download(&cost, sizeof(cost));
    selected_seed_device.download(&selected_seed, sizeof(selected_seed));
    position_error_device.download(&position_error, sizeof(position_error));

    if (success == 0) {
      std::ostringstream status;
      status << "pose candidate " << candidate_index << " failed";
      last_status = status.str();
      continue;
    }

    std::vector<float> position;
    std::vector<float> velocity;
    std::vector<float> acceleration;
    position_device.download(position);
    velocity_device.download(velocity);
    acceleration_device.download(acceleration);
    PlanOutcome outcome = make_outcome(
      std::move(position), std::move(velocity), std::move(acceleration), dof_count,
      steps, dt, joint_names);
    outcome.success = true;
    outcome.feasible = true;
    outcome.status = "success";
    outcome.goalset_index = static_cast<int>(candidate_index);
    outcome.seed_index = selected_seed;
    outcome.solver_time_s =
      std::chrono::duration<double>(end_time - start_time).count();
    outcome.motion_time_s = (steps > 0) ? static_cast<double>(steps - 1) * dt : 0.0;

    // The native benchmark reports the TIME-OPTIMIZED executable trajectory,
    // not the raw TrajOpt control points. plan_pose() leaves that trajectory
    // in the planner; fetch it so ROS and native report the same motion time.
    const auto timed = planner->timeopt_trajectory();
    if (timed.sample_count != nullptr) {
      int32_t timed_steps = 0;
      float timed_dt = 0.0F;
      float timed_duration = 0.0F;
      cuda_check(
        "timeopt/sample_count",
        cudaMemcpy(
          &timed_steps, timed.sample_count, sizeof(timed_steps),
          cudaMemcpyDeviceToHost));
      if (timed.dt != nullptr) {
        cuda_check(
          "timeopt/dt",
          cudaMemcpy(&timed_dt, timed.dt, sizeof(timed_dt), cudaMemcpyDeviceToHost));
      }
      if (timed.duration != nullptr) {
        cuda_check(
          "timeopt/duration",
          cudaMemcpy(
            &timed_duration, timed.duration, sizeof(timed_duration),
            cudaMemcpyDeviceToHost));
      }
      if (timed_steps > 0 && timed.position != nullptr &&
        timed.velocity != nullptr && timed.acceleration != nullptr)
      {
        const std::size_t count =
          static_cast<std::size_t>(timed_steps) * static_cast<std::size_t>(dof_count);
        std::vector<float> timed_position(count);
        std::vector<float> timed_velocity(count);
        std::vector<float> timed_acceleration(count);
        cuda_check(
          "timeopt/position",
          cudaMemcpy(
            timed_position.data(), timed.position, count * sizeof(float),
            cudaMemcpyDeviceToHost));
        cuda_check(
          "timeopt/velocity",
          cudaMemcpy(
            timed_velocity.data(), timed.velocity, count * sizeof(float),
            cudaMemcpyDeviceToHost));
        cuda_check(
          "timeopt/acceleration",
          cudaMemcpy(
            timed_acceleration.data(), timed.acceleration, count * sizeof(float),
            cudaMemcpyDeviceToHost));
        PlanOutcome timed_outcome = make_outcome(
          std::move(timed_position), std::move(timed_velocity),
          std::move(timed_acceleration), dof_count, timed_steps, timed_dt,
          joint_names);
        outcome.positions = std::move(timed_outcome.positions);
        outcome.velocity = std::move(timed_outcome.velocity);
        outcome.acceleration = std::move(timed_outcome.acceleration);
        outcome.motion_time_s =
          timed_duration > 0.0F ? static_cast<double>(timed_duration) :
          static_cast<double>(std::max(0, timed_steps - 1)) * timed_dt;
      }
    }
    outcome.position_error_mm = static_cast<double>(position_error) * 1000.0;
    outcome.candidates_generated = static_cast<std::uint32_t>(candidates.size());
    outcome.candidates_solved = 1;
    outcome.waypoints_planned = 1;
    if (!have_best || cost < best_cost) {
      best = std::move(outcome);
      best_cost = cost;
      have_best = true;
    }
  }

  if (!have_best) {
    PlanOutcome outcome;
    outcome.success = false;
    outcome.feasible = false;
    outcome.status = last_status;
    outcome.joint_names = joint_names;
    outcome.solver_time_s = 0.0;
    return outcome;
  }
  return best;
}

curobo_runtime::PlanOutcome CppBridge::Impl::plan_waypoints_impl(
  const curobo_runtime::PlanRequest & request, const OrderedWaypoints & waypoints)
{
  ensure_planner("plan_pose");
  const int dof_count = dof();
  const int tools = num_tools();
  if (static_cast<int>(request.start.size()) != dof_count) {
    fail("plan_pose", "start has " + std::to_string(request.start.size()) +
      " values, expected " + std::to_string(dof_count));
  }
  std::vector<float> start = request.start;
  {
    const int count = std::min(
      static_cast<int>(start.size()),
      static_cast<int>(std::min(joint_lower.size(), joint_upper.size())));
    for (int i = 0; i < count; ++i) {
      if (static_cast<std::size_t>(i) < joint_continuous.size() &&
        joint_continuous[static_cast<std::size_t>(i)] != 0)
      {
        continue;
      }
      const float lower = static_cast<float>(joint_lower[static_cast<std::size_t>(i)]);
      const float upper = static_cast<float>(joint_upper[static_cast<std::size_t>(i)]);
      start[static_cast<std::size_t>(i)] =
        std::clamp(start[static_cast<std::size_t>(i)], lower, upper);
    }
  }
  if (waypoints.count <= 0) {
    fail("plan_pose", "waypoint_count must be positive");
  }
  if (waypoints.count > 256) {
    fail("plan_pose", "cpp_runtime supports at most 256 ordered waypoints");
  }

  std::vector<std::string> frames = waypoints.tool_frames;
  if (frames.empty()) {
    frames = request.tool_frames;
  }
  int active_slot = 0;
  std::vector<int> tool_slots;
  if (!waypoints.multi_tool && tools > 1) {
    // A single-tool ordered path on a multi-tool artifact is valid: the named
    // tool follows the path and every other tool holds its current FK pose.
    if (frames.size() > 1) {
      fail("plan_pose", "single-tool ordered waypoints must name at most one frame");
    }
    active_slot = 0;
    if (!frames.empty()) {
      const auto it = std::find(tool_frames.begin(), tool_frames.end(), frames[0]);
      if (it == tool_frames.end()) {
        fail("plan_pose", "unknown waypoint tool frame '" + frames[0] + "'");
      }
      active_slot = static_cast<int>(std::distance(tool_frames.begin(), it));
    }
    for (int i = 0; i < tools; ++i) {
      tool_slots.push_back(i);
    }
  } else {
    if (!frames.empty()) {
      if (static_cast<int>(frames.size()) != tools) {
        fail("plan_pose", "ordered waypoints require one frame per artifact tool (" +
          std::to_string(tools) + "), got " + std::to_string(frames.size()));
      }
      std::vector<bool> used(static_cast<std::size_t>(tools), false);
      for (const auto & frame : frames) {
        const auto it = std::find(tool_frames.begin(), tool_frames.end(), frame);
        if (it == tool_frames.end()) {
          fail("plan_pose", "unknown waypoint tool frame '" + frame + "'");
        }
        const int index = static_cast<int>(std::distance(tool_frames.begin(), it));
        if (used[static_cast<std::size_t>(index)]) {
          fail("plan_pose", "duplicate waypoint tool frame '" + frame + "'");
        }
        used[static_cast<std::size_t>(index)] = true;
        tool_slots.push_back(index);
      }
    } else {
      for (int i = 0; i < tools; ++i) {
        tool_slots.push_back(i);
      }
    }
  }

  const int values = static_cast<int>(waypoints.poses.data.size());
  const int expected_single = waypoints.count * 7;
  const int expected_multi = waypoints.count * tools * 7;
  if (!waypoints.multi_tool && values != expected_single) {
    fail("plan_pose", "waypoints must hold K*7 values (K=" +
      std::to_string(waypoints.count) + "), got " + std::to_string(values));
  }
  if (waypoints.multi_tool && values != expected_multi) {
    fail("plan_pose", "tool_waypoints must hold K*T*7 values (K=" +
      std::to_string(waypoints.count) + ", T=" + std::to_string(tools) +
      "), got " + std::to_string(values));
  }

  std::vector<std::array<float, 7>> held_tools;
  held_tools.reserve(static_cast<std::size_t>(tools));
  for (int tool = 0; tool < tools; ++tool) {
    held_tools.push_back(tool_pose(start, tool, "plan_pose"));
  }
  std::vector<float> positions(
    static_cast<std::size_t>(waypoints.count) * tools * 3);
  std::vector<float> quaternions(
    static_cast<std::size_t>(waypoints.count) * tools * 4);
  for (int k = 0; k < waypoints.count; ++k) {
    for (int slot = 0; slot < tools; ++slot) {
      std::array<float, 7> pose = held_tools[static_cast<std::size_t>(slot)];
      if (waypoints.multi_tool) {
        const float * row = waypoints.poses.data.data() +
          (static_cast<std::size_t>(k) * tools + static_cast<std::size_t>(slot)) * 7;
        pose = pose_from_values(row);
      } else if (slot == active_slot) {
        const float * row = waypoints.poses.data.data() +
          static_cast<std::size_t>(k) * 7;
        pose = pose_from_values(row);
      }
      const int destination_tool = tool_slots[static_cast<std::size_t>(slot)];
      const std::size_t destination =
        (static_cast<std::size_t>(k) * tools + static_cast<std::size_t>(destination_tool));
      positions[destination * 3 + 0] = pose[0];
      positions[destination * 3 + 1] = pose[1];
      positions[destination * 3 + 2] = pose[2];
      quaternions[destination * 4 + 0] = pose[3];
      quaternions[destination * 4 + 1] = pose[4];
      quaternions[destination * 4 + 2] = pose[5];
      quaternions[destination * 4 + 3] = pose[6];
    }
  }

  const auto info = planner->info();
  const int steps = info.trajectory_steps;
  const int knots = info.num_knots;
  DeviceBuffer start_device;
  DeviceBuffer waypoint_position_device;
  DeviceBuffer waypoint_quaternion_device;
  DeviceBuffer goal_pos_device;
  DeviceBuffer goal_quat_device;
  DeviceBuffer knots_device;
  DeviceBuffer position_device;
  DeviceBuffer velocity_device;
  DeviceBuffer acceleration_device;
  DeviceBuffer jerk_device;
  DeviceBuffer success_device;
  DeviceBuffer cost_device;
  DeviceBuffer collision_device;
  DeviceBuffer dt_device;
  DeviceBuffer position_error_device;
  DeviceBuffer orientation_error_device;
  DeviceBuffer used_rrt_device;
  DeviceBuffer rrt_iterations_device;
  DeviceBuffer selected_seed_device;
  DeviceBuffer planning_attempts_device;

  start_device.upload(start);
  waypoint_position_device.upload(positions);
  waypoint_quaternion_device.upload(quaternions);
  const std::size_t final_index =
    static_cast<std::size_t>(waypoints.count - 1) * tools;
  std::vector<float> final_positions(
    positions.begin() + static_cast<std::ptrdiff_t>(final_index * 3),
    positions.begin() + static_cast<std::ptrdiff_t>((final_index + tools) * 3));
  std::vector<float> final_quaternions(
    quaternions.begin() + static_cast<std::ptrdiff_t>(final_index * 4),
    quaternions.begin() + static_cast<std::ptrdiff_t>((final_index + tools) * 4));
  goal_pos_device.upload(final_positions);
  goal_quat_device.upload(final_quaternions);
  knots_device.allocate(static_cast<std::size_t>(knots) * dof_count * sizeof(float));
  position_device.allocate(static_cast<std::size_t>(steps) * dof_count * sizeof(float));
  velocity_device.allocate(static_cast<std::size_t>(steps) * dof_count * sizeof(float));
  acceleration_device.allocate(static_cast<std::size_t>(steps) * dof_count * sizeof(float));
  jerk_device.allocate(static_cast<std::size_t>(steps) * dof_count * sizeof(float));
  success_device.allocate(sizeof(uint8_t));
  cost_device.allocate(sizeof(float));
  collision_device.allocate(sizeof(float));
  dt_device.allocate(sizeof(float));
  position_error_device.allocate(sizeof(float));
  orientation_error_device.allocate(sizeof(float));
  used_rrt_device.allocate(sizeof(uint8_t));
  rrt_iterations_device.allocate(sizeof(int32_t));
  selected_seed_device.allocate(sizeof(int32_t));
  planning_attempts_device.allocate(sizeof(int32_t));

  curobo::robot::WaypointMotionPlanningRequest native;
  native.motion.goal_positions = static_cast<const float *>(goal_pos_device.get());
  native.motion.goal_quaternions = static_cast<const float *>(goal_quat_device.get());
  native.motion.current_position = static_cast<const float *>(start_device.get());
  native.motion.optimized_knots = static_cast<float *>(knots_device.get());
  native.motion.trajectory_position = static_cast<float *>(position_device.get());
  native.motion.trajectory_velocity = static_cast<float *>(velocity_device.get());
  native.motion.trajectory_acceleration = static_cast<float *>(acceleration_device.get());
  native.motion.trajectory_jerk = static_cast<float *>(jerk_device.get());
  native.motion.success = static_cast<uint8_t *>(success_device.get());
  native.motion.cost = static_cast<float *>(cost_device.get());
  native.motion.collision_cost = static_cast<float *>(collision_device.get());
  native.motion.trajectory_dt = static_cast<float *>(dt_device.get());
  native.motion.position_error = static_cast<float *>(position_error_device.get());
  native.motion.orientation_error = static_cast<float *>(orientation_error_device.get());
  native.motion.used_rrt = static_cast<uint8_t *>(used_rrt_device.get());
  native.motion.rrt_iterations = static_cast<int32_t *>(rrt_iterations_device.get());
  native.motion.selected_seed = static_cast<int32_t *>(selected_seed_device.get());
  native.motion.planning_attempts = static_cast<int32_t *>(planning_attempts_device.get());
  native.motion.max_attempts = std::max(1, request.max_attempts);
  native.motion.enable_graph_attempt = request.enable_graph_attempt;
  native.motion.finetune_attempts = 1;
  native.motion.finetune_dt_scale = 0.6F;
  native.positions = static_cast<const float *>(waypoint_position_device.get());
  native.quaternions = static_cast<const float *>(waypoint_quaternion_device.get());
  native.num_waypoints = waypoints.count;
  native.path_position_tolerance = static_cast<float>(waypoints.path_position_tolerance);
  native.path_orientation_tolerance =
    static_cast<float>(waypoints.path_orientation_tolerance);
  native.final_position_tolerance = static_cast<float>(waypoints.final_position_tolerance);
  native.final_orientation_tolerance =
    static_cast<float>(waypoints.final_orientation_tolerance);
  native.track_orientation = waypoints.track_orientation;
  native.outer_iterations = std::max(1, waypoints.outer_iterations);

  const auto start_time = std::chrono::steady_clock::now();
  planner->plan_pose_waypoints(native);
  cuda_check("plan_pose", cudaDeviceSynchronize());
  const auto end_time = std::chrono::steady_clock::now();

  uint8_t success = 0;
  float dt = 0.0F;
  float position_error = 0.0F;
  int32_t selected_seed = 0;
  success_device.download(&success, sizeof(success));
  dt_device.download(&dt, sizeof(dt));
  position_error_device.download(&position_error, sizeof(position_error));
  selected_seed_device.download(&selected_seed, sizeof(selected_seed));

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
  outcome.status = outcome.success ? "success" : "waypoint planning failed";
  outcome.seed_index = selected_seed;
  outcome.solver_time_s =
    std::chrono::duration<double>(end_time - start_time).count();
  outcome.motion_time_s = (steps > 0) ? static_cast<double>(steps - 1) * dt : 0.0;
  outcome.position_error_mm = static_cast<double>(position_error) * 1000.0;
  outcome.candidates_generated = 1;
  outcome.candidates_solved = outcome.success ? 1 : 0;
  outcome.waypoints_planned = static_cast<std::uint32_t>(waypoints.count);
  outcome.waypoint_status.assign(
    static_cast<std::size_t>(waypoints.count), outcome.success ? 1 : 0);
  return outcome;
}

curobo_runtime::PlanOutcome CppBridge::plan_pose(
  curobo_runtime::RobotHandle, const curobo_runtime::PlanRequest & request)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("plan_pose");
  if (!impl_->native_scene_valid) {
    impl_->ensure_python_planner("plan_pose");
    return impl_->python->plan_pose(impl_->python_handle, request);
  }
  // Per-tool pose goals are not ordered waypoint paths. Route them straight to
  // the pose planner; the waypoint extractor probes optional fields on the
  // request and must not shadow a populated `tool_poses` goal.
  if (!request.tool_poses.data.empty()) {
    return impl_->plan_pose_impl(request);
  }
  // Ordered waypoint / Cartesian paths have a native implementation.
  const OrderedWaypoints waypoints = extract_ordered_waypoints(request);
  if (waypoints.present) {
    // Native is the only ordered-waypoint path. Do not fall back to the
    // Python planner here: its Torch/Dynamo recompile can block the service
    // for minutes during a task.
    return impl_->plan_waypoints_impl(request, waypoints);
  }
  // The native bridge implements this plan mode. Use it directly; the Python
  // planner is only the fallback for a missing native artifact (handled at
  // the top of this function), never a runtime recompile path.
  return impl_->plan_pose_impl(request);
}
}  // namespace curobo_cpp_runtime

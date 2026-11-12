// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/planners/mpc_planner.py.
#include "curobo_runtime/runtime.hpp"
#include "curobo_core/planners/mpc_planner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <iomanip>
#include <sstream>

#include <builtin_interfaces/msg/duration.hpp>

namespace curobo_core
{

namespace
{
const char * kUse = "use_cuda_graph";
}

MPCController::MPCController(
  curobo_runtime::IBridge * bridge, curobo_runtime::RobotHandle handle,
  std::vector<std::string> joint_names, rclcpp::Node * node)
: ReactiveController(bridge, handle, std::move(joint_names), node)
{
  base_link_ = (node_ != nullptr && node_->has_parameter("base_link")) ?
    node_->get_parameter("base_link").as_string() : "base_link";
}

// PORT: ReactiveController.ensure_solver / MPCController.build_solver
void MPCController::ensure_solver()
{
  if (built_) {
    return;
  }
  rclcpp::Node * node = node_;
  const auto param_double = [&](const char * name, double fallback) {
      return node->has_parameter(name) ? node->get_parameter(name).as_double() : fallback;
    };
  const auto param_int = [&](const char * name, int fallback) {
      return node->has_parameter(name) ? node->get_parameter(name).as_int() : fallback;
    };
  const auto param_str = [&](const char * name, const std::string & fallback) {
      return node->has_parameter(name) ? node->get_parameter(name).as_string() : fallback;
    };
  const auto param_bool = [&](const char * name, bool fallback) {
      return node->has_parameter(name) ? node->get_parameter(name).as_bool() : fallback;
    };

  step_dt_ = param_double("mpc_step_dt", 0.03);
  const int horizon = param_int("mpc_horizon_steps", 30);
  const int warm_iters = param_int("mpc_warm_start_iters", 5);
  const int cold_iters = param_int("mpc_cold_start_iters", 10);
  const std::string solver_type = param_str("mpc_solver_type", "mppi_acceleration");
  use_mppi_acceleration_ = (solver_type == "mppi_acceleration");
  const double command_interval = param_double("mpc_command_interval", 0.24);

  std::ostringstream cache;
  cache << "{\"cuboid\":" << param_int("collision_cache_cuboid", 32)
        << ",\"mesh\":" << param_int("collision_cache_mesh", 4)
        << ",\"voxel\":{\"layers\":1,\"dims\":[";
  const auto extent = node->has_parameter("mapper_extent_xyz") ?
    node->get_parameter("mapper_extent_xyz").as_double_array() :
    std::vector<double>{2.0, 2.0, 2.0};
  const double voxel_size = param_double("voxel_size", 0.05);
  const double margin = voxel_size > 0.0 ? 2.0 * voxel_size : 0.0;
  for (std::size_t i = 0; i < extent.size(); ++i) {
    cache << (i ? "," : "") << (extent[i] + margin);
  }
  cache << "],\"voxel_size\":" << voxel_size << "}}";

  std::ostringstream cfg;
  cfg << "{\"step_dt\":" << step_dt_
      << ",\"horizon\":" << horizon
      << ",\"warm_start_iters\":" << warm_iters
      << ",\"cold_start_iters\":" << cold_iters
      << ",\"solver_type\":\"" << solver_type << "\""
      << ",\"command_interval\":" << command_interval
      << ",\"mppi_num_particles\":" << param_int("mpc_mppi_num_particles", 400)
      << ",\"use_cuda_graph\":" << (param_bool(kUse, true) ? "true" : "false")
      << ",\"collision_cache\":" << cache.str() << "}";
  // PORT: building the MPC solver allocates/captures GPU buffers -> exclusive.
  {
    auto guard = curobo_runtime::Runtime::instance().barrier().exclusive();
    bridge_->mpc_build(handle_, cfg.str());
  }
  built_ = true;
  // The artifact owns the tool-frame order. Keep it so a single-pose goal is
  // reported against the frame the bridge's FK actually uses, and so a
  // bimanual model is never mistaken for a one-arm one.
  try {
    solver_tool_frames_ = bridge_->tool_frames(handle_);
  } catch (const std::exception & e) {
    RCLCPP_WARN(
      node_->get_logger(), "MPC: could not read the solver tool frames: %s", e.what());
  }

  // PORT: command_interval is read by the servo loop; expose it on the base.
  command_interval_ = command_interval;

  const std::string prefix = node_->get_name() + std::string("/");
  path_pub_ = node_->create_publisher<trajectory_msgs::msg::JointTrajectory>(
    prefix + "mpc_predicted_path", 10);
  goal_marker_pub_ = node_->create_publisher<visualization_msgs::msg::Marker>(
    prefix + "mpc_goal_marker", 10);
  cost_pub_ = node_->create_publisher<curobov2_ros_interfaces::msg::MpcCosts>(
    prefix + "mpc_costs", 10);
  // PORT: build_solver - the mpc_diag CSV is opened when mpc_debug is set.
  if (debug_enabled()) {
    csv_ = open_diag_csv(node_, "mpc_diag");
  }

  RCLCPP_INFO(
    node_->get_logger(),
    "MPC solver built: solver_type=%s, optimization_dt=%gs, horizon=%d, "
    "warm_start_iters=%d, cold_start_iters=%d",
    solver_type.c_str(), step_dt_, horizon, warm_iters, cold_iters);
}

// PORT: MPCController.update_world
void MPCController::update_world(const std::string & scene_json)
{
  try {
    auto guard = curobo_runtime::Runtime::instance().barrier().exclusive();
    bridge_->mpc_load_scene(handle_, scene_json);
  } catch (const std::exception & e) {
    RCLCPP_WARN(
      node_->get_logger(), "MPC solver built empty: current world does not fit (%s)", e.what());
  }
}

// PORT: MPCController.setup
bool MPCController::setup_solver(
  const std::vector<float> & start_q, const ReactiveGoal & goal)
{
  // A reactive goal is exactly ONE goalset: either one legacy Cartesian pose
  // or a simultaneous multi-tool goal (both OpenArm grasp frames). Candidate
  // sets with several poses belong to the open-loop planners.
  std::vector<std::string> frames;
  std::vector<std::array<double, 7>> poses;
  if (!goal.tool_poses.empty()) {
    if (goal.goalsets.size() > 1) {
      RCLCPP_ERROR(
        node_->get_logger(),
        "MPC multi-tool goal must be a single goalset (got %zu)", goal.goalsets.size());
      return false;
    }
    frames = goal.tool_frames;
    poses = goal.tool_poses;
  } else if (goal.goalsets.size() == 1 && goal.goalsets[0].size() == 1) {
    poses.push_back(goal.goalsets[0][0]);
  } else {
    std::ostringstream sizes;
    for (const auto & g : goal.goalsets) {
      sizes << g.size() << ",";
    }
    RCLCPP_ERROR(
      node_->get_logger(),
      "MPC requires exactly one goalset entry with a single pose or one "
      "multi-tool goalset (got %zu sets, poses_per_set=[%s]) - candidate sets "
      "belong to the open-loop planners (classic/joint space)",
      goal.goalsets.size(), sizes.str().c_str());
    return false;
  }
  if (!frames.empty() && frames.size() != poses.size()) {
    RCLCPP_ERROR(
      node_->get_logger(),
      "MPC: tool_frames and tool_poses must be parallel (got %zu frame(s), "
      "%zu pose(s))", frames.size(), poses.size());
    return false;
  }
  if (frames.empty() && poses.size() > 1) {
    RCLCPP_ERROR(
      node_->get_logger(),
      "MPC: a %zu-pose goal needs one tool frame per pose", poses.size());
    return false;
  }

  const auto target = target_pose_for_status(frames, poses);
  set_target(target);
  has_last_live_goal_ = false;

  const auto setup_outcome = bridge_->mpc_setup(handle_, start_q);
  const bool applied = apply_goal_poses(frames, poses, start_q, start_q);
  publish_goal_marker(poses, applied);

  if (debug_enabled() && setup_outcome.has_ee) {
    RCLCPP_DEBUG(
      node_->get_logger(),
      "MPC setup: goals=%zu current_EE_pos=[%.4f, %.4f, %.4f] "
      "first_goal=[%.4f, %.4f, %.4f] goal_applied=%d",
      poses.size(), setup_outcome.ee_position[0], setup_outcome.ee_position[1],
      setup_outcome.ee_position[2], target[0], target[1], target[2],
      static_cast<int>(applied));
  }
  return true;
}

// PORT: MPCController._apply_goal (single pose or simultaneous multi-tool)
bool MPCController::apply_goal_poses(
  const std::vector<std::string> & tool_frames,
  const std::vector<std::array<double, 7>> & tool_poses,
  const std::vector<float> & hold_q, const std::vector<float> & seed_q)
{
  if (tool_poses.empty()) {
    return false;
  }
  curobo_runtime::TensorF32 poses;
  poses.shape = {static_cast<std::int64_t>(tool_poses.size()), 7};
  poses.data.reserve(tool_poses.size() * 7);
  for (const auto & pose : tool_poses) {
    for (double value : pose) {
      poses.data.push_back(static_cast<float>(value));
    }
  }
  // PORT: _apply_goal with run_ik=True. Use the IK solution as the MPC's seed
  // (a bent configuration away from the straight-arm singularity) but keep the
  // Cartesian objective: switching to joint tracking made the arm swing to a
  // remote elbow branch instead of following the dragged target.
  const auto solved =
    bridge_->mpc_solve_goal_poses(handle_, tool_frames, poses, hold_q, seed_q);
  if (solved.success) {
    std::vector<float> q(
      solved.positions.data.begin(), solved.positions.data.end());
    // The IK solver happily returns a different elbow branch. Using such a
    // solution as the MPC's SEED makes the optimizer interpolate the whole arm
    // across the body in one window - which usually ends in a self-collision
    // and an infeasible step, so a small gizmo drag stalls the arm. Only accept
    // a seed that stays near the current configuration; otherwise seed from the
    // measured state and let the Cartesian objective pull the arm locally.
    float jump = 0.0F;
    const std::size_t n = std::min(q.size(), hold_q.size());
    for (std::size_t i = 0; i < n; ++i) {
      jump = std::max(jump, std::fabs(q[i] - hold_q[i]));
    }
    constexpr float kMaxSeedJumpRad = 0.7F;
    // Only single-tool goals get this guard. A multi-tool goal's IK solution is
    // the coherent PAIR (both arms at once); rejecting it leaves the optimizer
    // with no paired seed at all and the bimanual solve fails - which is what
    // made "one tool up, the other down" stutter.
    const bool single_tool_goal = tool_frames.size() <= 1;
    if (jump <= kMaxSeedJumpRad || !single_tool_goal) {
      if (bridge_->mpc_update_goal_poses(handle_, tool_frames, poses, false, q)) {
        remember_active_goal(tool_frames, tool_poses);
        return true;
      }
    } else {
      RCLCPP_DEBUG(
        node_->get_logger(),
        "MPC: IK solution is %.2f rad from the measured state - seeding from the "
        "state instead of the remote branch", static_cast<double>(jump));
    }
  }
  if (bridge_->mpc_update_goal_poses(handle_, tool_frames, poses, false, hold_q)) {
    remember_active_goal(tool_frames, tool_poses);
    return true;
  }
  bridge_->mpc_update_goal_poses(handle_, tool_frames, poses, false, hold_q);
  remember_active_goal(tool_frames, tool_poses);
  RCLCPP_WARN_THROTTLE(
    node_->get_logger(), *node_->get_clock(), 5000,
    "MPC: IK failed for goal pose - pose-only tracking (arm may not move)");
  return false;
}

// PORT: MPCController._reseed_toward_goal
void MPCController::reseed_toward_goal(
  const std::vector<std::string> & tool_frames,
  const std::vector<std::array<double, 7>> & tool_poses,
  const curobo_runtime::ReactiveState & state)
{
  if (tool_poses.empty()) {
    return;
  }
  try {
    curobo_runtime::TensorF32 poses;
    poses.shape = {static_cast<std::int64_t>(tool_poses.size()), 7};
    poses.data.reserve(tool_poses.size() * 7);
    for (const auto & pose : tool_poses) {
      for (double value : pose) {
        poses.data.push_back(static_cast<float>(value));
      }
    }
    const auto solved = bridge_->mpc_solve_goal_poses(
      handle_, tool_frames, poses, state.position, state.position);
    if (solved.success) {
      std::vector<float> q(solved.positions.data.begin(), solved.positions.data.end());
      bridge_->mpc_reseed(handle_, q);
    } else {
      RCLCPP_WARN_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 1000,
        "MPC: live retarget re-seed skipped (no IK solution for new pose)");
    }
  } catch (const std::exception & e) {
    RCLCPP_ERROR_THROTTLE(
      node_->get_logger(), *node_->get_clock(), 1000,
      "MPC: live retarget re-seed failed: %s", e.what());
  }
}

std::array<double, 7> MPCController::target_pose_for_status(
  const std::vector<std::string> & tool_frames,
  const std::vector<std::array<double, 7>> & tool_poses) const
{
  if (tool_poses.empty()) {
    return {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
  }
  // The bridge's FK reports the solver's FIRST tool frame. If the goal names
  // that frame, report its error; otherwise the first goal pose is the best
  // available signal (the goal still drives all named tools).
  if (!solver_tool_frames_.empty() && !tool_frames.empty()) {
    const auto it = std::find(
      tool_frames.begin(), tool_frames.end(), solver_tool_frames_.front());
    if (it != tool_frames.end()) {
      return tool_poses[static_cast<std::size_t>(it - tool_frames.begin())];
    }
  }
  return tool_poses.front();
}

void MPCController::remember_active_goal(
  const std::vector<std::string> & tool_frames,
  const std::vector<std::array<double, 7>> & tool_poses)
{
  active_goal_frames_ = tool_frames;
  active_goal_poses_ = tool_poses;
}

// PORT: mpc_planner.MPCController._csv_write (mpc_diag, mpc_debug only).
void MPCController::write_csv(
  const curobo_runtime::MpcStepOutcome & outcome, double solve_ms)
{
  if (csv_ == nullptr) {
    return;
  }
  const auto breakdown = bridge_->mpc_cost_breakdown(handle_);
  const double now = std::chrono::duration<double>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
  if (csv_t0_ == 0.0) {
    csv_t0_ = now;
    csv_t_prev_ = now;
  }
  const double dt_step_ms = (now - csv_t_prev_) * 1000.0;
  csv_t_prev_ = now;
  const double dt = step_dt_ > 0.0 ? step_dt_ : 0.03;
  constexpr double kRadToDeg = 180.0 / M_PI;

  // PORT: the native loop has one command per step (next_action), so the CSV is
  // per-command velocity continuity (vfirst/vlast), not per-window.
  const curobo_runtime::TensorF32 & src =
    (outcome.action.velocity.data.empty() || outcome.action.velocity.shape.empty()) ?
    outcome.action.position : outcome.action.velocity;
  const std::size_t dof = src.shape.empty() ? 0 : static_cast<std::size_t>(src.shape.back());
  const std::size_t rows = (dof > 0) ? src.data.size() / dof : 0;
  std::vector<double> vfirst, vlast;
  for (std::size_t i = 0; i < dof && rows > 0; ++i) {
    vfirst.push_back(src.data[i]);
    vlast.push_back(src.data[(rows - 1) * dof + i]);
  }
  const auto amax = [](const std::vector<double> & v) {
      double m = 0.0;
      for (double x : v) {
        m = std::max(m, std::abs(x));
      }
      return m;
    };
  // Accel per command: no intra-window term; only the batch-boundary term.
  double accel_bd = 0.0;
  if (!csv_last_vlast_.empty() && csv_last_vlast_.size() == vfirst.size()) {
    double m = 0.0;
    for (std::size_t i = 0; i < vfirst.size(); ++i) {
      m = std::max(m, std::abs(vfirst[i] - csv_last_vlast_[i]));
    }
    accel_bd = m * kRadToDeg / dt;
  }
  csv_last_vlast_ = vlast;

  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double pose_pos = outcome.has_pose_err ? outcome.pose_pos_err_m : nan;
  const double pose_rot = outcome.has_pose_err ? outcome.pose_rot_err_rad : nan;
  const auto f = [](double v, int prec) {
      std::ostringstream o;
      o << std::fixed << std::setprecision(prec) << v;
      return o.str();
    };

  std::vector<std::string> header = {
    "t_s", "dt_step_ms", "solve_ms", "fk_err_m", "vfirst_max_dps",
    "vlast_max_dps", "accel_win_max_dps2", "accel_boundary_dps2",
    "pose_pos_err_m", "pose_rot_err_rad", "cost_tool_pose_pos",
    "cost_tool_pose_orient", "cost_cspace", "con_self_collision",
    "con_scene_collision", "con_cspace_bound"};
  for (std::size_t i = 0; i < dof; ++i) {
    header.push_back("vfirst_j" + std::to_string(i + 1) + "_dps");
  }
  for (std::size_t i = 0; i < dof; ++i) {
    header.push_back("vlast_j" + std::to_string(i + 1) + "_dps");
  }
  csv_->write_header_once(header);

  std::vector<std::string> row = {
    f(now - csv_t0_, 3), f(dt_step_ms, 1), f(solve_ms, 1),
    f(get_position_error(), 5), f(amax(vfirst) * kRadToDeg, 2),
    f(amax(vlast) * kRadToDeg, 2), f(0.0, 1), f(accel_bd, 1),
    f(pose_pos, 5), f(pose_rot, 5),
    f(breakdown.cost_tool_pose_pos, 4), f(breakdown.cost_tool_pose_orient, 4),
    f(breakdown.cost_cspace, 4), f(breakdown.con_self_collision, 4),
    f(breakdown.con_scene_collision, 4), f(breakdown.con_cspace_bound, 4)};
  for (double v : vfirst) {
    row.push_back(f(v * kRadToDeg, 2));
  }
  for (double v : vlast) {
    row.push_back(f(v * kRadToDeg, 2));
  }
  csv_->writerow(row);
}

// PORT: mpc_planner.MPCController._publish_costs
void MPCController::publish_costs(const curobo_runtime::MpcStepOutcome & outcome)
{
  if (cost_pub_ == nullptr) {
    return;
  }
  const auto breakdown = bridge_->mpc_cost_breakdown(handle_);
  curobov2_ros_interfaces::msg::MpcCosts msg;
  msg.header.stamp = node_->get_clock()->now();
  msg.header.frame_id = base_link_;
  msg.fk_err_m = get_position_error();
  msg.pose_pos_err_m = outcome.has_pose_err ? outcome.pose_pos_err_m : 0.0;
  msg.pose_rot_err_rad = outcome.has_pose_err ? outcome.pose_rot_err_rad : 0.0;
  msg.cost_tool_pose_pos = breakdown.cost_tool_pose_pos;
  msg.cost_tool_pose_orient = breakdown.cost_tool_pose_orient;
  msg.cost_cspace = breakdown.cost_cspace;
  msg.con_self_collision = breakdown.con_self_collision;
  msg.con_scene_collision = breakdown.con_scene_collision;
  msg.con_cspace_bound = breakdown.con_cspace_bound;
  cost_pub_->publish(msg);
}

// PORT: MPCController.step
curobo_runtime::MpcStepOutcome MPCController::step_impl(
  const curobo_runtime::ReactiveState & state, bool paced)
{
  const auto t_solve = std::chrono::steady_clock::now();
  auto outcome = bridge_->mpc_step(handle_, state, paced);
  if (debug_enabled()) {
    // PORT: step_paced publishes the predicted path; step() publishes the
    // MpcCosts breakdown + writes the mpc_diag CSV (both mpc_debug only).
    if (paced) {
      publish_predicted_path(outcome);
      // Also emit the cost/constraint row for paced steps: an infeasible paced
      // goal is otherwise invisible except for the hold.
      const double solve_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_solve).count();
      publish_costs(outcome);
      write_csv(outcome, solve_ms);
    } else if (outcome.action.has_action) {
      const double solve_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_solve).count();
      publish_costs(outcome);
      write_csv(outcome, solve_ms);
    }
  }
  return outcome;
}

// PORT: MPCController.step_paced
curobo_runtime::MpcStepOutcome MPCController::step_paced_impl(
  const curobo_runtime::ReactiveState & state)
{
  return step_impl(state, true);
}

// PORT: MPCController.apply_live_goal
bool MPCController::apply_live_goal(
  const std::array<double, 7> & raw, const std::string & tool,
  const curobo_runtime::ReactiveState & state)
{
  // The RViz streams the active target continuously. Re-applying an unchanged
  // pose re-runs IK and re-seeds the MPC every 100 ms, which makes the arm
  // jitter; only react to a real target change.
  if (has_last_live_goal_ && tool == last_live_tool_) {
    double delta = 0.0;
    for (std::size_t i = 0; i < raw.size(); ++i) {
      delta = std::max(delta, std::abs(raw[i] - last_live_goal_[i]));
    }
    if (delta < 1e-4) {
      return true;
    }
  }
  last_live_goal_ = raw;
  last_live_tool_ = tool;
  has_last_live_goal_ = true;

  // On a multi-tool robot retarget the tool named by the live message; the
  // other tools keep their commanded targets. Without the name every gizmo
  // drag was applied to the first tool, which crosses the arms and makes the
  // solver reject the target as infeasible.
  std::vector<std::string> frames = active_goal_frames_;
  std::vector<std::array<double, 7>> poses = active_goal_poses_;
  std::size_t index = 0;
  if (!tool.empty() && !frames.empty()) {
    const auto it = std::find(frames.begin(), frames.end(), tool);
    if (it != frames.end()) {
      index = static_cast<std::size_t>(it - frames.begin());
    }
  }
  if (poses.empty()) {
    poses.push_back(raw);
    frames.clear();
  } else {
    index = std::min(index, poses.size() - 1);
    poses[index] = raw;
    if (!frames.empty() && frames.size() != poses.size()) {
      frames.clear();
    }
  }
  const auto target = target_pose_for_status(frames, poses);
  set_target(target);
  // apply_goal_poses already resets the solver with an IK-derived seed. The
  // old extra reseed_toward_goal() ran a second multi-seed IK on every live
  // message and then did nothing (mpc_has_goal_state is false in pose mode),
  // which saturated the bridge/GPU while dragging.
  const bool applied = apply_goal_poses(frames, poses, state.position, state.position);
  publish_goal_marker(poses, applied);
  return applied;
}

// PORT: MPCController._publish_goal_marker
void MPCController::publish_goal_marker(
  const std::vector<std::array<double, 7>> & tool_poses, bool applied)
{
  if (goal_marker_pub_ == nullptr) {
    return;
  }
  const auto now = node_->get_clock()->now();
  for (std::size_t i = 0; i < tool_poses.size(); ++i) {
    const auto & raw = tool_poses[i];
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = base_link_;
    marker.header.stamp = now;
    marker.ns = "mpc_goal";
    marker.id = static_cast<std::int32_t>(i);
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = raw[0];
    marker.pose.position.y = raw[1];
    marker.pose.position.z = raw[2];
    marker.pose.orientation.w = raw[3];
    marker.pose.orientation.x = raw[4];
    marker.pose.orientation.y = raw[5];
    marker.pose.orientation.z = raw[6];
    marker.scale.x = marker.scale.y = marker.scale.z = 0.03;
    marker.color.a = 1.0;
    marker.color.r = 1.0;
    marker.color.g = applied ? 0.0 : 0.5;
    marker.color.b = 0.0;
    goal_marker_pub_->publish(marker);
  }
}

// PORT: MPCController._publish_predicted_path (throttled to 10 Hz)
void MPCController::publish_predicted_path(const curobo_runtime::MpcStepOutcome & outcome)
{
  if (path_pub_ == nullptr || outcome.horizon_position.shape.size() != 2) {
    return;
  }
  const auto now = node_->get_clock()->now();
  if (has_published_path_ && (now - last_path_pub_at_).seconds() < 0.1) {
    return;
  }
  last_path_pub_at_ = now;
  has_published_path_ = true;

  const std::int64_t rows = outcome.horizon_position.shape[0];
  const std::int64_t cols = outcome.horizon_position.shape[1];
  trajectory_msgs::msg::JointTrajectory msg;
  msg.header.frame_id = base_link_;
  msg.header.stamp = now;
  msg.joint_names = joint_names_;
  for (std::int64_t i = 0; i < rows; ++i) {
    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions.reserve(cols);
    for (std::int64_t c = 0; c < cols; ++c) {
      const std::size_t idx = static_cast<std::size_t>(i * cols + c);
      point.positions.push_back(
        idx < outcome.horizon_position.data.size() ? outcome.horizon_position.data[idx] : 0.0);
    }
    const std::int64_t ns = static_cast<std::int64_t>(std::llround(step_dt_ * i * 1e9));
    point.time_from_start.sec = static_cast<std::int32_t>(ns / 1000000000);
    point.time_from_start.nanosec = static_cast<std::uint32_t>(ns % 1000000000);
    msg.points.push_back(std::move(point));
  }
  path_pub_->publish(msg);
}

// PORT: MPCController._debug_enabled
bool MPCController::debug_enabled()
{
  if (!node_->has_parameter("mpc_debug")) {
    node_->declare_parameter("mpc_debug", false);
  }
  return node_->get_parameter("mpc_debug").as_bool();
}

}  // namespace curobo_core

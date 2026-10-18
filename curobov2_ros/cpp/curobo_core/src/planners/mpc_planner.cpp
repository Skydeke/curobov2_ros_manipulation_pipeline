// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/planners/mpc_planner.py.
#include "curobo_runtime/runtime.hpp"
#include "curobo_core/planners/mpc_planner.hpp"

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
  for (std::size_t i = 0; i < extent.size(); ++i) {
    cache << (i ? "," : "") << extent[i];
  }
  cache << "],\"voxel_size\":" << param_double("voxel_size", 0.05) << "}}";

  std::ostringstream cfg;
  cfg << "{\"step_dt\":" << step_dt_
      << ",\"horizon\":" << horizon
      << ",\"warm_start_iters\":" << warm_iters
      << ",\"cold_start_iters\":" << cold_iters
      << ",\"solver_type\":\"" << solver_type << "\""
      << ",\"mppi_num_particles\":" << param_int("mpc_mppi_num_particles", 400)
      << ",\"use_cuda_graph\":" << (param_bool(kUse, true) ? "true" : "false")
      << ",\"collision_cache\":" << cache.str() << "}";
  // PORT: building the MPC solver allocates/captures GPU buffers -> exclusive.
  {
    auto guard = curobo_runtime::Runtime::instance().barrier().exclusive();
    bridge_->mpc_build(handle_, cfg.str());
  }
  built_ = true;

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
  if (!(goal.goalsets.size() == 1 && goal.goalsets[0].size() == 1)) {
    std::ostringstream sizes;
    for (const auto & g : goal.goalsets) {
      sizes << g.size() << ",";
    }
    RCLCPP_ERROR(
      node_->get_logger(),
      "MPC requires exactly one goalset entry with a single pose (got %zu sets, "
      "poses_per_set=[%s]) - candidate sets belong to the open-loop planners "
      "(classic/joint space)",
      goal.goalsets.size(), sizes.str().c_str());
    return false;
  }
  const auto & raw = goal.goalsets[0][0];
  set_target(raw);

  const auto setup_outcome = bridge_->mpc_setup(handle_, start_q);
  const bool applied = apply_goal(raw, start_q);
  publish_goal_marker(raw, applied);

  if (debug_enabled() && setup_outcome.has_ee) {
    RCLCPP_DEBUG(
      node_->get_logger(),
      "[MPC DIAG] setup: goal_xyz=[%.4f, %.4f, %.4f] start_EE_pos=[%.4f, %.4f, %.4f] "
      "fk_err=%.4fm goal_applied=%d",
      raw[0], raw[1], raw[2], setup_outcome.ee_position[0], setup_outcome.ee_position[1],
      setup_outcome.ee_position[2], get_position_error(), static_cast<int>(applied));
  }
  return true;
}

// PORT: MPCController._apply_goal
bool MPCController::apply_goal(
  const std::array<double, 7> & raw, const std::vector<float> & seed_q)
{
  if (bridge_->mpc_update_goal_pose(handle_, raw, false)) {
    return true;
  }
  const auto solved = bridge_->mpc_solve_goal_state(handle_, raw, seed_q);
  if (solved.success) {
    std::vector<float> q(
      solved.positions.data.begin(), solved.positions.data.end());
    bridge_->mpc_update_goal_pose(handle_, raw, false);
    bridge_->mpc_update_goal_state(handle_, q);
    RCLCPP_INFO(
      node_->get_logger(), "MPC: goal_state set via multi-seed IK (joint tracking on)");
    return true;
  }
  bridge_->mpc_update_goal_pose(handle_, raw, false);
  RCLCPP_WARN(
    node_->get_logger(), "MPC: IK failed for goal pose - pose-only tracking (arm may not move)");
  return false;
}

// PORT: MPCController._reseed_toward_goal
void MPCController::reseed_toward_goal(
  const std::array<double, 7> & raw, const curobo_runtime::ReactiveState & state)
{
  try {
    const auto solved = bridge_->mpc_solve_goal_state(handle_, raw, state.position);
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
  const std::array<double, 7> & raw, const curobo_runtime::ReactiveState & state)
{
  set_target(raw);
  const bool applied = apply_goal(raw, state.position);
  if (applied) {
    reseed_toward_goal(raw, state);
  }
  publish_goal_marker(raw, applied);
  return applied;
}

// PORT: MPCController._publish_goal_marker
void MPCController::publish_goal_marker(const std::array<double, 7> & raw, bool applied)
{
  if (goal_marker_pub_ == nullptr) {
    return;
  }
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = base_link_;
  marker.header.stamp = node_->get_clock()->now();
  marker.ns = "mpc_goal";
  marker.id = 0;
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

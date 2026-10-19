// SPDX-License-Identifier: Apache-2.0
// PORT: the planning service callbacks (see the header).
#include "curobo_core/core/planner_services.hpp"

#include <algorithm>
#include <mutex>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>
#include <set>
#include <stdexcept>

#include <sensor_msgs/msg/joint_state.hpp>

#include "curobo_core/core/obstacle_manager.hpp"
#include "curobo_core/core/collision_diagnostics.hpp"
#include "curobo_core/planners/mpc_planner.hpp"
#include "curobo_core/robot/robot_context.hpp"
#include "curobo_runtime/runtime.hpp"
#include "curobo_core/planners/reactive_controller.hpp"

namespace curobo_core
{

namespace
{
// PORT: SetPlanner enum id -> catalog key
const char * key_for_id(int id)
{
  for (const auto & entry : PlannerFactory::catalog()) {
    if (entry.id == id) {
      return entry.key;
    }
  }
  return nullptr;
}
// PORT: PlanningStats.considered shaping (single_planner._segment_considered_rows).
void append_considered_rows(
  const curobo_runtime::PlanOutcome & outcome,
  std::vector<curobov2_ros_interfaces::msg::ConsideredTrajectory> & out)
{
  for (const auto & r : outcome.considered) {
    curobov2_ros_interfaces::msg::ConsideredTrajectory row;
    row.problem = r.problem;
    row.segment = r.segment;
    row.goalset_candidate = r.goalset_candidate;
    row.seed = r.seed;
    row.success = r.success;
    row.cost = r.cost;
    row.waypoint_cost = r.waypoint_cost;
    row.path_length = r.path_length;
    row.clearance = r.clearance;
    row.max_waypoint_error = r.max_waypoint_error;
    row.solve_time = r.solve_time;
    out.push_back(row);
  }
}
// PORT: the start-position log's value list ("['0.000', ...]").
std::string format_values(const std::vector<float> & values)
{
  std::ostringstream out;
  out << "[" << std::fixed << std::setprecision(3);
  for (std::size_t i = 0; i < values.size(); ++i) {
    out << (i ? ", " : "") << values[i];
  }
  out << "]";
  return out.str();
}

// PORT: unified_planner_node._check_goal_request - validate a goal request
// against the active planner's rules so an invalid request never reaches the
// solver (cuRobo aborts on an oversized goalset, and the raw exception hides
// the real reason). Returns {ok, reason}.
std::pair<bool, std::string> check_goal_request(
  TrajectoryPlanner * planner, const curobov2_ros_interfaces::msg::TrajectoryGoal & goal,
  int max_goalset)
{
  const std::string name = planner->get_planner_name();
  const auto & goalsets = goal.goalsets;
  std::string sizes, joints;
  for (std::size_t i = 0; i < goalsets.size(); ++i) {
    sizes += (i ? ", " : "") + std::to_string(goalsets[i].poses.size());
    joints += (i ? ", " : "") + std::to_string(goalsets[i].target_joint_positions.size());
  }
  if (goalsets.empty()) {
    return {false, name + ": request must provide `goalsets` (one Goalset entry per segment)"};
  }
  for (const auto & g : goalsets) {
    if (g.poses.empty() && g.target_joint_positions.empty()) {
      return {false, name + ": every goalset entry must hold >= 1 pose or a "
        "non-empty target_joint_positions (got pose sizes [" + sizes +
        "], joint sizes [" + joints + "])"};
    }
  }
  if (!planner->is_open_loop()) {
    if (goalsets.size() != 1 || goalsets.front().poses.size() != 1) {
      return {false, name + ": requires exactly one goalset entry with a single "
        "pose, got " + std::to_string(goalsets.size()) + " set(s) of sizes [" +
        sizes + "]"};
    }
    return {true, ""};
  }
  for (const auto & g : goalsets) {
    if (static_cast<int>(g.poses.size()) > max_goalset) {
      int biggest = 0;
      for (const auto & h : goalsets) {
        biggest = std::max(biggest, static_cast<int>(h.poses.size()));
      }
      return {false, name + ": goalset of " + std::to_string(biggest) +
        " poses exceeds max_goalset=" + std::to_string(max_goalset) +
        " (per-segment cap; the goalset buffer is sized at solver build)"};
    }
  }
  return {true, ""};
}

// PORT: _build_collision_diagnostics._eval_state - attributed collisions at `q`
// (an empty attribution when there is nothing to evaluate).
curobo_runtime::CollisionAttribution collision_attribution_at(
  curobo_runtime::IBridge * bridge, curobo_runtime::RobotHandle handle,
  const std::vector<float> & q, bool evaluated)
{
  if (!evaluated || q.empty()) {
    return curobo_runtime::CollisionAttribution{};
  }
  curobo_runtime::TensorF32 tensor;
  tensor.shape = {1, static_cast<std::int64_t>(q.size())};
  tensor.data = q;
  try {
    return bridge->collision_attribution(handle, tensor);
  } catch (const std::exception &) {
    // Best-effort, like the Python's try/except -> empty StateCollisions.
    return curobo_runtime::CollisionAttribution{};
  }
}

// PORT: _build_collision_diagnostics - copy one state's attribution into its
// StateCollisions message (values rounded to 6 decimals, like the Python).
void fill_state_collisions(
  const std::string & label, const curobo_runtime::CollisionAttribution & sc,
  curobov2_ros_interfaces::msg::StateCollisions & out)
{
  for (const auto & c : sc.world_contacts) {
    curobov2_ros_interfaces::msg::WorldCollisionContact contact;
    contact.state_label = label;
    contact.link = c.link;
    contact.obstacle = c.obstacle;
    contact.depth = round6(c.depth);
    out.world_contacts.push_back(std::move(contact));
  }
  for (const auto & c : sc.self_contacts) {
    curobov2_ros_interfaces::msg::SelfCollisionContact contact;
    contact.state_label = label;
    contact.link_a = c.link_a;
    contact.link_b = c.link_b;
    contact.depth = round6(c.depth);
    out.self_contacts.push_back(std::move(contact));
  }
  for (const auto & c : sc.cspace_violations) {
    curobov2_ros_interfaces::msg::JointLimitViolation violation;
    violation.state_label = label;
    violation.joint = c.joint;
    violation.value = round6(c.value);
    violation.lower = round6(c.lower);
    violation.upper = round6(c.upper);
    out.cspace_violations.push_back(std::move(violation));
  }
}
}  // namespace

PlannerServices::PlannerServices(
  rclcpp::Node * node, curobo_runtime::IBridge * bridge,
  curobo_runtime::RobotHandle handle, std::vector<std::string> joint_names,
  ObstacleManager * obstacle_manager)
: node_(node), bridge_(bridge), handle_(handle), joint_names_(joint_names),
  obstacle_manager_(obstacle_manager), manager_(bridge, handle, joint_names, node)
{
  const std::string name = node_->get_name();
  set_planner_srv_ = node_->create_service<curobov2_ros_interfaces::srv::SetPlanner>(
    name + "/set_planner",
    std::bind(
      &PlannerServices::handle_set_planner, this,
      std::placeholders::_1, std::placeholders::_2));
  get_planners_srv_ = node_->create_service<curobov2_ros_interfaces::srv::GetPlanners>(
    name + "/get_planners",
    std::bind(
      &PlannerServices::handle_get_planners, this,
      std::placeholders::_1, std::placeholders::_2));
  generate_trajectory_srv_ =
    node_->create_service<curobov2_ros_interfaces::srv::TrajectoryGeneration>(
    name + "/generate_trajectory",
    std::bind(
      &PlannerServices::handle_generate_trajectory, this,
      std::placeholders::_1, std::placeholders::_2));
  generate_trajectory_batch_srv_ =
    node_->create_service<curobov2_ros_interfaces::srv::TrajectoryGenerationBatch>(
    name + "/trajectory_generation_batch",
    std::bind(
      &PlannerServices::handle_generate_trajectory_batch, this,
      std::placeholders::_1, std::placeholders::_2));
}

// PORT: joint_order.resolve_start_pose (name reorder, prefix padding from the
// robot's current pose, else positional).
std::vector<float> PlannerServices::resolve_start(
  const std::vector<std::string> & names, const std::vector<double> & positions) const
{
  const std::vector<float> current = current_pose_provider_ ? current_pose_provider_() :
    std::vector<float>{};
  const auto to_float = [](const std::vector<double> & values) {
      std::vector<float> out;
      out.reserve(values.size());
      for (double v : values) {
        out.push_back(static_cast<float>(v));
      }
      return out;
    };
  // PORT: `if not positions or not names: return positions or current`.
  if (positions.empty()) {
    return current;
  }
  if (names.empty()) {
    return to_float(positions);
  }
  // named_order: reorder by name when the name/position lengths agree.
  std::vector<float> ordered;
  bool named_ok = names.size() == positions.size();
  if (named_ok) {
    ordered.reserve(joint_names_.size());
    for (const std::string & want : joint_names_) {
      auto it = std::find(names.begin(), names.end(), want);
      if (it == names.end()) {
        named_ok = false;
        break;
      }
      ordered.push_back(
        static_cast<float>(positions[static_cast<std::size_t>(it - names.begin())]));
    }
  }
  if (!named_ok) {
    // PORT: name/position length disagree -> legacy positional.
    return to_float(positions);
  }
  // PORT: a short named pose must be the cspace PREFIX; pad the rest from the
  // robot's current pose. Any other partial naming is rejected.
  if (ordered.size() < joint_names_.size()) {
    std::set<std::string> named_set(names.begin(), names.end());
    std::set<std::string> prefix_set(
      joint_names_.begin(), joint_names_.begin() + static_cast<long>(ordered.size()));
    if (named_set != prefix_set) {
      std::string named_txt, prefix_txt;
      for (const auto & n : named_set) {
        named_txt += (named_txt.empty() ? "" : ", ") + n;
      }
      for (const auto & n : prefix_set) {
        prefix_txt += (prefix_txt.empty() ? "" : ", ") + n;
      }
      throw std::invalid_argument(
        "named start pose covers {" + named_txt + "}, which is not the cspace "
        "prefix {" + prefix_txt + "}. A partial start pose is read as a prefix "
        "and padded from the robot's current pose, so name either the full "
        "cspace or that prefix.");
    }
    const std::size_t padded = joint_names_.size() - ordered.size();
    for (std::size_t i = ordered.size(); i < joint_names_.size(); ++i) {
      ordered.push_back(i < current.size() ? current[i] : 0.0f);
    }
    // PORT: "Using start position from request (padded N DOF from current): ...".
    RCLCPP_INFO(
      node_->get_logger(), "Using start position from request (padded %zu DOF from "
      "current): %s", padded, format_values(ordered).c_str());
    return ordered;
  }
  // PORT: "Using start position from request: ...".
  RCLCPP_INFO(
    node_->get_logger(), "Using start position from request: %s",
    format_values(ordered).c_str());
  return ordered;
}

// Continuous-joint target normalisation (see the header). A full turn is the
// wrap period; the reported range of a continuous joint is 2*pi (cuRobo's
// [-2*pi, 2*pi] is 4*pi, comfortably >= one turn, so exactly one equivalent
// of any target falls in range except at the very edges).
void PlannerServices::wrap_full_turn_targets(curobo_runtime::PlanRequest & request) const
{
  if (request.joint_target.empty()) {
    return;
  }
  curobo_runtime::JointLimitsOutcome limits;
  try {
    limits = bridge_->joint_limits(handle_);
  } catch (const std::exception &) {
    return;  // limits unavailable: leave the target verbatim
  }
  const std::size_t dof = limits.joint_names.size();
  if (request.joint_target.size() != dof || limits.positions.data.size() < 2 * dof) {
    return;
  }
  constexpr double kTwoPi = 6.283185307179586;
  for (std::size_t j = 0; j < dof; ++j) {
    const double lo = limits.positions.data[j];
    const double hi = limits.positions.data[dof + j];
    // Bounded revolute/prismatic joints have a range shorter than a turn;
    // only full-turn (continuous) joints are periodic under 2*pi.
    if (!std::isfinite(lo) || !std::isfinite(hi) || hi - lo < kTwoPi - 1e-6) {
      continue;
    }
    const double start = j < request.start.size() ?
      static_cast<double>(request.start[j]) : static_cast<double>(request.joint_target[j]);
    const double target = request.joint_target[j];
    // cuRobo converts a continuous joint to revolute with limits of
    // +/-2*3.14 (=+/-6.28), which is ~0.0032 rad SHORT of a full turn. So the
    // +/-2*pi equivalent of a target can fall *just outside* [-6.28, 6.28]:
    // e.g. target 0 on a joint whose start is 5.997 -- the short way is
    // 2*pi = 6.2832, but the range tops out at 6.2800. Admit equivalents within
    // a small tolerance of the range and snap them to the boundary; the pose
    // error is that tolerance (<0.01 rad) yet it is what lets the SHORT path be
    // chosen instead of a full-turn sweep that usually self-collides.
    constexpr double kRangeTolerance = 0.01;
    double best = target;
    double best_dist = std::numeric_limits<double>::infinity();
    for (int k = -2; k <= 2; ++k) {
      double cand = target + static_cast<double>(k) * kTwoPi;
      if (cand < lo - kRangeTolerance || cand > hi + kRangeTolerance) {
        continue;  // equivalent genuinely outside the reported range
      }
      cand = std::clamp(cand, lo, hi);
      const double dist = std::abs(cand - start);
      if (dist < best_dist) {
        best_dist = dist;
        best = cand;
      }
    }
    if (best != target) {
      RCLCPP_INFO(
        node_->get_logger(),
        "Continuous joint %s: snapped target %.6f -> %.6f (nearest start %.6f)",
        limits.joint_names[j].c_str(), target, best, start);
    }
    request.joint_target[j] = static_cast<float>(best);
  }
}

// PORT: unified_planner_node._plan_trajectory_goal request assembly
curobo_runtime::PlanRequest PlannerServices::build_request(
  const curobov2_ros_interfaces::msg::TrajectoryGoal & goal) const
{
  curobo_runtime::PlanRequest plan_request;
  plan_request.start = resolve_start(goal.start_pose.name, goal.start_pose.position);
  // PORT: config max_attempts / enable_graph_attempt (planner config).
  plan_request.max_attempts = node_->get_parameter("max_attempts").as_int();
  plan_request.enable_graph_attempt = 1;
  plan_request.cartesian_step_size = goal.cartesian_step_size;
  plan_request.cartesian_precision = goal.cartesian_precision;

  const bool joint_space = !goal.goalsets.empty() &&
    !goal.goalsets.front().target_joint_positions.empty();
  if (joint_space) {
    for (double v : goal.goalsets.front().target_joint_positions) {
      plan_request.joint_target.push_back(static_cast<float>(v));
    }
    // Snap continuous-joint targets to the in-range equivalent nearest the
    // start so a negative / un-wrapped joint goal plans like its wrapped twin.
    wrap_full_turn_targets(plan_request);
  } else if (!goal.goalsets.empty()) {
    plan_request.goalset_count = static_cast<std::int64_t>(goal.goalsets.size());
    // PORT: single_planner._apply_pose_constraints - the FIRST non-empty
    // Goalset.trajectory_constraints holds its axes along the whole path.
    for (const auto & g : goal.goalsets) {
      if (!g.trajectory_constraints.empty()) {
        plan_request.trajectory_constraints.assign(
          g.trajectory_constraints.begin(), g.trajectory_constraints.end());
        break;
      }
    }
    const auto & poses = goal.goalsets.front().poses;
    plan_request.goalset.shape = {static_cast<std::int64_t>(poses.size()), 7};
    for (const auto & pose : poses) {
      plan_request.goalset.data.push_back(static_cast<float>(pose.position.x));
      plan_request.goalset.data.push_back(static_cast<float>(pose.position.y));
      plan_request.goalset.data.push_back(static_cast<float>(pose.position.z));
      plan_request.goalset.data.push_back(static_cast<float>(pose.orientation.w));
      plan_request.goalset.data.push_back(static_cast<float>(pose.orientation.x));
      plan_request.goalset.data.push_back(static_cast<float>(pose.orientation.y));
      plan_request.goalset.data.push_back(static_cast<float>(pose.orientation.z));
    }
  }
  return plan_request;
}

// PORT: single_planner._publish_plan_image
void PlannerServices::publish_plan_image(const curobo_runtime::PlanOutcome & outcome)
{
  if (!node_->has_parameter("publish_plan_debug_image") ||
    !node_->get_parameter("publish_plan_debug_image").as_bool())
  {
    return;
  }
  if (plan_image_pub_ == nullptr) {
    rclcpp::QoS qos(rclcpp::KeepLast(1));
    qos.transient_local();
    plan_image_pub_ = node_->create_publisher<sensor_msgs::msg::Image>(
      node_->get_name() + std::string("/motion_plan_debug"), qos);
  }
  try {
    const auto img = bridge_->render_plan_image(
      outcome.positions, outcome.velocity, outcome.acceleration,
      outcome.joint_names.empty() ? joint_names_ : outcome.joint_names, 0.025);
    if (!img.has_value || img.height <= 0 || img.width <= 0) {
      return;
    }
    sensor_msgs::msg::Image msg;
    msg.header.stamp = node_->get_clock()->now();
    msg.header.frame_id = node_->has_parameter("base_link") ?
      node_->get_parameter("base_link").as_string() : "";
    msg.height = static_cast<std::uint32_t>(img.height);
    msg.width = static_cast<std::uint32_t>(img.width);
    msg.encoding = img.channels == 3 ? "rgb8" : "mono8";
    msg.is_bigendian = false;
    msg.step = static_cast<std::uint32_t>(img.width * img.channels);
    msg.data = img.data;
    plan_image_pub_->publish(msg);
  } catch (const std::exception &) {
    // A debug image must never break a successful plan.
  }
}

// PORT: single_planner._stage_trajectory - project + buffer a trajectory into
// the RobotContext's command buffer. plan() calls this so a compute-only plan
// is rendered for preview (GhostStrategy) and can be streamed later; it does
// not execute anything.
void PlannerServices::stage_preview(
  const std::vector<sensor_msgs::msg::JointState> & trajectory)
{
  if (robot_context_ == nullptr || trajectory.empty()) {
    return;
  }

  std::vector<std::vector<double>> position;
  std::vector<std::vector<double>> velocity;
  std::vector<std::vector<double>> acceleration;
  std::vector<std::string> names;
  position.reserve(trajectory.size());
  velocity.reserve(trajectory.size());
  acceleration.reserve(trajectory.size());

  for (const auto & point : trajectory) {
    names = point.name.empty() ? joint_names_ : point.name;
    // cuRobo can return more columns than the named active joints (locked
    // fingers are part of the solver DOF). The command strategy expects one
    // name per column, so keep the leading columns that are named. This is the
    // same projection the action server applies before streaming.
    std::size_t count = std::min(names.size(), point.position.size());
    if (count == 0) {
      return;
    }
    names.resize(count);

    std::vector<double> pos(point.position.begin(), point.position.begin() + count);
    std::vector<double> vel(count, 0.0);
    if (point.velocity.size() >= count) {
      vel.assign(point.velocity.begin(), point.velocity.begin() + count);
    }
    position.push_back(std::move(pos));
    velocity.push_back(std::move(vel));
    acceleration.emplace_back(count, 0.0);
  }

  try {
    robot_context_->set_command(names, velocity, acceleration, position);
    RCLCPP_INFO(
      node_->get_logger(), "Trajectory sent to robot context for visualization");
  } catch (const std::exception & e) {
    // A preview must never turn a successful plan into a failed service call.
    RCLCPP_WARN(
      node_->get_logger(), "Could not stage trajectory preview: %s", e.what());
  }
}

// PORT: unified_planner_node._plan_trajectory_goal (result assembly)
void PlannerServices::fill_result(
  TrajectoryPlanner * planner, const curobov2_ros_interfaces::msg::TrajectoryGoal & goal,
  curobov2_ros_interfaces::msg::TrajectoryResult & out)
{
  // PORT: single_planner._plan_trajectory_goal -> lane_plan.call - serialize
  // every plan on the plan lane (the shared MotionPlanner is not thread-safe).
  std::lock_guard<std::recursive_mutex> plan_lane(plan_lane_mutex_);
  const auto t_total = std::chrono::steady_clock::now();
  // PORT: reset_seed_per_plan - rewind the solver RNG before every plan.
  if (node_->get_parameter("reset_seed_per_plan").as_bool()) {
    bridge_->reset_seed(handle_);
  }
  // PORT: unified_planner_node._check_goal_request (validation) +
  // _resolve_goalset_frames (the server owns every transform), both BEFORE the
  // solver so an invalid request never reaches it.
  curobov2_ros_interfaces::msg::TrajectoryGoal resolved_goal = goal;
  {
    const auto check = check_goal_request(
      planner, resolved_goal, static_cast<int>(node_->get_parameter("max_goalset").as_int()));
    if (!check.first) {
      out.success = false;
      out.message = check.second;
      out.selected_goal_index.push_back(-1);
      out.selected_seed_index.push_back(-1);
      out.stats.problems = 1;
      out.stats.waypoints_planned = static_cast<std::uint32_t>(goal.goalsets.size());
      // PORT: the Python returns the reason without logging.
      return;
    }
  }
  if (pose_resolver_) {
    for (auto & g : resolved_goal.goalsets) {
      if (g.poses.empty()) {
        continue;
      }
      try {
        g.poses = pose_resolver_(g.poses, g.frame_id);
        g.frame_id = "";
      } catch (const std::exception & e) {
        out.success = false;
        out.message = planner->get_planner_name() + ": " + e.what();
        out.selected_goal_index.push_back(-1);
        out.selected_seed_index.push_back(-1);
        out.stats.problems = 1;
        out.stats.waypoints_planned = static_cast<std::uint32_t>(goal.goalsets.size());
        RCLCPP_ERROR(node_->get_logger(), "Planning failed: %s", out.message.c_str());
        return;
      }
    }
  }

  // PORT: the Python planning callbacks wrap the plan in try/except and return
  // success=false with the message (e.g. an invalid partial start pose).
  curobo_runtime::PlanRequest request;
  try {
    request = build_request(resolved_goal);
  } catch (const std::exception & e) {
    out.success = false;
    out.message = std::string("Planning error: ") + e.what();
    out.selected_goal_index.push_back(-1);
    out.selected_seed_index.push_back(-1);
    out.stats.problems = 1;
    out.stats.waypoints_planned = static_cast<std::uint32_t>(goal.goalsets.size());
    return;
  }

  // A non-empty target_joint_positions is an unambiguous JOINT-space segment
  // (the RViz Joints tab / named poses send this without switching planner, and
  // the tab documents "a non-empty target_joint_positions switches the planner
  // to plan_cspace"). Route it to the joint-space planner for this request.
  TrajectoryPlanner * effective = planner;
  if (!request.joint_target.empty() && planner->is_open_loop() &&
    planner->get_planner_name().find("Joint Space") == std::string::npos)
  {
    TrajectoryPlanner * joint_space = manager_.get_planner("joint_space");
    if (joint_space != nullptr) {
      RCLCPP_INFO(
        node_->get_logger(),
        "Joint-space segment: using %s (was %s) for this request",
        joint_space->get_planner_name().c_str(), planner->get_planner_name().c_str());
      effective = joint_space;
    }
  }

  // PORT: single_planner._apply_pose_constraints' "holding axes along path"
  // log (ToolPoseCriteria order is [x, y, z, roll, pitch, yaw]).
  if (request.trajectory_constraints.size() == 6) {
    bool any_hold = false;
    for (int c : request.trajectory_constraints) {
      any_hold = any_hold || (c == 1);
    }
    if (any_hold) {
      const auto & tc = request.trajectory_constraints;
      const double tx = tc[0] == 1 ? 1.0 : 0.0;
      const double ty = tc[1] == 1 ? 1.0 : 0.0;
      const double tz = tc[2] == 1 ? 1.0 : 0.0;
      const double x = tc[3] == 1 ? 1.0 : 0.0;
      const double y = tc[4] == 1 ? 1.0 : 0.0;
      const double z = tc[5] == 1 ? 1.0 : 0.0;
      RCLCPP_INFO(
        node_->get_logger(), "%s: holding axes along path (x,y,z,roll,pitch,yaw)=[%g, %g, %g, %g, %g, %g]",
        effective->get_planner_name().c_str(), x, y, z, tx, ty, tz);
    }
  }

  // PORT: unified_planner_node "Planning with {planner.get_planner_name()}".
  RCLCPP_INFO(
    node_->get_logger(), "Planning with %s", effective->get_planner_name().c_str());

  // PORT: refresh_perception_world() - refresh the perception-based collision
  // world before planning so the plan accounts for the latest camera data.
  const auto t_setup = std::chrono::steady_clock::now();
  if (!node_->has_parameter("push_esdf_to_solvers") ||
    node_->get_parameter("push_esdf_to_solvers").as_bool())
  {
    try {
      // PORT: refresh_perception_world() = refresh_esdf() + update_all_solvers_
      // world(); the push is what makes the planner collide against the
      // camera-observed voxels (otherwise the published grid shows unchecked
      // voxels). Gated by should_refresh_world (open-loop: throttle bound) and
      // the ESDF rebuild + push both run under the exclusive barrier (the Python
      // holds gpu_lock around both; refresh_esdf mutates the shared scene).
      if (obstacle_manager_->should_refresh_world(false)) {
        auto guard = curobo_runtime::Runtime::instance().barrier().exclusive();
        if (obstacle_manager_->refresh_esdf()) {
          bridge_->world_push(handle_);
          obstacle_manager_->mark_world_pushed();
        }
      }
    } catch (const std::exception &) {
      // A perception refresh must never block planning.
    }
  }
  const auto t_world = std::chrono::steady_clock::now();

  PlannerResult result;
  try {
    result = effective->plan(request);
  } catch (const std::exception & e) {
    out.success = false;
    out.message = std::string("Planning error: ") + e.what();
    out.selected_goal_index.push_back(-1);
    out.selected_seed_index.push_back(-1);
    out.stats.problems = 1;
    out.stats.waypoints_planned = static_cast<std::uint32_t>(goal.goalsets.size());
    return;
  }
  const auto & outcome = result.trajectory;

  const auto t_plan = std::chrono::steady_clock::now();
  // PORT: unified_planner_node "[plan-perf] {planner}: ... end_to_end ...".
  {
    const auto ms = [](const auto & a, const auto & b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
      };
    RCLCPP_INFO(
      node_->get_logger(),
      "[plan-perf] %s: node_setup(start-state+config) %.1f ms, "
      "world_refresh(ESDF+world-push) %.1f ms, "
      "planner_plan(=curobo_solve+finalize) %.1f ms, end_to_end %.1f ms",
      effective->get_planner_name().c_str(), ms(t_total, t_setup),
      ms(t_setup, t_world), ms(t_world, t_plan), ms(t_total, t_plan));
  }

  if (!result.success) {
    // PORT: _build_collision_diagnostics - attributed collisions at the start
    // (always) and goal (joint-space segments only) configurations.
    const auto start_collisions =
      collision_attribution_at(bridge_, handle_, request.start, true);
    const auto goal_collisions = collision_attribution_at(
      bridge_, handle_, request.joint_target, !request.joint_target.empty());
    // PORT: unified_planner_node "Planning failed: {message}\n{_format_collision_feedback}".
    const std::string feedback =
      format_collision_feedback(start_collisions, goal_collisions);
    RCLCPP_ERROR(
      node_->get_logger(), "Planning failed: %s\n%s", result.message.c_str(),
      feedback.c_str());
    out.success = false;
    out.message = result.message;
    // PORT: _fill_collision_diagnostics (populated on failure only).
    fill_state_collisions("start", start_collisions, out.start_state_collisions);
    fill_state_collisions("goal", goal_collisions, out.end_state_collisions);
    out.selected_goal_index.push_back(-1);
    out.selected_seed_index.push_back(-1);
    // PORT: _build_planning_stats is ALWAYS populated, including on failure.
    out.stats.problems = outcome.problems;
    out.stats.waypoints_planned = static_cast<std::uint32_t>(goal.goalsets.size());
    out.stats.candidates_generated = outcome.candidates_generated;
    out.stats.candidates_solved = outcome.candidates_solved;
    out.stats.candidates_pruned = outcome.candidates_pruned;
    out.stats.solver_time_s = outcome.solver_time_s;
    out.stats.position_error_mm = outcome.position_error_mm;
    out.stats.motion_time_s = outcome.motion_time_s;
    // PORT: the considered rows ride the stats block on failure too.
    append_considered_rows(outcome, out.stats.considered);
    for (std::int16_t status : outcome.waypoint_status) {
      out.waypoint_status.push_back(status);
    }
    return;
  }

  const std::int64_t dof = outcome.positions.shape.size() > 1 ? outcome.positions.shape[1] : 0;
  const std::int64_t rows = outcome.positions.shape.empty() ? 0 : outcome.positions.shape[0];
  const std::int64_t vdof = outcome.velocity.shape.size() > 1 ? outcome.velocity.shape[1] : 0;
  for (std::int64_t t = 0; t < rows; ++t) {
    sensor_msgs::msg::JointState js;
    js.name = outcome.joint_names.empty() ? joint_names_ : outcome.joint_names;
    for (std::int64_t j = 0; j < dof; ++j) {
      js.position.push_back(outcome.positions.data[static_cast<std::size_t>(t * dof + j)]);
    }
    if (vdof > 0 &&
      outcome.velocity.data.size() >= static_cast<std::size_t>((t + 1) * vdof))
    {
      for (std::int64_t j = 0; j < vdof; ++j) {
        js.velocity.push_back(outcome.velocity.data[static_cast<std::size_t>(t * vdof + j)]);
      }
    }
    out.trajectory.push_back(js);
  }
  out.dt = 0.025;
  out.success = true;
  out.message = result.message;
  // PORT: unified_planner_node "Planning succeeded: {message} (trajectory: N
  // waypoints, dt: Xs)".
  RCLCPP_INFO(
    node_->get_logger(), "Planning succeeded: %s (trajectory: %zu waypoints, dt: %gs)",
    out.message.c_str(), out.trajectory.size(), out.dt);
  // PORT: single_planner "Successfully planned trajectory with {num_wp} waypoints".
  RCLCPP_INFO(
    node_->get_logger(), "%s: Successfully planned trajectory with %zu waypoint(s)",
    effective->get_planner_name().c_str(), out.trajectory.size());
  // PORT: single_planner "<planner> planner-phases: curobo_solve ... finalize ...".
  {
    const auto t_done = std::chrono::steady_clock::now();
    const auto ms = [](const auto & a, const auto & b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
      };
    RCLCPP_INFO(
      node_->get_logger(),
      "%s planner-phases: curobo_solve %.1f ms, finalize(result shaping) %.1f ms, "
      "planner_total %.1f ms",
      effective->get_planner_name().c_str(), ms(t_world, t_plan), ms(t_plan, t_done),
      ms(t_world, t_done));
  }
  // PORT: single_planner._selected_goal_indexes / _selected_seed_index (the
  // winner candidate / seed reported by the solver).
  out.selected_goal_index.push_back(static_cast<std::int16_t>(outcome.goalset_index));
  out.selected_seed_index.push_back(static_cast<std::int16_t>(outcome.seed_index));
  out.stats.problems = outcome.problems;
  out.stats.candidates_generated = outcome.candidates_generated;
  out.stats.candidates_solved = outcome.candidates_solved;
  out.stats.candidates_pruned = outcome.candidates_pruned;
  out.stats.waypoints_planned = outcome.waypoints_planned;
  // PORT: single_planner.plan() -> _stage_trajectory (preview buffer) and
  // _publish_plan_image (once per new plan).
  stage_preview(out.trajectory);
  publish_plan_image(outcome);
  out.stats.solver_time_s = outcome.solver_time_s;
  out.stats.position_error_mm = outcome.position_error_mm;
  out.stats.motion_time_s = outcome.motion_time_s;
  append_considered_rows(outcome, out.stats.considered);
  for (std::int16_t status : outcome.waypoint_status) {
    out.waypoint_status.push_back(status);
  }
}

// PORT: unified_planner_node.set_planner_callback
void PlannerServices::handle_set_planner(
  const std::shared_ptr<curobov2_ros_interfaces::srv::SetPlanner::Request> request,
  std::shared_ptr<curobov2_ros_interfaces::srv::SetPlanner::Response> response)
{
  const char * key = key_for_id(static_cast<int>(request->planner_type));
  if (key == nullptr) {
    response->success = false;
    response->message = "Unknown planner enum id: " + std::to_string(request->planner_type);
    return;
  }
  response->previous_planner = manager_.current() ? manager_.current()->get_planner_name() : "";
  manager_.set_current_planner(key);
  // PORT: the node pushes the shared scene to a freshly selected reactive solver
  // (MPC reloads its scene collision checker; the open-loop planner already sees
  // the shared world).
  if (auto * reactive = dynamic_cast<ReactiveController *>(manager_.current())) {
    reactive->ensure_solver();
    // PORT: the servo loop's periodic refresh_perception_world (ESDF refresh +
    // world_push) - same gate and effect as the plan path's push below.
    reactive->set_perception_refresh([this]() {
        if (node_->has_parameter("push_esdf_to_solvers") &&
          !node_->get_parameter("push_esdf_to_solvers").as_bool())
        {
          return;
        }
        // PORT: the Python holds gpu_lock (exclusive) around the whole
        // refresh_perception_world (ESDF rebuild + push) and gates it on
        // should_refresh_world(force=True): the servo only re-pushes when the
        // perception epoch advanced (a new depth frame / scene mutation).
        if (obstacle_manager_ != nullptr &&
          obstacle_manager_->should_refresh_world(true))
        {
          auto guard = curobo_runtime::Runtime::instance().barrier().exclusive();
          if (obstacle_manager_->refresh_esdf()) {
            bridge_->world_push(handle_);
            obstacle_manager_->mark_world_pushed();
          }
        }
      });
    if (obstacle_manager_ != nullptr) {
      reactive->update_world(obstacle_manager_->scene_json());
    }
  }
  response->current_planner = manager_.current() ? manager_.current()->get_planner_name() : "";
  response->success = true;
  response->message = std::string("Successfully switched to ") + response->current_planner;
  RCLCPP_INFO(node_->get_logger(), "Switched to planner: %s", response->current_planner.c_str());
}

// PORT: unified_planner_node.get_planners_callback
void PlannerServices::handle_get_planners(
  const std::shared_ptr<curobov2_ros_interfaces::srv::GetPlanners::Request>,
  std::shared_ptr<curobov2_ros_interfaces::srv::GetPlanners::Response> response)
{
  for (const auto & entry : PlannerFactory::catalog()) {
    response->planner_names.push_back(entry.display);
    response->planner_ids.push_back(static_cast<std::uint8_t>(entry.id));
  }
  response->current_planner_name =
    manager_.current() ? manager_.current()->get_planner_name() : "";
  response->current_planner_id = 0;
  for (const auto & entry : PlannerFactory::catalog()) {
    if (manager_.current_type() == entry.key) {
      response->current_planner_id = static_cast<std::uint8_t>(entry.id);
    }
  }
  response->success = true;
}

// PORT: unified_planner_node.generate_trajectory_callback
void PlannerServices::handle_generate_trajectory(
  const std::shared_ptr<curobov2_ros_interfaces::srv::TrajectoryGeneration::Request> request,
  std::shared_ptr<curobov2_ros_interfaces::srv::TrajectoryGeneration::Response> response)
{
  TrajectoryPlanner * planner = manager_.current();
  if (planner == nullptr || !planner->is_open_loop()) {
    response->response.success = false;
    response->response.message = "No planner selected";
    return;
  }
  fill_result(planner, request->request, response->response);
}

// PORT: unified_planner_node.trajectory_generation_batch_callback + _rollup_stats
void PlannerServices::handle_generate_trajectory_batch(
  const std::shared_ptr<curobov2_ros_interfaces::srv::TrajectoryGenerationBatch::Request> request,
  std::shared_ptr<curobov2_ros_interfaces::srv::TrajectoryGenerationBatch::Response> response)
{
  TrajectoryPlanner * planner = manager_.current();
  if (planner == nullptr || !planner->is_open_loop()) {
    response->success = false;
    response->error_msg = "No planner selected";
    return;
  }
  response->responses.resize(request->requests.size());
  bool all_ok = true;
  for (std::size_t i = 0; i < request->requests.size(); ++i) {
    fill_result(planner, request->requests[i], response->responses[i]);
    all_ok = all_ok && response->responses[i].success;
  }
  // PORT: _rollup_stats - sum scalar counts, times do not sum.
  auto & total = response->total_stats;
  total.problems = static_cast<std::uint32_t>(request->requests.size());
  total.solver_time_s = std::nan("");
  total.position_error_mm = std::nan("");
  for (const auto & r : response->responses) {
    total.candidates_generated += r.stats.candidates_generated;
    total.candidates_solved += r.stats.candidates_solved;
    total.candidates_pruned += r.stats.candidates_pruned;
    total.waypoints_planned += r.stats.waypoints_planned;
  }
  response->success = all_ok;
}

}  // namespace curobo_core

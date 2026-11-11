// SPDX-License-Identifier: Apache-2.0
// PORT: the planning service callbacks (see the header).
#include "curobo_core/core/planner_services.hpp"
#include "curobo_core/core/gpu_runtime.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <utility>
#include <vector>
#include <set>
#include <stdexcept>

#include <geometry_msgs/msg/pose.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include "curobo_core/core/obstacle_manager.hpp"
#include "curobo_core/core/collision_diagnostics.hpp"
#include "curobo_core/planners/mpc_planner.hpp"
#include "curobo_core/planners/zero_motion.hpp"
#include "curobo_core/robot/continuous_joints.hpp"
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
  std::string sizes, joints, tools;
  for (std::size_t i = 0; i < goalsets.size(); ++i) {
    sizes += (i ? ", " : "") + std::to_string(goalsets[i].poses.size());
    joints += (i ? ", " : "") + std::to_string(goalsets[i].target_joint_positions.size());
    tools += (i ? ", " : "") + std::to_string(goalsets[i].tool_poses.size());
  }
  if (goalsets.empty()) {
    return {false, name + ": request must provide `goalsets` (one Goalset entry per segment)"};
  }
  for (const auto & g : goalsets) {
    const bool has_waypoints = !g.waypoints.empty();
    const bool has_tool_waypoints = !g.tool_waypoints.empty();
    if (g.poses.empty() && g.target_joint_positions.empty() &&
      g.tool_poses.empty() && !has_waypoints && !has_tool_waypoints)
    {
      return {false, name + ": every goalset entry must hold >= 1 pose, a "
        "non-empty target_joint_positions, multi-tool poses, or an ordered "
        "waypoint path (got pose sizes [" +
        sizes + "], joint sizes [" + joints + "], tool pose sizes [" + tools + "])"};
    }
    // Ordered waypoint path (MULTIPOINT): the waypoint fields own the segment.
    // A joint target and a waypoint path are two different solves, and the
    // single-tool / multi-tool path layouts exclude each other.
    if ((has_waypoints || has_tool_waypoints) &&
      !g.target_joint_positions.empty())
    {
      return {false, name + ": a waypoint path cannot mix with "
        "target_joint_positions (use the joint-space planner for a joint goal)"};
    }
    if (has_waypoints && has_tool_waypoints) {
      return {false, name + ": waypoints and tool_waypoints are mutually "
        "exclusive; send one single-tool path or one waypoint-major multi-tool path"};
    }
    if (g.waypoint_count > 0 && !has_waypoints && !has_tool_waypoints) {
      return {false, name + ": waypoint_count=" + std::to_string(g.waypoint_count) +
        " is set but the goalset holds no waypoints or tool_waypoints"};
    }
    if (has_waypoints) {
      if (g.tool_frames.size() > 1) {
        return {false, name + ": waypoints is a single-tool path but tool_frames "
          "names " + std::to_string(g.tool_frames.size()) +
          " frame(s); use tool_waypoints for multiple tools"};
      }
      if (g.waypoint_count > 0 &&
        static_cast<std::size_t>(g.waypoint_count) != g.waypoints.size())
      {
        return {false, name + ": waypoint_count=" +
          std::to_string(g.waypoint_count) + " but waypoints holds " +
          std::to_string(g.waypoints.size()) + " pose(s)"};
      }
    }
    if (has_tool_waypoints) {
      if (g.tool_frames.empty()) {
        return {false, name + ": tool_waypoints requires tool_frames (one name "
          "per tracked tool)"};
      }
      if (g.waypoint_count > 0) {
        const std::size_t expected =
          static_cast<std::size_t>(g.waypoint_count) * g.tool_frames.size();
        if (g.tool_waypoints.size() != expected) {
          return {false, name + ": tool_waypoints holds " +
            std::to_string(g.tool_waypoints.size()) + " pose(s) for waypoint_count=" +
            std::to_string(g.waypoint_count) + " and " +
            std::to_string(g.tool_frames.size()) + " tool frame(s) (expected " +
            std::to_string(expected) + ")"};
        }
      } else if (g.tool_waypoints.size() % g.tool_frames.size() != 0) {
        return {false, name + ": tool_waypoints holds " +
          std::to_string(g.tool_waypoints.size()) + " pose(s), which is not a "
          "multiple of " + std::to_string(g.tool_frames.size()) + " tool frame(s)"};
      }
    }
    if (has_waypoints || has_tool_waypoints) {
      // cuRobo's interpolate_pose_waypoints accepts 1..256 ordered targets.
      const std::size_t waypoint_count = has_waypoints ?
        g.waypoints.size() :
        (g.waypoint_count > 0 ? static_cast<std::size_t>(g.waypoint_count) :
        (g.tool_frames.empty() ? 0 : g.tool_waypoints.size() / g.tool_frames.size()));
      if (waypoint_count < 1 || waypoint_count > 256) {
        return {false, name + ": an ordered waypoint path supports 1..256 "
          "waypoints, got " + std::to_string(waypoint_count)};
      }
      constexpr double kPi = 3.14159265358979323846;
      const auto bad_position = [](double v) {
          return !std::isfinite(v) || v <= 0.0;
        };
      const auto bad_orientation = [](double v) {
          return !std::isfinite(v) || v <= 0.0 || v > kPi;
        };
      if (bad_position(g.path_position_tolerance)) {
        return {false, name + ": path_position_tolerance must be finite and "
          "positive (metres), got " + std::to_string(g.path_position_tolerance)};
      }
      if (bad_orientation(g.path_orientation_tolerance)) {
        return {false, name + ": path_orientation_tolerance must be finite and "
          "in (0, pi] radians, got " +
          std::to_string(g.path_orientation_tolerance)};
      }
      if (bad_position(g.final_position_tolerance)) {
        return {false, name + ": final_position_tolerance must be finite and "
          "positive (metres), got " + std::to_string(g.final_position_tolerance)};
      }
      if (bad_orientation(g.final_orientation_tolerance)) {
        return {false, name + ": final_orientation_tolerance must be finite and "
          "in (0, pi] radians, got " +
          std::to_string(g.final_orientation_tolerance)};
      }
      if (g.outer_iterations < 1) {
        return {false, name + ": outer_iterations must be >= 1, got " +
          std::to_string(g.outer_iterations)};
      }
    }
    if (!g.tool_poses.empty() && g.tool_poses.size() != g.tool_frames.size()) {
      return {false, name + ": tool_frames and tool_poses must be parallel "
        "(got " + std::to_string(g.tool_frames.size()) + " frame(s), " +
        std::to_string(g.tool_poses.size()) + " pose(s))"};
    }
    if (!g.tool_poses.empty() && !g.target_joint_positions.empty()) {
      return {false, name + ": a goalset cannot mix multi-tool poses with "
        "target_joint_positions"};
    }
    if (!g.tool_trajectory_constraints.empty() &&
      g.tool_trajectory_constraints.size() != 6 * g.tool_frames.size())
    {
      return {false, name + ": tool_trajectory_constraints must be empty or "
        "exactly 6 * len(tool_frames) (got " +
        std::to_string(g.tool_trajectory_constraints.size()) + " value(s) for " +
        std::to_string(g.tool_frames.size()) + " frame(s))"};
    }
  }
  if (!planner->is_open_loop()) {
    const bool single_pose = goalsets.size() == 1 &&
      goalsets.front().poses.size() == 1 && goalsets.front().tool_poses.empty();
    const bool multi_tool = goalsets.size() == 1 &&
      !goalsets.front().tool_poses.empty();
    const bool waypoint_path = goalsets.size() == 1 &&
      (!goalsets.front().waypoints.empty() ||
      !goalsets.front().tool_waypoints.empty());
    if (!single_pose && !multi_tool && !waypoint_path) {
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
  // Dedicated mutually exclusive group: plans still serialize among themselves
  // (one plan lane, one GPU), but they no longer starve the quick services and
  // the viz timers that live in the default group.
  plan_callback_group_ = node_->create_callback_group(
    rclcpp::CallbackGroupType::MutuallyExclusive);
  generate_trajectory_srv_ =
    node_->create_service<curobov2_ros_interfaces::srv::TrajectoryGeneration>(
    name + "/generate_trajectory",
    std::bind(
      &PlannerServices::handle_generate_trajectory, this,
      std::placeholders::_1, std::placeholders::_2),
    rmw_qos_profile_services_default, plan_callback_group_);
  generate_trajectory_batch_srv_ =
    node_->create_service<curobov2_ros_interfaces::srv::TrajectoryGenerationBatch>(
    name + "/trajectory_generation_batch",
    std::bind(
      &PlannerServices::handle_generate_trajectory_batch, this,
      std::placeholders::_1, std::placeholders::_2),
    rmw_qos_profile_services_default, plan_callback_group_);
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

// Continuous joints are circles. cuRobo plans in R^n, so the finite
// [-2*pi, 2*pi] window its parser reports is used as a coordinate chart:
//   * the requested start is lifted to the measured branch (same physical pose),
//   * a joint target is lifted by SHORTEST PATH from that start (|delta| <= pi),
//   * one common 2*pi multiple shifts both into the window,
//   * fill_result subtracts that same constant from every output point.
// Bounded joints are left byte-for-byte untouched.
bool PlannerServices::prepare_continuous_joints(
  curobo_runtime::PlanRequest & request) const
{
  if (request.start.empty()) {
    return true;
  }
  curobo_runtime::JointLimitsOutcome limits;
  try {
    limits = bridge_->joint_limits(handle_);
  } catch (const std::exception & e) {
    RCLCPP_WARN(
      node_->get_logger(),
      "Continuous-joint preparation skipped: joint limits unavailable (%s)",
      e.what());
    return true;
  }
  const std::size_t dof = limits.joint_names.size();
  // request.start/joint_target are in the NODE's cspace order (joint_names_),
  // which is normally the same as the bridge's. Map by name so a reordered
  // bridge cannot apply a joint's shift to the wrong column.
  if (request.start.size() != joint_names_.size() ||
    joint_names_.size() != dof || limits.positions.data.size() < 2 * dof)
  {
    return true;
  }
  std::vector<double> measured;
  if (current_pose_provider_) {
    try {
      const auto pose = current_pose_provider_();
      measured.assign(pose.begin(), pose.end());
    } catch (const std::exception &) {
      measured.clear();
    }
  }
  const bool have_measured = measured.size() == dof;
  // MoveIt-compatible start-state tolerance. A plan whose requested start is
  // physically farther than this from the measured state is still solved (the
  // planner is also used for compute-only previews), but it will not execute.
  const double start_state_max_bounds_error =
    node_->has_parameter("start_state_max_bounds_error") ?
    node_->get_parameter("start_state_max_bounds_error").as_double() : 0.1;

  request.window_shift.assign(dof, 0.0);
  for (std::size_t c = 0; c < joint_names_.size(); ++c) {
    const std::string & name = joint_names_[c];
    const auto found =
      std::find(limits.joint_names.begin(), limits.joint_names.end(), name);
    if (found == limits.joint_names.end()) {
      continue;
    }
    const std::size_t j =
      static_cast<std::size_t>(std::distance(limits.joint_names.begin(), found));
    const double lo = limits.positions.data[j];
    const double hi = limits.positions.data[dof + j];
    // Explicit joint type first; a legacy bridge that reports no mask falls
    // back to the range width. Once the mask is known, it is authoritative.
    bool continuous = is_continuous_joint(continuous_joint_names_, name);
    if (j < limits.continuous.size()) {
      continuous = continuous || limits.continuous[j] != 0;
    } else if (!continuous) {
      continuous = is_continuous_range(lo, hi);
    }
    if (!continuous) {
      continue;
    }
    if (!std::isfinite(lo) || !std::isfinite(hi)) {
      throw std::runtime_error(
        "continuous joint '" + name + "' has non-finite limits; cannot build a chart");
    }

    const double requested_start = request.start[c];
    double start_lift = requested_start;
    if (have_measured) {
      // Same physical pose on the branch the robot actually reports. This is
      // used for EXECUTION continuity; it never changes the requested pose.
      start_lift = periodic_lift(start_lift, measured[c]);
      if (std::fabs(wrap_pi(start_lift - measured[c])) >
        start_state_max_bounds_error)
      {
        RCLCPP_WARN_THROTTLE(
          node_->get_logger(), *node_->get_clock(), 5000,
          "Requested start for continuous joint %s differs physically from the "
          "measured state by %.3f rad (start_state_max_bounds_error=%.3f); this "
          "plan is compute-only and will not match the robot at execution",
          name.c_str(), wrap_pi(start_lift - measured[c]),
          start_state_max_bounds_error);
      }
    }
    double target_lift = start_lift;
    const bool has_target = !request.joint_target.empty() &&
      request.joint_target.size() == dof;
    const double requested_target =
      has_target ? static_cast<double>(request.joint_target[c]) : start_lift;
    if (has_target) {
      // A single segment is a shortest-path move; multi-turn motion must be
      // requested as explicit intermediate waypoints.
      target_lift = start_lift + wrap_pi(requested_target - start_lift);
    }

    double shift = 0.0;
    if (!choose_window_shift(start_lift, target_lift, lo, hi, shift)) {
      throw std::runtime_error(
        "continuous joint '" + name + "' has a window narrower than one turn "
        "([" + std::to_string(lo) + ", " + std::to_string(hi) + "])");
    }
    request.window_shift[c] = shift;
    request.start[c] = static_cast<float>(start_lift + shift);
    if (has_target) {
      request.joint_target[c] = static_cast<float>(target_lift + shift);
    }
    if (std::fabs(start_lift - requested_start) > 1e-6 ||
      (has_target && std::fabs(target_lift - requested_target) > 1e-6))
    {
      RCLCPP_INFO(
        node_->get_logger(),
        "Continuous joint %s: requested start %.6f target %s -> shortest-path "
        "start %.6f target %s (window shift %+.6f rad)",
        name.c_str(), requested_start,
        has_target ? std::to_string(requested_target).c_str() : "(none)",
        start_lift, has_target ? std::to_string(target_lift).c_str() : "(none)",
        shift);
    }
  }
  return true;
}

// PORT: unified_planner_node._plan_trajectory_goal request assembly
curobo_runtime::PlanRequest PlannerServices::build_request(
  const curobov2_ros_interfaces::msg::TrajectoryGoal & goal) const
{
  curobo_runtime::PlanRequest plan_request;
  plan_request.start = resolve_start(goal.start_pose.name, goal.start_pose.position);
  // PORT: config max_attempts / enable_graph_attempt (planner config).
  plan_request.max_attempts = node_->get_parameter("max_attempts").as_int();
  // PORT: config enable_graph_attempt (Python: `config.get("enable_graph_attempt", 1)`).
  // cuRobo consults the PRM graph planner on attempts where
  // `current_attempt >= enable_graph_attempt`, so at 1 with max_attempts=1 it is
  // not consulted -- the reference behaviour.
  plan_request.enable_graph_attempt =
    node_->get_parameter("enable_graph_attempt").as_int();
  plan_request.cartesian_step_size = goal.cartesian_step_size;
  plan_request.cartesian_precision = goal.cartesian_precision;

  const bool waypoint_path = !goal.goalsets.empty() &&
    (!goal.goalsets.front().waypoints.empty() ||
    !goal.goalsets.front().tool_waypoints.empty());
  const bool joint_space = !goal.goalsets.empty() &&
    !goal.goalsets.front().target_joint_positions.empty();
  if (waypoint_path) {
    const auto & g = goal.goalsets.front();
    const auto append_pose = [](curobo_runtime::TensorF32 & tensor,
        const geometry_msgs::msg::Pose & pose) {
        tensor.data.push_back(static_cast<float>(pose.position.x));
        tensor.data.push_back(static_cast<float>(pose.position.y));
        tensor.data.push_back(static_cast<float>(pose.position.z));
        tensor.data.push_back(static_cast<float>(pose.orientation.w));
        tensor.data.push_back(static_cast<float>(pose.orientation.x));
        tensor.data.push_back(static_cast<float>(pose.orientation.y));
        tensor.data.push_back(static_cast<float>(pose.orientation.z));
      };
    // One ordered path, one goalset. `poses` keeps its candidate semantics for
    // every legacy request; a waypoint path simply does not use it.
    plan_request.goalset_count = 1;
    plan_request.tool_frames.assign(g.tool_frames.begin(), g.tool_frames.end());
    if (!g.tool_trajectory_constraints.empty()) {
      plan_request.tool_trajectory_constraints.assign(
        g.tool_trajectory_constraints.begin(), g.tool_trajectory_constraints.end());
    }
    if (!g.waypoints.empty()) {
      plan_request.waypoints.shape = {
        static_cast<std::int64_t>(g.waypoints.size()), 7};
      for (const auto & pose : g.waypoints) {
        append_pose(plan_request.waypoints, pose);
      }
    }
    if (!g.tool_waypoints.empty()) {
      plan_request.tool_waypoints.shape = {
        static_cast<std::int64_t>(g.tool_waypoints.size()), 7};
      for (const auto & pose : g.tool_waypoints) {
        append_pose(plan_request.tool_waypoints, pose);
      }
    }
    // K: explicit waypoint_count when the client/front-end supplied it, else
    // derived from the array layout (the validator already checked divisibility).
    std::int64_t count = static_cast<std::int64_t>(g.waypoint_count);
    if (count <= 0) {
      if (!g.waypoints.empty()) {
        count = static_cast<std::int64_t>(g.waypoints.size());
      } else if (!g.tool_frames.empty()) {
        count = static_cast<std::int64_t>(
          g.tool_waypoints.size() / g.tool_frames.size());
      }
    }
    plan_request.waypoint_count = count;
    plan_request.path_position_tolerance = g.path_position_tolerance;
    plan_request.path_orientation_tolerance = g.path_orientation_tolerance;
    plan_request.final_position_tolerance = g.final_position_tolerance;
    plan_request.final_orientation_tolerance = g.final_orientation_tolerance;
    plan_request.track_orientation = g.track_orientation;
    plan_request.outer_iterations = g.outer_iterations;
  } else if (joint_space) {
    for (double v : goal.goalsets.front().target_joint_positions) {
      plan_request.joint_target.push_back(static_cast<float>(v));
    }
  } else if (!goal.goalsets.empty() && !goal.goalsets.front().tool_frames.empty()) {
    const auto & g = goal.goalsets.front();
    // One goalset, N simultaneous per-tool targets.  The Cartesian planner's
    // is_cartesian_request checks goalset_count, and the Classic planner uses
    // this count only for the single-tool candidate path; setting it to 1 is
    // what lets a multi-tool CartesianPath reach the multi-tool solve.
    plan_request.goalset_count = 1;
    plan_request.tool_frames.assign(g.tool_frames.begin(), g.tool_frames.end());
    plan_request.tool_poses.shape = {
      static_cast<std::int64_t>(g.tool_poses.size()), 7};
    for (const auto & pose : g.tool_poses) {
      plan_request.tool_poses.data.push_back(static_cast<float>(pose.position.x));
      plan_request.tool_poses.data.push_back(static_cast<float>(pose.position.y));
      plan_request.tool_poses.data.push_back(static_cast<float>(pose.position.z));
      plan_request.tool_poses.data.push_back(static_cast<float>(pose.orientation.w));
      plan_request.tool_poses.data.push_back(static_cast<float>(pose.orientation.x));
      plan_request.tool_poses.data.push_back(static_cast<float>(pose.orientation.y));
      plan_request.tool_poses.data.push_back(static_cast<float>(pose.orientation.z));
    }
    // Per-tool Cartesian holds (flat 6 * N, parallel to tool_frames). Empty
    // means legacy: the single trajectory_constraints vector below applies to
    // the first named tool.
    plan_request.tool_trajectory_constraints.assign(
      g.tool_trajectory_constraints.begin(), g.tool_trajectory_constraints.end());
    // PORT: single_planner._apply_pose_constraints - the FIRST non-empty
    // Goalset.trajectory_constraints holds its axes along the whole path.
    for (const auto & gs : goal.goalsets) {
      if (!gs.trajectory_constraints.empty()) {
        plan_request.trajectory_constraints.assign(
          gs.trajectory_constraints.begin(), gs.trajectory_constraints.end());
        break;
      }
    }
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
  // Apply the continuous-joint chart to the assembled request, whatever kind
  // of goal it is: a pose goal still needs its start on a valid window branch,
  // and its result still has to be shifted back.
  prepare_continuous_joints(plan_request);
  return plan_request;
}

// PORT: single_planner._publish_plan_image
void PlannerServices::publish_plan_image(
  const curobo_runtime::PlanOutcome & outcome,
  const std::vector<double> & window_shift)
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
    // Plot the executed (unshifted) positions, not the shifted planner window.
    curobo_runtime::TensorF32 plotted = outcome.positions;
    const std::int64_t cols =
      plotted.shape.size() > 1 ? plotted.shape.back() : 0;
    if (!window_shift.empty() && cols > 0) {
      const std::size_t shift_count =
        std::min(window_shift.size(), static_cast<std::size_t>(cols));
      for (std::size_t i = 0; i + shift_count <= plotted.data.size(); i += cols) {
        for (std::size_t j = 0; j < shift_count; ++j) {
          plotted.data[i + j] -= static_cast<float>(window_shift[j]);
        }
      }
    }
    const auto img = bridge_->render_plan_image(
      plotted, outcome.velocity, outcome.acceleration,
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

// Copy the request's non-cspace joints (locked fingers) onto every planned
// waypoint. cuRobo's solver returns the lock value it planned with, which is
// the task's pre-command state, not the state the task will be in when this
// segment runs. The task constructor sends the segment's start pose with the
// commanded jaw value; use it for both the returned trajectory and the ghost
// preview so the two previews agree with the task chain.
void apply_start_locks(
  std::vector<sensor_msgs::msg::JointState> & trajectory,
  const sensor_msgs::msg::JointState & start,
  const std::vector<std::string> & cspace)
{
  if (start.name.empty() || start.position.empty()) {
    return;
  }
  std::vector<std::pair<std::string, double>> locks;
  const std::size_t n = std::min(start.name.size(), start.position.size());
  for (std::size_t i = 0; i < n; ++i) {
    if (std::find(cspace.begin(), cspace.end(), start.name[i]) == cspace.end()) {
      locks.emplace_back(start.name[i], start.position[i]);
    }
  }
  if (locks.empty()) {
    return;
  }
  for (auto & point : trajectory) {
    if (point.name.empty()) {
      point.name = cspace;
    }
    for (const auto & lock : locks) {
      const auto it = std::find(point.name.begin(), point.name.end(), lock.first);
      if (it == point.name.end()) {
        point.name.push_back(lock.first);
        point.position.push_back(lock.second);
        continue;
      }
      const std::size_t index = static_cast<std::size_t>(it - point.name.begin());
      if (index < point.position.size()) {
        point.position[index] = lock.second;
      } else {
        point.position.push_back(lock.second);
      }
    }
  }
}

// PORT: single_planner._stage_trajectory - project + buffer a trajectory into
// the RobotContext's command buffer. plan() calls this so a compute-only plan
// is rendered for preview (GhostStrategy) and can be streamed later; it does
// not execute anything.
void PlannerServices::stage_preview(
  const std::vector<sensor_msgs::msg::JointState> & trajectory, double plan_dt)
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
    robot_context_->set_command(names, velocity, acceleration, position, true, plan_dt);
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

  // Make the collision model use this leg's commanded non-cspace joints (the
  // gripper jaws) before solving. Otherwise a pre-grasp leg planned right after
  // "open gripper" can still collide against the previous closed jaw state.
  if (start_lock_callback_ && !resolved_goal.start_pose.name.empty()) {
    try {
      start_lock_callback_(resolved_goal.start_pose);
    } catch (const std::exception & e) {
      RCLCPP_WARN_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 5000,
        "Could not apply start-pose lock joints: %s", e.what());
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

  // The inverse routing: a pose/tool-frame request cannot be solved by the
  // joint-space planner (it would call plan_cspace with an empty q). Another
  // client (task constructor, Joints tab, a previous request) may have left
  // the server on Joint Space; solve THIS pose request with Classic instead of
  // leaving the planner in a mode that can only fail.
  if (request.joint_target.empty() && planner->is_open_loop() &&
    planner->get_planner_name().find("Joint Space") != std::string::npos)
  {
    TrajectoryPlanner * classic = manager_.get_planner("classic");
    if (classic != nullptr) {
      RCLCPP_INFO(
        node_->get_logger(),
        "Pose goal: using %s (was %s) for this request",
        classic->get_planner_name().c_str(), planner->get_planner_name().c_str());
      effective = classic;
    }
  }

  // PORT: single_planner._apply_pose_constraints' "holding axes along path"
  // log (ToolPoseCriteria order is [x, y, z, roll, pitch, yaw]).
  //
  // Always print the hold vector for the Cartesian planner, including the
  // all-zero free-space case.  A Cartesian leg that unexpectedly holds no
  // axis is exactly the failure this line needs to make visible; the old
  // `any_hold` guard hid it.
  const bool is_cartesian =
    effective->get_planner_name().find("Cartesian") != std::string::npos &&
    effective->get_planner_name().find("Waypoint") == std::string::npos;
  if (request.trajectory_constraints.size() == 6) {
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
  } else if (is_cartesian) {
    RCLCPP_INFO(
      node_->get_logger(),
      "%s: holding axes along path (x,y,z,roll,pitch,yaw)=[0, 0, 0, 0, 0, 0] (no constraints)",
      effective->get_planner_name().c_str());
  }

  // Per-tool holds (multi-tool CartesianPath): one [x,y,z,roll,pitch,yaw]
  // group per named tool, in `tool_frames` order.
  if (!request.tool_trajectory_constraints.empty() &&
    request.tool_trajectory_constraints.size() == 6 * request.tool_frames.size())
  {
    const auto & ttc = request.tool_trajectory_constraints;
    std::ostringstream holds;
    for (std::size_t i = 0; i < request.tool_frames.size(); ++i) {
      const auto * group = &ttc[6 * i];
      const double tx = group[0] == 1 ? 1.0 : 0.0;
      const double ty = group[1] == 1 ? 1.0 : 0.0;
      const double tz = group[2] == 1 ? 1.0 : 0.0;
      const double x = group[3] == 1 ? 1.0 : 0.0;
      const double y = group[4] == 1 ? 1.0 : 0.0;
      const double z = group[5] == 1 ? 1.0 : 0.0;
      holds << (i ? " " : "") << "[" << request.tool_frames[i] << "]=["
            << x << ", " << y << ", " << z << ", "
            << tx << ", " << ty << ", " << tz << "]";
    }
    RCLCPP_INFO(
      node_->get_logger(), "%s: holding axes per tool %s",
      effective->get_planner_name().c_str(), holds.str().c_str());
  }

  // PORT: unified_planner_node "Planning with {planner.get_planner_name()}".
  RCLCPP_INFO(
    node_->get_logger(), "Planning with %s", effective->get_planner_name().c_str());

  // PORT: refresh_perception_world() - refresh the perception-based collision
  // world before planning so the plan accounts for the latest camera data.
  const auto t_setup = std::chrono::steady_clock::now();
  bool world_ok = true;
  if (!node_->has_parameter("push_esdf_to_solvers") ||
    node_->get_parameter("push_esdf_to_solvers").as_bool())
  {
    try {
      // PORT: refresh_perception_world() = refresh_esdf() + update_all_solvers_
      // world(); the push is what makes the planner collide against the
      // camera-observed voxels (otherwise the published grid shows unchecked
      // voxels) and both run under the exclusive barrier (the Python holds
      // gpu_lock around both; refresh_esdf mutates the shared scene).
      //
      // force=true: a discrete plan must see the LATEST perception, not a
      // throttled snapshot. The old force=false path skipped a refresh whenever a
      // depth frame had arrived within esdf_refresh_min_interval of the last push,
      // so the solver solved against a world older than the wall it was looking
      // at. The throttle only exists to spare the servo loop; it must not gate a
      // one-shot plan.
      if (obstacle_manager_->should_refresh_world(true)) {
        auto guard = curobo_runtime::Runtime::instance().barrier().exclusive();
        if (obstacle_manager_->refresh_esdf()) {
          if (bridge_->world_push(handle_)) {
            obstacle_manager_->mark_world_pushed();
          } else {
            // The solver did not receive the perception world (camera voxels may
            // be missing). Never plan against it.
            obstacle_manager_->invalidate_world();
            world_ok = false;
          }
        }
      }
    } catch (const std::exception & e) {
      // A failed perception refresh is NOT ignorable: planning with it risks
      // driving straight through an obstacle the sensors can see.
      obstacle_manager_->invalidate_world();
      world_ok = false;
      RCLCPP_ERROR_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 5000,
        "Perception world refresh failed (%s): refusing to plan against an "
        "unverified collision world", e.what());
    }
  }
  if (!world_ok) {
    out.success = false;
    out.message =
      "Planning refused: the collision world could not be refreshed from "
      "perception (camera voxels may be missing)";
    out.selected_goal_index.push_back(-1);
    out.selected_seed_index.push_back(-1);
    out.stats.problems = 1;
    out.stats.waypoints_planned = static_cast<std::uint32_t>(goal.goalsets.size());
    // Throttled: the freshness gate keeps retrying while the push fails, so this
    // fires once per plan. Report the condition, do not flood the log.
    RCLCPP_ERROR_THROTTLE(
      node_->get_logger(), *node_->get_clock(), 5000,
      "Planning failed: %s", out.message.c_str());
    return;
  }
  const auto t_world = std::chrono::steady_clock::now();

  // cuRobo captures ONE set of optimizer CUDA graphs and replays them for BOTH
  // solve modes. A pose<->cspace switch changes the goal-buffer structure under
  // the captured graph, and the stale replay dies with a CUDA illegal memory
  // access (the `plan_cspace` crash). Make the solver's graphs match this mode
  // before solving, under the exclusive barrier (capture is process-global).
  if (resolve_use_cuda_graph(node_))
  {
    const bool is_cspace =
      effective->get_planner_name().find("Joint Space") != std::string::npos;
    try {
      auto guard = curobo_runtime::Runtime::instance().barrier().exclusive();
      bridge_->prepare_solve_mode(handle_, is_cspace ? "cspace" : "pose");
    } catch (const std::exception & e) {
      RCLCPP_WARN_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 5000,
        "solve-mode CUDA-graph preparation failed (%s)", e.what());
    }
  }

  // PORT: unified_planner_node._plan_lock - hold gpu_lock (the exclusive GPU
  // barrier) for the WHOLE solve while CUDA graphs are on. cuRobo re-captures
  // optimizer graphs mid-plan (`reset_shape` out of `_prepare_goal_buffer`) and
  // replays the rest; either one concurrent with the camera/ESDF lane faults the
  // context. The port locked only the world refresh and the solve-mode prep, so
  // the first mid-plan re-capture ("operation failed due to a previous error
  // during capture" / "illegal memory access") killed the server.
  const bool plan_exclusive =
    resolve_use_cuda_graph(node_) &&
    (!node_->has_parameter("gpu_exclusive_plans") ||
    node_->get_parameter("gpu_exclusive_plans").as_bool());

  PlannerResult result;
  try {
    std::optional<curobo_runtime::CaptureBarrier::ExclusiveGuard> plan_guard;
    if (plan_exclusive) {
      plan_guard.emplace(curobo_runtime::Runtime::instance().barrier().exclusive());
    }
    // PORT: single_planner.plan -> reset_seed_per_plan runs INSIDE plan(), i.e.
    // under _plan_lock. It rewinds the solver RNG through GPU sampling kernels
    // (_dist.update_samples), so issuing it outside the barrier let it land in
    // another thread's capture window and invalidate the capture.
    if (node_->get_parameter("reset_seed_per_plan").as_bool()) {
      bridge_->reset_seed(handle_);
    }
    result = effective->plan(request);
  } catch (const std::exception & e) {
    // A CUDA fatal error must not be reported as an ordinary "planning failed"
    // and then left to corrupt every later request: the context is dead.
    handle_gpu_fault("planning", e.what());
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
    // The solver may have caught the fault and returned it as a message instead
    // of throwing; catch that too so a corrupt context still stops the node.
    handle_gpu_fault("planning", result.message);
    // PORT: _build_collision_diagnostics - attributed collisions at the start
    // (always) and goal (joint-space segments only) configurations. The Python
    // ran this inside the plan-lane job (barrier held); it is FK + sphere
    // distance + a host sync, so it needs the barrier here too.
    curobo_runtime::CollisionAttribution start_collisions;
    curobo_runtime::CollisionAttribution goal_collisions;
    {
      auto attribution_guard =
        curobo_runtime::Runtime::instance().barrier().exclusive();
      start_collisions = collision_attribution_at(bridge_, handle_, request.start, true);
      goal_collisions = collision_attribution_at(
        bridge_, handle_, request.joint_target, !request.joint_target.empty());
    }
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
      // Remove the whole-turn window shift. It is a per-joint constant, so the
      // physical path (FK/collision) is unchanged; the first point lands on
      // the measured branch and no per-point rewrite happens after validation.
      const double shift = static_cast<std::size_t>(j) < request.window_shift.size() ?
        request.window_shift[static_cast<std::size_t>(j)] : 0.0;
      js.position.push_back(
        outcome.positions.data[static_cast<std::size_t>(t * dof + j)] - shift);
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
  // The native planner samples the TIME-OPTIMIZED executable trajectory on a
  // uniform grid whose interval is generally NOT interpolation_dt: a short
  // joint move can be sampled at ~13 ms while a long one is ~28 ms. The
  // velocity/acceleration feed-forward is consistent with THAT interval, so
  // report it instead of the old hardcoded 25 ms. A consumer that re-stamps
  // the points at 25 ms would otherwise ask for up to 2x the planned speed.
  const double fallback_dt =
    node_->has_parameter("interpolation_dt") ?
    node_->get_parameter("interpolation_dt").as_double() : 0.025;
  double sample_dt = fallback_dt;
  if (rows > 1 && std::isfinite(outcome.motion_time_s) && outcome.motion_time_s > 0.0) {
    const double candidate =
      outcome.motion_time_s / static_cast<double>(rows - 1);
    if (std::isfinite(candidate) && candidate > 1.0e-5 && candidate < 10.0) {
      sample_dt = candidate;
    }
  }
  out.dt = sample_dt;
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
  apply_start_locks(out.trajectory, resolved_goal.start_pose, joint_names_);
  stage_preview(out.trajectory, out.dt);
  publish_plan_image(outcome, request.window_shift);
  out.stats.solver_time_s = outcome.solver_time_s;
  out.stats.position_error_mm = outcome.position_error_mm;
  out.stats.motion_time_s = outcome.motion_time_s;
  append_considered_rows(outcome, out.stats.considered);
  for (std::int16_t status : outcome.waypoint_status) {
    out.waypoint_status.push_back(status);
  }
}

// PORT: gpu_lanes.py:GpuLane._run_job's fatal path, applied to the plan service
// results. A CUDA fatal error (illegal memory access, device-side assert, ...)
// leaves the process's single context permanently broken: every later CUDA call
// returns garbage or fails. `fill_result` catches those errors and turns them
// into a "Planning failed" response, which is exactly what kept the node limping
// after the crash. Detecting the marker here lets `exit_on_gpu_fault` do its job.
bool PlannerServices::handle_gpu_fault(
  const std::string & context, const std::string & message)
{
  if (!curobo_runtime::is_cuda_fatal_message(message)) {
    return false;
  }
  RCLCPP_FATAL(
    node_->get_logger(),
    "CUDA fatal error while %s: %s\n"
    "The GPU context is corrupt; no later plan can be trusted.",
    context.c_str(), message.c_str());

  bool exit_on_fault = true;
  if (node_->has_parameter("exit_on_gpu_fault")) {
    exit_on_fault = node_->get_parameter("exit_on_gpu_fault").as_bool();
  }
  if (exit_on_fault) {
    RCLCPP_FATAL(
      node_->get_logger(),
      "exit_on_gpu_fault=true: stopping instead of serving a corrupt context");
    rclcpp::shutdown();
    // Do NOT run destructors: tearing down (and especially synchronizing) a dead
    // CUDA context can block forever. Stop the process now.
    std::fflush(nullptr);
    std::_Exit(EXIT_FAILURE);
  }
  return true;
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
            if (bridge_->world_push(handle_)) {
              obstacle_manager_->mark_world_pushed();
            } else {
              // Keep retrying: a blind reactive controller must not be treated
              // as fresh.
              obstacle_manager_->invalidate_world();
            }
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


namespace
{
// PORT: unified_planner_node._plan_trajectory_goal -> lane_plan.call - run the
// plan body on the plan lane (PRIO_PLAN); inline when there is no lane or when
// this thread already is the lane worker.
template<class F>
auto on_plan_lane(curobo_runtime::GpuLane * lane, F && fn) -> decltype(fn())
{
  if (lane == nullptr || lane->owns_current_thread()) {
    return fn();
  }
  curobo_runtime::SubmitOptions opts;
  opts.prio = curobo_runtime::PRIO_PLAN;
  return lane->call(opts, std::forward<F>(fn));
}
}  // namespace

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
  on_plan_lane(
    lane_plan_, [&]() {fill_result(planner, request->request, response->response);});
  // Surface the same failure diagnostics at the service's top level. Clients
  // that only look at the outer response still get the start/goal collision
  // attribution when a plan they expected to succeed failed.
  response->start_state_collisions = response->response.start_state_collisions;
  response->end_state_collisions = response->response.end_state_collisions;
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
    on_plan_lane(lane_plan_, [&]() {
        fill_result(planner, request->requests[i], response->responses[i]);
      });
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

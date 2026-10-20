// SPDX-License-Identifier: Apache-2.0
// PORT: unified_planner_node.py set_planner/get_planners/generate_trajectory
// callbacks, delegating to planners/planner_factory.PlannerManager.
#ifndef CUROBO_CORE__CORE__PLANNER_SERVICES_HPP_
#define CUROBO_CORE__CORE__PLANNER_SERVICES_HPP_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include "curobov2_ros_interfaces/srv/get_planners.hpp"
#include "curobov2_ros_interfaces/srv/set_planner.hpp"
#include "curobov2_ros_interfaces/srv/trajectory_generation.hpp"
#include "curobov2_ros_interfaces/srv/trajectory_generation_batch.hpp"
#include "curobo_core/planners/planner_factory.hpp"
#include "curobo_runtime/bridge.hpp"

namespace curobo_core
{

class ObstacleManager;
class RobotContext;

class PlannerServices
{
public:
  PlannerServices(
    rclcpp::Node * node, curobo_runtime::IBridge * bridge,
    curobo_runtime::RobotHandle handle, std::vector<std::string> joint_names,
    ObstacleManager * obstacle_manager = nullptr);

  PlannerManager & manager() {return manager_;}

  // PORT: joint_order.resolve_start_pose's `current` argument - the robot's
  // current joint pose, padded into a partial/empty start request.
  void set_current_pose_provider(std::function<std::vector<float>()> provider)
  {current_pose_provider_ = std::move(provider);}

  // PORT: frame_utils.resolve_poses_to_base - the node owns the TF buffer, so
  // the resolver is injected (server owns every transform: a goalset arrives
  // tagged with its frame and non-base goalsets are resolved into base).
  void set_pose_resolver(
    std::function<std::vector<geometry_msgs::msg::Pose>(
      const std::vector<geometry_msgs::msg::Pose> &, const std::string &)> resolver)
  {pose_resolver_ = std::move(resolver);}

  // PORT: single_planner._publish_plan_image (latched, once per new plan).
  void publish_plan_image(const curobo_runtime::PlanOutcome & outcome);

  // PORT: single_planner._stage_trajectory - feed a successful plan to the
  // RobotContext so GhostStrategy publishes it on <node>/trajectory for
  // preview. This is what makes the Joints tab's compute-only "Plan" visible
  // in RViz; it never executes the trajectory.
  void stage_preview(const std::vector<sensor_msgs::msg::JointState> & trajectory);

  void set_robot_context(RobotContext * robot_context) {robot_context_ = robot_context;}

private:
  void handle_set_planner(
    const std::shared_ptr<curobov2_ros_interfaces::srv::SetPlanner::Request> request,
    std::shared_ptr<curobov2_ros_interfaces::srv::SetPlanner::Response> response);
  void handle_get_planners(
    const std::shared_ptr<curobov2_ros_interfaces::srv::GetPlanners::Request> request,
    std::shared_ptr<curobov2_ros_interfaces::srv::GetPlanners::Response> response);
  void handle_generate_trajectory(
    const std::shared_ptr<curobov2_ros_interfaces::srv::TrajectoryGeneration::Request> request,
    std::shared_ptr<curobov2_ros_interfaces::srv::TrajectoryGeneration::Response> response);
  void handle_generate_trajectory_batch(
    const std::shared_ptr<curobov2_ros_interfaces::srv::TrajectoryGenerationBatch::Request> request,
    std::shared_ptr<curobov2_ros_interfaces::srv::TrajectoryGenerationBatch::Response> response);

  // Shared single-problem solve + result assembly (used by the single and batch
  // surfaces; PORT: unified_planner._plan_trajectory_goal / _rollup_stats).
  void fill_result(
    TrajectoryPlanner * planner, const curobov2_ros_interfaces::msg::TrajectoryGoal & goal,
    curobov2_ros_interfaces::msg::TrajectoryResult & out);
  curobo_runtime::PlanRequest build_request(
    const curobov2_ros_interfaces::msg::TrajectoryGoal & goal) const;

  // PORT: gpu_lanes.py:GpuLane._run_job's fatal path. `fill_result` catches the
  // planner exception and returns it as a service result string, so a CUDA fatal
  // error (illegal memory access / device-side assert) never reaches
  // GpuLane::run_job's poison logic and the node keeps serving a context that can
  // only return garbage. This detects the marker and, when `exit_on_gpu_fault` is
  // set, stops the process instead of limping. Returns true when a fault was seen.
  bool handle_gpu_fault(const std::string & context, const std::string & message);

  // Continuous-joint target normalisation. A joint whose reported position
  // range spans a full turn (>= 2*pi) is continuous; cuRobo plans in the
  // NUMERIC joint space and never wraps, so a joint-space target written as
  // the "other" turn — in range but ~2*pi away from the start (e.g. -4.55
  // instead of +1.73) — asks for a full sweep the solver usually cannot find,
  // and a value just outside the range (e.g. 6.36) is rejected outright. Snap
  // each such target to the in-range full-turn equivalent NEAREST the start,
  // so a negative / un-wrapped target plans exactly like its wrapped twin
  // (the requested and snapped values are the same physical configuration).
  void wrap_full_turn_targets(curobo_runtime::PlanRequest & request) const;

  // PORT: joint_order.resolve_start_pose (name reorder, positional fallback).
  std::vector<float> resolve_start(
    const std::vector<std::string> & names, const std::vector<double> & positions) const;

  rclcpp::Node * node_;
  // PORT: unified_planner_node.lane_plan - EVERY plan is serialized on the 'plan'
  // lane, the sole owner of the shared MotionPlanner and its CUDA graphs. Without
  // it, overlapping requests (an MTC fan-out, a batch, an orchestrator retry)
  // race the solver's pose criteria / seed RNG / reset_shape and fail
  // intermittently. Recursive: the batch surface calls the single-problem path.
  std::recursive_mutex plan_lane_mutex_;
  std::function<std::vector<geometry_msgs::msg::Pose>(
    const std::vector<geometry_msgs::msg::Pose> &, const std::string &)> pose_resolver_;
  curobo_runtime::IBridge * bridge_;
  curobo_runtime::RobotHandle handle_;
  std::vector<std::string> joint_names_;
  ObstacleManager * obstacle_manager_;
  // PORT: single_planner._debug_img_pub (transient_local depth-1).
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr plan_image_pub_;
  std::function<std::vector<float>()> current_pose_provider_;
  RobotContext * robot_context_ = nullptr;
  PlannerManager manager_;

  rclcpp::Service<curobov2_ros_interfaces::srv::SetPlanner>::SharedPtr set_planner_srv_;
  rclcpp::Service<curobov2_ros_interfaces::srv::GetPlanners>::SharedPtr get_planners_srv_;
  rclcpp::Service<curobov2_ros_interfaces::srv::TrajectoryGeneration>::SharedPtr
    generate_trajectory_srv_;
  rclcpp::Service<curobov2_ros_interfaces::srv::TrajectoryGenerationBatch>::SharedPtr
    generate_trajectory_batch_srv_;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__CORE__PLANNER_SERVICES_HPP_

// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_cumotion_moveit/cumotion_interface.hpp"

#include "isaac_ros_cumotion_moveit/cumotion_planner_ids.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "moveit/planning_interface/planning_interface.hpp"
#include "moveit/planning_scene/planning_scene.hpp"
#include "moveit/robot_state/conversions.hpp"
#include "builtin_interfaces/msg/duration.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"

namespace nvidia
{
namespace isaac
{
namespace manipulation
{

namespace
{

/**
 * Convert the curobo generate_trajectory response (a list of JointState
 * waypoints + a dt) into a moveit trajectory_msgs/JointTrajectory. Joint names
 * are taken from the first waypoint; each subsequent waypoint is stamped with
 * time_from_start = i * dt.
 */
trajectory_msgs::msg::JointTrajectory toJointTrajectory(
  const std::vector<sensor_msgs::msg::JointState> & waypoints, double dt)
{
  trajectory_msgs::msg::JointTrajectory traj;
  if (waypoints.empty()) {
    return traj;
  }
  // First waypoint supplies joint names (all waypoints share the same order).
  traj.joint_names = waypoints.front().name;
  double t = 0.0;
  for (const auto & wp : waypoints) {
    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions = wp.position;
    point.velocities = wp.velocity;
    point.time_from_start = builtin_interfaces::msg::Duration();
    point.time_from_start.sec = static_cast<int>(t);
    point.time_from_start.nanosec = static_cast<uint32_t>((t - static_cast<int>(t)) * 1e9);
    traj.points.push_back(point);
    t += dt;
  }
  return traj;
}

}  // namespace

void CumotionInterface::solve(
  const planning_scene::PlanningSceneConstPtr & planning_scene,
  const planning_interface::MotionPlanRequest & request,
  planning_interface::MotionPlanDetailedResponse & response)
{
  RCLCPP_INFO(node_->get_logger(), "Planning trajectory");

  // Mirror MoveIt planning-scene obstacles into cuRobo's collision scene so the
  // plan accounts for them. Best-effort: a sync failure is logged, not fatal.
  if (!planner_busy) {
    service_client_->syncPlanningScene(planning_scene);
  }

  // Resolve the group's joint ordering from the robot model.
  std::vector<std::string> dof_joint_names;
  if (planning_scene && planning_scene->getRobotModel()) {
    const auto model = planning_scene->getRobotModel();
    const auto * group = model->getJointModelGroup(request.group_name);
    if (group) {
      dof_joint_names = group->getActiveJointModelNames();
    }
  }

  // Build the cuRobo request and issue the plan.
  isaac_ros_cumotion_interfaces::srv::TrajectoryGeneration::Request req;
  isaac_ros_cumotion_interfaces::srv::TrajectoryGeneration::Response res;
  const auto plan_start = std::chrono::steady_clock::now();
  if (!service_client_->plan(request, dof_joint_names, req, res)) {
    RCLCPP_ERROR(node_->get_logger(), "Planning failed");
    response.error_code.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
    planner_busy = false;
    return;
  }

  if (!res.success) {
    RCLCPP_ERROR_STREAM(
      node_->get_logger(), "cuRobo planning failed: " << res.message);
    response.error_code.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
    planner_busy = false;
    return;
  }
  RCLCPP_INFO(node_->get_logger(), "Trajectory success!");

  response.description.push_back("cuRobo (curobo_ros generate_trajectory)");
  response.error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
  response.planner_id = request.planner_id.empty() ? kAutoPlannerId : request.planner_id;
  const auto plan_end = std::chrono::steady_clock::now();
  response.processing_time.push_back(
    std::chrono::duration<double>(plan_end - plan_start).count());

  if (res.trajectory.empty()) {
    RCLCPP_ERROR(node_->get_logger(), "cuRobo returned an empty trajectory");
    response.error_code.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
    planner_busy = false;
    return;
  }

  auto result_traj = std::make_shared<robot_trajectory::RobotTrajectory>(
    planning_scene->getRobotModel(), request.group_name);

  moveit::core::RobotState robot_state(planning_scene->getRobotModel());
  moveit::core::robotStateMsgToRobotState(request.start_state, robot_state);

  trajectory_msgs::msg::JointTrajectory jt = toJointTrajectory(res.trajectory, res.dt);
  result_traj->setRobotTrajectoryMsg(robot_state, jt);

  response.trajectory.clear();
  response.trajectory.push_back(result_traj);

  planner_busy = false;
}

}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia

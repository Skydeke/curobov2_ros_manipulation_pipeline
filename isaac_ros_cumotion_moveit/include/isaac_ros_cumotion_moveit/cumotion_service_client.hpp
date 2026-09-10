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

#ifndef ISAAC_ROS_CUMOTION_SERVICE_CLIENT_H
#define ISAAC_ROS_CUMOTION_SERVICE_CLIENT_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "moveit/planning_interface/planning_interface.hpp"
#include "moveit/planning_scene/planning_scene.hpp"
#include "rclcpp/rclcpp.hpp"

#include "isaac_ros_cumotion_interfaces/srv/add_object.hpp"
#include "isaac_ros_cumotion_interfaces/srv/remove_object.hpp"
#include "isaac_ros_cumotion_interfaces/srv/set_planner.hpp"
#include "isaac_ros_cumotion_interfaces/srv/trajectory_generation.hpp"

namespace nvidia
{
namespace isaac
{
namespace manipulation
{

/**
 * ROS 2 service client for the curobo_ros unified trajectory planner node.
 *
 * Replaces the retired `cumotion/move_group` MoveGroup action client. The new
 * node exposes planning through the `generate_trajectory` service plus a
 * `set_planner` service to chose the active planner (CLASSIC for pose goals,
 * JOINT_SPACE for joint goals), and the `add_object`/`remove_object` services
 * to mirror obstacles into its collision scene.
 *
 * The service namespace (node name, e.g. `curobo_server`) is configurable via
 * the `cumotion_service_namespace` ROS parameter.
 *
 * Every service round-trip is bounded by the `cumotion_service_timeout`
 * parameter (seconds, default 60). The old fixed 5s bound raced the first
 * plan's cold-start cost (CUDA graph re-record + kernel compile), making MoveIt
 * drop plans the server was actively producing.
 */
class CumotionServiceClient
{
public:
  explicit CumotionServiceClient(const rclcpp::Node::SharedPtr & node);

  /// Select the active planner by type (see SetPlanner.srv constants).
  bool setPlanner(uint8_t planner_type, std::string & message);

  /// Plan a single trajectory; fills @p req and @p res from the MoveIt request.
  bool plan(
    const planning_interface::MotionPlanRequest & request,
    const std::vector<std::string> & dof_joint_names,
    isaac_ros_cumotion_interfaces::srv::TrajectoryGeneration::Request & req,
    isaac_ros_cumotion_interfaces::srv::TrajectoryGeneration::Response & res);

  /// Mirror the planning-scene collision objects into cuRobo (add/update/remove).
  bool syncPlanningScene(const planning_scene::PlanningSceneConstPtr & planning_scene);

private:
  template<typename Srv>
  bool callService(
    const typename rclcpp::Client<Srv>::SharedPtr & client,
    const typename Srv::Request::SharedPtr & req,
    typename Srv::Response::SharedPtr & res);

  std::shared_ptr<rclcpp::Node> node_;
  std::string ns_;  // service namespace, e.g. "curobo_server"
  double service_timeout_secs_;  // per-call service timeout (cumotion_service_timeout)

  rclcpp::Client<isaac_ros_cumotion_interfaces::srv::SetPlanner>::SharedPtr set_planner_client_;
  rclcpp::Client<isaac_ros_cumotion_interfaces::srv::TrajectoryGeneration>::SharedPtr
    traj_gen_client_;
  rclcpp::Client<isaac_ros_cumotion_interfaces::srv::AddObject>::SharedPtr add_object_client_;
  rclcpp::Client<isaac_ros_cumotion_interfaces::srv::RemoveObject>::SharedPtr remove_object_client_;

  /// Names of objects currently mirrored into cuRobo (for stale-object removal).
  std::vector<std::string> synced_object_names_;

  std::mutex sync_mutex_;
};

}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia

#endif  // ISAAC_ROS_CUMOTION_SERVICE_CLIENT_H

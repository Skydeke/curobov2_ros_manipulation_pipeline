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
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "moveit/planning_interface/planning_interface.hpp"
#include "moveit/planning_scene/planning_scene.hpp"
#include "rclcpp/rclcpp.hpp"

#include "isaac_ros_cumotion_interfaces/srv/add_object.hpp"
#include "isaac_ros_cumotion_interfaces/srv/attach_object.hpp"
#include "isaac_ros_cumotion_interfaces/srv/remove_object.hpp"
#include "isaac_ros_cumotion_interfaces/srv/set_planner.hpp"
#include "isaac_ros_cumotion_interfaces/srv/trajectory_generation.hpp"
#include "std_srvs/srv/trigger.hpp"

namespace moveit
{
namespace planning_interface
{
class PlanningSceneInterface;
}
}  // namespace moveit

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

  /// Mirror the planning-scene collision objects (world + attached bodies) into
  /// cuRobo (add/update/remove), multi-shape and mesh aware.
  bool syncPlanningScene(const planning_scene::PlanningSceneConstPtr & planning_scene);

  /// Reverse direction of syncPlanningScene: pull the objects cuRobo currently
  /// knows about (anything added via /add_object, not just MoveIt mirrors) and
  /// mirror them into a MoveIt planning scene as CollisionObjects. Names that
  /// were never part of a forward sync have no cached geometry and are skipped
  /// with a one-time warning (Sec 6d of the task-constructor plan). @p
  /// planning_frame becomes the CollisionObjects' header.frame_id.
  bool syncObstaclesFromServer(
    moveit::planning_interface::PlanningSceneInterface & psi,
    const std::string & planning_frame);

private:
  template<typename Srv>
  bool callService(
    const typename rclcpp::Client<Srv>::SharedPtr & client,
    const typename Srv::Request::SharedPtr & req,
    typename Srv::Response::SharedPtr & res);

  /// Remove one curobo-side object (member name) and drop its cached state.
  void removeCuroboObject(const std::string & name);

  /// Parse the newline-separated obstacle-name list returned by get_obstacles.
  static std::vector<std::string> parseServerObjectNames(const std::string & message);

  std::shared_ptr<rclcpp::Node> node_;
  std::string ns_;  // service namespace, e.g. "curobo_server"
  double service_timeout_secs_;  // per-call service timeout (cumotion_service_timeout)

  rclcpp::Client<isaac_ros_cumotion_interfaces::srv::SetPlanner>::SharedPtr set_planner_client_;
  rclcpp::Client<isaac_ros_cumotion_interfaces::srv::TrajectoryGeneration>::SharedPtr
    traj_gen_client_;
  rclcpp::Client<isaac_ros_cumotion_interfaces::srv::AddObject>::SharedPtr add_object_client_;
  rclcpp::Client<isaac_ros_cumotion_interfaces::srv::RemoveObject>::SharedPtr remove_object_client_;
  rclcpp::Client<isaac_ros_cumotion_interfaces::srv::AttachObject>::SharedPtr attach_object_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr detach_object_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr get_obstacles_client_;

  /// Obstacles mirrored into cuRobo, keyed by their MoveIt object id. Values are
  /// the expanded curobo-side names (`{id}` for single-shape objects, `{id}__{i}`
  /// per shape for multi-shape ones), so stale-object removal and re-sync stay
  /// keyed per original object id (Sec 6a).
  std::map<std::string, std::vector<std::string>> synced_groups_;
  /// Per curobo-side name, a signature of the last-synced geometry + pose. Used
  /// to skip unchanged objects on repeated syncs (avoid a remove/re-add churn
  /// on every plan in a static scene).
  std::map<std::string, std::string> synced_member_signatures_;
  /// Names currently attached in cuRobo (diff-driven via attach/detach services,
  /// Sec 6c).
  std::vector<std::string> synced_attached_names_;

  /// Per curobo-side name, the AddObject request that last mirrored it. Forward
  /// sync populates this; the reverse sync (syncObstaclesFromServer) replays it
  /// into MoveIt as a CollisionObject (Sec 6d). Keyed by the exact name sent to
  /// add_object (i.e. the expanded member name).
  std::map<std::string, isaac_ros_cumotion_interfaces::srv::AddObject::Request>
    cached_geometry_;

  /// Names mirrored into MoveIt by the reverse sync (stale-removal tracking).
  std::vector<std::string> mirrored_in_moveit_names_;

  std::mutex sync_mutex_;
};

}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia

#endif  // ISAAC_ROS_CUMOTION_SERVICE_CLIENT_H
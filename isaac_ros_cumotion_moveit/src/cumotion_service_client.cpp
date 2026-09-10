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

#include "isaac_ros_cumotion_moveit/cumotion_service_client.hpp"

#include "isaac_ros_cumotion_moveit/cumotion_planner_ids.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

#include "geometric_shapes/shapes.h"
#include "geometry_msgs/msg/pose.hpp"
#include "moveit_msgs/msg/collision_object.hpp"
#include "moveit_msgs/msg/constraints.hpp"
#include "rclcpp/rclcpp.hpp"

namespace nvidia
{
namespace isaac
{
namespace manipulation
{

namespace
{

// Default bound on a single service round-trip (seconds). 5s was the old
// hard-coded value, but the FIRST generate_trajectory after a planner switch /
// solver rebuild re-records the ESDF CUDA graph and compiles kernels in
// addition to the planning itself (~5-6s cold, ~4s warm with max_attempts=10),
// so a fixed 5s cap raced legitimate cold-start latency and MoveIt dropped
// plans the server was actively producing. 60s is generous yet still bounded;
// override via the `cumotion_service_timeout` ROS parameter.
constexpr double kDefaultServiceTimeoutSeconds = 60.0;

}  // namespace

CumotionServiceClient::CumotionServiceClient(const rclcpp::Node::SharedPtr & node)
: node_(node)
{
  // Service namespace: the name of the curobo_ros planner node. Overridable via
  // a parameter so the plugin can target any launched host node. Guard against
  // re-declaration (the launch file / node may already have declared it).
  if (!node_->has_parameter("cumotion_service_namespace")) {
    node_->declare_parameter<std::string>("cumotion_service_namespace", "curobo_server");
  }
  ns_ = "/" + node_->get_parameter("cumotion_service_namespace").as_string();

  // Per-call timeout (seconds) for every service the client issues. Must cover
  // the cold-start cost of the first plan after a solver rebuild (CUDA graph
  // re-record + kernel compile + planning), not just warm replans — see the
  // constant above. Same re-declaration guard as the namespace parameter.
  if (!node_->has_parameter("cumotion_service_timeout")) {
    node_->declare_parameter<double>(
      "cumotion_service_timeout", kDefaultServiceTimeoutSeconds);
  }
  service_timeout_secs_ = node_->get_parameter("cumotion_service_timeout").as_double();

  set_planner_client_ = node_->create_client<isaac_ros_cumotion_interfaces::srv::SetPlanner>(
    ns_ + "/set_planner");
  traj_gen_client_ =
    node_->create_client<isaac_ros_cumotion_interfaces::srv::TrajectoryGeneration>(
      ns_ + "/generate_trajectory");
  add_object_client_ = node_->create_client<isaac_ros_cumotion_interfaces::srv::AddObject>(
    ns_ + "/add_object");
  remove_object_client_ = node_->create_client<isaac_ros_cumotion_interfaces::srv::RemoveObject>(
    ns_ + "/remove_object");
}

template<typename Srv>
bool CumotionServiceClient::callService(
  const typename rclcpp::Client<Srv>::SharedPtr & client,
  const typename Srv::Request::SharedPtr & req,
  typename Srv::Response::SharedPtr & res)
{
  auto timeout = std::chrono::duration<double>(service_timeout_secs_);
  if (!client->wait_for_service(timeout)) {
    RCLCPP_ERROR_STREAM(
      node_->get_logger(), "Service not available: " << client->get_service_name());
    return false;
  }
  auto future = client->async_send_request(req);
  if (future.wait_for(timeout) != std::future_status::ready)
  {
    RCLCPP_ERROR_STREAM(
      node_->get_logger(),
      "Service call timed out (" << service_timeout_secs_ << "s): "
        << client->get_service_name());
    return false;
  }
  res = future.get();
  return true;
}

bool CumotionServiceClient::setPlanner(uint8_t planner_type, std::string & message)
{
  auto req = std::make_shared<isaac_ros_cumotion_interfaces::srv::SetPlanner::Request>();
  req->planner_type = planner_type;

  isaac_ros_cumotion_interfaces::srv::SetPlanner::Response::SharedPtr res;
  if (!callService<isaac_ros_cumotion_interfaces::srv::SetPlanner>(
      set_planner_client_, req, res)) {
    message = "Failed to reach set_planner service";
    return false;
  }
  message = res->message;
  return res->success;
}

bool CumotionServiceClient::plan(
  const planning_interface::MotionPlanRequest & request,
  const std::vector<std::string> & dof_joint_names,
  isaac_ros_cumotion_interfaces::srv::TrajectoryGeneration::Request & req,
  isaac_ros_cumotion_interfaces::srv::TrajectoryGeneration::Response & res)
{
  // Only the first goal constraint is honoured (a single start->goal plan).
  if (request.goal_constraints.empty()) {
    RCLCPP_ERROR(node_->get_logger(), "No goal constraints in request");
    return false;
  }
  const auto & constraint = request.goal_constraints.front();

  // Honor the requested start state (e.g. chosen in RViz) instead of letting
  // cuRobo fall back to the robot's current position.
  if (!request.start_state.joint_state.name.empty() &&
    request.start_state.joint_state.name.size() == request.start_state.joint_state.position.size())
  {
    bool have_start = true;
    // Fill the start state in the group's DOF ordering.
    for (const std::string & name : dof_joint_names) {
      const auto & js = request.start_state.joint_state;
      const auto it = std::find(js.name.begin(), js.name.end(), name);
      if (it == js.name.end()) {
        have_start = false;
        break;
      }
      req.start_pose.name.push_back(name);
      req.start_pose.position.push_back(
        js.position[static_cast<std::size_t>(std::distance(js.name.begin(), it))]);
    }
    if (have_start && !req.start_pose.name.empty()) {
      req.start_pose.header = request.start_state.joint_state.header;
    } else {
      req.start_pose.name.clear();
      req.start_pose.position.clear();
    }
  }

  // Planner selection: an explicit planner_id names a concrete cuRobo planner
  // and takes precedence; empty "cuMotion" (auto) picks from the goal type.
  std::string planner_msg;
  uint8_t planner_type = 0;
  bool auto_planner = true;
  if (!request.planner_id.empty() && request.planner_id != kAutoPlannerId &&
    plannerIdToType(request.planner_id, planner_type))
  {
    auto_planner = false;
  }

  if (!constraint.joint_constraints.empty()) {
    // JOINT goal, with targets in the group's dof order.
    std::vector<double> joint_positions;
    for (const std::string & name : dof_joint_names) {
      for (const auto & jc : constraint.joint_constraints) {
        if (jc.joint_name == name) {
          joint_positions.push_back(jc.position);
          break;
        }
      }
    }
    if (joint_positions.empty()) {
      RCLCPP_ERROR(node_->get_logger(), "No joint targets matched the group DOF names");
      return false;
    }
    // A joint-space goal can only be planned by the JOINT_SPACE planner; the
    // pose planners (Classic/Multipoint) consume `goalsets`, not
    // target_joint_positions, and would silently plan toward an empty goal.
    // Force JointSpace regardless of the requested planner_id.
    if (!auto_planner) {
      RCLCPP_WARN(node_->get_logger(),
        "Joint-space goal requested planner '%s'; forcing Joint Space Motion Generation",
        request.planner_id.c_str());
    }
    planner_type = isaac_ros_cumotion_interfaces::srv::SetPlanner::Request::JOINT_SPACE;
    if (!setPlanner(planner_type, planner_msg)) {
      RCLCPP_ERROR_STREAM(node_->get_logger(), "setPlanner failed: " << planner_msg);
      return false;
    }
    req.target_joint_positions = std::move(joint_positions);
  } else if (!constraint.position_constraints.empty() || !constraint.orientation_constraints.empty()) {
    // POSE goal -> classic (Cartesian) planner by default.
    // Collect waypoints from the position constraints the way MoveIt's
    // poseFromConstraint does: the goal position lives in the constraint-region
    // primitive pose plus the target_point_offset. Each waypoint becomes a
    // single-pose goalset segment (one candidate); Classic uses the first one.
    const auto extract_pose = [](const auto & pc) {
      geometry_msgs::msg::Pose p;
      p.position.x = pc.target_point_offset.x;
      p.position.y = pc.target_point_offset.y;
      p.position.z = pc.target_point_offset.z;
      if (!pc.constraint_region.primitives.empty() && !pc.constraint_region.primitive_poses.empty()) {
        const auto & prim_pose = pc.constraint_region.primitive_poses[0];
        p.position.x += prim_pose.position.x;
        p.position.y += prim_pose.position.y;
        p.position.z += prim_pose.position.z;
        p.orientation = prim_pose.orientation;
      }
      return p;
    };
    std::vector<geometry_msgs::msg::Pose> waypoints;
    for (const auto & pc : constraint.position_constraints) {
      waypoints.push_back(extract_pose(pc));
    }
    if (waypoints.empty() && !constraint.orientation_constraints.empty()) {
      // Orientation-only goal.
      geometry_msgs::msg::Pose p;
      p.orientation = constraint.orientation_constraints.front().orientation;
      waypoints.push_back(p);
    }
    // Classic uses the first waypoint as its single target pose.
    geometry_msgs::msg::Pose pose = waypoints.empty() ? geometry_msgs::msg::Pose()
      : waypoints.front();
    if (!constraint.orientation_constraints.empty() &&
      pose.orientation == geometry_msgs::msg::Quaternion())
    {
      pose.orientation = constraint.orientation_constraints.front().orientation;
      if (!waypoints.empty()) {
        waypoints.front().orientation = pose.orientation;
      }
    }
    if (auto_planner) {
      planner_type = isaac_ros_cumotion_interfaces::srv::SetPlanner::Request::JOINT_SPACE;
    }
    RCLCPP_INFO_STREAM(node_->get_logger(),
      "Pose goal: link=" << (constraint.position_constraints.empty()
        ? std::string("(none)") : constraint.position_constraints.front().link_name)
      << " waypoints=" << waypoints.size()
      << " goal=[x=" << pose.position.x << " y=" << pose.position.y
      << " z=" << pose.position.z << "]");
    for (const auto & wp : waypoints) {
      isaac_ros_cumotion_interfaces::msg::Goalset gset;
      gset.poses.push_back(wp);
      req.goalsets.push_back(gset);
    }
    if (!setPlanner(planner_type, planner_msg)) {
      RCLCPP_ERROR_STREAM(node_->get_logger(), "setPlanner failed: " << planner_msg);
      return false;
    }
  } else {
    RCLCPP_ERROR(node_->get_logger(), "Unsupported goal constraint type");
    return false;
  }

  auto req_ptr =
    std::make_shared<isaac_ros_cumotion_interfaces::srv::TrajectoryGeneration::Request>();
  *req_ptr = req;
  isaac_ros_cumotion_interfaces::srv::TrajectoryGeneration::Response::SharedPtr res_ptr;
  if (!callService<isaac_ros_cumotion_interfaces::srv::TrajectoryGeneration>(
      traj_gen_client_, req_ptr, res_ptr)) {
    RCLCPP_ERROR(node_->get_logger(), "generate_trajectory service call failed");
    return false;
  }
  res = *res_ptr;
  return res.success;
}

bool CumotionServiceClient::syncPlanningScene(
  const planning_scene::PlanningSceneConstPtr & planning_scene)
{
  if (!planning_scene) {
    return false;
  }

  std::lock_guard<std::mutex> lk(sync_mutex_);

  std::vector<std::string> wanted_names;
  const auto & world = planning_scene->getWorld();
  wanted_names = world->getObjectIds();

  // Remove stale objects no longer in the planning scene.
  for (const std::string & name : synced_object_names_) {
    if (std::find(wanted_names.begin(), wanted_names.end(), name) == wanted_names.end()) {
      auto req = std::make_shared<isaac_ros_cumotion_interfaces::srv::RemoveObject::Request>();
      req->name = name;
      isaac_ros_cumotion_interfaces::srv::RemoveObject::Response::SharedPtr res;
      callService<isaac_ros_cumotion_interfaces::srv::RemoveObject>(
        remove_object_client_, req, res);
    }
  }

  // Add new / updated objects (first primitive only; meshes are not mirrored).
  std::vector<std::string> new_synced;
  for (const std::string & name : wanted_names) {
    const auto obj = world->getObject(name);
    if (!obj) {
      continue;
    }

    if (obj->shapes_.empty() || obj->global_shape_poses_.empty()) {
      RCLCPP_DEBUG_STREAM(
        node_->get_logger(), "Skipping non-primitive collision object '" << name << "'");
      continue;
    }

    auto req = std::make_shared<isaac_ros_cumotion_interfaces::srv::AddObject::Request>();
    req->name = name;
    req->color.r = 0.8f;
    req->color.g = 0.8f;
    req->color.b = 0.8f;
    req->color.a = 1.0f;

    // Use the first shape's pose in the world frame.
    const Eigen::Isometry3d & obj_pose = obj->global_shape_poses_.front();
    const Eigen::Vector3d t = obj_pose.translation();
    req->pose.position.x = t.x();
    req->pose.position.y = t.y();
    req->pose.position.z = t.z();
    const Eigen::Quaterniond q(obj_pose.linear());
    req->pose.orientation.x = q.x();
    req->pose.orientation.y = q.y();
    req->pose.orientation.z = q.z();
    req->pose.orientation.w = q.w();

    const shapes::ShapeConstPtr & shape = obj->shapes_.front();
    switch (shape->type) {
      case shapes::BOX: {
        const auto * box = static_cast<const shapes::Box *>(shape.get());
        req->type = isaac_ros_cumotion_interfaces::srv::AddObject::Request::CUBOID;
        req->dimensions.x = box->size[0];
        req->dimensions.y = box->size[1];
        req->dimensions.z = box->size[2];
        break;
      }
      case shapes::SPHERE: {
        const auto * sphere = static_cast<const shapes::Sphere *>(shape.get());
        req->type = isaac_ros_cumotion_interfaces::srv::AddObject::Request::SPHERE;
        req->dimensions.x = sphere->radius;
        req->dimensions.y = sphere->radius;
        req->dimensions.z = sphere->radius;
        break;
      }
      case shapes::CYLINDER: {
        const auto * cylinder = static_cast<const shapes::Cylinder *>(shape.get());
        req->type = isaac_ros_cumotion_interfaces::srv::AddObject::Request::CYLINDER;
        req->dimensions.x = cylinder->radius;
        req->dimensions.y = cylinder->length;
        req->dimensions.z = cylinder->length;
        break;
      }
      default:
        RCLCPP_WARN_STREAM(
          node_->get_logger(), "Unsupported primitive type for '" << name << "', skipping");
        continue;
    }

    // Same-name object already present must be removed first (add_object
    // rejects duplicates).
    if (std::find(synced_object_names_.begin(), synced_object_names_.end(), name) !=
      synced_object_names_.end())
    {
      auto rm = std::make_shared<isaac_ros_cumotion_interfaces::srv::RemoveObject::Request>();
      rm->name = name;
      isaac_ros_cumotion_interfaces::srv::RemoveObject::Response::SharedPtr rm_res;
      callService<isaac_ros_cumotion_interfaces::srv::RemoveObject>(
        remove_object_client_, rm, rm_res);
    }

    isaac_ros_cumotion_interfaces::srv::AddObject::Response::SharedPtr res;
    if (callService<isaac_ros_cumotion_interfaces::srv::AddObject>(
        add_object_client_, req, res) &&
      res->success) {
      new_synced.push_back(name);
    } else {
      RCLCPP_WARN_STREAM(node_->get_logger(), "add_object failed for '" << name << "'");
    }
  }

  synced_object_names_ = std::move(new_synced);
  return true;
}

}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia

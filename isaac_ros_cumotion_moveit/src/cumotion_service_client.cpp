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
#include <cstdint>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

#include "geometric_shapes/shapes.h"
#include "geometry_msgs/msg/pose.hpp"
#include "moveit_msgs/msg/collision_object.hpp"
#include "moveit_msgs/msg/constraints.hpp"
#include "moveit/planning_scene_interface/planning_scene_interface.h"
#include "rclcpp/rclcpp.hpp"
#include "shape_msgs/msg/mesh.hpp"
#include "shape_msgs/msg/mesh_triangle.hpp"
#include "shape_msgs/msg/solid_primitive.hpp"

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

// curobo-side name for shape i of an n-shape CollisionObject. Single-shape
// objects keep the plain id so name-based flows (attach/detach, which look up
// the obstacle by its MoveIt id) keep working; multi-shape objects expand to
// {id}__{i} (Sec 6a).
std::string memberName(const std::string & base_name, std::size_t i, std::size_t n)
{
  return (n == 1U) ? base_name : base_name + "__" + std::to_string(i);
}

// FNV-1a over raw bytes; used to fingerprint mesh geometry cheaply.
std::uint64_t hashBytes(const unsigned char * data, std::size_t len)
{
  std::uint64_t h = 1469598103934665603ULL;
  for (std::size_t i = 0; i < len; ++i) {
    h ^= static_cast<std::uint64_t>(data[i]);
    h *= 1099511628211ULL;
  }
  return h;
}

// Deterministic fingerprint of one shape + its world pose, so identical shapes
// are not removed and re-added on every sync. Mesh vertices/triangles are
// hashed rather than stringified (they can be large).
std::string shapeSignature(const shapes::ShapeConstPtr & shape, const Eigen::Isometry3d & pose)
{
  std::ostringstream os;
  os << static_cast<int>(shape->type) << '|';
  const Eigen::Vector3d t = pose.translation();
  const Eigen::Quaterniond q(pose.linear());
  os << t.x() << ',' << t.y() << ',' << t.z() << ','
     << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w() << '|';
  switch (shape->type) {
    case shapes::BOX: {
      const auto * box = static_cast<const shapes::Box *>(shape.get());
      os << box->size[0] << ',' << box->size[1] << ',' << box->size[2];
      break;
    }
    case shapes::SPHERE: {
      const auto * sphere = static_cast<const shapes::Sphere *>(shape.get());
      os << sphere->radius;
      break;
    }
    case shapes::CYLINDER: {
      const auto * cylinder = static_cast<const shapes::Cylinder *>(shape.get());
      os << cylinder->radius << ',' << cylinder->length;
      break;
    }
    case shapes::MESH: {
      // shapes::Mesh stores raw arrays: vertices (double, vertex_count*3) and
      // triangles (unsigned int, triangle_count*3). Hash both byte-wise.
      const auto * mesh = static_cast<const shapes::Mesh *>(shape.get());
      os << std::hex
         << hashBytes(
              reinterpret_cast<const unsigned char *>(mesh->vertices),
              static_cast<std::size_t>(mesh->vertex_count) * 3U * sizeof(double))
         << ':' << hashBytes(
              reinterpret_cast<const unsigned char *>(mesh->triangles),
              static_cast<std::size_t>(mesh->triangle_count) * 3U * sizeof(unsigned int))
         << std::dec;
      break;
    }
    default:
      os << '?';
      break;
  }
  return os.str();
}

// Fill an AddObject request (type/pose/dimensions or inline mesh) from one
// MoveIt shape. Returns false for unsupported shape types (skipped, not fatal).
bool fillRequestFromShape(
  isaac_ros_cumotion_interfaces::srv::AddObject::Request & req,
  const shapes::ShapeConstPtr & shape,
  const Eigen::Isometry3d & obj_pose)
{
  const Eigen::Vector3d t = obj_pose.translation();
  const Eigen::Quaterniond q(obj_pose.linear());
  req.pose.position.x = t.x();
  req.pose.position.y = t.y();
  req.pose.position.z = t.z();
  req.pose.orientation.x = q.x();
  req.pose.orientation.y = q.y();
  req.pose.orientation.z = q.z();
  req.pose.orientation.w = q.w();

  switch (shape->type) {
    case shapes::BOX: {
      const auto * box = static_cast<const shapes::Box *>(shape.get());
      req.type = isaac_ros_cumotion_interfaces::srv::AddObject::Request::CUBOID;
      req.dimensions.x = box->size[0];
      req.dimensions.y = box->size[1];
      req.dimensions.z = box->size[2];
      return true;
    }
    case shapes::SPHERE: {
      const auto * sphere = static_cast<const shapes::Sphere *>(shape.get());
      req.type = isaac_ros_cumotion_interfaces::srv::AddObject::Request::SPHERE;
      req.dimensions.x = sphere->radius;
      req.dimensions.y = sphere->radius;
      req.dimensions.z = sphere->radius;
      return true;
    }
    case shapes::CYLINDER: {
      const auto * cylinder = static_cast<const shapes::Cylinder *>(shape.get());
      req.type = isaac_ros_cumotion_interfaces::srv::AddObject::Request::CYLINDER;
      req.dimensions.x = cylinder->radius;
      req.dimensions.y = cylinder->length;
      req.dimensions.z = cylinder->length;
      return true;
    }
    case shapes::MESH: {
      // MoveIt CollisionObject meshes are in-memory vertex/triangle arrays, not
      // a file on disk; pass them inline through the AddObject service (Sec 6b).
      // MoveIt meshes are already world-size, so the MESH scale is identity.
      // shapes::Mesh stores raw arrays: `vertices` is vertex_count*3 doubles,
      // `triangles` is triangle_count*3 unsigned ints (not Eigen vectors /
      // std::vectors), so iterate them with explicit strides.
      const auto * mesh = static_cast<const shapes::Mesh *>(shape.get());
      req.type = isaac_ros_cumotion_interfaces::srv::AddObject::Request::MESH;
      req.dimensions.x = 1.0;
      req.dimensions.y = 1.0;
      req.dimensions.z = 1.0;
      req.vertices.clear();
      req.vertices.reserve(mesh->vertex_count);
      const double * vptr = mesh->vertices;
      for (std::size_t i = 0; i < mesh->vertex_count; ++i, vptr += 3) {
        geometry_msgs::msg::Point p;
        p.x = vptr[0];
        p.y = vptr[1];
        p.z = vptr[2];
        req.vertices.push_back(p);
      }
      req.triangles.clear();
      req.triangles.reserve(static_cast<std::size_t>(mesh->triangle_count) * 3U);
      const unsigned int * tptr = mesh->triangles;
      for (std::size_t i = 0; i < mesh->triangle_count; ++i, tptr += 3) {
        req.triangles.push_back(tptr[0]);
        req.triangles.push_back(tptr[1]);
        req.triangles.push_back(tptr[2]);
      }
      return true;
    }
    default:
      return false;
  }
}

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
  attach_object_client_ =
    node_->create_client<isaac_ros_cumotion_interfaces::srv::AttachObject>(
      ns_ + "/attach_object");
  detach_object_client_ = node_->create_client<std_srvs::srv::Trigger>(ns_ + "/detach_object");
  get_obstacles_client_ = node_->create_client<std_srvs::srv::Trigger>(ns_ + "/get_obstacles");
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

void CumotionServiceClient::removeCuroboObject(const std::string & name)
{
  auto req = std::make_shared<isaac_ros_cumotion_interfaces::srv::RemoveObject::Request>();
  req->name = name;
  isaac_ros_cumotion_interfaces::srv::RemoveObject::Response::SharedPtr res;
  callService<isaac_ros_cumotion_interfaces::srv::RemoveObject>(remove_object_client_, req, res);
  synced_member_signatures_.erase(name);
  cached_geometry_.erase(name);
}

std::vector<std::string> CumotionServiceClient::parseServerObjectNames(
  const std::string & message)
{
  std::vector<std::string> names;
  std::istringstream iss(message);
  std::string line;
  while (std::getline(iss, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (!line.empty()) {
      names.push_back(line);
    }
  }
  return names;
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
      req.request.start_pose.name.push_back(name);
      req.request.start_pose.position.push_back(
        js.position[static_cast<std::size_t>(std::distance(js.name.begin(), it))]);
    }
    if (have_start && !req.request.start_pose.name.empty()) {
      req.request.start_pose.header = request.start_state.joint_state.header;
    } else {
      req.request.start_pose.name.clear();
      req.request.start_pose.position.clear();
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
    // pose planner (Classic) reads candidate `poses` from goalsets,
    // not `target_joint_positions`, and would silently plan toward an empty
    // goal. The joint target rides inside a Goalset (Goalset.target_joint_positions).
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
    isaac_ros_cumotion_interfaces::msg::Goalset joint_gset;
    joint_gset.target_joint_positions = std::move(joint_positions);
    req.request.goalsets.push_back(joint_gset);
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
      req.request.goalsets.push_back(gset);
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
  return res.response.success;
}

bool CumotionServiceClient::syncPlanningScene(
  const planning_scene::PlanningSceneConstPtr & planning_scene)
{
  if (!planning_scene) {
    return false;
  }

  std::lock_guard<std::mutex> lk(sync_mutex_);

  // Effective wanted set = world objects + attached bodies. An attached body is
  // REMOVED from the MoveIt world, but its obstacle must survive on the curobo
  // side: the attach flow only disables the name (the obstacle stays registered
  // so detach can re-enable it and so attachment geometry keeps existing), and
  // /attach_object itself fails when the named obstacle is not in the scene. So
  // attached ids are kept in the wanted set (Sec 6c).
  std::vector<std::string> wanted_names = planning_scene->getWorld()->getObjectIds();
  // MoveIt 2 (Jazzy) dropped PlanningScene::getAttachedBody*; attached bodies
  // live on the current RobotState (a const reference from getCurrentState()).
  // RobotState has no count/index API either: getAttachedBody() looks up by
  // NAME, so enumerate the whole set via getAttachedBodies().
  const moveit::core::RobotState & state = planning_scene->getCurrentState();
  std::vector<const moveit::core::AttachedBody *> attached_bodies;
  state.getAttachedBodies(attached_bodies);
  std::vector<std::string> attached_names;
  attached_names.reserve(attached_bodies.size());
  for (const auto * body : attached_bodies) {
    const std::string id = body->getName();
    attached_names.push_back(id);
    if (std::find(wanted_names.begin(), wanted_names.end(), id) == wanted_names.end()) {
      wanted_names.push_back(id);
    }
  }

  // Remove stale groups (base ids present nowhere any more). Also drops their
  // member signatures and cached geometry.
  for (auto it = synced_groups_.begin(); it != synced_groups_.end();) {
    const std::string & base = it->first;
    if (std::find(wanted_names.begin(), wanted_names.end(), base) == wanted_names.end()) {
      for (const std::string & member : it->second) {
        removeCuroboObject(member);
      }
      it = synced_groups_.erase(it);
    } else {
      ++it;
    }
  }

  // Add/update world obstacles, one curobo object per shape (Sec 6a), meshes
  // inline (Sec 6b). Rebuild the group map for the synced bases along the way.
  std::map<std::string, std::vector<std::string>> new_groups;
  for (const std::string & base : wanted_names) {
    // Attached body: its geometry lives in the attached link frame and moves
    // with the gripper — do not mirror it as a static world object. Its group
    // (mirrored back when it was still a world object) is carried over so the
    // obstacle stays registered for the attach/detach dance above.
    if (std::find(attached_names.begin(), attached_names.end(), base) != attached_names.end()) {
      auto prev = synced_groups_.find(base);
      if (prev != synced_groups_.end()) {
        new_groups[base] = prev->second;
      }
      continue;
    }

    const auto obj = planning_scene->getWorld()->getObject(base);
    if (!obj) {
      continue;
    }
    if (obj->shapes_.empty() || obj->global_shape_poses_.empty()) {
      RCLCPP_DEBUG_STREAM(
        node_->get_logger(), "Skipping empty collision object '" << base << "'");
      continue;
    }

    const std::size_t n = std::min(obj->shapes_.size(), obj->global_shape_poses_.size());
    std::vector<std::string> members;
    members.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      const std::string member = memberName(base, i, n);
      const std::string sig = shapeSignature(obj->shapes_[i], obj->global_shape_poses_[i]);

      // Unchanged since the last sync (same geometry + pose, still mirrored):
      // leave the server-side object alone.
      auto sig_it = synced_member_signatures_.find(member);
      if (sig_it != synced_member_signatures_.end() && sig_it->second == sig) {
        members.push_back(member);
        continue;
      }

      auto req = std::make_shared<isaac_ros_cumotion_interfaces::srv::AddObject::Request>();
      req->name = member;
      req->color.r = 0.8f;
      req->color.g = 0.8f;
      req->color.b = 0.8f;
      req->color.a = 1.0f;
      if (!fillRequestFromShape(*req, obj->shapes_[i], obj->global_shape_poses_[i])) {
        RCLCPP_DEBUG_STREAM(
          node_->get_logger(), "Unsupported shape type for '" << member << "', skipping");
        continue;
      }

      // add_object rejects duplicates: a same-name object from an earlier sync
      // (now with changed geometry) must be removed first.
      if (sig_it != synced_member_signatures_.end()) {
        removeCuroboObject(member);
      }

      isaac_ros_cumotion_interfaces::srv::AddObject::Response::SharedPtr res;
      if (!callService<isaac_ros_cumotion_interfaces::srv::AddObject>(
          add_object_client_, req, res) ||
        !res->success) {
        RCLCPP_WARN_STREAM(node_->get_logger(), "add_object failed for '" << member << "'");
        continue;
      }
      synced_member_signatures_[member] = sig;
      // Remember the geometry for the reverse sync (Sec 6d).
      cached_geometry_[member] = *req;
      members.push_back(member);
    }

    if (!members.empty()) {
      new_groups[base] = std::move(members);
    }
  }
  synced_groups_ = std::move(new_groups);

  // --- Attached bodies → curobo attach state (Sec 6c, MoveIt → cuRobo) ---
  std::vector<std::string> new_attached;
  new_attached.reserve(attached_names.size());
  for (const std::string & name : attached_names) {
    const bool already =
      std::find(synced_attached_names_.begin(), synced_attached_names_.end(), name) !=
      synced_attached_names_.end();
    if (!already) {
      auto req = std::make_shared<isaac_ros_cumotion_interfaces::srv::AttachObject::Request>();
      req->object_name = name;
      isaac_ros_cumotion_interfaces::srv::AttachObject::Response::SharedPtr res;
      if (callService<isaac_ros_cumotion_interfaces::srv::AttachObject>(
          attach_object_client_, req, res) &&
        res->success) {
        new_attached.push_back(name);
      } else {
        // Multi-shape objects are mirrored as {id}__{i} members; the server's
        // attach looks the id up verbatim, so a multi-shape attached body cannot
        // be attached by its base id — warn rather than silently diverge.
        RCLCPP_WARN_STREAM(
          node_->get_logger(),
          "attach_object failed for attached body '" << name
          << "' (multi-shape objects are mirrored as {id}__{i} members and must "
             "be attached by one of those names)");
      }
    } else {
      new_attached.push_back(name);
    }
  }
  for (const std::string & name : synced_attached_names_) {
    if (std::find(attached_names.begin(), attached_names.end(), name) == attached_names.end()) {
      auto req = std::make_shared<std_srvs::srv::Trigger::Request>();
      std_srvs::srv::Trigger::Response::SharedPtr res;
      callService<std_srvs::srv::Trigger>(detach_object_client_, req, res);
    }
  }
  synced_attached_names_ = std::move(new_attached);

  return true;
}

bool CumotionServiceClient::syncObstaclesFromServer(
  moveit::planning_interface::PlanningSceneInterface & psi,
  const std::string & planning_frame)
{
  std::lock_guard<std::mutex> lk(sync_mutex_);

  auto req = std::make_shared<std_srvs::srv::Trigger::Request>();
  std_srvs::srv::Trigger::Response::SharedPtr res;
  if (!callService<std_srvs::srv::Trigger>(get_obstacles_client_, req, res)) {
    RCLCPP_ERROR(node_->get_logger(), "get_obstacles service call failed");
    return false;
  }
  const std::vector<std::string> server_names = parseServerObjectNames(res->message);

  // Remove MoveIt mirrors that no longer exist in cuRobo.
  for (const std::string & name : mirrored_in_moveit_names_) {
    if (std::find(server_names.begin(), server_names.end(), name) == server_names.end()) {
      psi.removeCollisionObjects({name});
    }
  }

  std::vector<std::string> new_mirrored;
  new_mirrored.reserve(server_names.size());
  for (const std::string & name : server_names) {
    auto it = cached_geometry_.find(name);
    if (it == cached_geometry_.end()) {
      // The object was created directly on the server (perception / raw
      // add_object) and never passed through a forward sync, so we have no
      // geometry to replay. get_obstacles is names-only, hence the limitation.
      RCLCPP_WARN_ONCE(
        node_->get_logger(),
        "Server object '%s' has no cached geometry (not mirror of a MoveIt "
        "collision object); skipping cuRobo->MoveIt mirror", name.c_str());
      continue;
    }
    const auto & add_req = it->second;

    moveit_msgs::msg::CollisionObject co;
    co.id = name;
    co.header.frame_id = planning_frame;
    co.operation = moveit_msgs::msg::CollisionObject::ADD;
    co.pose = add_req.pose;

    bool built = false;
    switch (add_req.type) {
      case isaac_ros_cumotion_interfaces::srv::AddObject::Request::CUBOID: {
        shape_msgs::msg::SolidPrimitive prim;
        prim.type = shape_msgs::msg::SolidPrimitive::BOX;
        prim.dimensions[0] = add_req.dimensions.x;
        prim.dimensions[1] = add_req.dimensions.y;
        prim.dimensions[2] = add_req.dimensions.z;
        co.primitives.push_back(prim);
        co.primitive_poses.push_back(co.pose);
        built = true;
        break;
      }
      case isaac_ros_cumotion_interfaces::srv::AddObject::Request::SPHERE: {
        shape_msgs::msg::SolidPrimitive prim;
        prim.type = shape_msgs::msg::SolidPrimitive::SPHERE;
        prim.dimensions[0] = add_req.dimensions.x;
        co.primitives.push_back(prim);
        co.primitive_poses.push_back(co.pose);
        built = true;
        break;
      }
      case isaac_ros_cumotion_interfaces::srv::AddObject::Request::CYLINDER: {
        shape_msgs::msg::SolidPrimitive prim;
        prim.type = shape_msgs::msg::SolidPrimitive::CYLINDER;
        prim.dimensions[0] = add_req.dimensions.x;
        prim.dimensions[1] = add_req.dimensions.y;
        co.primitives.push_back(prim);
        co.primitive_poses.push_back(co.pose);
        built = true;
        break;
      }
      case isaac_ros_cumotion_interfaces::srv::AddObject::Request::CAPSULE:
        // moveit_msgs SolidPrimitive has no CAPSULE; cannot mirror faithfully.
        RCLCPP_WARN_STREAM(
          node_->get_logger(), "Skipping CAPSULE '" << name
          << "' in cuRobo->MoveIt mirror");
        continue;
      case isaac_ros_cumotion_interfaces::srv::AddObject::Request::MESH: {
        if (add_req.vertices.empty() || add_req.triangles.empty()) {
          RCLCPP_WARN_STREAM(
            node_->get_logger(), "Skipping file-path-only MESH '" << name
            << "' in cuRobo->MoveIt mirror (no inline geometry cached)");
          continue;
        }
        shape_msgs::msg::Mesh mesh;
        mesh.vertices.reserve(add_req.vertices.size());
        for (const auto & v : add_req.vertices) {
          mesh.vertices.push_back(v);
        }
        mesh.triangles.reserve(add_req.triangles.size() / 3U);
        for (std::size_t i = 0; i + 2U < add_req.triangles.size(); i += 3U) {
          shape_msgs::msg::MeshTriangle tri;
          tri.vertex_indices[0] = add_req.triangles[i];
          tri.vertex_indices[1] = add_req.triangles[i + 1U];
          tri.vertex_indices[2] = add_req.triangles[i + 2U];
          mesh.triangles.push_back(tri);
        }
        co.meshes.push_back(mesh);
        co.mesh_poses.push_back(co.pose);
        built = true;
        break;
      }
      default:
        RCLCPP_WARN_STREAM(
          node_->get_logger(), "Unsupported AddObject type for '" << name
          << "'; skipping mirror");
        continue;
    }
    if (!built) {
      continue;
    }

    // Idempotent: re-applying a CollisionObject with ADD replaces the previous
    // one, so unchanged or updated geometry is handled by the same call.
    psi.applyCollisionObject(co);
    new_mirrored.push_back(name);
  }

  mirrored_in_moveit_names_ = std::move(new_mirrored);
  return true;
}

}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia
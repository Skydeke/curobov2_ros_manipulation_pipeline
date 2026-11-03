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

#include "curobov2_ros_moveit/cumotion_service_client.hpp"

#include "curobov2_ros_moveit/cumotion_planner_ids.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

#include "geometric_shapes/mesh_operations.h"
#include "geometric_shapes/shapes.h"
#include "geometry_msgs/msg/pose.hpp"
#include "moveit_msgs/msg/collision_object.hpp"
#include "moveit_msgs/msg/constraints.hpp"
#include "moveit_msgs/msg/object_color.hpp"
#include "moveit/planning_scene_interface/planning_scene_interface.h"
#include "rclcpp/rclcpp.hpp"
#include "shape_msgs/msg/mesh.hpp"
#include "shape_msgs/msg/mesh_triangle.hpp"
#include "shape_msgs/msg/solid_primitive.hpp"
#include "std_msgs/msg/color_rgba.hpp"

namespace curobov2
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
  curobov2_ros_interfaces::srv::AddObject::Request & req,
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
      req.type = curobov2_ros_interfaces::srv::AddObject::Request::CUBOID;
      req.dimensions.x = box->size[0];
      req.dimensions.y = box->size[1];
      req.dimensions.z = box->size[2];
      return true;
    }
    case shapes::SPHERE: {
      const auto * sphere = static_cast<const shapes::Sphere *>(shape.get());
      req.type = curobov2_ros_interfaces::srv::AddObject::Request::SPHERE;
      req.dimensions.x = sphere->radius;
      req.dimensions.y = sphere->radius;
      req.dimensions.z = sphere->radius;
      return true;
    }
    case shapes::CYLINDER: {
      const auto * cylinder = static_cast<const shapes::Cylinder *>(shape.get());
      req.type = curobov2_ros_interfaces::srv::AddObject::Request::CYLINDER;
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
      req.type = curobov2_ros_interfaces::srv::AddObject::Request::MESH;
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

// One object exactly as `get_scene_objects` reports it. rosidl has no nested
// record type, so the service returns parallel arrays; this index-aligned view
// exists only inside this file.
struct ServerObject
{
  std::string name;
  std::int8_t type = 0;
  geometry_msgs::msg::Pose pose;
  geometry_msgs::msg::Vector3 dimensions;
  std_msgs::msg::ColorRGBA color;
  std::string mesh_file_path;
  bool attached = false;
};

geometry_msgs::msg::Pose poseFromEigen(const Eigen::Isometry3d & pose)
{
  geometry_msgs::msg::Pose out;
  out.position.x = pose.translation().x();
  out.position.y = pose.translation().y();
  out.position.z = pose.translation().z();
  const Eigen::Quaterniond q(pose.linear());
  out.orientation.x = q.x();
  out.orientation.y = q.y();
  out.orientation.z = q.z();
  out.orientation.w = q.w();
  return out;
}

void meshToMsg(const shapes::Mesh & mesh, shape_msgs::msg::Mesh & out)
{
  out.vertices.clear();
  out.vertices.reserve(mesh.vertex_count);
  const double * vptr = mesh.vertices;
  for (std::size_t i = 0; i < mesh.vertex_count; ++i, vptr += 3) {
    geometry_msgs::msg::Point p;
    p.x = vptr[0];
    p.y = vptr[1];
    p.z = vptr[2];
    out.vertices.push_back(p);
  }
  out.triangles.clear();
  out.triangles.reserve(mesh.triangle_count);
  const unsigned int * tptr = mesh.triangles;
  for (std::size_t i = 0; i < mesh.triangle_count; ++i, tptr += 3) {
    shape_msgs::msg::MeshTriangle tri;
    tri.vertex_indices[0] = tptr[0];
    tri.vertex_indices[1] = tptr[1];
    tri.vertex_indices[2] = tptr[2];
    out.triangles.push_back(tri);
  }
}

bool fillCollisionObjectFromServer(
  const ServerObject & object,
  const curobov2_ros_interfaces::srv::AddObject::Request * cached_geometry,
  const std::string & planning_frame,
  moveit_msgs::msg::CollisionObject & co,
  std_msgs::msg::ColorRGBA & color,
  std::string & reason)
{
  co = moveit_msgs::msg::CollisionObject();
  co.id = object.name;
  co.header.frame_id = planning_frame;
  co.operation = moveit_msgs::msg::CollisionObject::ADD;
  co.pose = object.pose;
  color = object.color;

  // CollisionObject primitive/mesh poses are relative to co.pose. The service
  // reports one world pose per object, so the shape pose is the identity; the
  // old implementation copied co.pose into primitive_poses, which transformed
  // every mirrored object twice.
  const geometry_msgs::msg::Pose identity;

  switch (object.type) {
    case curobov2_ros_interfaces::srv::AddObject::Request::CUBOID: {
      shape_msgs::msg::SolidPrimitive prim;
      prim.type = shape_msgs::msg::SolidPrimitive::BOX;
      prim.dimensions = {object.dimensions.x, object.dimensions.y, object.dimensions.z};
      co.primitives.push_back(prim);
      co.primitive_poses.push_back(identity);
      return true;
    }
    case curobov2_ros_interfaces::srv::AddObject::Request::SPHERE: {
      shape_msgs::msg::SolidPrimitive prim;
      prim.type = shape_msgs::msg::SolidPrimitive::SPHERE;
      prim.dimensions = {object.dimensions.x};
      co.primitives.push_back(prim);
      co.primitive_poses.push_back(identity);
      return true;
    }
    case curobov2_ros_interfaces::srv::AddObject::Request::CYLINDER: {
      shape_msgs::msg::SolidPrimitive prim;
      prim.type = shape_msgs::msg::SolidPrimitive::CYLINDER;
      prim.dimensions = {object.dimensions.x, object.dimensions.y};
      co.primitives.push_back(prim);
      co.primitive_poses.push_back(identity);
      return true;
    }
    case curobov2_ros_interfaces::srv::AddObject::Request::CAPSULE: {
      // moveit_msgs has no capsule primitive. A capsule is exactly the union of
      // a cylinder over the segment between the two sphere centres and two
      // spheres at its ends, so mirror it as those three primitives (kept in
      // the object frame, which is why their poses are offsets, not co.pose).
      const double radius = object.dimensions.x;
      const double segment = object.dimensions.y;
      shape_msgs::msg::SolidPrimitive cylinder;
      cylinder.type = shape_msgs::msg::SolidPrimitive::CYLINDER;
      cylinder.dimensions = {radius, segment};
      co.primitives.push_back(cylinder);
      co.primitive_poses.push_back(identity);
      for (double sign : {-1.0, 1.0}) {
        shape_msgs::msg::SolidPrimitive sphere;
        sphere.type = shape_msgs::msg::SolidPrimitive::SPHERE;
        sphere.dimensions = {radius};
        co.primitives.push_back(sphere);
        Eigen::Isometry3d offset = Eigen::Isometry3d::Identity();
        offset.translation().z() = sign * 0.5 * segment;
        co.primitive_poses.push_back(poseFromEigen(offset));
      }
      return true;
    }
    case curobov2_ros_interfaces::srv::AddObject::Request::MESH: {
      shape_msgs::msg::Mesh mesh;
      if (object.mesh_file_path.empty() && cached_geometry != nullptr &&
        !cached_geometry->vertices.empty() && !cached_geometry->triangles.empty())
      {
        // Inline geometry this client itself forwarded. get_scene_objects does
        // not echo vertex buffers (they can be huge), so this cache is the only
        // faithful source for such an object.
        mesh.vertices.reserve(cached_geometry->vertices.size());
        for (const auto & point : cached_geometry->vertices) {
          mesh.vertices.push_back(point);
        }
        mesh.triangles.reserve(cached_geometry->triangles.size() / 3U);
        for (std::size_t i = 0; i + 2U < cached_geometry->triangles.size(); i += 3U) {
          shape_msgs::msg::MeshTriangle tri;
          tri.vertex_indices[0] = cached_geometry->triangles[i];
          tri.vertex_indices[1] = cached_geometry->triangles[i + 1U];
          tri.vertex_indices[2] = cached_geometry->triangles[i + 2U];
          mesh.triangles.push_back(tri);
        }
      } else if (!object.mesh_file_path.empty()) {
        Eigen::Vector3d scale(object.dimensions.x, object.dimensions.y, object.dimensions.z);
        if (scale.x() <= 0.0 || scale.y() <= 0.0 || scale.z() <= 0.0) {
          scale = Eigen::Vector3d::Ones();
        }
        shapes::Mesh * loaded = shapes::createMeshFromResource(object.mesh_file_path, scale);
        if (loaded == nullptr) {
          reason = "could not load mesh file '" + object.mesh_file_path + "'";
          return false;
        }
        meshToMsg(*loaded, mesh);
        delete loaded;
      } else {
        reason = "inline mesh vertices are not echoed by get_scene_objects";
        return false;
      }
      co.meshes.push_back(mesh);
      co.mesh_poses.push_back(identity);
      return true;
    }
    default:
      reason = "unsupported object type " + std::to_string(static_cast<int>(object.type));
      return false;
  }
}

std::string serverObjectSignature(
  const ServerObject & object,
  const curobov2_ros_interfaces::srv::AddObject::Request * cached_geometry)
{
  std::ostringstream os;
  os << static_cast<int>(object.type) << '|'
     << object.pose.position.x << ',' << object.pose.position.y << ','
     << object.pose.position.z << ',' << object.pose.orientation.x << ','
     << object.pose.orientation.y << ',' << object.pose.orientation.z << ','
     << object.pose.orientation.w << '|'
     << object.dimensions.x << ',' << object.dimensions.y << ','
     << object.dimensions.z << '|' << object.mesh_file_path;
  if (object.mesh_file_path.empty() && cached_geometry != nullptr) {
    // Hash inline buffers instead of stringifying them: a mesh can be hundreds
    // of thousands of vertices and this signature is recomputed every plan.
    os << '|' << cached_geometry->vertices.size() << ','
       << cached_geometry->triangles.size() << ','
       << std::hex
       << hashBytes(
            reinterpret_cast<const unsigned char *>(cached_geometry->vertices.data()),
            cached_geometry->vertices.size() * sizeof(geometry_msgs::msg::Point))
       << ':' << hashBytes(
            reinterpret_cast<const unsigned char *>(cached_geometry->triangles.data()),
            cached_geometry->triangles.size() * sizeof(std::int32_t))
       << std::dec;
  }
  return os.str();
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

  // Dedicated client node/executor (see the header): move_group may be
  // single-threaded, and all these calls are synchronous. A unique name avoids
  // duplicate-node warnings when one context per planning group is created.
  static std::atomic<unsigned> instance_counter{0};
  client_node_ = rclcpp::Node::make_shared(
    "curobov2_moveit_client_" + std::to_string(++instance_counter));
  client_executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
  client_executor_->add_node(client_node_);
  client_spinner_ = std::thread([this]() {client_executor_->spin();});

  set_planner_client_ =
    client_node_->create_client<curobov2_ros_interfaces::srv::SetPlanner>(
      ns_ + "/set_planner");
  traj_gen_client_ =
    client_node_->create_client<curobov2_ros_interfaces::srv::TrajectoryGeneration>(
      ns_ + "/generate_trajectory");
  add_object_client_ = client_node_->create_client<curobov2_ros_interfaces::srv::AddObject>(
    ns_ + "/add_object");
  remove_object_client_ =
    client_node_->create_client<curobov2_ros_interfaces::srv::RemoveObject>(
      ns_ + "/remove_object");
  attach_object_client_ =
    client_node_->create_client<curobov2_ros_interfaces::srv::AttachObject>(
      ns_ + "/attach_object");
  detach_object_client_ =
    client_node_->create_client<std_srvs::srv::Trigger>(ns_ + "/detach_object");
  get_scene_objects_client_ =
    client_node_->create_client<curobov2_ros_interfaces::srv::GetSceneObjects>(
      ns_ + "/get_scene_objects");
}

CumotionServiceClient::~CumotionServiceClient()
{
  if (client_executor_) {
    client_executor_->cancel();
  }
  if (client_spinner_.joinable()) {
    client_spinner_.join();
  }
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
  auto req = std::make_shared<curobov2_ros_interfaces::srv::RemoveObject::Request>();
  req->name = name;
  curobov2_ros_interfaces::srv::RemoveObject::Response::SharedPtr res;
  callService<curobov2_ros_interfaces::srv::RemoveObject>(remove_object_client_, req, res);
  synced_member_signatures_.erase(name);
  cached_geometry_.erase(name);
}

void CumotionServiceClient::removeMoveItMirror(
  moveit::planning_interface::PlanningSceneInterface & psi,
  const std::string & name)
{
  psi.removeCollisionObjects({name});
}

bool CumotionServiceClient::setPlanner(uint8_t planner_type, std::string & message)
{
  auto req = std::make_shared<curobov2_ros_interfaces::srv::SetPlanner::Request>();
  req->planner_type = planner_type;

  curobov2_ros_interfaces::srv::SetPlanner::Response::SharedPtr res;
  if (!callService<curobov2_ros_interfaces::srv::SetPlanner>(
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
  curobov2_ros_interfaces::srv::TrajectoryGeneration::Request & req,
  curobov2_ros_interfaces::srv::TrajectoryGeneration::Response & res)
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
    planner_type = curobov2_ros_interfaces::srv::SetPlanner::Request::JOINT_SPACE;
    if (!setPlanner(planner_type, planner_msg)) {
      RCLCPP_ERROR_STREAM(node_->get_logger(), "setPlanner failed: " << planner_msg);
      return false;
    }
    curobov2_ros_interfaces::msg::Goalset joint_gset;
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
      planner_type = curobov2_ros_interfaces::srv::SetPlanner::Request::JOINT_SPACE;
    }
    RCLCPP_INFO_STREAM(node_->get_logger(),
      "Pose goal: link=" << (constraint.position_constraints.empty()
        ? std::string("(none)") : constraint.position_constraints.front().link_name)
      << " waypoints=" << waypoints.size()
      << " goal=[x=" << pose.position.x << " y=" << pose.position.y
      << " z=" << pose.position.z << "]");
    for (const auto & wp : waypoints) {
      curobov2_ros_interfaces::msg::Goalset gset;
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
    std::make_shared<curobov2_ros_interfaces::srv::TrajectoryGeneration::Request>();
  *req_ptr = req;
  curobov2_ros_interfaces::srv::TrajectoryGeneration::Response::SharedPtr res_ptr;
  if (!callService<curobov2_ros_interfaces::srv::TrajectoryGeneration>(
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
  std::vector<std::string> world_names = planning_scene->getWorld()->getObjectIds();
  std::vector<std::string> wanted_names = world_names;
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

  // A reverse-mirrored object is owned by cuRobo: the previous reverse pass put
  // it in MoveIt only so RViz and other MoveIt clients can see it. If it is
  // gone from the MoveIt world now, the user deleted it there — delete it on the
  // server too instead of resurrecting it from the server in this plan's
  // reverse pass. (Attached bodies live in the RobotState, not the world.)
  for (auto it = mirrored_in_moveit_names_.begin(); it != mirrored_in_moveit_names_.end();) {
    const std::string & name = *it;
    const bool in_world =
      std::find(world_names.begin(), world_names.end(), name) != world_names.end();
    const bool is_attached =
      std::find(attached_names.begin(), attached_names.end(), name) != attached_names.end();
    if (!in_world && !is_attached && mirrored_seen_in_moveit_.count(name) != 0U) {
      removeCuroboObject(name);
      mirrored_in_moveit_signatures_.erase(name);
      mirrored_seen_in_moveit_.erase(name);
      it = mirrored_in_moveit_names_.erase(it);
    } else {
      if (in_world) {
        mirrored_seen_in_moveit_.insert(name);
      }
      ++it;
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

    // A reverse mirror: the server already owns this object (it may have been
    // added by perception / the task constructor / another client), and the
    // reverse pass below is what keeps MoveIt in step with it. Do not forward it
    // again — that would either collide with the server's copy or churn the
    // geometry every plan.
    if (std::find(
        mirrored_in_moveit_names_.begin(), mirrored_in_moveit_names_.end(), base) !=
      mirrored_in_moveit_names_.end())
    {
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

      auto req = std::make_shared<curobov2_ros_interfaces::srv::AddObject::Request>();
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

      curobov2_ros_interfaces::srv::AddObject::Response::SharedPtr res;
      bool added = callService<curobov2_ros_interfaces::srv::AddObject>(
        add_object_client_, req, res) && res->success;
      // A name the server already holds but this client did not put there (a
      // direct add_object / perception object whose id collides with a MoveIt
      // object) is a duplicate to the server. MoveIt is the forward authority,
      // so replace the server-side object with the MoveIt geometry once.
      if (!added && res && res->message.find("already exists") != std::string::npos) {
        removeCuroboObject(member);
        added = callService<curobov2_ros_interfaces::srv::AddObject>(
          add_object_client_, req, res) && res->success;
      }
      if (!added) {
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
      auto req = std::make_shared<curobov2_ros_interfaces::srv::AttachObject::Request>();
      req->object_name = name;
      curobov2_ros_interfaces::srv::AttachObject::Response::SharedPtr res;
      if (callService<curobov2_ros_interfaces::srv::AttachObject>(
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

  // Remember what MoveIt already contained so the reverse pass (which runs
  // immediately after this one) can tell a server-only object from one the
  // forward pass just made.
  last_moveit_world_names_ = std::move(world_names);
  last_moveit_attached_names_ = attached_names;

  return true;
}

bool CumotionServiceClient::syncObstaclesFromServer(
  moveit::planning_interface::PlanningSceneInterface & psi,
  const std::string & planning_frame)
{
  std::lock_guard<std::mutex> lk(sync_mutex_);

  // Ask for the SOLVER's full scene, not a names-only listing: the objects that
  // matter most in the reverse direction (perception, direct add_object calls,
  // other clients) were never forwarded by this process, so there is no local
  // geometry to replay for them.
  auto req =
    std::make_shared<curobov2_ros_interfaces::srv::GetSceneObjects::Request>();
  curobov2_ros_interfaces::srv::GetSceneObjects::Response::SharedPtr res;
  if (!callService<curobov2_ros_interfaces::srv::GetSceneObjects>(
      get_scene_objects_client_, req, res))
  {
    RCLCPP_ERROR(node_->get_logger(), "get_scene_objects service call failed");
    return false;
  }
  if (!res->success) {
    RCLCPP_ERROR_STREAM(
      node_->get_logger(), "get_scene_objects refused: " << res->message);
    return false;
  }

  // Unpack the parallel arrays defensively: a short array must not silently
  // attribute one object's pose or size to another.
  std::vector<ServerObject> objects;
  objects.reserve(res->names.size());
  for (std::size_t i = 0; i < res->names.size(); ++i) {
    if (res->names[i].empty()) {
      continue;
    }
    if (i >= res->types.size() || i >= res->poses.size() ||
      i >= res->dimensions.size() || i >= res->colors.size() ||
      i >= res->mesh_file_paths.size() || i >= res->attached.size())
    {
      RCLCPP_WARN_STREAM(
        node_->get_logger(),
        "get_scene_objects returned a short array for '" << res->names[i]
        << "'; omitting it instead of mirroring misaligned values");
      continue;
    }
    ServerObject object;
    object.name = res->names[i];
    object.type = res->types[i];
    object.pose = res->poses[i];
    object.dimensions = res->dimensions[i];
    object.color = res->colors[i];
    object.mesh_file_path = res->mesh_file_paths[i];
    object.attached = res->attached[i];
    objects.push_back(std::move(object));
  }

  // Names the forward pass already represents in MoveIt (user world objects
  // and things this client forwarded) must not be mirrored back as if the
  // server had created them.
  std::set<std::string> forward_members;
  for (const auto & group : synced_groups_) {
    forward_members.insert(group.second.begin(), group.second.end());
  }
  const std::set<std::string> moveit_world(
    last_moveit_world_names_.begin(), last_moveit_world_names_.end());
  const std::set<std::string> moveit_attached(
    last_moveit_attached_names_.begin(), last_moveit_attached_names_.end());

  // Existing mirrors: keep the ones the server still reports as world objects.
  // A mirror that disappeared from the server (or became attached) must be
  // removed from MoveIt; a stale world copy of an attached object would make
  // MoveIt collide against the payload twice.
  std::vector<std::string> new_mirrored;
  new_mirrored.reserve(objects.size());
  std::set<std::string> mirrored_names(
    mirrored_in_moveit_names_.begin(), mirrored_in_moveit_names_.end());
  for (const std::string & name : mirrored_in_moveit_names_) {
    const auto it = std::find_if(
      objects.begin(), objects.end(),
      [&name](const ServerObject & object) {return object.name == name;});
    const bool still_world = it != objects.end() && !it->attached;
    if (!still_world) {
      removeMoveItMirror(psi, name);
      mirrored_in_moveit_signatures_.erase(name);
      mirrored_seen_in_moveit_.erase(name);
      continue;
    }
    new_mirrored.push_back(name);
  }

  for (const ServerObject & object : objects) {
    if (object.attached) {
      // Attach state is owned by the forward pass (MoveIt AttachedBody ->
      // attach_object). get_scene_objects does not report WHICH link an object
      // is attached to, so a server-side attach cannot be reconstructed as a
      // MoveIt AttachedCollisionObject without guessing; leave it out of the
      // world rather than showing a bogus free-floating obstacle.
      continue;
    }

    const bool was_mirror = mirrored_names.count(object.name) != 0U;
    if (!was_mirror) {
      // A forward-synced object is already in MoveIt (possibly under a different
      // member name for a multi-shape CollisionObject). Leave it alone.
      if (forward_members.count(object.name) != 0U ||
        moveit_world.count(object.name) != 0U ||
        moveit_attached.count(object.name) != 0U)
      {
        continue;
      }
    }

    const auto cached = cached_geometry_.find(object.name);
    const curobov2_ros_interfaces::srv::AddObject::Request * cached_geometry =
      cached == cached_geometry_.end() ? nullptr : &cached->second;
    const std::string sig = serverObjectSignature(object, cached_geometry);

    if (was_mirror) {
      const auto sig_it = mirrored_in_moveit_signatures_.find(object.name);
      const bool seen = mirrored_seen_in_moveit_.count(object.name) != 0U;
      if (seen && sig_it != mirrored_in_moveit_signatures_.end() && sig_it->second == sig) {
        continue;  // unchanged since the last reverse pass
      }
    }

    moveit_msgs::msg::CollisionObject co;
    std_msgs::msg::ColorRGBA color;
    std::string reason;
    if (!fillCollisionObjectFromServer(
        object, cached_geometry, planning_frame, co, color, reason))
    {
      RCLCPP_WARN_ONCE(
        node_->get_logger(),
        "Skipping cuRobo->MoveIt mirror for '%s': %s",
        object.name.c_str(), reason.c_str());
      if (was_mirror) {
        removeMoveItMirror(psi, object.name);
        mirrored_in_moveit_signatures_.erase(object.name);
        mirrored_seen_in_moveit_.erase(object.name);
        new_mirrored.erase(
          std::remove(new_mirrored.begin(), new_mirrored.end(), object.name),
          new_mirrored.end());
      }
      continue;
    }

    // Publish the mirror asynchronously. A synchronous apply_planning_scene
    // call here would wait for move_group's planning-scene lock, which the
    // planning callback that called us is holding — a deadlock. ADD replaces an
    // existing object with the same id, so this is also the update path.
    moveit_msgs::msg::ObjectColor object_color;
    object_color.id = object.name;
    object_color.color = color;
    psi.addCollisionObjects({co}, {object_color});
    mirrored_in_moveit_signatures_[object.name] = sig;
    if (!was_mirror) {
      new_mirrored.push_back(object.name);
    }
  }

  // Drop signatures for names that are no longer mirrored.
  const std::set<std::string> current(new_mirrored.begin(), new_mirrored.end());
  for (auto it = mirrored_in_moveit_signatures_.begin();
    it != mirrored_in_moveit_signatures_.end();)
  {
    if (current.count(it->first) == 0U) {
      it = mirrored_in_moveit_signatures_.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = mirrored_seen_in_moveit_.begin();
    it != mirrored_seen_in_moveit_.end();)
  {
    if (current.count(*it) == 0U) {
      it = mirrored_seen_in_moveit_.erase(it);
    } else {
      ++it;
    }
  }

  mirrored_in_moveit_names_ = std::move(new_mirrored);
  return true;
}

}  // namespace manipulation
}  // namespace curobov2
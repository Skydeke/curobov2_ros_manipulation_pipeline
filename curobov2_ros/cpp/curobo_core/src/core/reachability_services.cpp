// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/core/reachability_services.py.
#include "curobo_core/core/reachability_services.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/bool.hpp>

#include <Eigen/Geometry>

#include "curobo_core/core/frame_utils.hpp"

namespace curobo_core
{

namespace
{
bool is_base_frame(const std::string & frame, const std::string & base)
{
  // PORT: '"\" == base: no-op' - an empty or literal backslash frame means base.
  return frame.empty() || frame == base;
}
}  // namespace

ReachabilityServices::ReachabilityServices(
  rclcpp::Node * node, IKServices * ik_services, std::string base_link,
  std::shared_ptr<tf2_ros::Buffer> tf_buffer)
: node_(node), ik_services_(ik_services), base_link_(std::move(base_link)),
  tf_buffer_(std::move(tf_buffer))
{
  generate_rm_srv_ = node_->create_service<curobov2_ros_interfaces::srv::GenerateRM>(
    std::string(node_->get_name()) + "/generate_rm",
    std::bind(
      &ReachabilityServices::handle_generate_rm, this,
      std::placeholders::_1, std::placeholders::_2));
  RCLCPP_INFO(
    node_->get_logger(),
    "Reachability services registered (/<node>/generate_rm, grid cap %d cells)",
    kMaxGridCells);
}

// PORT: reachability_services._grid_size
std::pair<int, int> ReachabilityServices::grid_size(std::uint32_t gx_in, std::uint32_t gy_in)
{
  int gx = std::max(1, static_cast<int>(gx_in));
  int gy = std::max(1, static_cast<int>(gy_in));
  if (gx * gy > kMaxGridCells) {
    const double scale = std::sqrt(
      static_cast<double>(kMaxGridCells) / static_cast<double>(gx * gy));
    gx = std::max(1, static_cast<int>(static_cast<double>(gx) * scale));
    gy = std::max(1, static_cast<int>(static_cast<double>(gy) * scale));
    while (gx * gy > kMaxGridCells) {
      if (gx > gy) {
        gx -= 1;
      } else {
        gy -= 1;
      }
    }
  }
  return {gx, gy};
}

// PORT: reachability_services._build_goals
std::vector<Pose7> ReachabilityServices::build_goals(
  const geometry_msgs::msg::Pose & plane, double size_x, double size_y, int gx, int gy)
{
  const auto quat = normalize_quat_wxyz(
    {plane.orientation.w, plane.orientation.x, plane.orientation.y, plane.orientation.z});
  const Eigen::Matrix3d rot = quat_to_matrix(quat);
  // +90deg about the plane's local Y: (x,y,z) -> (-z,y,x).
  const auto tool_quat = quat_multiply(
    quat, {std::cos(M_PI / 4.0), 0.0, std::sin(M_PI / 4.0), 0.0});
  const Eigen::Vector3d origin(plane.position.x, plane.position.y, plane.position.z);

  const double hx = size_x / 2.0;
  const double hy = size_y / 2.0;
  const auto axis = [](double half, int n) {
      std::vector<double> out(static_cast<std::size_t>(n), 0.0);
      if (n > 1) {
        for (int i = 0; i < n; ++i) {
          out[static_cast<std::size_t>(i)] =
            -half + static_cast<double>(i) * (2.0 * half) / static_cast<double>(n - 1);
        }
      }
      return out;
    };
  const auto us = axis(hx, gx);
  const auto vs = axis(hy, gy);

  std::vector<Pose7> goals;
  goals.reserve(static_cast<std::size_t>(gx) * static_cast<std::size_t>(gy));
  for (int iy = 0; iy < gy; ++iy) {
    for (int ix = 0; ix < gx; ++ix) {
      const Eigen::Vector3d local(us[static_cast<std::size_t>(ix)],
        vs[static_cast<std::size_t>(iy)], 0.0);
      const Eigen::Vector3d world = origin + rot * local;
      Pose7 pose;
      pose << world.x(), world.y(), world.z(),
        tool_quat[0], tool_quat[1], tool_quat[2], tool_quat[3];
      goals.push_back(pose);
    }
  }
  return goals;
}

// PORT: reachability_services._generate_rm_callback
void ReachabilityServices::handle_generate_rm(
  const std::shared_ptr<curobov2_ros_interfaces::srv::GenerateRM::Request> request,
  std::shared_ptr<curobov2_ros_interfaces::srv::GenerateRM::Response> response)
{
  const auto t0 = std::chrono::steady_clock::now();
  try {
    const std::string req_frame = is_base_frame(request->frame_id, base_link_) ?
      base_link_ : request->frame_id;

    geometry_msgs::msg::Pose plane_req;
    plane_req.position.x = request->plane_position_x;
    plane_req.position.y = request->plane_position_y;
    plane_req.position.z = request->plane_position_z;
    plane_req.orientation.x = request->plane_orientation_x;
    plane_req.orientation.y = request->plane_orientation_y;
    plane_req.orientation.z = request->plane_orientation_z;
    plane_req.orientation.w = request->plane_orientation_w;

    geometry_msgs::msg::Pose plane_base;
    if (req_frame != base_link_) {
      if (tf_buffer_ == nullptr) {
        throw std::runtime_error(
          "cannot transform '" + req_frame + "' -> '" + base_link_ + "': no TF buffer");
      }
      plane_base = apply_transform(
        plane_req, get_transform(*tf_buffer_, base_link_, req_frame));
    } else {
      plane_base = plane_req;
    }

    const auto [gx, gy] = grid_size(request->grid_size_x, request->grid_size_y);
    auto goals = build_goals(
      plane_base, request->plane_size_x, request->plane_size_y, gx, gy);

    std::vector<std::vector<float>> positions;
    std::vector<std::uint8_t> flags;
    std::vector<std::string> joint_names;
    const bool ok = ik_services_->solve_poses(goals, positions, flags, joint_names);
    if (!ok) {
      positions.assign(goals.size(), {});
      flags.assign(goals.size(), 0);
    }

    if (req_frame != base_link_ && tf_buffer_ != nullptr) {
      const auto tf_back = get_transform(*tf_buffer_, req_frame, base_link_);
      for (auto & goal : goals) {
        goal = from_ros_pose(apply_transform(to_ros_pose(goal), tf_back));
      }
    }

    auto & metrics = response->metrics;
    metrics.header.stamp = node_->get_clock()->now();
    metrics.header.frame_id = req_frame;
    metrics.map_id = map_id_;
    map_id_ += 1;
    metrics.plane_size_x = request->plane_size_x;
    metrics.plane_size_y = request->plane_size_y;
    metrics.grid_size_x = static_cast<std::uint32_t>(gx);
    metrics.grid_size_y = static_cast<std::uint32_t>(gy);
    metrics.plane_pose.position.x = request->plane_position_x;
    metrics.plane_pose.position.y = request->plane_position_y;
    metrics.plane_pose.position.z = request->plane_position_z;
    const auto quat = normalize_quat_wxyz(
      {request->plane_orientation_w, request->plane_orientation_x,
        request->plane_orientation_y, request->plane_orientation_z});
    metrics.plane_pose.orientation.w = quat[0];
    metrics.plane_pose.orientation.x = quat[1];
    metrics.plane_pose.orientation.y = quat[2];
    metrics.plane_pose.orientation.z = quat[3];

    for (std::size_t i = 0; i < goals.size(); ++i) {
      metrics.goals.push_back(to_ros_pose(goals[i]));
      sensor_msgs::msg::JointState js;
      std_msgs::msg::Bool valid;
      if (ok && i < flags.size() && flags[i] != 0 && i < positions.size()) {
        js.name = joint_names;
        js.position.assign(positions[i].begin(), positions[i].end());
        valid.data = true;
      }
      metrics.joint_states.push_back(std::move(js));
      metrics.joint_states_valid.push_back(valid);
    }
    metrics.n_total = static_cast<std::uint32_t>(goals.size());
    std::uint32_t solved = 0;
    for (const auto & v : metrics.joint_states_valid) {
      if (v.data) {
        solved += 1;
      }
    }
    metrics.n_solved = solved;
    metrics.solve_time_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();

    response->success = ok;
    response->message = ok ?
      ("solved " + std::to_string(solved) + "/" + std::to_string(goals.size())) :
      "IK solve failed; map shown all-red";
  } catch (const std::exception & e) {
    RCLCPP_ERROR(node_->get_logger(), "Reachability generate_rm failed: %s", e.what());
    response->success = false;
    response->message = e.what();
  }
}

}  // namespace curobo_core

// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/core/reachability_services.py
//
// Reachability-map service: solves a uniform grid of IK goals on a user plane
// and returns per-cell outcomes so an RViz display can render the map. Reuses
// the shared IK solver from IKServices.
#ifndef CUROBO_CORE__CORE__REACHABILITY_SERVICES_HPP_
#define CUROBO_CORE__CORE__REACHABILITY_SERVICES_HPP_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>

#include "curobov2_ros_interfaces/srv/generate_rm.hpp"
#include "curobo_core/convert.hpp"
#include "curobo_core/core/ik_services.hpp"

namespace curobo_core
{

class ReachabilityServices
{
public:
  ReachabilityServices(
    rclcpp::Node * node, IKServices * ik_services, std::string base_link,
    std::shared_ptr<tf2_ros::Buffer> tf_buffer);

  // PORT: reachability_services._MAX_GRID_CELLS
  static constexpr int kMaxGridCells = 1200;

  // PORT: reachability_services._grid_size
  static std::pair<int, int> grid_size(std::uint32_t gx_in, std::uint32_t gy_in);
  // PORT: reachability_services._build_goals
  static std::vector<Pose7> build_goals(
    const geometry_msgs::msg::Pose & plane, double size_x, double size_y, int gx, int gy);

private:
  // PORT: reachability_services._generate_rm_callback
  void handle_generate_rm(
    const std::shared_ptr<curobov2_ros_interfaces::srv::GenerateRM::Request> request,
    std::shared_ptr<curobov2_ros_interfaces::srv::GenerateRM::Response> response);

  rclcpp::Node * node_;
  IKServices * ik_services_;
  std::string base_link_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::uint32_t map_id_ = 0;
  rclcpp::Service<curobov2_ros_interfaces::srv::GenerateRM>::SharedPtr generate_rm_srv_;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__CORE__REACHABILITY_SERVICES_HPP_

// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/lasers/laser_strategy.py
//
// Abstract base for laser strategies: TF buffer/listener, static-extrinsics
// parsing, and the shared `_resolve_pose_7` helper.
#ifndef CUROBO_CORE__LASERS__LASER_STRATEGY_HPP_
#define CUROBO_CORE__LASERS__LASER_STRATEGY_HPP_

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace curobo_core
{

class LaserStrategy
{
public:
  LaserStrategy(
    rclcpp::Node * node, const std::string & laser_name, const std::string & topic,
    const std::string & frame_id,
    const std::optional<std::vector<double>> & extrinsics = std::nullopt);
  virtual ~LaserStrategy() = default;

  const std::string & name() const {return name_;}

protected:
  // PORT: laser_strategy._parse_extrinsics (7-element pose or 4x4 matrix)
  std::optional<std::array<double, 7>> parse_extrinsics(
    const std::optional<std::vector<double>> & extrinsics);

  // PORT: laser_strategy._resolve_pose_7 - returns (has_pose, pose). A static
  // extrinsic yields (true, nullopt): the context holds the pose.
  std::pair<bool, std::optional<std::array<double, 7>>> resolve_pose_7(
    bool has_stamp, const builtin_interfaces::msg::Time & stamp);

  rclcpp::Node * node_;
  std::string name_;
  std::string topic_;
  std::string frame_id_;
  std::string base_frame_;
  std::optional<std::array<double, 7>> static_pose_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__LASERS__LASER_STRATEGY_HPP_

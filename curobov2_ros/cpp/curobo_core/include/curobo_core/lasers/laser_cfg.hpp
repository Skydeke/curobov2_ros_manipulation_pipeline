// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/lasers/laser_cfg.py
//
// Single source of truth for the `laser_*` perception params.
#ifndef CUROBO_CORE__LASERS__LASER_CFG_HPP_
#define CUROBO_CORE__LASERS__LASER_CFG_HPP_

#include <optional>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

namespace curobo_core
{

// PORT: laser_cfg.DEFAULT_LASER_FRAME_RATE_HZ
inline constexpr double DEFAULT_LASER_FRAME_RATE_HZ = 10.0;

// PORT: laser_cfg.PerceptionLaserCfg
struct PerceptionLaserCfg
{
  int laser_index = 0;
  std::string name;
  std::string topic;
  std::string frame_id;
  std::optional<std::vector<double>> extrinsics;
  std::string laser_type = "pointcloud";
  double frame_rate_hz = DEFAULT_LASER_FRAME_RATE_HZ;
  double range_min_m = 0.1;
  double range_max_m = 10.0;
  double elevation_min_rad = -1.5707963;
  double elevation_max_rad = 1.5707963;

  // PORT: PARAM_*
  static constexpr const char * PARAM_TOPIC = "laser_topic";
  static constexpr const char * PARAM_FRAME = "laser_frame";
  static constexpr const char * PARAM_EXTRINSICS = "laser_extrinsics";
  static constexpr const char * PARAM_LASER_TYPE = "laser_type";
  static constexpr const char * PARAM_FRAME_RATE = "laser_frame_rate_hz";
  static constexpr const char * PARAM_RANGE_MIN = "laser_range_min_m";
  static constexpr const char * PARAM_RANGE_MAX = "laser_range_max_m";
  static constexpr const char * PARAM_ELEVATION_MIN = "laser_elevation_min_rad";
  static constexpr const char * PARAM_ELEVATION_MAX = "laser_elevation_max_rad";
  static constexpr const char * PARAM_IMAGE_HEIGHT = "laser_image_height";
  static constexpr const char * PARAM_IMAGE_WIDTH = "laser_image_width";

  // PORT: declare / num_lasers / get_image_height / get_image_width / from_node
  static void declare(rclcpp::Node * node);
  static int num_lasers(rclcpp::Node * node);
  static int get_image_height(rclcpp::Node * node);
  static int get_image_width(rclcpp::Node * node);
  static PerceptionLaserCfg from_node(rclcpp::Node * node, int laser_index = 0);
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__LASERS__LASER_CFG_HPP_

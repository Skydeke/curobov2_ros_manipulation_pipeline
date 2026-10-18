// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/lasers/laser_cfg.py.
#include "curobo_core/lasers/laser_cfg.hpp"

#include "curobo_core/cameras/camera_cfg.hpp"   // parse_floats

namespace curobo_core
{

namespace
{
template<class T>
void declare_param(rclcpp::Node * node, const std::string & name, const T & value)
{
  if (!node->has_parameter(name)) {
    node->declare_parameter<T>(name, value);
  }
}

std::string entry_string(rclcpp::Node * node, const std::string & name, int index)
{
  if (!node->has_parameter(name)) {
    return "";
  }
  const auto values = node->get_parameter(name).as_string_array();
  if (index < 0 || static_cast<std::size_t>(index) >= values.size()) {
    return "";
  }
  const std::string value = values[static_cast<std::size_t>(index)];
  return value.find_first_not_of(" \t\n\r") == std::string::npos ? "" : value;
}

double entry_double(rclcpp::Node * node, const std::string & name, int index, double fallback)
{
  if (!node->has_parameter(name)) {
    return fallback;
  }
  const auto values = node->get_parameter(name).as_double_array();
  if (index < 0 || static_cast<std::size_t>(index) >= values.size()) {
    return fallback;
  }
  return values[static_cast<std::size_t>(index)];
}
}  // namespace

// PORT: PerceptionLaserCfg.declare
void PerceptionLaserCfg::declare(rclcpp::Node * node)
{
  declare_param(node, PARAM_TOPIC, std::vector<std::string>{""});
  declare_param(node, PARAM_FRAME, std::vector<std::string>{""});
  declare_param(node, PARAM_EXTRINSICS, std::vector<std::string>{""});
  declare_param(node, PARAM_LASER_TYPE, std::vector<std::string>{"pointcloud"});
  declare_param(node, PARAM_FRAME_RATE, std::vector<double>{0.0});
  declare_param(node, PARAM_RANGE_MIN, std::vector<double>{0.0});
  declare_param(node, PARAM_RANGE_MAX, std::vector<double>{0.0});
  declare_param(node, PARAM_ELEVATION_MIN, std::vector<double>{0.0});
  declare_param(node, PARAM_ELEVATION_MAX, std::vector<double>{0.0});
  declare_param(node, PARAM_IMAGE_HEIGHT, 0);
  declare_param(node, PARAM_IMAGE_WIDTH, 0);
}

// PORT: PerceptionLaserCfg.num_lasers
int PerceptionLaserCfg::num_lasers(rclcpp::Node * node)
{
  if (!node->has_parameter(PARAM_TOPIC)) {
    return 0;
  }
  return static_cast<int>(node->get_parameter(PARAM_TOPIC).as_string_array().size());
}

int PerceptionLaserCfg::get_image_height(rclcpp::Node * node)
{
  return node->has_parameter(PARAM_IMAGE_HEIGHT) ?
    static_cast<int>(node->get_parameter(PARAM_IMAGE_HEIGHT).as_int()) : 0;
}

int PerceptionLaserCfg::get_image_width(rclcpp::Node * node)
{
  return node->has_parameter(PARAM_IMAGE_WIDTH) ?
    static_cast<int>(node->get_parameter(PARAM_IMAGE_WIDTH).as_int()) : 0;
}

// PORT: PerceptionLaserCfg.from_node
PerceptionLaserCfg PerceptionLaserCfg::from_node(rclcpp::Node * node, int laser_index)
{
  PerceptionLaserCfg cfg;
  cfg.laser_index = laser_index;
  cfg.name = "laser_" + std::to_string(laser_index);
  cfg.topic = entry_string(node, PARAM_TOPIC, laser_index);
  cfg.frame_id = entry_string(node, PARAM_FRAME, laser_index);
  cfg.extrinsics = parse_floats(entry_string(node, PARAM_EXTRINSICS, laser_index));

  std::string type = entry_string(node, PARAM_LASER_TYPE, laser_index);
  if (type.empty()) {
    type = "pointcloud";
  }
  if (type != "pointcloud" && type != "scan") {
    type = "pointcloud";
  }
  cfg.laser_type = type;

  double rate = entry_double(node, PARAM_FRAME_RATE, laser_index, 0.0);
  if (rate <= 0.0) {
    rate = DEFAULT_LASER_FRAME_RATE_HZ;
  }
  cfg.frame_rate_hz = rate;

  const double rmin = entry_double(node, PARAM_RANGE_MIN, laser_index, 0.0);
  const double rmax = entry_double(node, PARAM_RANGE_MAX, laser_index, 0.0);
  cfg.range_min_m = rmin > 0.0 ? rmin : 0.1;
  cfg.range_max_m = rmax > 0.0 ? rmax : 10.0;

  const double emin = entry_double(node, PARAM_ELEVATION_MIN, laser_index, 0.0);
  const double emax = entry_double(node, PARAM_ELEVATION_MAX, laser_index, 0.0);
  cfg.elevation_min_rad = emin != 0.0 ? emin : -1.5707963;
  cfg.elevation_max_rad = emax != 0.0 ? emax : 1.5707963;
  return cfg;
}

}  // namespace curobo_core

// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/core/laser_system_manager.py.
#include "curobo_core/core/laser_system_manager.hpp"

#include <map>

#include "curobo_core/lasers/laser_pointcloud_strategy.hpp"
#include "curobo_core/lasers/laser_scan_strategy.hpp"

namespace curobo_core
{

LaserSystemManager::LaserSystemManager(
  rclcpp::Node * node, curobo_runtime::IBridge * bridge,
  curobo_runtime::RobotHandle handle, curobo_runtime::GpuLane * lane)
: node_(node), bridge_(bridge), handle_(handle), lane_(lane)
{
  configure();
}

// PORT: LaserSystemManager._configure
void LaserSystemManager::configure()
{
  PerceptionLaserCfg::declare(node_);
  const int num = PerceptionLaserCfg::num_lasers(node_);
  if (num == 0) {
    RCLCPP_INFO(node_->get_logger(), "No lasers configured (laser_topic unset).");
    return;
  }
  std::map<std::string, PerceptionLaserCfg> cfgs_by_name;
  for (int i = 0; i < num; ++i) {
    PerceptionLaserCfg cfg = PerceptionLaserCfg::from_node(node_, i);
    if (cfg.topic.empty()) {
      continue;
    }
    cfgs_by_name[cfg.name] = cfg;
  }
  if (cfgs_by_name.empty()) {
    RCLCPP_INFO(node_->get_logger(), "No lasers configured (all laser_topic entries empty).");
    return;
  }

  lidar_image_height_ = PerceptionLaserCfg::get_image_height(node_);
  lidar_image_width_ = PerceptionLaserCfg::get_image_width(node_);
  if (lidar_image_height_ <= 0 || lidar_image_width_ <= 0) {
    RCLCPP_ERROR(
      node_->get_logger(),
      "%zu laser(s) configured but 'laser_image_height' / 'laser_image_width' are not "
      "positive (%dx%d). LiDAR integration disabled — set both to the range-image "
      "resolution (e.g. 1x360 for a 2D scan, 64x1024 for a 3D lidar).",
      cfgs_by_name.size(), lidar_image_height_, lidar_image_width_);
    return;
  }

  laser_context_ = std::make_unique<LaserContext>(
    node_, lidar_image_height_, lidar_image_width_, bridge_, handle_, lane_);

  for (auto & [name, cfg] : cfgs_by_name) {
    if (lidar_image_height_ == 1) {
      RCLCPP_INFO(
        node_->get_logger(), "Laser '%s' is planar (H=1): forcing elevation range to [0, 0] rad.",
        cfg.name.c_str());
      cfg.elevation_min_rad = 0.0;
      cfg.elevation_max_rad = 0.0;
    }
    laser_cfgs_.push_back(cfg);
    RCLCPP_INFO(
      node_->get_logger(),
      "Laser '%s' (index %d): type=%s, topic=%s, frame=%s, range=[%.2f, %.2f] m, "
      "elevation=[%.3f, %.3f] rad, rate=%.1f Hz, image=%dx%d",
      cfg.name.c_str(), cfg.laser_index, cfg.laser_type.c_str(), cfg.topic.c_str(),
      cfg.frame_id.c_str(), cfg.range_min_m, cfg.range_max_m, cfg.elevation_min_rad,
      cfg.elevation_max_rad, cfg.frame_rate_hz, lidar_image_width_, lidar_image_height_);

    if (cfg.laser_type == "scan") {
      auto strategy = std::make_shared<LaserScanLaserStrategy>(
        node_, cfg.name, cfg.topic, cfg.frame_id, cfg.extrinsics, laser_context_.get(),
        lidar_image_height_, lidar_image_width_, cfg.range_min_m, cfg.range_max_m,
        cfg.elevation_min_rad, cfg.elevation_max_rad);
      laser_context_->add_laser(cfg.name, cfg.frame_rate_hz);
      (void)strategy;
    } else if (cfg.laser_type == "pointcloud") {
      auto strategy = std::make_shared<PointCloudLaserStrategy>(
        node_, cfg.name, cfg.topic, cfg.frame_id, cfg.extrinsics, laser_context_.get(),
        lidar_image_height_, lidar_image_width_, cfg.range_min_m, cfg.range_max_m,
        cfg.elevation_min_rad, cfg.elevation_max_rad);
      laser_context_->add_laser(cfg.name, cfg.frame_rate_hz);
      (void)strategy;
    } else {
      RCLCPP_ERROR(
        node_->get_logger(),
        "Unknown/unsupported laser type: %s (supported: 'pointcloud' (PointCloud2) or "
        "'scan' (LaserScan))", cfg.laser_type.c_str());
    }
  }
  laser_context_->set_laser_cfgs(laser_cfgs_);
  num_lasers_ = static_cast<int>(cfgs_by_name.size());
}

}  // namespace curobo_core

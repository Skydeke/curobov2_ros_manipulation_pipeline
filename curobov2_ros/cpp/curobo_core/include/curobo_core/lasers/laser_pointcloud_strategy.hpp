// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/lasers/laser_pointcloud_strategy.py
#ifndef CUROBO_CORE__LASERS__LASER_POINTCLOUD_STRATEGY_HPP_
#define CUROBO_CORE__LASERS__LASER_POINTCLOUD_STRATEGY_HPP_

#include <array>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#include <sensor_msgs/msg/point_cloud2.hpp>

#include "curobo_core/lasers/laser_context.hpp"
#include "curobo_core/lasers/laser_strategy.hpp"

namespace curobo_core
{

class PointCloudLaserStrategy : public LaserStrategy
{
public:
  PointCloudLaserStrategy(
    rclcpp::Node * node, const std::string & laser_name, const std::string & topic,
    const std::string & frame_id, const std::optional<std::vector<double>> & extrinsics,
    LaserContext * context, int image_height, int image_width, double range_min_m,
    double range_max_m, double elevation_min_rad, double elevation_max_rad,
    rclcpp::CallbackGroup::SharedPtr callback_group = nullptr);
  ~PointCloudLaserStrategy() override;

  // PORT: _callback_pointcloud
  void callback_pointcloud(const sensor_msgs::msg::PointCloud2 & msg);
  // PORT: destroy
  void destroy();

private:
  void integrate_loop();

  LaserContext * context_;
  int image_height_;
  int image_width_;
  double range_min_m_;
  double range_max_m_;
  double elevation_min_rad_;
  double elevation_max_rad_;

  std::deque<std::pair<std::vector<std::array<double, 3>>, builtin_interfaces::msg::Time>>
  queue_;
  std::mutex mutex_;
  std::condition_variable cond_;
  bool stop_ = false;
  std::thread worker_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__LASERS__LASER_POINTCLOUD_STRATEGY_HPP_

// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/lasers/laser_scan_strategy.py.
#include "curobo_core/lasers/laser_scan_strategy.hpp"

#include <cmath>

#include "curobo_core/lasers/range_image.hpp"

namespace curobo_core
{

LaserScanLaserStrategy::LaserScanLaserStrategy(
  rclcpp::Node * node, const std::string & laser_name, const std::string & topic,
  const std::string & frame_id, const std::optional<std::vector<double>> & extrinsics,
  LaserContext * context, int image_height, int image_width, double range_min_m,
  double range_max_m, double elevation_min_rad, double elevation_max_rad,
  rclcpp::CallbackGroup::SharedPtr callback_group)
: LaserStrategy(node, laser_name, topic, frame_id, extrinsics), context_(context),
  image_height_(image_height), image_width_(image_width), range_min_m_(range_min_m),
  range_max_m_(range_max_m), elevation_min_rad_(elevation_min_rad),
  elevation_max_rad_(elevation_max_rad)
{
  rclcpp::SubscriptionOptions options;
  if (callback_group) {
    options.callback_group = callback_group;
  }
  subscription_ = node_->create_subscription<sensor_msgs::msg::LaserScan>(
    topic_, 1,
    std::bind(
      &LaserScanLaserStrategy::callback_laserscan, this, std::placeholders::_1), options);
  worker_ = std::thread(&LaserScanLaserStrategy::integrate_loop, this);
  RCLCPP_INFO(
    node_->get_logger(),
    "[%s] LaserScanLaserStrategy subscribed to %s (HxW=%dx%d, range=[%.2f, %.2f] m)",
    name_.c_str(), topic_.c_str(), image_height_, image_width_, range_min_m_, range_max_m_);
}

LaserScanLaserStrategy::~LaserScanLaserStrategy()
{
  destroy();
}

// PORT: laser_scan_strategy.destroy
void LaserScanLaserStrategy::destroy()
{
  {
    std::lock_guard<std::mutex> lk(mutex_);
    stop_ = true;
  }
  cond_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
}

// PORT: _callback_laserscan (ranges -> [r cos, r sin, 0])
void LaserScanLaserStrategy::callback_laserscan(const sensor_msgs::msg::LaserScan & msg)
{
  try {
    std::vector<std::array<double, 3>> xyz;
    for (std::size_t i = 0; i < msg.ranges.size(); ++i) {
      const float r = msg.ranges[i];
      if (!std::isfinite(r)) {
        continue;
      }
      const double angle = msg.angle_min + static_cast<double>(i) * msg.angle_increment;
      xyz.push_back({static_cast<double>(r) * std::cos(angle),
        static_cast<double>(r) * std::sin(angle), 0.0});
    }
    {
      std::lock_guard<std::mutex> lk(mutex_);
      queue_.clear();
      queue_.emplace_back(std::move(xyz), msg.header.stamp);
    }
    cond_.notify_one();
  } catch (const std::exception & e) {
    RCLCPP_ERROR(node_->get_logger(), "[%s] LaserScan callback error: %s", name_.c_str(), e.what());
  }
}

// PORT: laser_scan_strategy._integrate_loop
void LaserScanLaserStrategy::integrate_loop()
{
  while (true) {
    std::vector<std::array<double, 3>> xyz;
    builtin_interfaces::msg::Time stamp;
    {
      std::unique_lock<std::mutex> lk(mutex_);
      cond_.wait(lk, [&] {return stop_ || !queue_.empty();});
      if (stop_ && queue_.empty()) {
        return;
      }
      if (queue_.empty()) {
        continue;
      }
      xyz = std::move(queue_.front().first);
      stamp = queue_.front().second;
      queue_.clear();
    }
    const auto [ok, pose] = resolve_pose_7(true, stamp);
    if (!ok) {
      continue;
    }
    try {
      const auto image = make_range_image(
        xyz, image_height_, image_width_, range_min_m_, range_max_m_, elevation_min_rad_,
        elevation_max_rad_);
      context_->publish_frame(name_, image.range, image.rgb, pose);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(node_->get_logger(), "[%s] Projection failed: %s", name_.c_str(), e.what());
    }
  }
}

}  // namespace curobo_core

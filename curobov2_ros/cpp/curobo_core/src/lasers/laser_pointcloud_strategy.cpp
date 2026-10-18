// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/lasers/laser_pointcloud_strategy.py.
#include "curobo_core/lasers/laser_pointcloud_strategy.hpp"

#include <sensor_msgs/point_cloud2_iterator.hpp>

#include "curobo_core/lasers/range_image.hpp"

namespace curobo_core
{

PointCloudLaserStrategy::PointCloudLaserStrategy(
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
  subscription_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
    topic_, 1,
    std::bind(
      &PointCloudLaserStrategy::callback_pointcloud, this, std::placeholders::_1), options);
  worker_ = std::thread(&PointCloudLaserStrategy::integrate_loop, this);
  RCLCPP_INFO(
    node_->get_logger(),
    "[%s] PointCloudLaserStrategy subscribed to %s (HxW=%dx%d, range=[%.2f, %.2f] m)",
    name_.c_str(), topic_.c_str(), image_height_, image_width_, range_min_m_, range_max_m_);
}

PointCloudLaserStrategy::~PointCloudLaserStrategy()
{
  destroy();
}

// PORT: laser_pointcloud_strategy.destroy
void PointCloudLaserStrategy::destroy()
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

// PORT: _callback_pointcloud (ros2_numpy.pointcloud2_to_array -> xyz)
void PointCloudLaserStrategy::callback_pointcloud(const sensor_msgs::msg::PointCloud2 & msg)
{
  try {
    std::vector<std::array<double, 3>> xyz;
    const std::size_t n = static_cast<std::size_t>(msg.width) * msg.height;
    xyz.reserve(n);
    sensor_msgs::PointCloud2ConstIterator<float> it_x(msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> it_y(msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> it_z(msg, "z");
    for (; it_x != it_x.end(); ++it_x, ++it_y, ++it_z) {
      xyz.push_back({static_cast<double>(*it_x), static_cast<double>(*it_y),
        static_cast<double>(*it_z)});
    }
    {
      std::lock_guard<std::mutex> lk(mutex_);
      queue_.clear();               // deque(maxlen=1)
      queue_.emplace_back(std::move(xyz), msg.header.stamp);
    }
    cond_.notify_one();
  } catch (const std::exception & e) {
    RCLCPP_ERROR(node_->get_logger(), "[%s] PointCloud callback error: %s", name_.c_str(), e.what());
  }
}

// PORT: laser_pointcloud_strategy._integrate_loop
void PointCloudLaserStrategy::integrate_loop()
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
    // PORT: resolve_pose_7 (static -> the context holds the pose).
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

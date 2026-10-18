// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/lasers/laser_context.py.
#include "curobo_runtime/runtime.hpp"
#include "curobo_core/lasers/laser_context.hpp"

#include <chrono>

namespace curobo_core
{

LaserContext::LaserContext(
  rclcpp::Node * node, int image_height, int image_width,
  curobo_runtime::IBridge * bridge, curobo_runtime::RobotHandle handle)
: node_(node), image_height_(image_height), image_width_(image_width),
  bridge_(bridge), handle_(handle)
{
  worker_ = std::thread(&LaserContext::integrate_loop, this);
}

LaserContext::~LaserContext()
{
  destroy();
}

// PORT: laser_context.destroy
void LaserContext::destroy()
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

// PORT: laser_context.add_laser
void LaserContext::add_laser(const std::string & name, double frame_rate_hz)
{
  std::lock_guard<std::mutex> lk(mutex_);
  frame_rates_[name] = frame_rate_hz;
  RCLCPP_INFO(
    node_->get_logger(), "Added laser strategy '%s', rate=%g Hz", name.c_str(),
    frame_rate_hz);
}

// PORT: laser_context.set_laser_cfgs
void LaserContext::set_laser_cfgs(const std::vector<PerceptionLaserCfg> & cfgs)
{
  std::lock_guard<std::mutex> lk(mutex_);
  cfgs_ = cfgs;
}

// PORT: laser_context.publish_frame
void LaserContext::publish_frame(
  const std::string & name, const std::vector<float> & range_image,
  const std::vector<std::uint8_t> &, const std::optional<std::array<double, 7>> & pose_7)
{
  {
    std::lock_guard<std::mutex> lk(mutex_);
    range_frames_[name] = range_image;
    if (pose_7.has_value()) {
      poses_[name] = *pose_7;
    }
  }
  cond_.notify_one();
}

// PORT: laser_context._integrate_loop / _maybe_integrate
void LaserContext::integrate_loop()
{
  while (true) {
    std::vector<PerceptionLaserCfg> cfgs;
    std::vector<std::array<double, 7>> poses;
    std::vector<float> batched;
    int n = 0;
    {
      std::unique_lock<std::mutex> lk(mutex_);
      cond_.wait_for(lk, std::chrono::milliseconds(50), [&] {return stop_;});
      if (stop_) {
        return;
      }
      // All configured lasers must have published at least once.
      bool ready = !cfgs_.empty();
      for (const auto & cfg : cfgs_) {
        if (range_frames_.find(cfg.name) == range_frames_.end()) {
          ready = false;
          break;
        }
      }
      if (!ready) {
        continue;
      }
      cfgs = cfgs_;
      n = static_cast<int>(cfgs.size());
      batched.reserve(static_cast<std::size_t>(n) * image_height_ * image_width_);
      for (const auto & cfg : cfgs) {
        const auto & frame = range_frames_[cfg.name];
        batched.insert(batched.end(), frame.begin(), frame.end());
        poses.push_back(poses_.count(cfg.name) ? poses_[cfg.name] :
          std::array<double, 7>{0, 0, 0, 1, 0, 0, 0});
      }
      // Consume the frames so the next integrate waits for fresh ones.
      range_frames_.clear();
    }

    curobo_runtime::TensorF32 range;
    range.shape = {n, image_height_, image_width_};
    range.data = batched;
    curobo_runtime::TensorF32 pose_t;
    pose_t.shape = {n, 7};
    for (const auto & p : poses) {
      for (double v : p) {
        pose_t.data.push_back(static_cast<float>(v));
      }
    }
    curobo_runtime::TensorF32 valid;
    valid.shape = {n, 2};
    curobo_runtime::TensorF32 elev;
    elev.shape = {n, 2};
    for (const auto & cfg : cfgs) {
      valid.data.push_back(static_cast<float>(cfg.range_min_m));
      valid.data.push_back(static_cast<float>(cfg.range_max_m));
      elev.data.push_back(static_cast<float>(cfg.elevation_min_rad));
      elev.data.push_back(static_cast<float>(cfg.elevation_max_rad));
    }
    try {
      auto gpu_guard = curobo_runtime::Runtime::instance().barrier().shared();
      bridge_->lidar_integrate(handle_, range, pose_t, valid, elev);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(node_->get_logger(), "Laser batch integrate failed: %s", e.what());
    }
  }
}

// PORT: laser_context.get_total_frame_rate_hz
double LaserContext::get_total_frame_rate_hz() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  double total = 0.0;
  for (const auto & [name, rate] : frame_rates_) {
    total += rate;
  }
  return total;
}

// PORT: laser_context.get_laser_names
std::vector<std::string> LaserContext::get_laser_names() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  std::vector<std::string> names;
  names.reserve(frame_rates_.size());
  for (const auto & [name, rate] : frame_rates_) {
    names.push_back(name);
  }
  return names;
}

}  // namespace curobo_core

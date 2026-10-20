// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/lasers/laser_context.py
//
// Merges every laser's freshest range image into ONE batched LidarObservation and
// integrates it through the bridge. All lasers share the range-image HxW.
#ifndef CUROBO_CORE__LASERS__LASER_CONTEXT_HPP_
#define CUROBO_CORE__LASERS__LASER_CONTEXT_HPP_

#include <array>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "curobo_core/lasers/laser_cfg.hpp"
#include "curobo_runtime/bridge.hpp"
#include "curobo_runtime/gpu_lanes.hpp"

namespace curobo_core
{

class LaserContext
{
public:
  LaserContext(
    rclcpp::Node * node, int image_height, int image_width,
    curobo_runtime::IBridge * bridge, curobo_runtime::RobotHandle handle,
    curobo_runtime::GpuLane * lane = nullptr);
  ~LaserContext();

  // PORT: laser_context.add_laser (rate bookkeeping; strategies built by
  // LaserSystemManager).
  void add_laser(const std::string & name, double frame_rate_hz);
  // PORT: laser_context.set_laser_cfgs
  void set_laser_cfgs(const std::vector<PerceptionLaserCfg> & cfgs);
  // PORT: laser_context.publish_frame
  void publish_frame(
    const std::string & name, const std::vector<float> & range_image,
    const std::vector<std::uint8_t> & rgb_image,
    const std::optional<std::array<double, 7>> & pose_7);
  // PORT: laser_context.get_total_frame_rate_hz / get_laser_names
  double get_total_frame_rate_hz() const;
  std::vector<std::string> get_laser_names() const;
  // PORT: laser_context.destroy
  void destroy();

private:
  void integrate_loop();

  rclcpp::Node * node_;
  int image_height_;
  int image_width_;
  curobo_runtime::IBridge * bridge_;
  curobo_runtime::RobotHandle handle_;
  // PORT: node.lane_perception (shared with the depth integrates).
  curobo_runtime::GpuLane * lane_ = nullptr;

  mutable std::mutex mutex_;
  std::condition_variable cond_;
  bool stop_ = false;
  std::map<std::string, double> frame_rates_;
  std::vector<PerceptionLaserCfg> cfgs_;
  std::map<std::string, std::vector<float>> range_frames_;
  std::map<std::string, std::array<double, 7>> poses_;
  std::thread worker_;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__LASERS__LASER_CONTEXT_HPP_

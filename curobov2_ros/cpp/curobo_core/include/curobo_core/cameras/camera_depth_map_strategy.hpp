// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/cameras/camera_depth_map_strategy.py
//
// Subscribes to a depth stream, decodes it on the ROS callback thread, resolves
// the camera pose via TF (or a static extrinsic), and integrates into the shared
// Mapper on a dedicated worker thread (dropping the oldest frame when a new one
// arrives). The Mapper lives in the bridge; this class calls
// `bridge->depth_integrate`.
#ifndef CUROBO_CORE__CAMERAS__CAMERA_DEPTH_MAP_STRATEGY_HPP_
#define CUROBO_CORE__CAMERAS__CAMERA_DEPTH_MAP_STRATEGY_HPP_

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

#include "curobo_core/cameras/camera_strategy.hpp"
#include "curobo_runtime/bridge.hpp"
#include "curobo_runtime/gpu_lanes.hpp"

namespace curobo_core
{


class DepthMapCameraStrategy : public CameraStrategy
{
public:
  DepthMapCameraStrategy(
    rclcpp::Node * node, const std::string & camera_name, const std::string & topic,
    const std::string & camera_info_topic, const std::string & frame_id,
    const std::optional<std::vector<double>> & intrinsics,
    const std::optional<std::vector<double>> & extrinsics,
    curobo_runtime::IBridge * bridge, curobo_runtime::RobotHandle handle,
    rclcpp::CallbackGroup::SharedPtr callback_group = nullptr,
    curobo_runtime::GpuLane * lane = nullptr,
    int mapper_image_width = 0, int mapper_image_height = 0,
    double max_update_rate_hz = 0.0);
  ~DepthMapCameraStrategy() override;

  // PORT: callback_depth_map
  void callback_depth_map(const sensor_msgs::msg::Image & msg);
  // PORT: destroy
  void destroy();

  // PORT: camera_depth_map_strategy's `node.perception_epoch += 1` after a
  // successful integrate - drives the ESDF freshness gate (should_refresh_world).
  void set_on_integrated(std::function<void()> cb) {on_integrated_ = std::move(cb);}

private:
  void integrate_loop();
  bool integrate_once(const curobo_runtime::TensorF32 & depth, const Pose7d & pose);

  curobo_runtime::IBridge * bridge_;
  curobo_runtime::RobotHandle handle_;
  curobo_runtime::GpuLane * lane_;
  std::string base_frame_;
  std::function<void()> on_integrated_;
  // Mapper projection resolution (mapper_image_width/height). A depth frame whose
  // (H, W) differs is resampled (and its intrinsics scaled) before integration.
  int mapper_image_width_ = 0;
  int mapper_image_height_ = 0;

  std::deque<std::pair<curobo_runtime::TensorF32, Pose7d>> queue_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cond_;
  bool stop_ = false;
  std::thread worker_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__CAMERAS__CAMERA_DEPTH_MAP_STRATEGY_HPP_

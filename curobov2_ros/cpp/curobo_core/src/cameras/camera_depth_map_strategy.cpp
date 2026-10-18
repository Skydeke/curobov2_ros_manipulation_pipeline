// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/cameras/camera_depth_map_strategy.py.
#include "curobo_runtime/runtime.hpp"
#include "curobo_core/cameras/camera_depth_map_strategy.hpp"

#include <chrono>

#include <tf2/exceptions.h>
#include <tf2/time.h>

#include "curobo_core/cameras/image_decode.hpp"
#include "curobo_core/core/frame_utils.hpp"

namespace curobo_core
{

DepthMapCameraStrategy::DepthMapCameraStrategy(
  rclcpp::Node * node, const std::string & camera_name, const std::string & topic,
  const std::string & camera_info_topic, const std::string & frame_id,
  const std::optional<std::vector<double>> & intrinsics,
  const std::optional<std::vector<double>> & extrinsics,
  curobo_runtime::IBridge * bridge, curobo_runtime::RobotHandle handle,
  rclcpp::CallbackGroup::SharedPtr callback_group, curobo_runtime::GpuLane * lane)
: CameraStrategy(node, camera_name, topic, camera_info_topic, frame_id, intrinsics, extrinsics),
  bridge_(bridge), handle_(handle), lane_(lane)
{
  base_frame_ = node_->has_parameter("base_link") ?
    node_->get_parameter("base_link").as_string() : "base_0";

  // PORT: create_subscription(Image, topic, callback_depth_map, 1, ...)
  rclcpp::SubscriptionOptions options;
  if (callback_group) {
    options.callback_group = callback_group;
  }
  subscription_ = node_->create_subscription<sensor_msgs::msg::Image>(
    topic_, 1,
    std::bind(&DepthMapCameraStrategy::callback_depth_map, this, std::placeholders::_1),
    options);
  worker_ = std::thread(&DepthMapCameraStrategy::integrate_loop, this);
  RCLCPP_INFO(
    node_->get_logger(), "DepthMap camera initialized with depth topic: %s", topic_.c_str());
}

DepthMapCameraStrategy::~DepthMapCameraStrategy()
{
  destroy();
}

// PORT: camera_depth_map_strategy.destroy
void DepthMapCameraStrategy::destroy()
{
  {
    std::lock_guard<std::mutex> lk(queue_mutex_);
    stop_ = true;
  }
  queue_cond_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
}

// PORT: camera_depth_map_strategy.callback_depth_map (decode + TF + handoff)
void DepthMapCameraStrategy::callback_depth_map(const sensor_msgs::msg::Image & msg)
{
  bool unsupported = false;
  const DepthFrame frame = decode_depth(msg, unsupported);
  if (unsupported) {
    RCLCPP_WARN_THROTTLE(
      node_->get_logger(), *node_->get_clock(), 2000,
      "Unsupported depth encoding: %s, trying 32FC1", msg.encoding.c_str());
  }

  Pose7d pose{};
  if (static_pose_.has_value()) {
    // PORT: "Using static extrinsics from config file"
    pose = *static_pose_;
  } else {
    try {
      geometry_msgs::msg::TransformStamped transform;
      if (!frame_id_.empty()) {
        try {
          transform = tf_buffer_->lookupTransform(
            base_frame_, frame_id_, tf2::TimePoint(
              std::chrono::nanoseconds(
                static_cast<std::int64_t>(msg.header.stamp.sec) * 1000000000 +
                msg.header.stamp.nanosec)));
        } catch (const tf2::TransformException &) {
          transform = tf_buffer_->lookupTransform(base_frame_, frame_id_, tf2::TimePointZero);
        }
      } else {
        return;
      }
      const auto & t = transform.transform.translation;
      const auto & r = transform.transform.rotation;
      const auto q = normalize_quat_wxyz({r.w, r.x, r.y, r.z});
      pose = {t.x, t.y, t.z, q[0], q[1], q[2], q[3]};
    } catch (const std::exception & ex) {
      RCLCPP_WARN_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 2000,
        "Could not transform %s to %s: %s. Dropping this depth frame (no integration).",
        base_frame_.c_str(), frame_id_.c_str(), ex.what());
      return;
    }
  }

  curobo_runtime::TensorF32 depth;
  depth.shape = {static_cast<std::int64_t>(frame.height), static_cast<std::int64_t>(frame.width)};
  depth.data = frame.data;
  {
    std::lock_guard<std::mutex> lk(queue_mutex_);
    queue_.clear();                       // deque(maxlen=1): keep the freshest
    queue_.emplace_back(std::move(depth), pose);
  }
  queue_cond_.notify_one();
}

// PORT: camera_depth_map_strategy._integrate_loop / _integrate_once
void DepthMapCameraStrategy::integrate_loop()
{
  while (true) {
    std::pair<curobo_runtime::TensorF32, Pose7d> item;
    {
      std::unique_lock<std::mutex> lk(queue_mutex_);
      queue_cond_.wait(lk, [&] {return stop_ || !queue_.empty();});
      if (stop_ && queue_.empty()) {
        return;
      }
      if (queue_.empty()) {
        continue;
      }
      item = std::move(queue_.front());
      queue_.clear();
    }
    integrate_once(item.first, item.second);
  }
}

bool DepthMapCameraStrategy::integrate_once(
  const curobo_runtime::TensorF32 & depth, const Pose7d & pose)
{
  try {
    const auto intrinsics = intrinsics_copy();
    if (!intrinsics.has_value()) {
      // PORT: no intrinsics -> nothing to project with.
      return false;
    }
    // PORT: camera_depth_map_strategy's torch_sync CPU/GPU ordering bridge
    // (off unless the torch_sync param is set).
    const auto maybe_sync = [this]() {
        if (node_ != nullptr && node_->has_parameter("torch_sync") &&
          node_->get_parameter("torch_sync").as_bool())
        {
          bridge_->torch_synchronize();
        }
      };
    curobo_runtime::TensorF32 k;
    k.shape = {3, 3};
    k.data.assign(intrinsics->begin(), intrinsics->end());
    if (lane_ != nullptr) {
      auto future = lane_->submit_if_idle(
        [this, &depth, &k, &pose]() {bridge_->depth_integrate(handle_, depth, k, pose);});
      // PORT: a busy lane drops the frame.
      if (future.has_value() && on_integrated_) {
        on_integrated_();   // PORT: node.perception_epoch += 1
      }
      if (future.has_value()) {
        maybe_sync();
      }
      return future.has_value();
    }
    // PORT: a depth integrate is GPU work -> shared barrier.
    auto gpu_guard = curobo_runtime::Runtime::instance().barrier().shared();
    bridge_->depth_integrate(handle_, depth, k, pose);
    if (on_integrated_) {
      on_integrated_();   // PORT: node.perception_epoch += 1
    }
    maybe_sync();
    return true;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(node_->get_logger(), "Depth integrate failed: %s", e.what());
    return false;
  }
}

}  // namespace curobo_core

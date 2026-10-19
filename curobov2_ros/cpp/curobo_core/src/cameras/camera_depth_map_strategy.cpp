// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/cameras/camera_depth_map_strategy.py.
#include "curobo_runtime/runtime.hpp"
#include "curobo_core/cameras/camera_depth_map_strategy.hpp"

#include <array>
#include <chrono>
#include <cstdint>

#include <tf2/exceptions.h>
#include <tf2/time.h>

#include "curobo_core/cameras/camera_cfg.hpp"
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
  rclcpp::CallbackGroup::SharedPtr callback_group, curobo_runtime::GpuLane * lane,
  int mapper_image_width, int mapper_image_height)
: CameraStrategy(node, camera_name, topic, camera_info_topic, frame_id, intrinsics, extrinsics),
  bridge_(bridge), handle_(handle), lane_(lane),
  mapper_image_width_(mapper_image_width), mapper_image_height_(mapper_image_height)
{
  // Map is integrated in `mapper_frame` (a fixed world frame) when set, else the
  // robot base — see integration_frame().
  base_frame_ = integration_frame(node_);

  // The mapper's depth-projection resolution is resolved by the owning
  // CameraSystemManager (explicit mapper_image_width/height, else the largest
  // configured camera) and passed in here; this strategy resamples every frame
  // to it so the Mapper's pre-allocated kernel accepts it. 0 keeps the native
  // frame size (unit tests / no mapper).

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
    // The frame the pixels are in: the camera_info header's frame when it has
    // arrived (driver-provided), else the configured `camera_frame`.
    const std::string camera = frame_for_lookup();
    if (camera.empty()) {
      return;
    }
    // Integrate in the fixed world frame when one exists (map -> odom -> base):
    // the first candidate TF can resolve is the Mapper's grid frame. Using the
    // base makes the map ride the robot, so a fixed frame is preferred.
    geometry_msgs::msg::TransformStamped transform;
    std::string base_frame;
    for (const std::string & candidate : integration_frame_candidates(node_)) {
      try {
        transform = get_transform(*tf_buffer_, candidate, camera);
        base_frame = candidate;
        break;
      } catch (const std::exception &) {
        // Not connected to the camera: try the next candidate.
      }
    }
    if (base_frame.empty()) {
      RCLCPP_WARN_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 2000,
        "Could not transform any of {map, odom, %s} to %s. Dropping this depth "
        "frame (no integration).", base_frame_.c_str(), camera.c_str());
      return;
    }
    const auto & t = transform.transform.translation;
    const auto & r = transform.transform.rotation;
    const auto q = normalize_quat_wxyz({r.w, r.x, r.y, r.z});
    pose = {t.x, t.y, t.z, q[0], q[1], q[2], q[3]};
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
    // PORT: camera_depth_map_strategy.callback_depth_map resizes the frame to
    // the Mapper's projection resolution (mapper_image_width/height) and scales
    // the intrinsics by the same factor. The Mapper's GPU kernel is allocated
    // once for exactly that (H, W), so a mismatched frame is rejected here
    // ("depth_images shape mismatch").
    curobo_runtime::TensorF32 frame = depth;
    std::array<double, 9> k_scaled = *intrinsics;
    if (mapper_image_width_ > 0 && mapper_image_height_ > 0 &&
      depth.shape.size() == 2 && depth.data.size() ==
      static_cast<std::size_t>(depth.shape[0] * depth.shape[1]))
    {
      const auto src_h = static_cast<std::uint32_t>(depth.shape[0]);
      const auto src_w = static_cast<std::uint32_t>(depth.shape[1]);
      const auto dst_h = static_cast<std::uint32_t>(mapper_image_height_);
      const auto dst_w = static_cast<std::uint32_t>(mapper_image_width_);
      if (src_h != dst_h || src_w != dst_w) {
        frame.data = resize_depth_bilinear(depth.data, src_h, src_w, dst_h, dst_w);
        frame.shape = {static_cast<std::int64_t>(dst_h), static_cast<std::int64_t>(dst_w)};
        const double sx = static_cast<double>(dst_w) / static_cast<double>(src_w);
        const double sy = static_cast<double>(dst_h) / static_cast<double>(src_h);
        k_scaled[0] *= sx;   // fx
        k_scaled[2] *= sx;   // cx
        k_scaled[4] *= sy;   // fy
        k_scaled[5] *= sy;   // cy
      }
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
    k.data.assign(k_scaled.begin(), k_scaled.end());
    if (lane_ != nullptr) {
      // Capture BY VALUE: the lane runs the job asynchronously, after this
      // frame's locals are gone, so a captured reference would dangle.
      auto future = lane_->submit_if_idle(
        [this, frame, k, pose]() {bridge_->depth_integrate(handle_, frame, k, pose);});
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
    bridge_->depth_integrate(handle_, frame, k, pose);
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

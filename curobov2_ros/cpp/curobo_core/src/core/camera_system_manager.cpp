// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/core/camera_system_manager.py.
#include "curobo_core/core/camera_system_manager.hpp"

#include <chrono>
#include <utility>

#include <rclcpp/wait_for_message.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include "curobo_core/cameras/camera_depth_map_strategy.hpp"

namespace curobo_core
{

CameraSystemManager::CameraSystemManager(
  rclcpp::Node * node, curobo_runtime::IBridge * bridge,
  curobo_runtime::RobotHandle handle, curobo_runtime::GpuLane * lane,
  std::function<void()> on_integrated,
  rclcpp::CallbackGroup::SharedPtr callback_group)
: node_(node), bridge_(bridge), handle_(handle), lane_(lane),
  on_integrated_(std::move(on_integrated)),
  callback_group_(std::move(callback_group))
{
  configure();
}

// Probe the configured cameras' camera_info topics and keep the LARGEST size.
// The shared Mapper allocates its projective scratch buffer ONCE for a single
// (H, W), so the kernel is sized to the best camera and every other camera is
// resampled to it downstream. Bounded per camera so a silent/absent topic
// cannot hang startup.
std::pair<int, int> CameraSystemManager::detect_max_camera_resolution(
  const std::vector<PerceptionCameraCfg> & cfgs)
{
  int best_w = 0;
  int best_h = 0;
  if (!node_->has_parameter("mapper_resolution_probe_timeout_s")) {
    node_->declare_parameter("mapper_resolution_probe_timeout_s", 3.0);
  }
  const double timeout_s =
    node_->get_parameter("mapper_resolution_probe_timeout_s").as_double();
  for (const auto & cfg : cfgs) {
    if (cfg.camera_info_topic.empty()) {
      continue;
    }
    // Best-effort QoS: compatible whether the driver publishes camera_info
    // reliable or best-effort.
    auto sub = node_->create_subscription<sensor_msgs::msg::CameraInfo>(
      cfg.camera_info_topic, rclcpp::SensorDataQoS(),
      [](sensor_msgs::msg::CameraInfo::ConstSharedPtr) {});
    sensor_msgs::msg::CameraInfo info;
    bool ok = false;
    try {
      ok = rclcpp::wait_for_message(
        info, sub, node_->get_node_base_interface()->get_context(),
        std::chrono::duration<double>(timeout_s));
    } catch (const std::exception & e) {
      RCLCPP_WARN(
        node_->get_logger(), "camera_info probe on '%s' failed: %s",
        cfg.camera_info_topic.c_str(), e.what());
    }
    if (!ok || info.width == 0 || info.height == 0) {
      RCLCPP_WARN(
        node_->get_logger(),
        "No camera_info on '%s' within %.1fs; camera '%s' does not constrain "
        "the mapper resolution", cfg.camera_info_topic.c_str(), timeout_s,
        cfg.name.c_str());
      continue;
    }
    RCLCPP_INFO(
      node_->get_logger(), "Camera '%s' reports %ux%u on %s", cfg.name.c_str(),
      info.width, info.height, cfg.camera_info_topic.c_str());
    if (static_cast<long>(info.width) * static_cast<long>(info.height) >
      static_cast<long>(best_w) * best_h)
    {
      best_w = static_cast<int>(info.width);
      best_h = static_cast<int>(info.height);
    }
  }
  return {best_w, best_h};
}

// PORT: CameraSystemManager._configure
void CameraSystemManager::configure()
{
  PerceptionCameraCfg::declare(node_);
  const int num = PerceptionCameraCfg::num_cameras(node_);
  std::vector<PerceptionCameraCfg> cfgs;
  for (int i = 0; i < num; ++i) {
    PerceptionCameraCfg cfg = PerceptionCameraCfg::from_node(node_, i);
    if (cfg.depth_topic.empty()) {
      continue;   // inactive slot
    }
    cfgs.push_back(cfg);
  }
  camera_cfgs_ = cfgs;

  // Resolve the Mapper's depth-projection resolution BEFORE the strategies are
  // built, so the strategies and the Mapper agree on one (H, W). An explicit
  // mapper_image_width/height (>0) wins; otherwise take the LARGEST size any
  // configured camera reports (the kernel is sized to the best camera, the
  // rest are resampled to it). Keep a positive fallback so a camera whose
  // camera_info never arrives still matches the Mapper.
  mapper_image_width_ = node_->has_parameter("mapper_image_width") ?
    static_cast<int>(node_->get_parameter("mapper_image_width").as_int()) : 0;
  mapper_image_height_ = node_->has_parameter("mapper_image_height") ?
    static_cast<int>(node_->get_parameter("mapper_image_height").as_int()) : 0;

  if (mapper_image_width_ <= 0 || mapper_image_height_ <= 0) {
    const auto detected = detect_max_camera_resolution(cfgs);
    if (detected.first > 0 && detected.second > 0) {
      mapper_image_width_ = detected.first;
      mapper_image_height_ = detected.second;
    }
  }
  if (!cfgs.empty() && (mapper_image_width_ <= 0 || mapper_image_height_ <= 0)) {
    mapper_image_width_ = DEFAULT_MAPPER_IMAGE_WIDTH;
    mapper_image_height_ = DEFAULT_MAPPER_IMAGE_HEIGHT;
    RCLCPP_WARN(
      node_->get_logger(),
      "No camera reported a resolution and mapper_image_width/height are unset; "
      "mapper depth projection falls back to %dx%d",
      mapper_image_width_, mapper_image_height_);
  }
  if (mapper_image_width_ > 0 && mapper_image_height_ > 0) {
    RCLCPP_INFO(
      node_->get_logger(), "Mapper depth projection resolution: %dx%d",
      mapper_image_width_, mapper_image_height_);
  }

  if (cfgs.empty()) {
    RCLCPP_WARN(
      node_->get_logger(),
      "No camera feeds the Mapper: set 'camera_topic' (with a camera_purpose of "
      "'all' or 'esdf') at launch.");
    camera_context_ = std::make_unique<CameraContext>(node_);
    return;
  }

  camera_context_ = std::make_unique<CameraContext>(node_);
  // PORT: callback_group=getattr(node, '_perception_callback_group', None).
  rclcpp::CallbackGroup::SharedPtr group = callback_group_;
  for (const auto & cfg : cfgs) {
    RCLCPP_INFO(
      node_->get_logger(),
      "Camera '%s' (index %d): raw=%s, purpose=%s, mapper_input=%s, "
      "frame_rate_hz=%.1f, frame=%s",
      cfg.name.c_str(), cfg.camera_index, cfg.depth_topic.c_str(), cfg.purpose.c_str(),
      cfg.mapper_topic.c_str(), cfg.frame_rate_hz, cfg.frame_id.c_str());

    if (!cfg.for_esdf()) {
      continue;   // segmentation-only camera: no mapper strategy
    }
    auto strategy = std::make_shared<DepthMapCameraStrategy>(
      node_, cfg.name, cfg.mapper_topic, cfg.camera_info_topic, cfg.frame_id,
      cfg.intrinsics, cfg.extrinsics, bridge_, handle_, group, lane_,
      mapper_image_width_, mapper_image_height_);
    // PORT: the depth strategy bumps node.perception_epoch on integrate.
    strategy->set_on_integrated(on_integrated_);
    camera_context_->add_camera(cfg.name, strategy, cfg.frame_rate_hz);
  }
}

}  // namespace curobo_core

// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/core/camera_system_manager.py.
#include "curobo_core/core/camera_system_manager.hpp"

#include "curobo_core/cameras/camera_depth_map_strategy.hpp"

namespace curobo_core
{

CameraSystemManager::CameraSystemManager(
  rclcpp::Node * node, curobo_runtime::IBridge * bridge,
  curobo_runtime::RobotHandle handle, curobo_runtime::GpuLane * lane,
  std::function<void()> on_integrated)
: node_(node), bridge_(bridge), handle_(handle), lane_(lane),
  on_integrated_(std::move(on_integrated))
{
  configure();
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
  rclcpp::CallbackGroup::SharedPtr group = nullptr;
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
      cfg.intrinsics, cfg.extrinsics, bridge_, handle_, group, lane_);
    // PORT: the depth strategy bumps node.perception_epoch on integrate.
    strategy->set_on_integrated(on_integrated_);
    camera_context_->add_camera(cfg.name, strategy, cfg.frame_rate_hz);
  }
}

}  // namespace curobo_core

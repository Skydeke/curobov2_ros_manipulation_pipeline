// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/core/camera_system_manager.py
//
// Owns the perception cameras: declares/reads the `camera_*` param block, builds
// the CameraContext and one DepthMapCameraStrategy per mapper-feeding camera.
#ifndef CUROBO_CORE__CORE__CAMERA_SYSTEM_MANAGER_HPP_
#define CUROBO_CORE__CORE__CAMERA_SYSTEM_MANAGER_HPP_

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "curobo_core/cameras/camera_cfg.hpp"
#include "curobo_core/cameras/camera_context.hpp"
#include "curobo_runtime/bridge.hpp"
#include "curobo_runtime/gpu_lanes.hpp"

namespace curobo_core
{


class CameraSystemManager
{
public:
  CameraSystemManager(
    rclcpp::Node * node, curobo_runtime::IBridge * bridge,
    curobo_runtime::RobotHandle handle, curobo_runtime::GpuLane * lane = nullptr,
    std::function<void()> on_integrated = nullptr,
    rclcpp::CallbackGroup::SharedPtr callback_group = nullptr);

  CameraContext * get_camera_context() {return camera_context_.get();}
  const std::vector<PerceptionCameraCfg> & camera_cfgs() const {return camera_cfgs_;}

  /// Depth-projection resolution the shared Mapper is (or will be) built at.
  /// Resolved in configure(): an explicit mapper_image_width/height (>0) wins,
  /// otherwise the LARGEST size any configured camera reports on its
  /// camera_info topic. 0 means "not resolved" (no camera configured), and the
  /// node then falls back to DEFAULT_MAPPER_IMAGE_*.
  int mapper_image_width() const {return mapper_image_width_;}
  int mapper_image_height() const {return mapper_image_height_;}

private:
  void configure();

  /// Probe each configured camera's camera_info topic (bounded) and return the
  /// largest reported (width, height); {0, 0} when none answers.
  std::pair<int, int> detect_max_camera_resolution(
    const std::vector<PerceptionCameraCfg> & cfgs);

  rclcpp::Node * node_;
  curobo_runtime::IBridge * bridge_;
  curobo_runtime::RobotHandle handle_;
  curobo_runtime::GpuLane * lane_;
  std::function<void()> on_integrated_;
  rclcpp::CallbackGroup::SharedPtr callback_group_;
  std::unique_ptr<CameraContext> camera_context_;
  std::vector<PerceptionCameraCfg> camera_cfgs_;
  int mapper_image_width_ = 0;
  int mapper_image_height_ = 0;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__CORE__CAMERA_SYSTEM_MANAGER_HPP_

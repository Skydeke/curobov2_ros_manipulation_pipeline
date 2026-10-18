// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/core/camera_system_manager.py
//
// Owns the perception cameras: declares/reads the `camera_*` param block, builds
// the CameraContext and one DepthMapCameraStrategy per mapper-feeding camera.
#ifndef CUROBO_CORE__CORE__CAMERA_SYSTEM_MANAGER_HPP_
#define CUROBO_CORE__CORE__CAMERA_SYSTEM_MANAGER_HPP_

#include <functional>
#include <memory>
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
    std::function<void()> on_integrated = nullptr);

  CameraContext * get_camera_context() {return camera_context_.get();}
  const std::vector<PerceptionCameraCfg> & camera_cfgs() const {return camera_cfgs_;}

private:
  void configure();

  rclcpp::Node * node_;
  curobo_runtime::IBridge * bridge_;
  curobo_runtime::RobotHandle handle_;
  curobo_runtime::GpuLane * lane_;
  std::function<void()> on_integrated_;
  std::unique_ptr<CameraContext> camera_context_;
  std::vector<PerceptionCameraCfg> camera_cfgs_;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__CORE__CAMERA_SYSTEM_MANAGER_HPP_

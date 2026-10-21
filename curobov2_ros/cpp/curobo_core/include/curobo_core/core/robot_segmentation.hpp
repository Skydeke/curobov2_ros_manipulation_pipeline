// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/core/robot_segmentation.py
//
// Owns the shared mask dict and the set_mask / remove_mask service callbacks,
// and builds one RobotSegmentationCameraStrategy per segmentation camera.
#ifndef CUROBO_CORE__CORE__ROBOT_SEGMENTATION_HPP_
#define CUROBO_CORE__CORE__ROBOT_SEGMENTATION_HPP_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "curobov2_ros_interfaces/srv/remove_object.hpp"
#include "curobov2_ros_interfaces/srv/set_mask.hpp"
#include "curobo_core/cameras/camera_cfg.hpp"
#include "curobo_core/cameras/camera_robot_segmentation_strategy.hpp"
#include "curobo_runtime/bridge.hpp"

namespace curobo_core
{

class RobotSegmentation
{
public:
  RobotSegmentation(
    rclcpp::Node * node, curobo_runtime::IBridge * bridge,
    curobo_runtime::RobotHandle handle, const std::vector<PerceptionCameraCfg> & camera_cfgs,
    std::function<std::vector<double>()> joint_pose,
    rclcpp::CallbackGroup::SharedPtr callback_group = nullptr);

  // PORT: RobotSegmentation.set_mask_callback
  void handle_set_mask(
    const std::shared_ptr<curobov2_ros_interfaces::srv::SetMask::Request> request,
    std::shared_ptr<curobov2_ros_interfaces::srv::SetMask::Response> response);
  // PORT: RobotSegmentation.remove_mask_callback
  void handle_remove_mask(
    const std::shared_ptr<curobov2_ros_interfaces::srv::RemoveObject::Request> request,
    std::shared_ptr<curobov2_ros_interfaces::srv::RemoveObject::Response> response);

  // PORT: the masks dict serialized for the bridge (masks_json provider).
  std::string masks_json() const;
  void destroy();

  std::size_t mask_count() const {return masks_.size();}

private:
  bool mesh_file_exists(const std::string & path) const;

  rclcpp::Node * node_;
  curobo_runtime::IBridge * bridge_;
  curobo_runtime::RobotHandle handle_;
  std::string base_frame_;
  double distance_threshold_ = 0.05;
  double mask_margin_ = 0.0;
  std::map<std::string, std::string> masks_;   // name -> mask JSON
  std::vector<std::shared_ptr<RobotSegmentationCameraStrategy>> streams_;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__CORE__ROBOT_SEGMENTATION_HPP_

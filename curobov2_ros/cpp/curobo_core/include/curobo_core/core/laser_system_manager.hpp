// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/core/laser_system_manager.py
#ifndef CUROBO_CORE__CORE__LASER_SYSTEM_MANAGER_HPP_
#define CUROBO_CORE__CORE__LASER_SYSTEM_MANAGER_HPP_

#include <memory>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "curobo_core/lasers/laser_cfg.hpp"
#include "curobo_core/lasers/laser_context.hpp"
#include "curobo_runtime/bridge.hpp"

namespace curobo_core
{

class LaserSystemManager
{
public:
  LaserSystemManager(
    rclcpp::Node * node, curobo_runtime::IBridge * bridge,
    curobo_runtime::RobotHandle handle);

  LaserContext * get_laser_context() {return laser_context_.get();}
  const std::vector<PerceptionLaserCfg> & laser_cfgs() const {return laser_cfgs_;}
  int num_lasers() const {return num_lasers_;}

private:
  void configure();

  rclcpp::Node * node_;
  curobo_runtime::IBridge * bridge_;
  curobo_runtime::RobotHandle handle_;
  std::unique_ptr<LaserContext> laser_context_;
  std::vector<PerceptionLaserCfg> laser_cfgs_;
  int num_lasers_ = 0;
  int lidar_image_height_ = 0;
  int lidar_image_width_ = 0;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__CORE__LASER_SYSTEM_MANAGER_HPP_

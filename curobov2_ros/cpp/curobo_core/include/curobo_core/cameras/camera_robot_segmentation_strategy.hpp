// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/cameras/camera_robot_segmentation_strategy.py
//
// Masks the robot (and user mask shapes) out of the raw depth stream and
// republishes the masked depth on a derived topic. The masking itself (FK
// spheres + shape tests) is one bridge call (`robot_segment`).
#ifndef CUROBO_CORE__CAMERAS__CAMERA_ROBOT_SEGMENTATION_STRATEGY_HPP_
#define CUROBO_CORE__CAMERAS__CAMERA_ROBOT_SEGMENTATION_STRATEGY_HPP_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "curobo_core/cameras/camera_strategy.hpp"
#include "curobo_core/cameras/image_decode.hpp"
#include "curobo_runtime/bridge.hpp"

namespace curobo_core
{

class RobotSegmentationCameraStrategy : public CameraStrategy
{
public:
  RobotSegmentationCameraStrategy(
    rclcpp::Node * node, const std::string & camera_name, const std::string & topic,
    const std::string & camera_info_topic, const std::string & frame_id,
    const std::optional<std::vector<double>> & intrinsics,
    const std::optional<std::vector<double>> & extrinsics,
    curobo_runtime::IBridge * bridge, curobo_runtime::RobotHandle handle,
    std::function<std::vector<double>(std::int64_t)> joint_pose_at,
    std::function<std::string()> masks_json,
    rclcpp::CallbackGroup::SharedPtr callback_group = nullptr);
  ~RobotSegmentationCameraStrategy() override;

  // PORT: callback_depth / _mask_depth_image
  void callback_depth(const sensor_msgs::msg::Image & msg);
  void destroy();

  const std::string & output_topic() const {return output_topic_;}

private:
  void publish_robot_pointcloud(
    const std::vector<float> & depth, const std::vector<float> & masked,
    std::uint32_t h, std::uint32_t w, const std::array<double, 9> & k,
    const std::array<double, 7> & cam_pose, const builtin_interfaces::msg::Time & stamp);
  void publish_masked(
    const std::vector<float> & masked, std::uint32_t h, std::uint32_t w,
    const std::string & encoding, const builtin_interfaces::msg::Time & stamp);

  curobo_runtime::IBridge * bridge_;
  curobo_runtime::RobotHandle handle_;
  std::function<std::vector<double>(std::int64_t)> joint_pose_at_;
  std::function<std::string()> masks_json_;
  std::string output_topic_;
  std::string depth_frame_id_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
  // PORT: _debug_topic -> robot_pointcloud_debug (the masked-OUT points).
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr robot_pointcloud_pub_;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__CAMERAS__CAMERA_ROBOT_SEGMENTATION_STRATEGY_HPP_

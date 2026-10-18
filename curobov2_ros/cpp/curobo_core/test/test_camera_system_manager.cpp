// SPDX-License-Identifier: Apache-2.0
// CameraSystemManager: params -> camera strategies. No GPU (FakeBridge).
#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "curobo_core/core/camera_system_manager.hpp"
#include "curobo_runtime/fake_bridge.hpp"

using curobo_core::CameraSystemManager;

TEST(CameraSystemManager, BuildsAMapperCameraFromParams)
{
  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
  }
  rclcpp::NodeOptions options;
  options.parameter_overrides({
    rclcpp::Parameter("camera_topic", std::vector<std::string>{"/cam/depth/image"}),
    rclcpp::Parameter("camera_info_topic", std::vector<std::string>{"/cam/depth/camera_info"}),
    rclcpp::Parameter("camera_frame_rate_hz", std::vector<double>{15.0}),
    rclcpp::Parameter("camera_purpose", std::vector<std::string>{"all"}),
    rclcpp::Parameter("enable_robot_segmentation", true),
  });
  auto node = std::make_shared<rclcpp::Node>("unified_planner", options);
  curobo_runtime::FakeBridge bridge;
  CameraSystemManager manager(node.get(), &bridge, 1);

  ASSERT_EQ(manager.camera_cfgs().size(), 1u);
  EXPECT_EQ(manager.camera_cfgs()[0].name, "depth_camera_0");
  auto * context = manager.get_camera_context();
  ASSERT_NE(context, nullptr);
  EXPECT_EQ(context->get_camera_names(), (std::vector<std::string>{"depth_camera_0"}));
  // frame-rate aggregate drives the TSDF decay normalisation.
  EXPECT_DOUBLE_EQ(context->get_total_frame_rate_hz(), 15.0);
}

TEST(CameraSystemManager, EmptyTopicSlotIsInactive)
{
  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
  }
  rclcpp::NodeOptions options;
  options.parameter_overrides({
    rclcpp::Parameter("camera_topic", std::vector<std::string>{""}),
  });
  auto node = std::make_shared<rclcpp::Node>("unified_planner", options);
  curobo_runtime::FakeBridge bridge;
  CameraSystemManager manager(node.get(), &bridge, 1);
  EXPECT_TRUE(manager.camera_cfgs().empty());
  EXPECT_TRUE(manager.get_camera_context()->get_camera_names().empty());
}

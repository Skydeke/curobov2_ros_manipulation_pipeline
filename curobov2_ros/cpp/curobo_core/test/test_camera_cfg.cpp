// SPDX-License-Identifier: Apache-2.0
// Unit tests for camera_cfg (pure topic/param logic). No GPU.
#include <gtest/gtest.h>

#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "curobo_core/cameras/camera_cfg.hpp"

using curobo_core::masked_depth_topic;
using curobo_core::parse_floats;
using curobo_core::PerceptionCameraCfg;

TEST(CameraCfg, MaskedDepthTopicReplacesTheLeaf)
{
  // PORT: masked_depth_topic
  EXPECT_EQ(masked_depth_topic("/kortex_vision/depth/image"), "/kortex_vision/depth/masked_depth");
  EXPECT_EQ(masked_depth_topic("image"), "masked_depth");
  EXPECT_EQ(masked_depth_topic(""), "");
  EXPECT_EQ(masked_depth_topic("/a/b/"), "/a/masked_depth");
}

TEST(CameraCfg, ParseFloats)
{
  // PORT: _parse_floats
  auto v = parse_floats("1,2,3");
  ASSERT_TRUE(v.has_value());
  EXPECT_EQ(*v, (std::vector<double>{1.0, 2.0, 3.0}));
  auto k = parse_floats("1,0,2,0,3,4,0,0,1");
  ASSERT_TRUE(k.has_value());
  EXPECT_EQ(k->size(), 9u);
  EXPECT_FALSE(parse_floats("").has_value());
  EXPECT_FALSE(parse_floats("not,a,float").has_value());
}

TEST(CameraCfg, FromNodeDerivesTopicsAndDefaults)
{
  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
  }
  rclcpp::NodeOptions options;
  options.parameter_overrides({
    rclcpp::Parameter("camera_topic", std::vector<std::string>{"/cam/depth/image"}),
    rclcpp::Parameter("camera_info_topic", std::vector<std::string>{"/cam/depth/camera_info"}),
    rclcpp::Parameter("enable_robot_segmentation", true),
  });
  auto node = std::make_shared<rclcpp::Node>("camera_cfg_test", options);

  PerceptionCameraCfg::declare(node.get());
  EXPECT_EQ(PerceptionCameraCfg::num_cameras(node.get()), 1);

  const auto cfg = PerceptionCameraCfg::from_node(node.get(), 0);
  EXPECT_EQ(cfg.camera_index, 0);
  EXPECT_EQ(cfg.name, "depth_camera_0");
  EXPECT_EQ(cfg.depth_topic, "/cam/depth/image");
  EXPECT_EQ(cfg.camera_info_topic, "/cam/depth/camera_info");
  EXPECT_EQ(cfg.purpose, "all");
  EXPECT_TRUE(cfg.for_esdf());
  EXPECT_TRUE(cfg.for_segmentation());
  EXPECT_EQ(cfg.masked_output_topic, "/cam/depth/masked_depth");
  // segmentation on + purpose all -> the mapper consumes the MASKED topic.
  EXPECT_EQ(cfg.mapper_topic, "/cam/depth/masked_depth");
  EXPECT_DOUBLE_EQ(cfg.frame_rate_hz, 30.0);
}

// SPDX-License-Identifier: Apache-2.0
// Depth-map camera strategy + camera context registry (FakeBridge: the Mapper
// lives behind the bridge, so no GPU). The node is spun with a single executor
// (tests may spin).
#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "curobo_core/cameras/camera_context.hpp"
#include "curobo_core/cameras/camera_depth_map_strategy.hpp"
#include "curobo_runtime/fake_bridge.hpp"

using namespace std::chrono_literals;
using curobo_core::CameraContext;
using curobo_core::DepthMapCameraStrategy;

namespace
{
rclcpp::Node::SharedPtr make_node()
{
  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
  }
  return std::make_shared<rclcpp::Node>("unified_planner");
}

sensor_msgs::msg::Image depth_image(std::uint32_t h, std::uint32_t w)
{
  sensor_msgs::msg::Image msg;
  msg.height = h;
  msg.width = w;
  msg.encoding = "32FC1";
  msg.step = w * 4;
  msg.data.resize(h * w * 4);
  auto * p = reinterpret_cast<float *>(msg.data.data());
  for (std::uint32_t i = 0; i < h * w; ++i) {
    p[i] = 1.0f;
  }
  return msg;
}
}  // namespace

TEST(CameraDepth, IntegratesADepthFrameThroughTheBridge)
{
  auto node = make_node();
  curobo_runtime::FakeBridge bridge;
  std::vector<double> intrinsics{100.0, 0, 50.0, 0, 100.0, 50.0, 0, 0, 1};
  std::vector<double> extrinsics{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
  DepthMapCameraStrategy camera(
    node.get(), "depth_camera_0", "/cam/depth/image", "/cam/depth/camera_info", "",
    intrinsics, extrinsics, &bridge, 1);

  auto pub = node->create_publisher<sensor_msgs::msg::Image>("/cam/depth/image", 1);
  pub->publish(depth_image(4, 4));

  const auto deadline = std::chrono::steady_clock::now() + 3s;
  while (std::chrono::steady_clock::now() < deadline && bridge.depth_integrates() == 0) {
    rclcpp::spin_some(node);
    std::this_thread::sleep_for(10ms);
  }
  EXPECT_GE(bridge.depth_integrates(), 1);
}

TEST(CameraContext, RegistryAndFrameRateAggregate)
{
  auto node = make_node();
  curobo_runtime::FakeBridge bridge;
  CameraContext context(node.get());
  auto camera = std::make_shared<DepthMapCameraStrategy>(
    node.get(), "depth_camera_0", "/cam/depth/image", "/cam/depth/camera_info", "",
    std::vector<double>{100, 0, 50, 0, 100, 50, 0, 0, 1},
    std::vector<double>{0, 0, 0, 1, 0, 0, 0}, &bridge, 1);
  context.add_camera("depth_camera_0", camera, 30.0);
  EXPECT_EQ(context.get_camera_names(), (std::vector<std::string>{"depth_camera_0"}));
  EXPECT_DOUBLE_EQ(context.get_total_frame_rate_hz(), 30.0);
  EXPECT_NE(context.get("depth_camera_0"), nullptr);
  context.remove_camera("depth_camera_0");
  EXPECT_TRUE(context.get_camera_names().empty());
}

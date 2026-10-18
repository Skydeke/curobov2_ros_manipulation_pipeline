// SPDX-License-Identifier: Apache-2.0
// Laser config + context (FakeBridge: the Mapper lives behind the bridge).
#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "curobo_core/lasers/laser_cfg.hpp"
#include "curobo_core/lasers/laser_context.hpp"
#include "curobo_runtime/fake_bridge.hpp"

using namespace std::chrono_literals;
using curobo_core::LaserContext;
using curobo_core::PerceptionLaserCfg;

namespace
{
rclcpp::Node::SharedPtr node()
{
  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
  }
  return std::make_shared<rclcpp::Node>("unified_planner");
}
}  // namespace

TEST(LaserCfg, DefaultsFromNode)
{
  auto n = node();
  PerceptionLaserCfg::declare(n.get());
  PerceptionLaserCfg cfg = PerceptionLaserCfg::from_node(n.get(), 0);
  EXPECT_EQ(cfg.name, "laser_0");
  EXPECT_EQ(cfg.laser_type, "pointcloud");
  EXPECT_DOUBLE_EQ(cfg.frame_rate_hz, 10.0);
  EXPECT_DOUBLE_EQ(cfg.range_min_m, 0.1);
  EXPECT_DOUBLE_EQ(cfg.range_max_m, 10.0);
  EXPECT_NEAR(cfg.elevation_min_rad, -1.5707963, 1e-9);
  EXPECT_NEAR(cfg.elevation_max_rad, 1.5707963, 1e-9);
}

TEST(LaserContext, PublishFrameIntegratesThroughTheBridge)
{
  auto n = node();
  curobo_runtime::FakeBridge bridge;
  LaserContext context(n.get(), /*H=*/1, /*W=*/4, &bridge, 1);
  PerceptionLaserCfg cfg;
  cfg.name = "laser_0";
  cfg.topic = "/lidar";
  context.set_laser_cfgs({cfg});
  context.add_laser("laser_0", 10.0);
  EXPECT_DOUBLE_EQ(context.get_total_frame_rate_hz(), 10.0);

  std::vector<float> range(4, 0.0f);
  range[2] = 1.0f;
  std::vector<std::uint8_t> rgb(12, 0);
  context.publish_frame("laser_0", range, rgb, std::array<double, 7>{0, 0, 0, 1, 0, 0, 0});

  const auto deadline = std::chrono::steady_clock::now() + 3s;
  while (std::chrono::steady_clock::now() < deadline && bridge.lidar_integrates() == 0) {
    std::this_thread::sleep_for(10ms);
  }
  EXPECT_GE(bridge.lidar_integrates(), 1);
  EXPECT_EQ(context.get_laser_names(), (std::vector<std::string>{"laser_0"}));
}

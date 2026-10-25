// SPDX-License-Identifier: Apache-2.0
// RobotSegmentation mask services (FakeBridge). No GPU.
#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <optional>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "curobo_core/core/robot_segmentation.hpp"
#include "curobo_runtime/fake_bridge.hpp"
#include "curobov2_ros_interfaces/srv/remove_object.hpp"
#include "curobov2_ros_interfaces/srv/set_mask.hpp"

using curobo_core::PerceptionCameraCfg;
using curobo_core::RobotSegmentation;

TEST(RobotSegmentation, SetAndRemoveMask)
{
  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
  }
  auto node = std::make_shared<rclcpp::Node>("unified_planner");
  curobo_runtime::FakeBridge bridge;
  RobotSegmentation segmentation(node.get(), &bridge, 1, {}, nullptr);

  auto request = std::make_shared<curobov2_ros_interfaces::srv::SetMask::Request>();
  request->name = "cage";
  request->type = 0;   // CUBOID
  request->dimensions.x = 0.5;
  request->dimensions.y = 0.5;
  request->dimensions.z = 0.5;
  auto response = std::make_shared<curobov2_ros_interfaces::srv::SetMask::Response>();
  segmentation.handle_set_mask(request, response);
  EXPECT_TRUE(response->success);
  EXPECT_EQ(response->message, "Mask 'cage' set (type=0, frame='base_link')");
  EXPECT_EQ(segmentation.mask_count(), 1u);
  EXPECT_NE(segmentation.masks_json().find("cage"), std::string::npos);

  // Non-positive dimensions are rejected.
  auto bad = std::make_shared<curobov2_ros_interfaces::srv::SetMask::Request>();
  bad->name = "bad";
  bad->dimensions.x = 0.0;
  auto bad_response = std::make_shared<curobov2_ros_interfaces::srv::SetMask::Response>();
  segmentation.handle_set_mask(bad, bad_response);
  EXPECT_FALSE(bad_response->success);
  EXPECT_EQ(bad_response->message, "Mask dimensions must be positive");

  auto remove = std::make_shared<curobov2_ros_interfaces::srv::RemoveObject::Request>();
  remove->name = "cage";
  auto remove_response = std::make_shared<curobov2_ros_interfaces::srv::RemoveObject::Response>();
  segmentation.handle_remove_mask(remove, remove_response);
  EXPECT_TRUE(remove_response->success);
  EXPECT_EQ(remove_response->message, "Mask 'cage' removed");
  EXPECT_EQ(segmentation.mask_count(), 0u);

  // Removing a missing mask is idempotent.
  auto remove2 = std::make_shared<curobov2_ros_interfaces::srv::RemoveObject::Request>();
  remove2->name = "cage";
  auto remove2_response = std::make_shared<curobov2_ros_interfaces::srv::RemoveObject::Response>();
  segmentation.handle_remove_mask(remove2, remove2_response);
  EXPECT_TRUE(remove2_response->success);
  EXPECT_EQ(remove2_response->message, "Mask 'cage' not present (nothing to remove)");
}

// A mask is given in its own frame_id; the bridge tests points in the base
// frame, so the pose must be converted (and unknown frames skipped, not guessed).
TEST(RobotSegmentation, MaskPoseIsConvertedToBaseFrame)
{
  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
  }
  auto node = std::make_shared<rclcpp::Node>("unified_planner");
  curobo_runtime::FakeBridge bridge;
  RobotSegmentation segmentation(node.get(), &bridge, 1, {}, nullptr);

  auto request = std::make_shared<curobov2_ros_interfaces::srv::SetMask::Request>();
  request->name = "obj";
  request->type = 0;
  request->frame_id = "map";
  request->pose.position.x = 1.0;
  request->pose.orientation.w = 1.0;
  request->dimensions.x = request->dimensions.y = request->dimensions.z = 0.2;
  auto response = std::make_shared<curobov2_ros_interfaces::srv::SetMask::Response>();
  segmentation.handle_set_mask(request, response);
  ASSERT_TRUE(response->success);

  // base <- map is a pure +2 m translation in y: the mask must move with it.
  const auto json = segmentation.masks_json(
    [](const std::string & frame) -> std::optional<std::array<double, 7>> {
      if (frame == "map") {
        return std::array<double, 7>{0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0};
      }
      return std::nullopt;
    });
  EXPECT_NE(json.find("\"pos\":[1,2,0]"), std::string::npos) << json;

  // Unresolvable frame -> mask skipped rather than placed at a guessed pose.
  const auto skipped = segmentation.masks_json(
    [](const std::string &) -> std::optional<std::array<double, 7>> {return std::nullopt;});
  EXPECT_EQ(skipped.find("obj"), std::string::npos) << skipped;
}

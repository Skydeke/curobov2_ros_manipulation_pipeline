// SPDX-License-Identifier: Apache-2.0
// In-process tests for the C++ FKServices: the /warmup_fk + /fk service wiring,
// message assembly and the ported helper semantics, over FakeBridge (no GPU,
// no interpreter).
#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include "curobo_core/core/fk_services.hpp"
#include "curobo_runtime/fake_bridge.hpp"
#include "curobov2_ros_interfaces/srv/fk.hpp"
#include "curobov2_ros_interfaces/srv/warmup_fk.hpp"

using namespace std::chrono_literals;
using curobo_core::FKServices;
using curobo_runtime::FakeBridge;

namespace
{
std::vector<std::string> joints()
{
  return {"panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7"};
}

class FKServicesTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }
  }
  void TearDown() override {}
};

template<class FutureT>
void spin_until(rclcpp::Node::SharedPtr node, FutureT & future)
{
  rclcpp::spin_until_future_complete(node, future, 10s);
}
}  // namespace

TEST_F(FKServicesTest, WarmupMessageThenReadyMessage)
{
  auto node = std::make_shared<rclcpp::Node>("unified_planner");
  FakeBridge bridge;
  FKServices services(node.get(), &bridge, 1, joints());

  EXPECT_EQ(services.warmup(1), "FK model ready (batch_size=1)");
  EXPECT_EQ(services.warmup(1), "FK model already ready (batch_size=1)");
  EXPECT_TRUE(services.initialized());
  EXPECT_EQ(services.batch(), 1);
}

TEST_F(FKServicesTest, FkServiceReturnsPosesAndValidity)
{
  auto node = std::make_shared<rclcpp::Node>("unified_planner");
  FakeBridge bridge;
  FKServices services(node.get(), &bridge, 1, joints());
  ASSERT_EQ(services.warmup(1), "FK model ready (batch_size=1)");

  auto client = node->create_client<curobov2_ros_interfaces::srv::Fk>("unified_planner/fk");
  ASSERT_TRUE(client->wait_for_service(5s));
  auto request = std::make_shared<curobov2_ros_interfaces::srv::Fk::Request>();
  sensor_msgs::msg::JointState js;
  js.position = {0.0, -1.3, 0.0, -2.5, 0.0, 1.5, 0.8};
  request->joint_states.push_back(js);

  auto future = client->async_send_request(request);
  spin_until(node, future);
  ASSERT_EQ(future.wait_for(0s), std::future_status::ready);
  auto response = future.get();

  ASSERT_EQ(response->poses.size(), 1u);
  // FakeBridge::fk is the documented deterministic double:
  // [x,y,z,qw,qx,qy,qz] = [q0, q1, q2, 1, q3, q4, q5].
  EXPECT_NEAR(response->poses[0].position.x, 0.0, 1e-6);
  EXPECT_NEAR(response->poses[0].position.y, -1.3, 1e-6);
  EXPECT_NEAR(response->poses[0].orientation.w, 1.0, 1e-6);
  EXPECT_NEAR(response->poses[0].orientation.x, -2.5, 1e-6);
  EXPECT_NEAR(response->poses[0].orientation.y, 0.0, 1e-6);
  EXPECT_NEAR(response->poses[0].orientation.z, 1.5, 1e-6);
  ASSERT_EQ(response->poses_valid.size(), 1u);
  EXPECT_TRUE(response->poses_valid[0].data);
}

TEST_F(FKServicesTest, PositionsForModelReordersByNameThenTruncates)
{
  auto node = std::make_shared<rclcpp::Node>("unified_planner");
  FakeBridge bridge;
  FKServices services(node.get(), &bridge, 1, joints());

  // A FULL set of names in a different order is reordered into model order.
  std::vector<std::string> names{
    "panda_joint7", "panda_joint6", "panda_joint5", "panda_joint4",
    "panda_joint3", "panda_joint2", "panda_joint1"};
  std::vector<double> positions{7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
  const auto reordered = services.positions_for_model(positions, names);
  ASSERT_EQ(reordered.size(), 7u);
  EXPECT_DOUBLE_EQ(reordered[0], 1.0);   // panda_joint1
  EXPECT_DOUBLE_EQ(reordered[6], 7.0);   // panda_joint7

  // A PARTIAL name set cannot index the whole model: fall back to a positional
  // truncation (PORT: the KeyError branch).
  const auto partial = services.positions_for_model({9.0, 8.0}, {"panda_joint1", "panda_joint2"});
  ASSERT_EQ(partial.size(), 2u);
  EXPECT_DOUBLE_EQ(partial[0], 9.0);

  // Nameless state is truncated to the model DOF.
  const auto truncated = services.positions_for_model({1, 2, 3, 4, 5, 6, 7, 8, 9}, {});
  ASSERT_EQ(truncated.size(), 7u);
  EXPECT_DOUBLE_EQ(truncated[6], 7.0);
}

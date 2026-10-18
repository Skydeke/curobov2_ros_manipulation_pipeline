// SPDX-License-Identifier: Apache-2.0
// Standalone launcher for the M2/M3 E2E runs (same node the component
// registers).
//
// There is exactly ONE executor and it is spun once, here — matching the
// component_container_mt posture (design section 5). The node itself never
// spins and never calls a blocking spin helper; callbacks are driven only by
// this executor (or the component container's).
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>

#include "curobo_core/core/unified_planner_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<curobo_core::UnifiedPlannerNode>();

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}

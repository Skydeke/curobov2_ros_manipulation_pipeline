// SPDX-License-Identifier: Apache-2.0
// Registers UnifiedPlannerNode as a component (libcurobo_planner_component.so).
#include <rclcpp_components/register_node_macro.hpp>

#include "curobo_core/core/unified_planner_node.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(curobo_core::UnifiedPlannerNode)

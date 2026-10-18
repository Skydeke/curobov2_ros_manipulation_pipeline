// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of the collision-diagnostics formatting in
// curobov2_ros/core/unified_planner_node.py (_build_collision_diagnostics /
// _format_state_collisions / _format_collision_feedback).
#ifndef CUROBO_CORE__CORE__COLLISION_DIAGNOSTICS_HPP_
#define CUROBO_CORE__CORE__COLLISION_DIAGNOSTICS_HPP_

#include <string>

#include "curobo_runtime/bridge.hpp"

namespace curobo_core
{

// PORT: unified_planner_node._format_state_collisions - a multi-line summary of
// one state's attribution: one row per contact, columns aligned, deepest
// contact first (joint-limit rows sort last).
std::string format_state_collisions(
  const std::string & label, const curobo_runtime::CollisionAttribution & sc);

// PORT: unified_planner_node._format_collision_feedback.
std::string format_collision_feedback(
  const curobo_runtime::CollisionAttribution & start,
  const curobo_runtime::CollisionAttribution & goal);

// PORT: the Python round(x, 6) applied to every contact value before it enters
// the StateCollisions message.
double round6(double value);

}  // namespace curobo_core

#endif  // CUROBO_CORE__CORE__COLLISION_DIAGNOSTICS_HPP_

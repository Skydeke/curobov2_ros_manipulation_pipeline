// SPDX-License-Identifier: Apache-2.0
// PORT: the two GPU-runtime policies that the Python node applied everywhere and
// the port had scattered/incomplete: `resolve_use_cuda_graph` and the
// fatal-CUDA stop. Kept inline in one header so every service uses the same
// rule.
#ifndef CUROBO_CORE__CORE__GPU_RUNTIME_HPP_
#define CUROBO_CORE__CORE__GPU_RUNTIME_HPP_

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "curobo_runtime/gpu_lanes.hpp"

namespace curobo_core
{

// PORT: config_wrapper.resolve_use_cuda_graph.
//
// Precedence: the ``CUROBO_USE_CUDA_GRAPH`` env var wins if set
// (``0``/``false``/``no``/``off`` -> False, anything else -> True), else the ROS
// ``use_cuda_graph`` parameter, else ``default``. The env var is the documented
// escape hatch for the MPC->Classic stale-graph replay, so every consumer must
// resolve the flag the same way - reading the raw parameter silently ignored it.
inline bool resolve_use_cuda_graph(rclcpp::Node * node, bool default_value = true)
{
  const char * env = std::getenv("CUROBO_USE_CUDA_GRAPH");
  if (env != nullptr) {
    std::string value(env);
    std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) {return static_cast<char>(std::tolower(c));});
    return !(value == "0" || value == "false" || value == "no" || value == "off");
  }
  if (node != nullptr && node->has_parameter("use_cuda_graph")) {
    return node->get_parameter("use_cuda_graph").as_bool();
  }
  return default_value;
}

// PORT: unified_planner_node.handle_gpu_fault / _on_gpu_poisoned.
//
// A CUDA fatal marker ("illegal memory access", "during capture", ...) means the
// context is dead: no later plan can be trusted. Returns false when the message
// is not a fatal CUDA error (the caller keeps its ordinary error path); true
// when it is - and, if ``exit_on_gpu_fault`` is set, does not return at all.
inline bool handle_gpu_fault(
  rclcpp::Node * node, const std::string & context, const std::string & message)
{
  if (!curobo_runtime::is_cuda_fatal_message(message)) {
    return false;
  }
  if (node != nullptr) {
    RCLCPP_FATAL(
      node->get_logger(), "CUDA fatal error while %s: %s", context.c_str(),
      message.c_str());
  }
  const bool exit_flag =
    node == nullptr || !node->has_parameter("exit_on_gpu_fault") ||
    node->get_parameter("exit_on_gpu_fault").as_bool();
  if (!exit_flag) {
    return true;
  }
  if (node != nullptr) {
    RCLCPP_FATAL(
      node->get_logger(),
      "The GPU context is corrupt; no later plan can be trusted.");
    RCLCPP_FATAL(
      node->get_logger(),
      "exit_on_gpu_fault=true: stopping instead of serving a corrupt context");
  }
  rclcpp::shutdown();
  std::fflush(nullptr);
  std::_Exit(EXIT_FAILURE);
  return true;   // unreachable
}

}  // namespace curobo_core

#endif  // CUROBO_CORE__CORE__GPU_RUNTIME_HPP_

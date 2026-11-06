// SPDX-License-Identifier: Apache-2.0
// curobo_trajectory_planner_cpp: one executable for curobo_runtime=python,
// cpp and auto. The selected IBridge is installed in curobo_runtime::Runtime
// BEFORE curobo_core::UnifiedPlannerNode is constructed, so the ROS surface is
// exactly the existing one and the node name stays curobo_server via launch.
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>

#include "curobo_core/core/unified_planner_node.hpp"
#include "curobo_runtime/runtime.hpp"
#include "curobo_cpp_runtime/runtime_selector.hpp"

namespace
{

std::string environment_or(const char * name, const std::string & fallback)
{
  const char * value = std::getenv(name);
  return value == nullptr ? fallback : std::string(value);
}

std::string parameter_or(
  const rclcpp::Node & node, const std::string & name, const std::string & fallback)
{
  if (node.has_parameter(name)) {
    const auto value = node.get_parameter(name);
    if (value.get_type() == rclcpp::ParameterType::PARAMETER_STRING) {
      return value.as_string();
    }
  }
  return fallback;
}

int parameter_or_int(const rclcpp::Node & node, const std::string & name, int fallback)
{
  if (node.has_parameter(name)) {
    const auto value = node.get_parameter(name);
    if (value.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER) {
      return static_cast<int>(value.as_int());
    }
  }
  return fallback;
}

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions probe_options;
  // The probe only reads launch parameter overrides. It is destroyed before
  // the real node is built, so the final node name remains curobo_server. The
  // real node must NOT auto-declare overrides: UnifiedPlannerNode declares the
  // same parameters itself and declare_parameter() would throw.
  probe_options.automatically_declare_parameters_from_overrides(true);

  curobo_cpp_runtime::RuntimeSelectionRequest request;
  try {
    auto probe = std::make_shared<rclcpp::Node>("curobo_runtime_resolver", probe_options);
    const std::string runtime = parameter_or(
      *probe, "curobo_runtime",
      environment_or("CUROBO_RUNTIME", "auto"));
    request.mode = curobo_cpp_runtime::parse_runtime_mode(runtime);
    request.artifact_root = parameter_or(
      *probe, "curobo_artifact_root",
      environment_or("CUROBO_ARTIFACT_ROOT", "/opt/curobo/artifacts"));
    request.robot = parameter_or(
      *probe, "robot", environment_or("CUROBO_ROBOT", ""));
    if (request.robot == "emulator") {
      request.robot = environment_or("CUROBO_ROBOT", "");
    }
    request.robot_config_file = parameter_or(
      *probe, "robot_config_file",
      environment_or("CUROBO_ROBOT_CONFIG", ""));
    request.sm_arch = parameter_or(
      *probe, "curobo_artifact_sm",
      environment_or("CUROBO_ARTIFACT_SM", ""));
    request.mode = curobo_cpp_runtime::parse_runtime_mode(runtime);
  } catch (const std::exception & error) {
    RCLCPP_ERROR(
      rclcpp::get_logger("curobo_cpp_runtime"),
      "could not read runtime selection parameters: %s", error.what());
    rclcpp::shutdown();
    return 2;
  }

  curobo_cpp_runtime::RuntimeSelection selection;
  try {
    selection = curobo_cpp_runtime::RuntimeSelector::resolve(request);
    RCLCPP_INFO(
      rclcpp::get_logger("curobo_cpp_runtime"),
      "curobo_runtime=%s -> %s (%s)",
      curobo_cpp_runtime::to_string(selection.requested_mode).c_str(),
      selection.backend.c_str(), selection.reason.c_str());
  } catch (const std::exception & error) {
    RCLCPP_ERROR(
      rclcpp::get_logger("curobo_cpp_runtime"),
      "runtime selection failed: %s", error.what());
    rclcpp::shutdown();
    return 2;
  }

  curobo_cpp_runtime::CppBridgeSettings settings;
  settings.device_ordinal = 0;
  try {
    auto probe = std::make_shared<rclcpp::Node>("curobo_runtime_resolver", probe_options);
    settings.device_ordinal = parameter_or_int(*probe, "curobo_device_ordinal", 0);
    settings.robot_config_file = request.robot_config_file;
    settings.robot = request.robot;
  } catch (const std::exception &) {
  }

  std::unique_ptr<curobo_runtime::IBridge> bridge;
  try {
    bridge = curobo_cpp_runtime::RuntimeSelector::create_bridge(selection, settings);
  } catch (const std::exception & error) {
    // Explicit cpp must fail loudly; auto never reaches this for a missing
    // artifact because resolve() already selected python.
    RCLCPP_FATAL(
      rclcpp::get_logger("curobo_cpp_runtime"),
      "could not create the %s bridge: %s",
      selection.backend.c_str(), error.what());
    rclcpp::shutdown();
    return 3;
  }

  if (!bridge) {
    RCLCPP_FATAL(
      rclcpp::get_logger("curobo_cpp_runtime"), "no bridge was constructed");
    rclcpp::shutdown();
    return 3;
  }

  // Install before UnifiedPlannerNode's constructor calls Runtime::start().
  // `set_bridge_for_test` is the existing public seam for this; the bridge is
  // then the process's single bridge for the whole node lifetime.
  curobo_runtime::Runtime::instance().set_bridge_for_test(std::move(bridge));

  auto node = std::make_shared<curobo_core::UnifiedPlannerNode>(rclcpp::NodeOptions());
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}

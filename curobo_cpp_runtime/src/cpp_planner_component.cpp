// SPDX-License-Identifier: Apache-2.0
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include "curobo_core/core/unified_planner_node.hpp"
#include "curobo_runtime/runtime.hpp"
#include "curobo_cpp_runtime/runtime_selector.hpp"

namespace curobo_cpp_runtime
{
namespace
{

std::string environment_or(const char * name, const std::string & fallback)
{
  const char * value = std::getenv(name);
  return value == nullptr ? fallback : std::string(value);
}

const rclcpp::Parameter * find_override(
  const rclcpp::NodeOptions & options, const std::string & name)
{
  for (const auto & parameter : options.parameter_overrides()) {
    if (parameter.get_name() == name) {
      return &parameter;
    }
  }
  return nullptr;
}

std::string string_override(
  const rclcpp::NodeOptions & options, const std::string & name,
  const std::string & fallback)
{
  const rclcpp::Parameter * parameter = find_override(options, name);
  if (parameter != nullptr &&
    parameter->get_type() == rclcpp::ParameterType::PARAMETER_STRING)
  {
    return parameter->as_string();
  }
  return fallback;
}

int int_override(
  const rclcpp::NodeOptions & options, const std::string & name, int fallback)
{
  const rclcpp::Parameter * parameter = find_override(options, name);
  if (parameter != nullptr &&
    parameter->get_type() == rclcpp::ParameterType::PARAMETER_INTEGER)
  {
    return static_cast<int>(parameter->as_int());
  }
  return fallback;
}

const rclcpp::NodeOptions & install_bridge(const rclcpp::NodeOptions & options)
{
  curobo_runtime::Runtime & runtime = curobo_runtime::Runtime::instance();
  if (runtime.started()) {
    throw std::runtime_error(
            "UnifiedPlannerCppComponent must be loaded before any component "
            "that starts the curobo runtime");
  }

  RuntimeSelectionRequest request;
  request.mode = parse_runtime_mode(
    string_override(
      options, "curobo_runtime", environment_or("CUROBO_RUNTIME", "auto")));
  request.artifact_root = string_override(
    options, "curobo_artifact_root",
    environment_or("CUROBO_ARTIFACT_ROOT", "/opt/curobo/artifacts"));
  request.robot = string_override(
    options, "robot", environment_or("CUROBO_ROBOT", ""));
  if (request.robot == "emulator") {
    request.robot = environment_or("CUROBO_ROBOT", "");
  }
  request.robot_config_file = string_override(
    options, "robot_config_file", environment_or("CUROBO_ROBOT_CONFIG", ""));
  request.sm_arch = string_override(
    options, "curobo_artifact_sm", environment_or("CUROBO_ARTIFACT_SM", ""));

  const RuntimeSelection selection = RuntimeSelector::resolve(request);
  RCLCPP_INFO(
    rclcpp::get_logger("curobo_cpp_runtime"), "curobo_runtime=%s -> %s (%s)",
    to_string(selection.requested_mode).c_str(), selection.backend.c_str(),
    selection.reason.c_str());

  CppBridgeSettings settings;
  settings.device_ordinal = int_override(options, "curobo_device_ordinal", 0);
  settings.robot = request.robot;
  settings.robot_config_file = request.robot_config_file;

  std::unique_ptr<curobo_runtime::IBridge> bridge =
    RuntimeSelector::create_bridge(selection, settings);
  if (!bridge) {
    throw std::runtime_error("runtime selector returned no bridge");
  }

  runtime.set_bridge_for_test(std::move(bridge));
  return options;
}

}  // namespace

class UnifiedPlannerCppComponent : public curobo_core::UnifiedPlannerNode
{
public:
  explicit UnifiedPlannerCppComponent(const rclcpp::NodeOptions & options)
  : curobo_core::UnifiedPlannerNode(install_bridge(options))
  {
  }
};

}  // namespace curobo_cpp_runtime

RCLCPP_COMPONENTS_REGISTER_NODE(curobo_cpp_runtime::UnifiedPlannerCppComponent)

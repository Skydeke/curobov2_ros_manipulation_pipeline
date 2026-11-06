// SPDX-License-Identifier: Apache-2.0
#include "curobo_cpp_runtime/runtime_selector.hpp"

#include <sstream>
#include <stdexcept>

#include "curobo_runtime/py_bridge.hpp"
#include "curobo_cpp_runtime/cpp_bridge.hpp"

namespace curobo_cpp_runtime
{

std::unique_ptr<curobo_runtime::IBridge> RuntimeSelector::create_bridge(
  const RuntimeSelection & selection, const CppBridgeSettings & settings)
{
  if (selection.selected_mode == RuntimeMode::Python) {
    return std::make_unique<curobo_runtime::PyBridge>();
  }

  if (selection.selected_mode != RuntimeMode::Cpp) {
    throw std::runtime_error("RuntimeSelector: unexpected runtime mode");
  }

  if (!selection.artifact_valid) {
    std::ostringstream message;
    message << "curobo_runtime=cpp requested but no usable native artifact";
    if (!selection.artifact.directory.empty()) {
      message << " at '" << selection.artifact.directory << "'";
    }
    if (!selection.artifact.error.empty()) {
      message << ": " << selection.artifact.error;
    }
    message << ". Re-export the artifact or run curobo_runtime:=python.";
    throw std::runtime_error(message.str());
  }
  // `auto` already keeps an incomplete capability profile on Python (see
  // resolve_runtime_selection).  An explicit cpp request is allowed through:
  // the artifact is present, so start the native bridge and let any missing
  // native method report its own precise error instead of refusing here.
  CppBridgeSettings resolved = settings;
  if (resolved.artifact_directory.empty()) {
    resolved.artifact_directory = selection.artifact.directory;
  }
  if (resolved.robot.empty()) {
    resolved.robot = selection.artifact.robot;
  }
  if (resolved.robot_config_file.empty()) {
    resolved.robot_config_file = selection.artifact.manifest.curobo_config;
  }
  // Both `auto` and explicit `cpp` prefer native and fall back per method.
  // Explicit `cpp` remains strict about the artifact itself (missing artifacts
  // were rejected above); a partial capability profile only means some calls
  // must use Python.
  const bool python_fallback = true;
  return std::make_unique<CppBridge>(
    resolved.artifact_directory, resolved.robot, resolved.robot_config_file,
    resolved.device_ordinal, python_fallback);
}

}  // namespace curobo_cpp_runtime

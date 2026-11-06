// SPDX-License-Identifier: Apache-2.0
// Runtime selection + bridge construction. The decision itself lives in
// runtime_selection.hpp; this header adds the ROS/backend-dependent factory.
#ifndef CUROBO_CPP_RUNTIME__RUNTIME_SELECTOR_HPP_
#define CUROBO_CPP_RUNTIME__RUNTIME_SELECTOR_HPP_

#include <memory>
#include <string>

#include "curobo_runtime/bridge.hpp"
#include "curobo_cpp_runtime/runtime_selection.hpp"

namespace curobo_cpp_runtime
{

struct CppBridgeSettings
{
  std::string artifact_directory;
  std::string robot;
  std::string robot_config_file;
  int device_ordinal{0};
};

class RuntimeSelector
{
public:
  static RuntimeSelection resolve(const RuntimeSelectionRequest & request)
  {
    return resolve_runtime_selection(request);
  }

  // Construct the selected backend. For python this is curobov2_ros's
  // PyBridge; for cpp it is CppBridge. An explicit cpp request that cannot
  // be satisfied throws: cpp never silently falls back to Python.
  static std::unique_ptr<curobo_runtime::IBridge> create_bridge(
    const RuntimeSelection & selection,
    const CppBridgeSettings & settings = CppBridgeSettings());
};

}  // namespace curobo_cpp_runtime

#endif  // CUROBO_CPP_RUNTIME__RUNTIME_SELECTOR_HPP_

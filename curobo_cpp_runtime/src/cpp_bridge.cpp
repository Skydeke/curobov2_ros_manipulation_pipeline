// SPDX-License-Identifier: Apache-2.0
#include "cpp_bridge_impl.hpp"

namespace curobo_cpp_runtime
{

CppBridge::CppBridge(
  std::string artifact_directory, std::string robot_name,
  std::string robot_config_file, int device_ordinal, bool python_fallback)
: impl_(std::make_unique<Impl>())
{
  impl_->artifact_dir = std::move(artifact_directory);
  impl_->robot = std::move(robot_name);
  impl_->config_file = std::move(robot_config_file);
  impl_->device_ordinal = device_ordinal;
  impl_->python_fallback = python_fallback;
}

CppBridge::~CppBridge() = default;

const std::string & CppBridge::artifact_directory() const
{
  return impl_->artifact_dir;
}

const std::string & CppBridge::robot_name() const
{
  return impl_->robot;
}

}  // namespace curobo_cpp_runtime

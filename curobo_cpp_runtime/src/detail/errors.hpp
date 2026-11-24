// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stdexcept>
#include <string>

namespace curobo_cpp_runtime
{
namespace detail
{

[[noreturn]] inline void unsupported(const char * method)
{
  throw std::runtime_error(
          std::string(method) +
          " is not implemented by curobo_cpp_runtime (cpp_runtime API gap). "
          "Run this robot/service with curobo_runtime:=python.");
}

[[noreturn]] inline void fail(const char * method, const std::string & message)
{
  throw std::runtime_error(std::string(method) + ": " + message);
}

}  // namespace detail
}  // namespace curobo_cpp_runtime

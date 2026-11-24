// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>

#include "curobo_cpp_runtime/detail/mini_json.hpp"

namespace curobo_cpp_runtime
{
namespace detail
{

std::string json_string(
  const json::Value & document, const std::string & key,
  const std::string & fallback = "");

int json_int(const json::Value & document, const std::string & key, int fallback);

double json_double(
  const json::Value & document, const std::string & key, double fallback);

}  // namespace detail
}  // namespace curobo_cpp_runtime

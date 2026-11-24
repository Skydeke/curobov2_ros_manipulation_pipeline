// SPDX-License-Identifier: Apache-2.0
#include "detail/json_helpers.hpp"

namespace curobo_cpp_runtime
{
namespace detail
{

std::string json_string(
  const json::Value & document, const std::string & key,
  const std::string & fallback)
{
  return document.string_at(key, fallback);
}

int json_int(const json::Value & document, const std::string & key, int fallback)
{
  return static_cast<int>(document.number_at(key, static_cast<double>(fallback)));
}

double json_double(
  const json::Value & document, const std::string & key, double fallback)
{
  return document.number_at(key, fallback);
}

}  // namespace detail
}  // namespace curobo_cpp_runtime

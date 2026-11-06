// SPDX-License-Identifier: Apache-2.0
#include "curobo_cpp_runtime/runtime_selection.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace curobo_cpp_runtime
{

RuntimeMode parse_runtime_mode(const std::string & value)
{
  std::string text = value;
  text.erase(
    text.begin(),
    std::find_if(
      text.begin(), text.end(),
      [](unsigned char c) {return std::isspace(c) == 0;}));
  text.erase(
    std::find_if(
      text.rbegin(), text.rend(),
      [](unsigned char c) {return std::isspace(c) == 0;}).base(),
    text.end());
  std::transform(
    text.begin(), text.end(), text.begin(),
    [](unsigned char c) {return static_cast<char>(std::tolower(c));});
  if (text.empty() || text == "auto") {
    return RuntimeMode::Auto;
  }
  if (text == "python" || text == "py") {
    return RuntimeMode::Python;
  }
  if (text == "cpp" || text == "native") {
    return RuntimeMode::Cpp;
  }
  throw std::invalid_argument(
    "invalid curobo_runtime='" + value + "' (expected python, cpp or auto)");
}

std::string to_string(RuntimeMode mode)
{
  switch (mode) {
    case RuntimeMode::Auto: return "auto";
    case RuntimeMode::Python: return "python";
    case RuntimeMode::Cpp: return "cpp";
  }
  return "auto";
}

RuntimeMode choose_runtime_mode(
  RuntimeMode requested, bool artifact_valid, bool capability_profile_complete)
{
  if (requested == RuntimeMode::Python) {
    return RuntimeMode::Python;
  }
  if (requested == RuntimeMode::Cpp) {
    // Explicit cpp is strict. The factory raises with the resolver's error
    // instead of silently falling back to Python (Plan.md 5.1).
    return RuntimeMode::Cpp;
  }
  // `auto` always starts the hybrid native bridge when the artifact is
  // usable. The hybrid dispatches unsupported methods (and scene mutations
  // that cannot be mirrored) to Python at call time, so a partial capability
  // profile no longer forces the whole node onto Python.
  (void)capability_profile_complete;
  return artifact_valid ? RuntimeMode::Cpp : RuntimeMode::Python;
}

RuntimeSelection resolve_runtime_selection(const RuntimeSelectionRequest & request)
{
  RuntimeSelection selection;
  selection.requested_mode = request.mode;

  ArtifactBundleResolver resolver(request.artifact_root, request.sm_arch);
  selection.artifact = resolver.resolve(
    request.robot, request.robot_config_file, request.sm_arch);
  selection.artifact_valid = selection.artifact.valid;
  selection.capability_profile_complete =
    selection.artifact.capability_profile_complete;
  selection.hash_mismatch = selection.artifact.hash_available &&
    !selection.artifact.hash_matches;

  selection.selected_mode = choose_runtime_mode(
    request.mode, selection.artifact_valid, selection.capability_profile_complete);
  selection.backend = selection.selected_mode == RuntimeMode::Cpp ? "cpp" : "python";

  if (request.mode == RuntimeMode::Python) {
    selection.reason = "curobo_runtime=python";
  } else if (request.mode == RuntimeMode::Cpp) {
    selection.reason = "curobo_runtime=cpp";
  } else if (selection.selected_mode == RuntimeMode::Cpp) {
    selection.reason = selection.capability_profile_complete ?
      "auto: artifact and native capability profile are complete" :
      "auto: hybrid native bridge (Python fallback for incomplete capabilities)";
  } else if (!selection.artifact_valid) {
    selection.reason = "auto: " + (selection.artifact.error.empty() ?
      std::string("native artifact unavailable") : selection.artifact.error);
  } else if (!selection.capability_profile_complete) {
    selection.reason = "auto: native capability profile is incomplete";
    if (!selection.artifact.missing_capabilities.empty()) {
      selection.reason += " (missing:";
      for (const auto & name : selection.artifact.missing_capabilities) {
        selection.reason += " " + name;
      }
      selection.reason += ")";
    }
  } else {
    selection.reason = "auto: python fallback";
  }
  return selection;
}

}  // namespace curobo_cpp_runtime

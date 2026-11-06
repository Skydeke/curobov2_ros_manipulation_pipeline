// SPDX-License-Identifier: Apache-2.0
// Pure runtime-selection policy (no ROS, no CUDA, no bridge construction).
//
// Keeping the decision here lets the package unit-test `python` / `cpp` /
// `auto` without an artifact, a GPU or a ROS graph.
#ifndef CUROBO_CPP_RUNTIME__RUNTIME_SELECTION_HPP_
#define CUROBO_CPP_RUNTIME__RUNTIME_SELECTION_HPP_

#include <string>

#include "curobo_cpp_runtime/artifact_bundle.hpp"

namespace curobo_cpp_runtime
{

enum class RuntimeMode
{
  Auto,
  Python,
  Cpp,
};

struct RuntimeSelectionRequest
{
  RuntimeMode mode{RuntimeMode::Auto};
  std::string artifact_root{"/opt/curobo/artifacts"};
  std::string robot;
  std::string robot_config_file;
  // Optional sm_XX override (tests and non-local build hosts). Empty means
  // detect the current CUDA device.
  std::string sm_arch;
};

struct RuntimeSelection
{
  RuntimeMode requested_mode{RuntimeMode::Auto};
  RuntimeMode selected_mode{RuntimeMode::Python};
  std::string backend;                       // "python" or "cpp"
  std::string reason;
  bool artifact_valid{false};
  bool capability_profile_complete{false};
  bool hash_mismatch{false};
  ArtifactBundle artifact;

  bool selected_cpp() const {return selected_mode == RuntimeMode::Cpp;}
};

// "auto", "python", "cpp" (case-insensitive). Throws std::invalid_argument for
// any other value: a typo must not silently select a backend.
RuntimeMode parse_runtime_mode(const std::string & value);
std::string to_string(RuntimeMode mode);

// Pure policy:
//   Python -> Python
//   Cpp    -> Cpp (the caller/factory must then enforce artifact validity)
//   Auto   -> Cpp only when the artifact is valid AND the capability profile
//             is complete, otherwise Python.
RuntimeMode choose_runtime_mode(
  RuntimeMode requested, bool artifact_valid, bool capability_profile_complete);

// Resolve the artifact and apply choose_runtime_mode. Does not throw for a
// missing artifact; the factory turns an explicit cpp failure into a clear
// exception and auto falls back to Python.
RuntimeSelection resolve_runtime_selection(const RuntimeSelectionRequest & request);

}  // namespace curobo_cpp_runtime

#endif  // CUROBO_CPP_RUNTIME__RUNTIME_SELECTION_HPP_

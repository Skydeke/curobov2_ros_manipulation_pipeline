// SPDX-License-Identifier: Apache-2.0
// Artifact bundle discovery for the native cpp_runtime.
//
// Layout produced by the export step:
//
//   <root>/<robot>/<sm_XX>/
//     manifest.json                    generated robot metadata
//     robot_kinematics_manifest.json   cpp_runtime artifact
//     robot_ik_manifest.json
//     robot_trajopt_manifest.json
//     robot_motion_manifest.json
//     ...
//
// This file only reads generated JSON and binaries. It never parses YAML.
#ifndef CUROBO_CPP_RUNTIME__ARTIFACT_BUNDLE_HPP_
#define CUROBO_CPP_RUNTIME__ARTIFACT_BUNDLE_HPP_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace curobo_cpp_runtime
{

// The generated manifest's robot metadata (Plan.md section 5.5). Only the
// fields the bridge needs are lifted out; the complete document is still on
// disk and remains the single generated source of truth.
struct ArtifactManifest
{
  bool present{false};
  std::string path;

  std::string schema_version;
  std::string robot;
  std::string base_link;
  std::string curobo_config;
  std::string strategy;
  std::map<std::string, std::string> strategy_params;
  std::string arch;

  // sha256 of `source_file`, when the exporter recorded one.
  std::string source_hash;
  std::string source_file;

  std::vector<std::string> home_names;
  std::vector<double> home_positions;

  // The descriptor's named_joint_configs / gripper_controllers, flattened
  // exactly like RobotDescriptionOutcome so GetJointInfo can report them on the
  // native runtime as well (the RViz Joints tab's "Named pose" combo). The
  // export step already copies both maps into manifest.json; they are parsed
  // here instead of YAML, per the "C++ never parses YAML" rule.
  std::vector<std::string> named_config_names;
  std::vector<std::string> named_config_joint_names;
  std::vector<double> named_config_positions;
  std::vector<std::int64_t> named_config_sizes;
  std::vector<std::string> gripper_names;
  std::vector<std::string> gripper_types;
  std::vector<std::string> gripper_controllers;
  std::vector<std::string> gripper_action_ns;
  std::vector<std::string> gripper_joint_lists;  // space-separated joints
  std::vector<std::string> gripper_command_joints;
  std::vector<double> gripper_open_positions;
  std::vector<double> gripper_close_positions;
  std::vector<double> gripper_max_efforts;
  std::vector<double> gripper_time_from_start;

  std::vector<std::string> joint_names;
  std::vector<double> joint_lower;
  std::vector<double> joint_upper;
  std::vector<std::string> link_names;
  std::vector<std::string> tool_frames;

  std::vector<std::string> locked_joint_names;
  std::vector<double> locked_positions;
  std::vector<std::string> unpinned_lock_joint_names;

  // Native capability profile. Empty means "not provided"; auto must then
  // stay on Python (Plan.md 5.1 / 6.3).
  bool capability_profile_present{false};
  bool capability_profile_complete{false};
  std::map<std::string, bool> capabilities;
  std::vector<std::string> required_capabilities;
  std::vector<std::string> missing_capabilities;

  // Fixed artifact dimensions that are useful before constructing a solver.
  int num_dof{0};
  int num_tool_frames{0};
  int interpolation_buffer_size{0};

  // Scene capacity from the export, if the generated manifest carries it.
  int scene_max_cuboids{64};
  int scene_max_capsules{64};
  int scene_max_cylinders{32};
  int scene_max_meshes{16};
  int scene_max_voxel_grids{0};
  int scene_max_voxels_per_grid{0};
};

struct ArtifactBundle
{
  bool valid{false};
  std::string error;

  std::string root;
  std::string robot;
  std::string sm_arch;                 // "sm_89" (empty when undetectable)
  std::string directory;               // <root>/<robot>/<sm_XX>

  ArtifactManifest manifest;

  bool manifest_present{false};
  bool arch_matches{false};
  bool has_required_files{false};
  bool hash_available{false};
  bool hash_matches{true};

  bool capability_profile_complete{false};
  std::vector<std::string> missing_capabilities;

  bool exists() const {return valid;}
};

// The capability keys the selector requires before it will choose cpp under
// `auto`. A generated capability profile may add a "required" array; otherwise
// this is the default set. Methods that throw in CppBridge are intentionally
// absent (scene/attachments/mapper/perception are still Python fallbacks).
const std::vector<std::string> & default_required_capabilities();

// Return "sm_XX" for the current CUDA device, or "" when CUDA is unavailable.
std::string current_cuda_arch();

// Normalize "89", "sm_89", "compute_89" to "sm_89". Returns "" for junk.
std::string normalize_arch(const std::string & value);

// Lowercase/trim a robot name for manifest comparisons.
std::string normalize_robot_name(const std::string & value);

// SHA-256 of arbitrary bytes / a file. Returns "" when the file cannot be read.
std::string sha256_hex(const std::string & bytes);
std::string sha256_file_hex(const std::string & path);

// Read only the generated manifest.json at `dir` (no layout validation).
ArtifactManifest load_artifact_manifest(const std::string & dir);

// Inspect one concrete artifact directory. `robot` and `config_file` are used
// for identity and optional source-hash validation.
ArtifactBundle inspect_artifact_directory(
  const std::string & directory,
  const std::string & robot = "",
  const std::string & config_file = "");

class ArtifactBundleResolver
{
public:
  ArtifactBundleResolver(
    std::string artifact_root = "/opt/curobo/artifacts",
    std::string sm_arch_override = "");

  // Resolve <root>/<robot>/<sm_XX>. An explicit `sm_arch` argument overrides
  // the resolver's override and the detected CUDA architecture.
  ArtifactBundle resolve(
    const std::string & robot,
    const std::string & config_file = "",
    const std::string & sm_arch = "") const;

  const std::string & root() const {return root_;}
  const std::string & sm_arch_override() const {return sm_arch_override_;}

private:
  std::string root_;
  std::string sm_arch_override_;
};

}  // namespace curobo_cpp_runtime

#endif  // CUROBO_CPP_RUNTIME__ARTIFACT_BUNDLE_HPP_

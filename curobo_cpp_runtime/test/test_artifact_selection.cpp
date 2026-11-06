// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "curobo_cpp_runtime/artifact_bundle.hpp"
#include "curobo_cpp_runtime/runtime_selection.hpp"

namespace fs = std::filesystem;
using curobo_cpp_runtime::ArtifactBundle;
using curobo_cpp_runtime::ArtifactBundleResolver;
using curobo_cpp_runtime::RuntimeMode;
using curobo_cpp_runtime::RuntimeSelection;
using curobo_cpp_runtime::RuntimeSelectionRequest;
using curobo_cpp_runtime::choose_runtime_mode;
using curobo_cpp_runtime::normalize_arch;
using curobo_cpp_runtime::parse_runtime_mode;
using curobo_cpp_runtime::resolve_runtime_selection;
using curobo_cpp_runtime::sha256_hex;

namespace
{

class TemporaryTree
{
public:
  TemporaryTree()
  {
    directory_ = fs::temp_directory_path() /
      ("curobo_cpp_selection_" + std::to_string(std::rand()));
    fs::remove_all(directory_);
    fs::create_directories(directory_);
  }

  ~TemporaryTree()
  {
    std::error_code error;
    fs::remove_all(directory_, error);
  }

  const fs::path & path() const {return directory_;}

  fs::path artifact(
    const std::string & robot, const std::string & arch = "sm_89") const
  {
    return directory_ / "artifacts" / robot / arch;
  }

private:
  fs::path directory_;
};

void write_file(const fs::path & path, const std::string & contents)
{
  fs::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(stream.good()) << path;
  stream << contents;
}

std::string capability(const std::string & name)
{
  return "\"" + name + "\":true";
}

std::string complete_manifest(
  const std::string & robot, const std::string & arch,
  const std::string & hash, bool complete = true)
{
  std::string capabilities;
  const std::vector<std::string> required = {
    "robot_model", "fk", "ik", "ik_multi", "validate", "plan_pose",
    "plan_cspace", "plan_pose_waypoints", "reset_seed", "tool_frames",
    "joint_limits", "set_link_collision", "scene_set", "scene_get"};
  for (std::size_t i = 0; i < required.size(); ++i) {
    if (i != 0) {
      capabilities += ",";
    }
    capabilities += capability(required[i]);
  }
  return std::string("{\n") +
    "  \"schema_version\": 1,\n" +
    "  \"robot\": \"" + robot + "\",\n" +
    "  \"arch\": \"" + arch + "\",\n" +
    "  \"source_hash\": \"" + hash + "\",\n" +
    "  \"base_link\": \"base_link\",\n" +
    "  \"joint_names\": [\"j1\", \"j2\"],\n" +
    "  \"joint_limits\": {\"lower\": [-1.0, -2.0], \"upper\": [1.0, 2.0]},\n" +
    "  \"tool_frames\": [\"tool\"],\n" +
    "  \"capabilities\": {" + capabilities +
    (complete ? ",\"complete\":true" : ",\"complete\":false") + "}\n" +
    "}\n";
}

void create_artifact(const fs::path & artifact, const std::string & manifest)
{
  write_file(artifact / "manifest.json", manifest);
  write_file(artifact / "robot_kinematics_manifest.json", "{}\n");
  write_file(artifact / "robot_ik_manifest.json", "{}\n");
  write_file(artifact / "robot_trajopt_manifest.json", "{}\n");
  write_file(artifact / "robot_motion_manifest.json", "{}\n");
}

}  // namespace

TEST(ArtifactSelection, ResolvesByRobotAndArch)
{
  TemporaryTree tree;
  const std::string config_text = "robot: openarm\n";
  const fs::path config = tree.path() / "robots" / "openarm.yaml";
  write_file(config, config_text);
  create_artifact(
    tree.artifact("openarm"),
    complete_manifest("openarm", "sm_89", sha256_hex(config_text)));

  ArtifactBundleResolver resolver(tree.path() / "artifacts", "sm_89");
  const ArtifactBundle bundle = resolver.resolve("openarm", config.string());
  ASSERT_TRUE(bundle.valid) << bundle.error;
  EXPECT_TRUE(bundle.arch_matches);
  EXPECT_TRUE(bundle.hash_available);
  EXPECT_TRUE(bundle.hash_matches);
  EXPECT_TRUE(bundle.capability_profile_complete);
  EXPECT_EQ(bundle.manifest.robot, "openarm");
  EXPECT_EQ(bundle.manifest.joint_names, (std::vector<std::string>{"j1", "j2"}));
  EXPECT_EQ(bundle.manifest.tool_frames, (std::vector<std::string>{"tool"}));
  EXPECT_EQ(bundle.manifest.num_dof, 2);
}

TEST(ArtifactSelection, WrongArchDoesNotResolve)
{
  TemporaryTree tree;
  create_artifact(
    tree.artifact("openarm", "sm_89"),
    complete_manifest("openarm", "sm_89", ""));

  ArtifactBundleResolver resolver(tree.path() / "artifacts", "sm_90");
  const ArtifactBundle bundle = resolver.resolve("openarm");
  EXPECT_FALSE(bundle.valid);
  EXPECT_NE(bundle.error.find("sm_90"), std::string::npos) << bundle.error;
}

TEST(ArtifactSelection, MissingManifestDoesNotResolve)
{
  TemporaryTree tree;
  const fs::path artifact = tree.artifact("openarm");
  fs::create_directories(artifact);
  write_file(artifact / "robot_kinematics_manifest.json", "{}\n");
  write_file(artifact / "robot_ik_manifest.json", "{}\n");
  write_file(artifact / "robot_trajopt_manifest.json", "{}\n");
  write_file(artifact / "robot_motion_manifest.json", "{}\n");

  ArtifactBundleResolver resolver(tree.path() / "artifacts", "sm_89");
  const ArtifactBundle bundle = resolver.resolve("openarm");
  EXPECT_FALSE(bundle.valid);
  EXPECT_NE(bundle.error.find("manifest.json"), std::string::npos) << bundle.error;
}

TEST(ArtifactSelection, SourceHashMismatchDoesNotResolve)
{
  TemporaryTree tree;
  const fs::path config = tree.path() / "robots" / "openarm.yaml";
  write_file(config, "robot: changed\n");
  create_artifact(
    tree.artifact("openarm"),
    complete_manifest("openarm", "sm_89", sha256_hex("robot: original\n")));

  ArtifactBundleResolver resolver(tree.path() / "artifacts", "sm_89");
  const ArtifactBundle bundle = resolver.resolve("openarm", config.string());
  EXPECT_FALSE(bundle.valid);
  EXPECT_TRUE(bundle.hash_available);
  EXPECT_FALSE(bundle.hash_matches);
  EXPECT_NE(bundle.error.find("hash"), std::string::npos) << bundle.error;
}

TEST(ArtifactSelection, IncompleteCapabilityProfileSelectsHybridCpp)
{
  TemporaryTree tree;
  create_artifact(
    tree.artifact("openarm"),
    complete_manifest("openarm", "sm_89", "", false));

  ArtifactBundleResolver resolver(tree.path() / "artifacts", "sm_89");
  const ArtifactBundle bundle = resolver.resolve("openarm");
  ASSERT_TRUE(bundle.valid) << bundle.error;
  EXPECT_FALSE(bundle.capability_profile_complete);

  const RuntimeMode mode = choose_runtime_mode(
    RuntimeMode::Auto, bundle.valid, bundle.capability_profile_complete);
  EXPECT_EQ(mode, RuntimeMode::Cpp);
}

TEST(ArtifactSelection, ParsesGeneratedNestedManifest)
{
  TemporaryTree tree;
  const fs::path artifact = tree.artifact("openarm");
  write_file(
    artifact / "manifest.json",
    R"({
      "arch": "sm_89",
      "tool_frames": ["left_tool", "right_tool"],
      "robot": {
        "name": "openarm",
        "base_link": "base_link",
        "curobo_config": "/tmp/openarm.yml"
      },
      "files": {
        "robot_config": "/tmp/robots/openarm.yaml",
        "curobo_config": "/tmp/openarm.yml"
      },
      "source_hash": "deadbeef",
      "home": {"joint_names": ["j1", "j2"], "positions": [0.1, 0.2]},
      "lock_joints": {"pinned": {"finger": 0.04}, "unpinned": ["finger2"]},
      "cspace": {
        "joint_names": ["j1", "j2"],
        "limits": {
          "position_lower": [-1.0, -2.0],
          "position_upper": [1.0, 2.0]
        }
      }
    })");
  write_file(artifact / "robot_kinematics_manifest.json", "{}\n");
  write_file(artifact / "robot_ik_manifest.json", "{}\n");
  write_file(artifact / "robot_trajopt_manifest.json", "{}\n");
  write_file(artifact / "robot_motion_manifest.json", "{}\n");

  ArtifactBundleResolver resolver(tree.path() / "artifacts", "sm_89");
  const ArtifactBundle bundle = resolver.resolve("openarm");
  ASSERT_TRUE(bundle.valid) << bundle.error;
  EXPECT_EQ(bundle.manifest.robot, "openarm");
  EXPECT_EQ(bundle.manifest.base_link, "base_link");
  EXPECT_EQ(bundle.manifest.curobo_config, "/tmp/openarm.yml");
  EXPECT_EQ(bundle.manifest.source_file, "/tmp/robots/openarm.yaml");
  EXPECT_EQ(bundle.manifest.home_names, (std::vector<std::string>{"j1", "j2"}));
  EXPECT_EQ(bundle.manifest.locked_joint_names, (std::vector<std::string>{"finger"}));
  ASSERT_EQ(bundle.manifest.locked_positions.size(), 1u);
  EXPECT_DOUBLE_EQ(bundle.manifest.locked_positions[0], 0.04);
  EXPECT_EQ(bundle.manifest.unpinned_lock_joint_names, (std::vector<std::string>{"finger2"}));
  EXPECT_EQ(bundle.manifest.num_dof, 2);
  ASSERT_EQ(bundle.manifest.joint_lower.size(), 2u);
  EXPECT_DOUBLE_EQ(bundle.manifest.joint_lower[1], -2.0);
  EXPECT_DOUBLE_EQ(bundle.manifest.joint_upper[1], 2.0);
}

TEST(RuntimeSelection, ParsesOnlyKnownModes)
{
  EXPECT_EQ(parse_runtime_mode("auto"), RuntimeMode::Auto);
  EXPECT_EQ(parse_runtime_mode("PYTHON"), RuntimeMode::Python);
  EXPECT_EQ(parse_runtime_mode("cpp"), RuntimeMode::Cpp);
  EXPECT_EQ(parse_runtime_mode(""), RuntimeMode::Auto);
  EXPECT_THROW(parse_runtime_mode("native2"), std::invalid_argument);
}

TEST(RuntimeSelection, PolicyIsStrictForCppAndFallbackForAuto)
{
  EXPECT_EQ(choose_runtime_mode(RuntimeMode::Python, true, true), RuntimeMode::Python);
  EXPECT_EQ(choose_runtime_mode(RuntimeMode::Cpp, false, false), RuntimeMode::Cpp);
  EXPECT_EQ(choose_runtime_mode(RuntimeMode::Auto, true, true), RuntimeMode::Cpp);
  EXPECT_EQ(choose_runtime_mode(RuntimeMode::Auto, true, false), RuntimeMode::Cpp);
  EXPECT_EQ(choose_runtime_mode(RuntimeMode::Auto, false, true), RuntimeMode::Python);
}

TEST(RuntimeSelection, AutoResolvesHybridCppThenFallsBackWhenMissing)
{
  TemporaryTree tree;
  create_artifact(
    tree.artifact("openarm"),
    complete_manifest("openarm", "sm_89", ""));

  RuntimeSelectionRequest request;
  request.mode = RuntimeMode::Auto;
  request.artifact_root = (tree.path() / "artifacts").string();
  request.robot = "openarm";
  request.sm_arch = "sm_89";

  RuntimeSelection selection = resolve_runtime_selection(request);
  ASSERT_TRUE(selection.artifact_valid) << selection.reason;
  EXPECT_TRUE(selection.capability_profile_complete);
  EXPECT_EQ(selection.selected_mode, RuntimeMode::Cpp);
  EXPECT_EQ(selection.backend, "cpp");

  // An incomplete profile still starts the hybrid cpp bridge; unsupported
  // calls fall back to Python at dispatch time.
  write_file(
    tree.artifact("openarm") / "manifest.json",
    complete_manifest("openarm", "sm_89", "", false));
  selection = resolve_runtime_selection(request);
  EXPECT_TRUE(selection.artifact_valid) << selection.reason;
  EXPECT_EQ(selection.selected_mode, RuntimeMode::Cpp);
  EXPECT_NE(selection.reason.find("hybrid"), std::string::npos);

  // A missing artifact is the only auto case that starts pure Python.
  request.robot = "missing";
  selection = resolve_runtime_selection(request);
  EXPECT_FALSE(selection.artifact_valid);
  EXPECT_EQ(selection.selected_mode, RuntimeMode::Python);
}

TEST(RuntimeSelection, ExplicitCppOnMissingArtifactRemainsCpp)
{
  TemporaryTree tree;
  RuntimeSelectionRequest request;
  request.mode = RuntimeMode::Cpp;
  request.artifact_root = (tree.path() / "artifacts").string();
  request.robot = "missing";
  request.sm_arch = "sm_89";

  const RuntimeSelection selection = resolve_runtime_selection(request);
  EXPECT_FALSE(selection.artifact_valid);
  // The factory turns this into a throw; the selection itself must not
  // silently change to python.
  EXPECT_EQ(selection.selected_mode, RuntimeMode::Cpp);
  EXPECT_EQ(selection.backend, "cpp");
}

TEST(ArtifactSelection, NormalizesArchitectures)
{
  EXPECT_EQ(normalize_arch("sm_89"), "sm_89");
  EXPECT_EQ(normalize_arch("89"), "sm_89");
  EXPECT_EQ(normalize_arch("compute_90"), "sm_90");
  EXPECT_EQ(normalize_arch("bogus"), "");
}

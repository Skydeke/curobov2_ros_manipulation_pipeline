// SPDX-License-Identifier: Apache-2.0
#include "curobo_cpp_runtime/artifact_bundle.hpp"

#ifdef CUROBO_CPP_RUNTIME_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "curobo_cpp_runtime/detail/mini_json.hpp"

namespace curobo_cpp_runtime
{
namespace
{

std::string read_file(const std::string & path, std::string * error = nullptr)
{
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    if (error != nullptr) {
      *error = "cannot open '" + path + "'";
    }
    return {};
  }
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  if (!stream.good() && !stream.eof()) {
    if (error != nullptr) {
      *error = "cannot read '" + path + "'";
    }
    return {};
  }
  return buffer.str();
}

bool file_exists(const std::string & path)
{
  std::error_code error;
  return std::filesystem::is_regular_file(path, error);
}

std::string trim(const std::string & value)
{
  std::size_t begin = 0;
  while (begin < value.size() &&
    std::isspace(static_cast<unsigned char>(value[begin])) != 0)
  {
    ++begin;
  }
  std::size_t end = value.size();
  while (end > begin &&
    std::isspace(static_cast<unsigned char>(value[end - 1])) != 0)
  {
    --end;
  }
  return value.substr(begin, end - begin);
}

bool parse_bool(const json::Value * value, bool fallback = false)
{
  if (value == nullptr) {
    return fallback;
  }
  if (value->is_bool()) {
    return value->as_bool();
  }
  if (value->is_number()) {
    return value->as_double() != 0.0;
  }
  if (value->is_string()) {
    const std::string text = trim(value->as_string());
    return text == "true" || text == "1" || text == "yes" || text == "complete";
  }
  return fallback;
}

std::vector<std::string> strings(const json::Value * value)
{
  std::vector<std::string> out;
  if (value == nullptr || !value->is_array()) {
    return out;
  }
  for (const auto & entry : value->as_array()) {
    if (entry.is_string()) {
      out.push_back(entry.as_string());
    }
  }
  return out;
}

std::vector<double> numbers(const json::Value * value)
{
  std::vector<double> out;
  if (value == nullptr || !value->is_array()) {
    return out;
  }
  for (const auto & entry : value->as_array()) {
    if (entry.is_number()) {
      out.push_back(entry.as_double());
    }
  }
  return out;
}

const json::Value * first(const json::Value & document, std::initializer_list<const char *> keys)
{
  for (const char * key : keys) {
    if (const json::Value * value = document.find_path(key)) {
      return value;
    }
  }
  return nullptr;
}

void load_scene_capacities(const json::Value & node, ArtifactManifest & manifest)
{
  const json::Value * scene = first(node, {"scene_capacities", "scene"});
  if (scene == nullptr || !scene->is_object()) {
    return;
  }
  const json::Value * cuboid = scene->find("cuboid");
  const json::Value * capsule = scene->find("capsule");
  const json::Value * cylinder = scene->find("cylinder");
  const json::Value * mesh = scene->find("mesh");
  const json::Value * voxel = scene->find("voxel");
  if (cuboid != nullptr) {manifest.scene_max_cuboids = static_cast<int>(cuboid->as_double(manifest.scene_max_cuboids));}
  if (capsule != nullptr) {manifest.scene_max_capsules = static_cast<int>(capsule->as_double(manifest.scene_max_capsules));}
  if (cylinder != nullptr) {manifest.scene_max_cylinders = static_cast<int>(cylinder->as_double(manifest.scene_max_cylinders));}
  if (mesh != nullptr) {manifest.scene_max_meshes = static_cast<int>(mesh->as_double(manifest.scene_max_meshes));}
  if (voxel != nullptr) {manifest.scene_max_voxel_grids = static_cast<int>(voxel->as_double(manifest.scene_max_voxel_grids));}
  if (const json::Value * count = scene->find("voxels_per_grid")) {
    manifest.scene_max_voxels_per_grid = static_cast<int>(count->as_double());
  }
}

void load_capabilities(
  const json::Value & node, const std::string & directory, ArtifactManifest & manifest)
{
  const json::Value * profile = first(node, {"native_capability_profile", "capability_profile"});
  bool complete_override = false;
  bool has_complete_override = false;

  const auto consume_profile = [&](const json::Value & object) {
      if (!object.is_object()) {
        return;
      }
      manifest.capability_profile_present = true;
      for (const auto & entry : object.as_object()) {
        if (entry.first == "required" || entry.first == "complete") {
          continue;
        }
        if (entry.second.is_bool() || entry.second.is_string() || entry.second.is_number()) {
          manifest.capabilities[entry.first] = parse_bool(&entry.second);
        }
      }
      if (const json::Value * capabilities = object.find("capabilities")) {
        if (capabilities->is_object()) {
          for (const auto & entry : capabilities->as_object()) {
            manifest.capabilities[entry.first] = parse_bool(&entry.second);
          }
        }
      }
      if (const json::Value * required = object.find("required")) {
        const auto names = strings(required);
        if (!names.empty()) {
          manifest.required_capabilities = names;
        }
      }
      if (const json::Value * complete = object.find("complete")) {
        complete_override = parse_bool(complete);
        has_complete_override = true;
      }
    };

  if (const json::Value * capabilities = node.find("capabilities")) {
    if (capabilities->is_object()) {
      manifest.capability_profile_present = true;
      for (const auto & entry : capabilities->as_object()) {
        if (entry.first == "complete" || entry.first == "required") {
          continue;
        }
        manifest.capabilities[entry.first] = parse_bool(&entry.second);
      }
      if (const json::Value * required = capabilities->find("required")) {
        const auto names = strings(required);
        if (!names.empty()) {
          manifest.required_capabilities = names;
        }
      }
      if (const json::Value * complete = capabilities->find("complete")) {
        complete_override = parse_bool(complete);
        has_complete_override = true;
      }
    }
  }
  if (profile != nullptr) {
    consume_profile(*profile);
  }

  // Optional sidecar file. The main manifest wins on conflicts.
  const std::string sidecar = directory + "/capability_profile.json";
  if (file_exists(sidecar)) {
    try {
      const json::Value sidecar_doc = json::parse(read_file(sidecar));
      if (!manifest.capability_profile_present) {
        consume_profile(sidecar_doc);
      } else {
        ArtifactManifest temporary;
        consume_profile(sidecar_doc);
        for (const auto & entry : temporary.capabilities) {
          manifest.capabilities.emplace(entry.first, entry.second);
        }
      }
    } catch (const std::exception &) {
      // An unreadable sidecar is simply an absent profile; auto stays Python.
    }
  }

  const auto & required = manifest.required_capabilities.empty() ?
    default_required_capabilities() : manifest.required_capabilities;
  manifest.missing_capabilities.clear();
  bool all_required = true;
  for (const auto & name : required) {
    const auto it = manifest.capabilities.find(name);
    if (it == manifest.capabilities.end() || !it->second) {
      all_required = false;
      manifest.missing_capabilities.push_back(name);
    }
  }
  // A profile that explicitly says `complete: true` is authoritative. If it
  // also lists `required`, every listed capability must still be true.
  manifest.capability_profile_complete =
    manifest.capability_profile_present &&
    (has_complete_override ? (complete_override && all_required) : all_required);
}

void merge_kinematics_manifest(
  const std::string & directory, ArtifactManifest & manifest)
{
  const std::string path = directory + "/robot_kinematics_manifest.json";
  if (!file_exists(path)) {
    return;
  }
  try {
    const json::Value doc = json::parse(read_file(path));
    if (manifest.arch.empty()) {
      manifest.arch = doc.string_at("arch");
    }
    if (manifest.num_dof == 0) {
      manifest.num_dof = static_cast<int>(doc.number_at("num_dof"));
    }
    if (manifest.num_tool_frames == 0) {
      manifest.num_tool_frames = static_cast<int>(doc.number_at("num_tool_frames"));
    }
    if (manifest.joint_names.empty()) {
      manifest.joint_names = strings(doc.find("joint_names"));
    }
    if (manifest.tool_frames.empty()) {
      manifest.tool_frames = strings(doc.find("tool_frames"));
    }
    if (manifest.link_names.empty()) {
      manifest.link_names = strings(doc.find("link_names"));
    }
    if (manifest.base_link.empty()) {
      manifest.base_link = doc.string_at("base_link");
    }
    if (manifest.curobo_config.empty()) {
      manifest.curobo_config = doc.string_at("source.robot_config");
    }
    if (manifest.source_file.empty()) {
      manifest.source_file = doc.string_at("source.robot_config");
    }
  } catch (const std::exception &) {
  }
}

void merge_trajopt_manifest(
  const std::string & directory, ArtifactManifest & manifest)
{
  const std::string path = directory + "/robot_trajopt_manifest.json";
  if (!file_exists(path)) {
    return;
  }
  try {
    const json::Value doc = json::parse(read_file(path));
    const int capacity = static_cast<int>(doc.number_at("interpolation_buffer_size"));
    if (capacity > 0) {
      manifest.interpolation_buffer_size = capacity;
    }
    load_scene_capacities(doc, manifest);
  } catch (const std::exception &) {
  }
}

void merge_ik_manifest(
  const std::string & directory, ArtifactManifest & manifest)
{
  const std::string path = directory + "/robot_ik_manifest.json";
  if (!file_exists(path)) {
    return;
  }
  try {
    const json::Value doc = json::parse(read_file(path));
    load_scene_capacities(doc, manifest);
    const int dof = manifest.num_dof > 0 ?
      manifest.num_dof : static_cast<int>(doc.number_at("num_dof"));
    if (dof <= 0 || !manifest.joint_lower.empty()) {
      return;
    }
    const json::Value * tensors = doc.find("tensors");
    if (tensors == nullptr) {
      return;
    }
    const json::Value * bounds = tensors->find("ik_position_bounds");
    if (bounds == nullptr) {
      return;
    }
    const std::string file = bounds->string_at("file");
    if (file.empty()) {
      return;
    }
    const std::string bytes = read_file(directory + "/" + file);
    const std::size_t expected = static_cast<std::size_t>(2 * dof) * sizeof(float);
    if (bytes.size() != expected) {
      return;
    }
    std::vector<float> values(static_cast<std::size_t>(2 * dof));
    std::memcpy(values.data(), bytes.data(), expected);
    manifest.joint_lower.assign(values.begin(), values.begin() + dof);
    manifest.joint_upper.assign(values.begin() + dof, values.end());
  } catch (const std::exception &) {
  }
}

void merge_generated_manifests(const std::string & directory, ArtifactManifest & manifest)
{
  merge_kinematics_manifest(directory, manifest);
  merge_trajopt_manifest(directory, manifest);
  merge_ik_manifest(directory, manifest);
}

}  // namespace

const std::vector<std::string> & default_required_capabilities()
{
  static const std::vector<std::string> required = {
    "robot_model",
    "fk",
    "ik",
    "ik_multi",
    "validate",
    "plan_pose",
    "plan_cspace",
    "plan_pose_waypoints",
    "reset_seed",
    "tool_frames",
    "joint_limits",
    "set_link_collision",
    "scene_set",
    "scene_get",
  };
  return required;
}

std::string normalize_arch(const std::string & value)
{
  std::string text = trim(value);
  if (text.empty()) {
    return {};
  }
  if (text.rfind("compute_", 0) == 0) {
    text = "sm_" + text.substr(std::strlen("compute_"));
  } else if (text.rfind("sm_", 0) != 0) {
    bool digits = true;
    for (char c : text) {
      if (std::isdigit(static_cast<unsigned char>(c)) == 0) {
        digits = false;
        break;
      }
    }
    if (digits) {
      text = "sm_" + text;
    }
  }
  if (text.rfind("sm_", 0) != 0 || text.size() <= 3) {
    return {};
  }
  return text;
}

std::string normalize_robot_name(const std::string & value)
{
  std::string out = trim(value);
  std::transform(
    out.begin(), out.end(), out.begin(),
    [](unsigned char c) {return static_cast<char>(std::tolower(c));});
  return out;
}

std::string strip_config_suffix(std::string name)
{
  static const std::array<const char *, 4> suffixes = {
    ".curobo.yml", ".curobo.yaml", ".yml", ".yaml"};
  for (const char * suffix : suffixes) {
    const std::string ending(suffix);
    if (name.size() > ending.size() &&
      name.compare(name.size() - ending.size(), ending.size(), ending) == 0)
    {
      name.erase(name.size() - ending.size());
      break;
    }
  }
  return name;
}

std::string artifact_key_from_path(const std::string & value)
{
  if (value.find('/') == std::string::npos && value.find('\\') == std::string::npos) {
    return trim(value);
  }
  return strip_config_suffix(std::filesystem::path(value).filename().string());
}

std::string current_cuda_arch()
{
#ifdef CUROBO_CPP_RUNTIME_HAVE_CUDA
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return {};
  }
  int major = 0;
  int minor = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) !=
    cudaSuccess)
  {
    return {};
  }
  if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) !=
    cudaSuccess)
  {
    return {};
  }
  return "sm_" + std::to_string(major) + std::to_string(minor);
#else
  return {};
#endif
}

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4). Kept local so the runtime package has no OpenSSL
// dependency; the hash is over the YAML bytes only.
// ---------------------------------------------------------------------------
namespace
{
constexpr std::array<std::uint32_t, 64> kSha256K = {
  0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U,
  0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
  0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U,
  0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
  0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
  0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
  0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
  0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
  0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU,
  0x5b9cca4fU, 0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
  0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

inline std::uint32_t rotr(std::uint32_t value, std::uint32_t bits)
{
  return (value >> bits) | (value << (32U - bits));
}

std::string sha256(const std::string & bytes)
{
  std::array<std::uint32_t, 8> state = {
    0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
    0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  std::vector<std::uint8_t> message(bytes.begin(), bytes.end());
  const std::uint64_t bit_length = static_cast<std::uint64_t>(message.size()) * 8U;
  message.push_back(0x80U);
  while (message.size() % 64U != 56U) {
    message.push_back(0U);
  }
  for (int shift = 56; shift >= 0; shift -= 8) {
    message.push_back(static_cast<std::uint8_t>((bit_length >> shift) & 0xFFU));
  }

  for (std::size_t offset = 0; offset < message.size(); offset += 64U) {
    std::array<std::uint32_t, 64> w{};
    for (std::size_t i = 0; i < 16U; ++i) {
      w[i] = (static_cast<std::uint32_t>(message[offset + 4U * i]) << 24U) |
        (static_cast<std::uint32_t>(message[offset + 4U * i + 1U]) << 16U) |
        (static_cast<std::uint32_t>(message[offset + 4U * i + 2U]) << 8U) |
        static_cast<std::uint32_t>(message[offset + 4U * i + 3U]);
    }
    for (std::size_t i = 16U; i < 64U; ++i) {
      const std::uint32_t s0 = rotr(w[i - 15U], 7U) ^ rotr(w[i - 15U], 18U) ^
        (w[i - 15U] >> 3U);
      const std::uint32_t s1 = rotr(w[i - 2U], 17U) ^ rotr(w[i - 2U], 19U) ^
        (w[i - 2U] >> 10U);
      w[i] = w[i - 16U] + s0 + w[i - 7U] + s1;
    }
    std::uint32_t a = state[0];
    std::uint32_t b = state[1];
    std::uint32_t c = state[2];
    std::uint32_t d = state[3];
    std::uint32_t e = state[4];
    std::uint32_t f = state[5];
    std::uint32_t g = state[6];
    std::uint32_t h = state[7];
    for (std::size_t i = 0; i < 64U; ++i) {
      const std::uint32_t s1 = rotr(e, 6U) ^ rotr(e, 11U) ^ rotr(e, 25U);
      const std::uint32_t ch = (e & f) ^ ((~e) & g);
      const std::uint32_t temp1 = h + s1 + ch + kSha256K[i] + w[i];
      const std::uint32_t s0 = rotr(a, 2U) ^ rotr(a, 13U) ^ rotr(a, 22U);
      const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = s0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  }

  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (std::uint32_t value : state) {
    out << std::setw(8) << value;
  }
  return out.str();
}
}  // namespace

std::string sha256_hex(const std::string & bytes)
{
  return sha256(bytes);
}

std::string sha256_file_hex(const std::string & path)
{
  std::string error;
  const std::string bytes = read_file(path, &error);
  if (!error.empty()) {
    return {};
  }
  return sha256(bytes);
}

ArtifactManifest load_artifact_manifest(const std::string & dir)
{
  ArtifactManifest manifest;
  const std::string path = dir + "/manifest.json";
  if (!file_exists(path)) {
    merge_generated_manifests(dir, manifest);
    return manifest;
  }
  try {
    const json::Value document = json::parse(read_file(path));
    manifest.present = true;
    manifest.path = path;

    if (const json::Value * schema = document.find("schema_version")) {
      if (schema->is_number()) {
        manifest.schema_version = std::to_string(static_cast<long long>(schema->as_double()));
      } else if (schema->is_string()) {
        manifest.schema_version = schema->as_string();
      }
    }
    manifest.robot = document.string_at("robot", document.string_at("robot_name"));
    manifest.base_link = document.string_at("base_link");
    manifest.curobo_config = document.string_at(
      "curobo_config", document.string_at("source.robot_config"));
    if (const json::Value * robot_node = document.find("robot")) {
      if (robot_node->is_object()) {
        manifest.robot = robot_node->string_at("name", manifest.robot);
        manifest.base_link = robot_node->string_at("base_link", manifest.base_link);
        manifest.curobo_config = robot_node->string_at(
          "curobo_config", manifest.curobo_config);
      }
    }
    manifest.strategy = document.string_at(
      "strategy", document.string_at("control_strategy"));
    if (const json::Value * params = document.find("strategy_params")) {
      if (params->is_object()) {
        for (const auto & entry : params->as_object()) {
          if (entry.second.is_string()) {
            manifest.strategy_params[entry.first] = entry.second.as_string();
          } else if (entry.second.is_number()) {
            manifest.strategy_params[entry.first] =
              std::to_string(entry.second.as_double());
          } else if (entry.second.is_bool()) {
            manifest.strategy_params[entry.first] =
              entry.second.as_bool() ? "true" : "false";
          }
        }
      }
    }
    manifest.arch = document.string_at("arch", document.string_at("target_arch"));
    manifest.curobo_config = document.string_at(
      "files.curobo_config", manifest.curobo_config);
    manifest.source_hash = document.string_at(
      "source_hash", document.string_at("source.file_hash"));
    manifest.source_file = document.string_at(
      "source_file", document.string_at("source.file"));
    manifest.source_file = document.string_at(
      "files.robot_config", manifest.source_file);

    manifest.home_names = strings(
      first(document, {"home_names", "home.names", "home.joint_names"}));
    manifest.home_positions = numbers(
      first(document, {"home_positions", "home.positions"}));
    manifest.joint_names = strings(
      first(document, {"joint_names", "cspace.joint_names", "cspace.names"}));
    manifest.link_names = strings(document.find("link_names"));
    manifest.tool_frames = strings(first(document, {"tool_frames", "tools"}));
    manifest.locked_joint_names = strings(
      first(document, {"locked_joint_names", "locked_joints.names"}));
    manifest.locked_positions = numbers(
      first(document, {"locked_positions", "locked_joints.positions"}));
    manifest.unpinned_lock_joint_names = strings(
      first(document, {"unpinned_lock_joint_names", "unpinned_lock_joints"}));

    if (const json::Value * lock_joints = document.find("lock_joints")) {
      if (lock_joints->is_object()) {
        if (const json::Value * pinned = lock_joints->find("pinned")) {
          if (pinned->is_object()) {
            manifest.locked_joint_names.clear();
            manifest.locked_positions.clear();
            for (const auto & entry : pinned->as_object()) {
              if (entry.second.is_number()) {
                manifest.locked_joint_names.push_back(entry.first);
                manifest.locked_positions.push_back(entry.second.as_double());
              }
            }
          }
        }
        if (const json::Value * unpinned = lock_joints->find("unpinned")) {
          const auto names = strings(unpinned);
          if (!names.empty()) {
            manifest.unpinned_lock_joint_names = names;
          }
        }
      }
    }

    if (const json::Value * limits = first(
        document, {"joint_limits", "cspace.joint_limits", "cspace.limits"}))
    {
      if (limits->is_object()) {
        manifest.joint_lower = numbers(
          first(*limits, {"lower", "position_lower", "position_min"}));
        manifest.joint_upper = numbers(
          first(*limits, {"upper", "position_upper", "position_max"}));
      } else if (limits->is_array() && limits->as_array().size() == 2) {
        manifest.joint_lower = numbers(&limits->as_array()[0]);
        manifest.joint_upper = numbers(&limits->as_array()[1]);
      }
    }
    if (manifest.joint_lower.empty()) {
      manifest.joint_lower = numbers(
        first(document, {"joint_lower", "joint_position_lower", "cspace.lower"}));
    }
    if (manifest.joint_upper.empty()) {
      manifest.joint_upper = numbers(
        first(document, {"joint_upper", "joint_position_upper", "cspace.upper"}));
    }

    manifest.num_dof = static_cast<int>(document.number_at("num_dof"));
    manifest.num_tool_frames = static_cast<int>(document.number_at("num_tool_frames"));
    manifest.interpolation_buffer_size = static_cast<int>(
      document.number_at("interpolation_buffer_size"));
    if (manifest.num_dof == 0 && !manifest.joint_names.empty()) {
      manifest.num_dof = static_cast<int>(manifest.joint_names.size());
    }
    if (manifest.num_tool_frames == 0 && !manifest.tool_frames.empty()) {
      manifest.num_tool_frames = static_cast<int>(manifest.tool_frames.size());
    }
    load_scene_capacities(document, manifest);
    load_capabilities(document, dir, manifest);

    merge_generated_manifests(dir, manifest);
  } catch (const std::exception & error) {
    manifest.present = false;
    manifest.path = path + " (" + error.what() + ")";
    merge_generated_manifests(dir, manifest);
  }
  return manifest;
}

ArtifactBundle inspect_artifact_directory(
  const std::string & directory,
  const std::string & robot,
  const std::string & config_file)
{
  ArtifactBundle bundle;
  bundle.directory = directory;
  bundle.robot = robot;
  bundle.sm_arch = normalize_arch(std::filesystem::path(directory).filename().string());
  bundle.root = std::filesystem::path(directory).parent_path().parent_path().string();
  bundle.manifest = load_artifact_manifest(directory);
  bundle.manifest_present = bundle.manifest.present;

  const std::string kinematics = directory + "/robot_kinematics_manifest.json";
  const std::string ik = directory + "/robot_ik_manifest.json";
  const std::string trajopt = directory + "/robot_trajopt_manifest.json";
  const std::string motion = directory + "/robot_motion_manifest.json";
  bundle.has_required_files =
    file_exists(kinematics) && file_exists(ik) && file_exists(trajopt) &&
    file_exists(motion);

  if (!bundle.manifest_present) {
    bundle.error = "artifact manifest.json is missing or unreadable in '" + directory + "'";
    return bundle;
  }

  if (!bundle.has_required_files) {
    bundle.error = "artifact is incomplete in '" + directory +
      "' (expected robot_kinematics, robot_ik, robot_trajopt and robot_motion manifests)";
    return bundle;
  }

  if (!robot.empty() && !bundle.manifest.robot.empty()) {
    const std::string requested = normalize_robot_name(robot);
    const std::string recorded = normalize_robot_name(bundle.manifest.robot);
    if (requested != recorded &&
      requested.find(recorded) == std::string::npos &&
      recorded.find(requested) == std::string::npos)
    {
      bundle.error = "artifact belongs to robot '" + bundle.manifest.robot +
        "', requested '" + robot + "'";
      return bundle;
    }
  }

  bundle.arch_matches = !bundle.sm_arch.empty() ?
    (normalize_arch(bundle.manifest.arch) == normalize_arch(bundle.sm_arch)) :
    !normalize_arch(bundle.manifest.arch).empty();
  if (!bundle.arch_matches) {
    bundle.error = "artifact arch '" + bundle.manifest.arch +
      "' does not match requested '" + bundle.sm_arch + "'";
    return bundle;
  }

  // Hash the configured source YAML when the exporter recorded a hash. A
  // missing local copy is not a mismatch: the image may not ship the source
  // tree. A present-but-different file is a hard mismatch.
  if (!bundle.manifest.source_hash.empty()) {
    std::vector<std::string> hash_targets;
    const auto add_target = [&](const std::string & path) {
        if (!path.empty() &&
          std::find(hash_targets.begin(), hash_targets.end(), path) == hash_targets.end())
        {
          hash_targets.push_back(path);
        }
      };
    if (!bundle.manifest.source_file.empty()) {
      // The generated manifest names the exact file its source_hash covers.
      add_target(bundle.manifest.source_file);
    } else {
      add_target(config_file);
      add_target(bundle.manifest.curobo_config);
    }

    bool saw_readable = false;
    bool matched = false;
    for (const auto & hash_target : hash_targets) {
      if (!file_exists(hash_target)) {
        continue;
      }
      const std::string actual = sha256_file_hex(hash_target);
      if (actual.empty()) {
        continue;
      }
      saw_readable = true;
      if (actual == bundle.manifest.source_hash) {
        matched = true;
        break;
      }
    }
    if (saw_readable) {
      bundle.hash_available = true;
      bundle.hash_matches = matched;
      if (!bundle.hash_matches) {
        bundle.error = "artifact source hash does not match the configured YAML; "
          "re-export the artifact";
        return bundle;
      }
    }
  }

  bundle.capability_profile_complete = bundle.manifest.capability_profile_complete;
  bundle.missing_capabilities = bundle.manifest.missing_capabilities;
  bundle.valid = true;
  return bundle;
}

ArtifactBundleResolver::ArtifactBundleResolver(
  std::string artifact_root, std::string sm_arch_override)
: root_(std::move(artifact_root)), sm_arch_override_(std::move(sm_arch_override))
{
  if (root_.empty()) {
    root_ = "/opt/curobo/artifacts";
  }
}

ArtifactBundle ArtifactBundleResolver::resolve(
  const std::string & robot, const std::string & config_file,
  const std::string & sm_arch) const
{
  ArtifactBundle bundle;
  bundle.root = root_;
  bundle.robot = robot;

  std::string arch = normalize_arch(sm_arch);
  if (arch.empty()) {
    arch = normalize_arch(sm_arch_override_);
  }
  if (arch.empty()) {
    arch = current_cuda_arch();
  }
  bundle.sm_arch = arch;

  if (robot.empty() && config_file.empty()) {
    bundle.error = "no robot name was provided for artifact resolution";
    return bundle;
  }
  if (arch.empty()) {
    bundle.error =
      "no CUDA device architecture available; set curobo_artifact_sm or run on a GPU";
    return bundle;
  }

  // `robot` is the robot *descriptor* path in some launches (e.g.
  // openarm.yaml), while the artifact directory is named after the cuRobo
  // config (`openarm_bimanual`). Try the config basename first, then the
  // descriptor basename, so both plain robot names and paths resolve.
  std::string robot_name = artifact_key_from_path(robot);
  if (robot_name.empty()) {
    robot_name = artifact_key_from_path(config_file);
  }
  std::vector<std::string> candidates;
  const auto add_candidate = [&candidates](const std::string & value) {
      const std::string key = artifact_key_from_path(value);
      if (key.empty()) {
        return;
      }
      if (std::find(candidates.begin(), candidates.end(), key) == candidates.end()) {
        candidates.push_back(key);
      }
    };
  add_candidate(config_file);
  add_candidate(robot);

  std::string last_error;
  std::string last_directory;
  bool inspected_any = false;
  for (const std::string & key : candidates) {
    const std::string directory = root_ + "/" + key + "/" + arch;
    last_directory = directory;
    std::error_code error;
    if (!std::filesystem::is_directory(directory, error)) {
      if (last_error.empty()) {
        last_error = "artifact directory does not exist: '" + directory + "'";
      }
      continue;
    }
    ArtifactBundle candidate = inspect_artifact_directory(directory, robot_name, config_file);
    candidate.root = root_;
    candidate.robot = robot_name;
    candidate.sm_arch = arch;
    candidate.directory = directory;
    if (candidate.valid) {
      return candidate;
    }
    // Keep the first inspected invalid bundle: callers/tests use its hash and
    // manifest diagnostics, not just the final error string.
    if (!inspected_any) {
      bundle = candidate;
      inspected_any = true;
    }
    last_error = candidate.error;
  }

  if (!inspected_any) {
    bundle.error = last_error.empty() ?
      "no artifact directory was found under '" + root_ + "'" : last_error;
    bundle.directory = last_directory;
    return bundle;
  }
  bundle.error = last_error;
  bundle.root = root_;
  bundle.robot = robot_name;
  bundle.sm_arch = arch;
  bundle.directory = last_directory;
  return bundle;
}

}  // namespace curobo_cpp_runtime

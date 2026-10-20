// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/cameras/camera_cfg.py.
#include "curobo_core/cameras/camera_cfg.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <sstream>

namespace curobo_core
{

// PORT: camera_cfg.masked_depth_topic
std::string masked_depth_topic(const std::string & depth_topic)
{
  std::string topic = depth_topic;
  while (!topic.empty() && topic.back() == '/') {
    topic.pop_back();
  }
  if (topic.empty()) {
    return "";
  }
  const std::size_t idx = topic.rfind('/');
  if (idx == std::string::npos) {
    return "masked_depth";
  }
  return topic.substr(0, idx) + "/masked_depth";
}

// The frame the depth is integrated in: ALWAYS the configured robot base
// (`base_link`). Depth and the Mapper's grid box must share ONE frame, and the
// grid is a fixed volume allocated once, so this is pinned — and
// `mapper_grid_center` is base-relative, so the voxels ride the robot. That is
// deliberate: the solver's voxel collision cache is built around `base_link`, so
// integrating in a world frame (map/odom) would leave the ESDF pushed into the
// cache in a different frame from the one the cache is expressed in.
std::vector<std::string> integration_frame_candidates(rclcpp::Node * node)
{
  std::vector<std::string> frames;
  std::string frame;
  if (node != nullptr && node->has_parameter("base_link")) {
    frame = node->get_parameter("base_link").as_string();
  }
  frames.push_back(frame.empty() ? "base_0" : frame);
  return frames;
}

// Single pick for callers WITHOUT a TF buffer (marker/service metadata): the
// configured robot base, same frame as integration_frame_candidates().
std::string integration_frame(rclcpp::Node * node)
{
  return (node != nullptr && node->has_parameter("base_link")) ?
         node->get_parameter("base_link").as_string() : "base_0";
}

// PORT: camera_cfg._parse_floats
std::optional<std::vector<double>> parse_floats(const std::string & text)
{
  if (text.empty()) {
    return std::nullopt;
  }
  std::vector<double> out;
  std::stringstream stream(text);
  std::string part;
  while (std::getline(stream, part, ',')) {
    std::string trimmed;
    for (char c : part) {
      if (!std::isspace(static_cast<unsigned char>(c))) {
        trimmed += c;
      }
    }
    if (trimmed.empty()) {
      continue;
    }
    try {
      out.push_back(std::stod(trimmed));
    } catch (const std::exception &) {
      return std::nullopt;
    }
  }
  if (out.empty()) {
    return std::nullopt;
  }
  return out;
}

namespace
{
// PORT: camera_cfg._declare_param
template<class T>
void declare_param(rclcpp::Node * node, const std::string & name, const T & value)
{
  if (!node->has_parameter(name)) {
    node->declare_parameter<T>(name, value);
  }
}

// PORT: camera_cfg._entry
std::string entry_string(rclcpp::Node * node, const std::string & name, int index)
{
  if (!node->has_parameter(name)) {
    return "";
  }
  const auto values = node->get_parameter(name).as_string_array();
  if (index < 0 || static_cast<std::size_t>(index) >= values.size()) {
    return "";
  }
  std::string value = values[static_cast<std::size_t>(index)];
  // Python: an all-whitespace string entry means "unset".
  const bool blank = value.find_first_not_of(" \t\n\r") == std::string::npos;
  return blank ? "" : value;
}

double entry_double(rclcpp::Node * node, const std::string & name, int index, double fallback)
{
  if (!node->has_parameter(name)) {
    return fallback;
  }
  const auto values = node->get_parameter(name).as_double_array();
  if (index < 0 || static_cast<std::size_t>(index) >= values.size()) {
    return fallback;
  }
  return values[static_cast<std::size_t>(index)];
}
}  // namespace

// PORT: PerceptionCameraCfg.declare
void PerceptionCameraCfg::declare(rclcpp::Node * node)
{
  declare_param(node, PARAM_TOPIC, std::vector<std::string>{""});
  declare_param(node, PARAM_CAMERA_INFO, std::vector<std::string>{""});
  declare_param(node, PARAM_FRAME, std::vector<std::string>{""});
  declare_param(node, PARAM_INTRINSICS, std::vector<std::string>{""});
  declare_param(node, PARAM_EXTRINSICS, std::vector<std::string>{""});
  declare_param(node, PARAM_FRAME_RATE, std::vector<double>{0.0});
  declare_param(node, PARAM_PURPOSE, std::vector<std::string>{PURPOSE_ALL});
}

// PORT: PerceptionCameraCfg.num_cameras
int PerceptionCameraCfg::num_cameras(rclcpp::Node * node)
{
  if (!node->has_parameter(PARAM_TOPIC)) {
    return 0;
  }
  return static_cast<int>(node->get_parameter(PARAM_TOPIC).as_string_array().size());
}

// PORT: PerceptionCameraCfg.from_node
PerceptionCameraCfg PerceptionCameraCfg::from_node(rclcpp::Node * node, int camera_index)
{
  PerceptionCameraCfg cfg;
  cfg.camera_index = camera_index;
  cfg.name = "depth_camera_" + std::to_string(camera_index);
  cfg.depth_topic = entry_string(node, PARAM_TOPIC, camera_index);
  cfg.camera_info_topic = entry_string(node, PARAM_CAMERA_INFO, camera_index);
  cfg.frame_id = entry_string(node, PARAM_FRAME, camera_index);
  cfg.intrinsics = parse_floats(entry_string(node, PARAM_INTRINSICS, camera_index));
  cfg.extrinsics = parse_floats(entry_string(node, PARAM_EXTRINSICS, camera_index));

  double rate = entry_double(node, PARAM_FRAME_RATE, camera_index, 0.0);
  if (rate <= 0.0) {
    rate = DEFAULT_CAMERA_FRAME_RATE_HZ;
  }
  cfg.frame_rate_hz = rate;

  std::string purpose = entry_string(node, PARAM_PURPOSE, camera_index);
  if (purpose.empty()) {
    purpose = PURPOSE_ALL;
  }
  std::transform(purpose.begin(), purpose.end(), purpose.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
  });
  // Python also strips; the entry helper already rejects blank.
  if (purpose != PURPOSE_ALL && purpose != PURPOSE_ESDF && purpose != PURPOSE_SEGMENTATION) {
    purpose = PURPOSE_ALL;
  }
  cfg.purpose = purpose;

  cfg.masked_output_topic = masked_depth_topic(cfg.depth_topic);

  // PORT: mapper_topic derivation.
  const bool segmentation_enabled = node->has_parameter("enable_robot_segmentation") ?
    node->get_parameter("enable_robot_segmentation").as_bool() : true;
  if (segmentation_enabled && !cfg.depth_topic.empty() &&
    (purpose == PURPOSE_SEGMENTATION || purpose == PURPOSE_ALL))
  {
    cfg.mapper_topic = cfg.masked_output_topic;
  } else {
    cfg.mapper_topic = cfg.depth_topic;
  }
  return cfg;
}

}  // namespace curobo_core

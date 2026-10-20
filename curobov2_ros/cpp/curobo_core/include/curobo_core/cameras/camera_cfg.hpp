// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/cameras/camera_cfg.py
//
// Single source of truth for the perception cameras: the `camera_*` param block
// and the derived masked/mapper topics. Pure ROS-param handling.
#ifndef CUROBO_CORE__CAMERAS__CAMERA_CFG_HPP_
#define CUROBO_CORE__CAMERAS__CAMERA_CFG_HPP_

#include <optional>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

namespace curobo_core
{

// PORT: camera_cfg.DEFAULT_CAMERA_FRAME_RATE_HZ
inline constexpr double DEFAULT_CAMERA_FRAME_RATE_HZ = 30.0;
// Fallback Mapper depth-projection resolution used when nothing better is
// known: no camera is configured, or none answered on its camera_info topic
// within the probe timeout. Matches the node's historical default. When any
// camera reports a size, the manager sizes the shared Mapper to the LARGEST
// one (see CameraSystemManager::configure) and the depth strategies resample
// the other cameras to it.
inline constexpr int DEFAULT_MAPPER_IMAGE_WIDTH = 640;
inline constexpr int DEFAULT_MAPPER_IMAGE_HEIGHT = 480;
// PORT: PURPOSE_*
inline constexpr const char * PURPOSE_ESDF = "esdf";
inline constexpr const char * PURPOSE_SEGMENTATION = "segmentation";
inline constexpr const char * PURPOSE_ALL = "all";

// PORT: camera_cfg.masked_depth_topic
std::string masked_depth_topic(const std::string & depth_topic);

// Frame the depth is integrated in — the Mapper's TSDF/ESDF frame — as a list of
// candidates to try against TF: ALWAYS the configured robot base (`base_link`).
// `mapper_grid_center` is base-relative, so the voxels ride the robot; a world
// frame (map/odom) would not match the solver's voxel collision cache, which is
// built around base_link. Kept as a list so the caller's try-each-in-order loop
// stays unchanged.
std::vector<std::string> integration_frame_candidates(rclcpp::Node * node);

// Single pick for callers without a TF buffer (marker/service metadata): the
// configured robot base, same frame as integration_frame_candidates().
std::string integration_frame(rclcpp::Node * node);

// PORT: camera_cfg._parse_floats
std::optional<std::vector<double>> parse_floats(const std::string & text);

// PORT: camera_cfg.PerceptionCameraCfg
struct PerceptionCameraCfg
{
  int camera_index = 0;
  std::string name;
  std::string depth_topic;
  std::string camera_info_topic;
  std::string frame_id;
  std::optional<std::vector<double>> intrinsics;
  std::optional<std::vector<double>> extrinsics;
  double frame_rate_hz = DEFAULT_CAMERA_FRAME_RATE_HZ;
  std::string purpose = PURPOSE_ALL;
  std::string masked_output_topic;
  std::string mapper_topic;

  // PORT: for_esdf / for_segmentation
  bool for_esdf() const
  {
    return purpose == PURPOSE_ESDF || purpose == PURPOSE_ALL;
  }
  bool for_segmentation() const
  {
    return purpose == PURPOSE_SEGMENTATION || purpose == PURPOSE_ALL;
  }

  // PORT: PARAM_*
  static constexpr const char * PARAM_TOPIC = "camera_topic";
  static constexpr const char * PARAM_CAMERA_INFO = "camera_info_topic";
  static constexpr const char * PARAM_FRAME = "camera_frame";
  static constexpr const char * PARAM_INTRINSICS = "camera_intrinsics";
  static constexpr const char * PARAM_EXTRINSICS = "camera_extrinsics";
  static constexpr const char * PARAM_FRAME_RATE = "camera_frame_rate_hz";
  static constexpr const char * PARAM_PURPOSE = "camera_purpose";

  // PORT: PerceptionCameraCfg.declare
  static void declare(rclcpp::Node * node);
  // PORT: PerceptionCameraCfg.num_cameras
  static int num_cameras(rclcpp::Node * node);
  // PORT: PerceptionCameraCfg.from_node
  static PerceptionCameraCfg from_node(rclcpp::Node * node, int camera_index = 0);
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__CAMERAS__CAMERA_CFG_HPP_

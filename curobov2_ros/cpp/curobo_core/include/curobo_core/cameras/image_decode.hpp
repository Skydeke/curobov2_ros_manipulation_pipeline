// SPDX-License-Identifier: Apache-2.0
// PORT: the depth decode/encode in
//   cameras/camera_depth_map_strategy.py (CvBridge.imgmsg_to_cv2)
//   cameras/camera_robot_segmentation_strategy.py (depth_tensor_to_image_msg)
//
// C++ replaces cv_bridge here (C4): ROS image decode/encode is ROS I/O.
#ifndef CUROBO_CORE__CAMERAS__IMAGE_DECODE_HPP_
#define CUROBO_CORE__CAMERAS__IMAGE_DECODE_HPP_

#include <string>
#include <vector>

#include <sensor_msgs/msg/image.hpp>

namespace curobo_core
{

// One decoded depth frame (metres), row-major.
struct DepthFrame
{
  std::uint32_t height = 0;
  std::uint32_t width = 0;
  std::vector<float> data;        // metres
  std::string encoding;           // "16UC1" / "32FC1"
};

// PORT: camera_depth_map_strategy.callback_depth_map - 16UC1 is millimetres
// (divided by 1000); 32FC1 is metres; anything else is reported so the caller
// can warn and try 32FC1.
DepthFrame decode_depth(const sensor_msgs::msg::Image & msg, bool & unsupported);

// PORT: _create_pointcloud2_msg / depth_tensor_to_image_msg - encode a depth
// frame as an Image with the given encoding ("32FC1" or "16UC1").
sensor_msgs::msg::Image encode_depth(
  const std::vector<float> & data, std::uint32_t height, std::uint32_t width,
  const std::string & encoding, const builtin_interfaces::msg::Time & stamp);

// PORT: camera_depth_map_strategy.callback_depth_map - the Mapper allocates its
// depth-projection kernel once for the configured mapper_image_width/height, so
// every frame must be resampled to that exact resolution before integration.
// Bilinear (align_corners=false), the cv2.INTER_LINEAR / torch bilinear
// convention. `src` is row-major metres of size src_h * src_w.
std::vector<float> resize_depth_bilinear(
  const std::vector<float> & src, std::uint32_t src_h, std::uint32_t src_w,
  std::uint32_t dst_h, std::uint32_t dst_w);

}  // namespace curobo_core

#endif  // CUROBO_CORE__CAMERAS__IMAGE_DECODE_HPP_

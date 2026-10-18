// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of the depth decode/encode paths (cv_bridge replacement).
#include "curobo_core/cameras/image_decode.hpp"

#include <cstring>

namespace curobo_core
{

// PORT: camera_depth_map_strategy.callback_depth_map
DepthFrame decode_depth(const sensor_msgs::msg::Image & msg, bool & unsupported)
{
  unsupported = false;
  DepthFrame frame;
  frame.height = msg.height;
  frame.width = msg.width;
  frame.encoding = msg.encoding;
  const std::size_t pixels = static_cast<std::size_t>(msg.height) * msg.width;
  frame.data.resize(pixels);

  if (msg.encoding == "16UC1") {
    // PORT: depth_img.astype(float32) / 1000.0 (millimetres -> metres).
    const auto * src = reinterpret_cast<const std::uint16_t *>(msg.data.data());
    for (std::size_t i = 0; i < pixels; ++i) {
      frame.data[i] = static_cast<float>(src[i]) / 1000.0f;
    }
  } else if (msg.encoding == "32FC1") {
    std::memcpy(frame.data.data(), msg.data.data(), pixels * sizeof(float));
  } else {
    // PORT: "Unsupported depth encoding: {encoding}, trying 32FC1"
    unsupported = true;
    frame.encoding = "32FC1";
    if (msg.data.size() >= pixels * sizeof(float)) {
      std::memcpy(frame.data.data(), msg.data.data(), pixels * sizeof(float));
    }
  }
  return frame;
}

// PORT: depth_tensor_to_image_msg
sensor_msgs::msg::Image encode_depth(
  const std::vector<float> & data, std::uint32_t height, std::uint32_t width,
  const std::string & encoding, const builtin_interfaces::msg::Time & stamp)
{
  sensor_msgs::msg::Image msg;
  msg.header.stamp = stamp;
  msg.height = height;
  msg.width = width;
  const std::size_t pixels = static_cast<std::size_t>(height) * width;
  if (encoding == "16UC1") {
    msg.encoding = "16UC1";
    msg.step = width * 2;
    msg.data.resize(pixels * 2);
    auto * dst = reinterpret_cast<std::uint16_t *>(msg.data.data());
    for (std::size_t i = 0; i < pixels && i < data.size(); ++i) {
      dst[i] = static_cast<std::uint16_t>(data[i] * 1000.0f);
    }
  } else {
    msg.encoding = "32FC1";
    msg.step = width * 4;
    msg.data.resize(pixels * 4);
    std::memcpy(msg.data.data(), data.data(),
      std::min(pixels, data.size()) * sizeof(float));
  }
  msg.is_bigendian = false;
  return msg;
}

}  // namespace curobo_core

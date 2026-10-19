// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of the depth decode/encode paths (cv_bridge replacement).
#include "curobo_core/cameras/image_decode.hpp"

#include <algorithm>
#include <cmath>
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
    // Depth images mark "no return" with inf (RealSense/Astra/Gazebo), not 0.
    // Replace every non-finite sample with 0 so it is treated as invalid
    // downstream: a raw inf reaching the mapper/robot-segmentation raises and
    // silently stops the depth feed (the map then freezes). Real depth is finite
    // and positive, so 0 is unambiguous.
    const auto * src = reinterpret_cast<const float *>(msg.data.data());
    for (std::size_t i = 0; i < pixels; ++i) {
      const float v = src[i];
      frame.data[i] = std::isfinite(v) ? v : 0.0f;
    }
  } else {
    // PORT: "Unsupported depth encoding: {encoding}, trying 32FC1"
    unsupported = true;
    frame.encoding = "32FC1";
    if (msg.data.size() >= pixels * sizeof(float)) {
      const auto * src = reinterpret_cast<const float *>(msg.data.data());
      for (std::size_t i = 0; i < pixels; ++i) {
        const float v = src[i];
        frame.data[i] = std::isfinite(v) ? v : 0.0f;
      }
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

// PORT: camera_depth_map_strategy.callback_depth_map - resample a depth frame to
// the Mapper's projection resolution. Bilinear (align_corners=false): the source
// pixel for an output center is (o + 0.5) * src / dst - 0.5, matching
// cv2.resize(..., INTER_LINEAR) and torch.nn.functional.interpolate(mode="bilinear",
// align_corners=False).
std::vector<float> resize_depth_bilinear(
  const std::vector<float> & src, std::uint32_t src_h, std::uint32_t src_w,
  std::uint32_t dst_h, std::uint32_t dst_w)
{
  std::vector<float> dst(static_cast<std::size_t>(dst_h) * dst_w);
  if (src_h == 0 || src_w == 0 || dst_h == 0 || dst_w == 0) {
    return dst;
  }
  if (src_h == dst_h && src_w == dst_w) {
    return src;
  }
  const double scale_y = static_cast<double>(src_h) / static_cast<double>(dst_h);
  const double scale_x = static_cast<double>(src_w) / static_cast<double>(dst_w);
  const auto clamp_index = [](double v, std::uint32_t n) -> std::uint32_t {
      if (v <= 0.0) {
        return 0;
      }
      const std::uint32_t max_i = n - 1;
      if (v >= static_cast<double>(max_i)) {
        return max_i;
      }
      return static_cast<std::uint32_t>(v);
    };
  for (std::uint32_t oy = 0; oy < dst_h; ++oy) {
    const double sy = (static_cast<double>(oy) + 0.5) * scale_y - 0.5;
    const std::uint32_t y0 = clamp_index(sy, src_h);
    const std::uint32_t y1 = clamp_index(sy + 1.0, src_h);
    const double wy = std::max(0.0, std::min(1.0, sy - static_cast<double>(y0)));
    for (std::uint32_t ox = 0; ox < dst_w; ++ox) {
      const double sx = (static_cast<double>(ox) + 0.5) * scale_x - 0.5;
      const std::uint32_t x0 = clamp_index(sx, src_w);
      const std::uint32_t x1 = clamp_index(sx + 1.0, src_w);
      const double wx = std::max(0.0, std::min(1.0, sx - static_cast<double>(x0)));
      const float v00 = src[static_cast<std::size_t>(y0) * src_w + x0];
      const float v01 = src[static_cast<std::size_t>(y0) * src_w + x1];
      const float v10 = src[static_cast<std::size_t>(y1) * src_w + x0];
      const float v11 = src[static_cast<std::size_t>(y1) * src_w + x1];
      const double top = v00 * (1.0 - wx) + v01 * wx;
      const double bottom = v10 * (1.0 - wx) + v11 * wx;
      dst[static_cast<std::size_t>(oy) * dst_w + ox] =
        static_cast<float>(top * (1.0 - wy) + bottom * wy);
    }
  }
  return dst;
}

}  // namespace curobo_core

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
  // Per-axis taps/weights computed once per row/column instead of per pixel.
  const double scale_y = static_cast<double>(src_h) / static_cast<double>(dst_h);
  const double scale_x = static_cast<double>(src_w) / static_cast<double>(dst_w);
  struct Tap {std::uint32_t i0, i1; float w;};
  const auto make_taps = [](std::uint32_t dn, std::uint32_t sn, double scale) {
      std::vector<Tap> taps(dn);
      const std::uint32_t max_i = sn - 1;
      for (std::uint32_t o = 0; o < dn; ++o) {
        const double s = (static_cast<double>(o) + 0.5) * scale - 0.5;
        const auto clampi = [&](double v) -> std::uint32_t {
            return v <= 0.0 ? 0u : (v >= static_cast<double>(max_i) ? max_i :
                   static_cast<std::uint32_t>(v));
          };
        const std::uint32_t i0 = clampi(s);
        taps[o] = {i0, clampi(s + 1.0),
          static_cast<float>(std::max(0.0, std::min(1.0, s - static_cast<double>(i0))))};
      }
      return taps;
    };
  const std::vector<Tap> xt = make_taps(dst_w, src_w, scale_x);
  const std::vector<Tap> yt = make_taps(dst_h, src_h, scale_y);
  for (std::uint32_t oy = 0; oy < dst_h; ++oy) {
    const float * r0 = src.data() + static_cast<std::size_t>(yt[oy].i0) * src_w;
    const float * r1 = src.data() + static_cast<std::size_t>(yt[oy].i1) * src_w;
    const float wy = yt[oy].w;
    float * out = dst.data() + static_cast<std::size_t>(oy) * dst_w;
    for (std::uint32_t ox = 0; ox < dst_w; ++ox) {
      const Tap & t = xt[ox];
      const float top = r0[t.i0] + (r0[t.i1] - r0[t.i0]) * t.w;
      const float bot = r1[t.i0] + (r1[t.i1] - r1[t.i0]) * t.w;
      out[ox] = top + (bot - top) * wy;
    }
  }
  return dst;
}

}  // namespace curobo_core

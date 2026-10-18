// SPDX-License-Identifier: Apache-2.0
// Depth decode/encode (cv_bridge replacement). No GPU.
#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "curobo_core/cameras/image_decode.hpp"

using curobo_core::decode_depth;
using curobo_core::encode_depth;

namespace
{
sensor_msgs::msg::Image make_image(
  const std::string & encoding, std::uint32_t h, std::uint32_t w,
  const std::vector<std::uint8_t> & data)
{
  sensor_msgs::msg::Image msg;
  msg.height = h;
  msg.width = w;
  msg.encoding = encoding;
  msg.data = data;
  return msg;
}
}  // namespace

TEST(ImageDecode, SixteenBitMillimetresBecomeMetres)
{
  // 2 pixels: 0 mm, 1500 mm -> 0.0 m, 1.5 m
  std::vector<std::uint8_t> raw(4);
  auto * p = reinterpret_cast<std::uint16_t *>(raw.data());
  p[0] = 0;
  p[1] = 1500;
  bool unsupported = true;
  const auto frame = decode_depth(make_image("16UC1", 1, 2, raw), unsupported);
  EXPECT_FALSE(unsupported);
  EXPECT_EQ(frame.height, 1u);
  EXPECT_EQ(frame.width, 2u);
  ASSERT_EQ(frame.data.size(), 2u);
  EXPECT_FLOAT_EQ(frame.data[0], 0.0f);
  EXPECT_FLOAT_EQ(frame.data[1], 1.5f);
}

TEST(ImageDecode, ThirtyTwoBitFloatPassesThrough)
{
  std::vector<float> values{0.25f, 2.5f};
  std::vector<std::uint8_t> raw(values.size() * sizeof(float));
  std::memcpy(raw.data(), values.data(), raw.size());
  bool unsupported = true;
  const auto frame = decode_depth(make_image("32FC1", 1, 2, raw), unsupported);
  EXPECT_FALSE(unsupported);
  EXPECT_FLOAT_EQ(frame.data[0], 0.25f);
  EXPECT_FLOAT_EQ(frame.data[1], 2.5f);
}

TEST(ImageDecode, UnknownEncodingIsReportedAndFallsBack)
{
  // PORT: "Unsupported depth encoding: ... , trying 32FC1"
  std::vector<float> values{3.0f};
  std::vector<std::uint8_t> raw(sizeof(float));
  std::memcpy(raw.data(), values.data(), raw.size());
  bool unsupported = false;
  const auto frame = decode_depth(make_image("bayer", 1, 1, raw), unsupported);
  EXPECT_TRUE(unsupported);
  EXPECT_EQ(frame.encoding, "32FC1");
  EXPECT_FLOAT_EQ(frame.data[0], 3.0f);
}

TEST(ImageDecode, EncodeRoundTripsThroughSixteenBit)
{
  std::vector<float> metres{0.0f, 1.5f};
  const auto msg = encode_depth(metres, 1, 2, "16UC1", builtin_interfaces::msg::Time());
  EXPECT_EQ(msg.encoding, "16UC1");
  EXPECT_EQ(msg.step, 4u);
  bool unsupported = false;
  const auto back = decode_depth(msg, unsupported);
  EXPECT_FLOAT_EQ(back.data[0], 0.0f);
  EXPECT_FLOAT_EQ(back.data[1], 1.5f);
}

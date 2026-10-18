// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/cameras/camera_robot_segmentation_strategy.py.
#include "curobo_runtime/runtime.hpp"
#include "curobo_core/cameras/camera_robot_segmentation_strategy.hpp"

#include <tf2/exceptions.h>
#include <tf2/time.h>

#include <cstring>

#include <Eigen/Geometry>
#include <sensor_msgs/msg/point_field.hpp>

#include "curobo_core/core/frame_utils.hpp"

namespace curobo_core
{

namespace
{
// PORT: camera_robot_segmentation_strategy._debug_topic / masked_depth_topic
std::string leaf_topic(const std::string & topic, const std::string & leaf)
{
  std::string t = topic;
  while (!t.empty() && t.back() == '/') {
    t.pop_back();
  }
  const std::size_t idx = t.rfind('/');
  if (idx == std::string::npos) {
    return leaf;
  }
  return t.substr(0, idx) + "/" + leaf;
}
}  // namespace

RobotSegmentationCameraStrategy::RobotSegmentationCameraStrategy(
  rclcpp::Node * node, const std::string & camera_name, const std::string & topic,
  const std::string & camera_info_topic, const std::string & frame_id,
  const std::optional<std::vector<double>> & intrinsics,
  const std::optional<std::vector<double>> & extrinsics,
  curobo_runtime::IBridge * bridge, curobo_runtime::RobotHandle handle,
  std::function<std::vector<double>()> joint_pose, std::function<std::string()> masks_json,
  rclcpp::CallbackGroup::SharedPtr callback_group)
: CameraStrategy(node, camera_name, topic, camera_info_topic, frame_id, intrinsics, extrinsics),
  bridge_(bridge), handle_(handle), joint_pose_(std::move(joint_pose)),
  masks_json_(std::move(masks_json))
{
  output_topic_ = leaf_topic(topic_, "masked_depth");
  rclcpp::SubscriptionOptions options;
  if (callback_group) {
    options.callback_group = callback_group;
  }
  subscription_ = node_->create_subscription<sensor_msgs::msg::Image>(
    topic_, 1, std::bind(&RobotSegmentationCameraStrategy::callback_depth, this,
    std::placeholders::_1), options);
  publisher_ = node_->create_publisher<sensor_msgs::msg::Image>(output_topic_, 10);
  robot_pointcloud_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>(
    leaf_topic(topic_, "robot_pointcloud_debug"), 10);
  RCLCPP_INFO(
    node_->get_logger(), "[%s] robot segmentation: %s -> %s (frame=%s)",
    name_.c_str(), topic_.c_str(), output_topic_.c_str(),
    frame_id_.empty() ? "?" : frame_id_.c_str());
}

RobotSegmentationCameraStrategy::~RobotSegmentationCameraStrategy()
{
  destroy();
}

void RobotSegmentationCameraStrategy::destroy()
{
  subscription_.reset();
  publisher_.reset();
}

// PORT: callback_depth / _mask_depth_image
void RobotSegmentationCameraStrategy::callback_depth(const sensor_msgs::msg::Image & msg)
{
  try {
    bool unsupported = false;
    const DepthFrame frame = decode_depth(msg, unsupported);
    if (unsupported) {
      RCLCPP_WARN_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 2000,
        "[%s] Unsupported depth encoding: %s", name_.c_str(), msg.encoding.c_str());
    }
    depth_frame_id_ = msg.header.frame_id;

    const auto intrinsics = intrinsics_copy();
    if (!intrinsics.has_value()) {
      return;
    }
    curobo_runtime::TensorF32 depth;
    depth.shape = {static_cast<std::int64_t>(frame.height), static_cast<std::int64_t>(frame.width)};
    depth.data = frame.data;

    // intrinsics_meta: [fx, fy, cx, cy] from the 3x3 K.
    curobo_runtime::TensorF32 k;
    k.shape = {4};
    k.data = {
      static_cast<float>((*intrinsics)[0]), static_cast<float>((*intrinsics)[4]),
      static_cast<float>((*intrinsics)[2]), static_cast<float>((*intrinsics)[5])};

    std::array<double, 7> cam_pose{0, 0, 0, 1, 0, 0, 0};
    if (static_pose_.has_value()) {
      cam_pose = *static_pose_;
    } else if (!frame_id_.empty()) {
      try {
        const auto transform = tf_buffer_->lookupTransform(
          node_->has_parameter("base_link") ? node_->get_parameter("base_link").as_string() :
          "base_0", frame_id_, tf2::TimePointZero);
        const auto & t = transform.transform.translation;
        const auto & r = transform.transform.rotation;
        const auto q = normalize_quat_wxyz({r.w, r.x, r.y, r.z});
        cam_pose = {t.x, t.y, t.z, q[0], q[1], q[2], q[3]};
      } catch (const tf2::TransformException &) {
        return;
      }
    }

    const std::vector<double> q = joint_pose_ ? joint_pose_() : std::vector<double>{};
    std::vector<float> qf(q.begin(), q.end());
    auto gpu_guard = curobo_runtime::Runtime::instance().barrier().shared();
    const auto masked = bridge_->robot_segment(
      handle_, depth, k, cam_pose, qf, masks_json_ ? masks_json_() : "[]");
    publish_masked(masked.data, frame.height, frame.width, msg.header.stamp);

    // PORT: the points masked OUT as robot, in the base frame, for debug.
    // Gated on an actual subscriber (the transform + serialization is not free).
    if (robot_pointcloud_pub_ != nullptr &&
      robot_pointcloud_pub_->get_subscription_count() > 0)
    {
      publish_robot_pointcloud(
        frame.data, masked.data, frame.height, frame.width, *intrinsics, cam_pose,
        msg.header.stamp);
    }
  } catch (const std::exception & e) {
    RCLCPP_DEBUG_THROTTLE(
      node_->get_logger(), *node_->get_clock(), 2000, "robot_segmentation frame dropped (%s)",
      e.what());
  }
}

// PORT: _create_pointcloud2_msg (the masked-out robot points, base frame).
void RobotSegmentationCameraStrategy::publish_robot_pointcloud(
  const std::vector<float> & depth, const std::vector<float> & masked,
  std::uint32_t h, std::uint32_t w, const std::array<double, 9> & k,
  const std::array<double, 7> & cam_pose, const builtin_interfaces::msg::Time & stamp)
{
  const double fx = k[0], fy = k[4], cx = k[2], cy = k[5];
  const Eigen::Matrix3d rot = quat_to_matrix({cam_pose[3], cam_pose[4], cam_pose[5], cam_pose[6]});
  const Eigen::Vector3d t(cam_pose[0], cam_pose[1], cam_pose[2]);
  std::vector<float> points;
  for (std::uint32_t v = 0; v < h; ++v) {
    for (std::uint32_t u = 0; u < w; ++u) {
      const std::size_t idx = static_cast<std::size_t>(v) * w + u;
      const float z = idx < depth.size() ? depth[idx] : 0.0f;
      const float m = idx < masked.size() ? masked[idx] : 0.0f;
      if (!(z > 0.0f) || m != 0.0f) {
        continue;   // kept or invalid -> not a robot point
      }
      const Eigen::Vector3d p_cam(
        (static_cast<double>(u) - cx) * z / fx, (static_cast<double>(v) - cy) * z / fy, z);
      const Eigen::Vector3d p_base = rot * p_cam + t;
      points.push_back(static_cast<float>(p_base.x()));
      points.push_back(static_cast<float>(p_base.y()));
      points.push_back(static_cast<float>(p_base.z()));
    }
  }
  sensor_msgs::msg::PointCloud2 msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = node_->has_parameter("base_link") ?
    node_->get_parameter("base_link").as_string() : "base_link";
  msg.height = 1;
  msg.width = static_cast<std::uint32_t>(points.size() / 3);
  msg.is_bigendian = false;
  msg.is_dense = true;
  msg.point_step = 12;
  msg.row_step = 12 * msg.width;
  sensor_msgs::msg::PointField field;
  field.datatype = sensor_msgs::msg::PointField::FLOAT32;
  field.count = 1;
  field.name = "x"; field.offset = 0; msg.fields.push_back(field);
  field.name = "y"; field.offset = 4; msg.fields.push_back(field);
  field.name = "z"; field.offset = 8; msg.fields.push_back(field);
  msg.data.resize(points.size() * sizeof(float));
  std::memcpy(msg.data.data(), points.data(), msg.data.size());
  robot_pointcloud_pub_->publish(msg);
}

void RobotSegmentationCameraStrategy::publish_masked(
  const std::vector<float> & masked, std::uint32_t h, std::uint32_t w,
  const builtin_interfaces::msg::Time & stamp)
{
  // PORT: depth_tensor_to_image_msg (32FC1).
  auto msg = encode_depth(masked, h, w, "32FC1", stamp);
  msg.header.frame_id = depth_frame_id_;
  publisher_->publish(msg);
}

}  // namespace curobo_core

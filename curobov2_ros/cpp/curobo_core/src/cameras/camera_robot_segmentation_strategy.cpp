// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/cameras/camera_robot_segmentation_strategy.py.
#include "curobo_runtime/runtime.hpp"
#include "curobo_core/cameras/camera_robot_segmentation_strategy.hpp"

#include <algorithm>
#include <cstring>
#include <map>

#include <Eigen/Geometry>
#include <sensor_msgs/msg/point_field.hpp>

#include "curobo_core/core/frame_utils.hpp"
#include "curobo_core/cameras/camera_cfg.hpp"

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
  std::function<std::vector<double>(std::int64_t)> joint_pose_at,
  std::function<std::string(const MaskFrameResolver &)> masks_json,
  rclcpp::CallbackGroup::SharedPtr callback_group, double max_update_rate_hz)
: CameraStrategy(node, camera_name, topic, camera_info_topic, frame_id, intrinsics, extrinsics,
    max_update_rate_hz),
  bridge_(bridge), handle_(handle), joint_pose_at_(std::move(joint_pose_at)),
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
  // MoveIt's octomap inputs are rate-limited by `max_update_rate`; the
  // segmentation pass (a TF lookup plus an exclusive-barrier GPU masking job per
  // frame) is limited by this camera's `camera_frame_rate_hz`. Gate before the
  // decode so a frame this node will not look at costs nothing at all.
  if (!allow_frame()) {
    return;
  }
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
    // The raw frame is only needed again for the (subscriber-gated) debug cloud;
    // otherwise hand the buffer over instead of copying ~3.7 MB per frame.
    const bool want_cloud = robot_pointcloud_pub_ != nullptr &&
      robot_pointcloud_pub_->get_subscription_count() > 0;
    if (want_cloud) {
      depth.data = frame.data;
    } else {
      depth.data = std::move(frame.data);
    }

    // intrinsics_meta: [fx, fy, cx, cy] from the 3x3 K.
    curobo_runtime::TensorF32 k;
    k.shape = {4};
    k.data = {
      static_cast<float>((*intrinsics)[0]), static_cast<float>((*intrinsics)[4]),
      static_cast<float>((*intrinsics)[2]), static_cast<float>((*intrinsics)[5])};

    const std::int64_t stamp_ns =
      static_cast<std::int64_t>(msg.header.stamp.sec) * 1000000000LL +
      static_cast<std::int64_t>(msg.header.stamp.nanosec);

    std::array<double, 7> cam_pose{0, 0, 0, 1, 0, 0, 0};
    bool have_pose = static_pose_.has_value();
    if (have_pose) {
      cam_pose = *static_pose_;
    } else if (!frame_id_.empty()) {
      // PORT: _transform_points_to_base - camera -> base at the CAPTURE stamp
      // (not the latest), so a moving wrist camera masks the right pixels.
      for (const std::string & base : integration_frame_candidates(node_)) {
        const auto transform =
          lookup_transform_at(*tf_buffer_, base, frame_id_, msg.header.stamp);
        if (!transform.has_value()) {
          continue;
        }
        const auto & t = transform->transform.translation;
        const auto & r = transform->transform.rotation;
        const auto q = normalize_quat_wxyz({r.w, r.x, r.y, r.z});
        cam_pose = {t.x, t.y, t.z, q[0], q[1], q[2], q[3]};
        have_pose = true;
        break;
      }
    }
    if (!have_pose) {
      // PORT: _transform_points_to_base None -> only a KNOWN pose can be masked.
      // Drop the frame: publishing an all-zero "masked" depth would hand the
      // mapper an empty frame to integrate (its TSDF/decay kernels run for no
      // data, at the camera rate, for as long as the transform stays
      // unavailable) and would silently replace depth the mapper may already
      // have counted. A missing frame costs nothing; an unmasked or empty one
      // costs either a permanent voxel error or wasted GPU time.
      RCLCPP_WARN_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 2000,
        "[%s] TF %s -> base unavailable; dropping this depth frame (no masked "
        "depth published, no mapper integration)",
        name_.c_str(), frame_id_.c_str());
      return;
    }

    // PORT: _joint_pose_at - joint pose at the frame's capture time, live pose
    // while the timestamped feedback buffer is still filling.
    const std::vector<double> q =
      joint_pose_at_ ? joint_pose_at_(stamp_ns) : std::vector<double>{};
    std::vector<float> qf(q.begin(), q.end());
    // PORT: camera_robot_segmentation_strategy.callback_depth - the Python takes
    // the EXCLUSIVE gpu_lock here (blocking), not a shared lane hold: the CPU->GPU
    // transfer + FK masking must not overlap any other GPU job, and a concurrent
    // CUDA-graph capture would invalidate it. shared() admitted exactly that.
    // Mask poses are given in their own frame_id (map, odom, a tool link...).
    // Express them in the base frame AT THE CAPTURE STAMP, like the camera pose;
    // the bridge tests points in the base frame, so an unconverted mask lands in
    // the wrong place and deletes whatever real depth sits there.
    // One TF lookup per distinct frame per image, and all of it BEFORE the
    // exclusive GPU barrier: a lookup can wait up to its timeout for TF to catch
    // up, and nothing else may touch the GPU while the barrier is held.
    std::map<std::string, std::optional<std::array<double, 7>>> mask_frame_cache;
    const MaskFrameResolver resolve_mask_frame =
      [this, &msg, &mask_frame_cache](const std::string & frame)
      -> std::optional<std::array<double, 7>> {
        const std::vector<std::string> bases = integration_frame_candidates(node_);
        if (frame.empty() || std::find(bases.begin(), bases.end(), frame) != bases.end()) {
          return std::array<double, 7>{0, 0, 0, 1, 0, 0, 0};
        }
        const auto cached = mask_frame_cache.find(frame);
        if (cached != mask_frame_cache.end()) {
          return cached->second;
        }
        std::optional<std::array<double, 7>> result;
        for (const std::string & base : bases) {
          const auto tf = lookup_transform_at(*tf_buffer_, base, frame, msg.header.stamp);
          if (!tf.has_value()) {
            continue;
          }
          const auto & t = tf->transform.translation;
          const auto & r = tf->transform.rotation;
          const auto qn = normalize_quat_wxyz({r.w, r.x, r.y, r.z});
          result = std::array<double, 7>{t.x, t.y, t.z, qn[0], qn[1], qn[2], qn[3]};
          break;
        }
        mask_frame_cache[frame] = result;
        return result;
      };
    const std::string masks_json = masks_json_ ? masks_json_(resolve_mask_frame) : "[]";
    auto gpu_guard = curobo_runtime::Runtime::instance().barrier().exclusive();
    const auto masked = bridge_->robot_segment(handle_, depth, k, cam_pose, qf, masks_json);
    publish_masked(masked.data, frame.height, frame.width, frame.encoding, msg.header.stamp);

    // PORT: the points masked OUT as robot, in the base frame, for debug.
    // Gated on an actual subscriber (the transform + serialization is not free).
    if (want_cloud) {
      publish_robot_pointcloud(
        frame.data, masked.data, frame.height, frame.width, *intrinsics, cam_pose,
        msg.header.stamp);
    }
  } catch (const std::exception & e) {
    // WARN, not DEBUG: a persistent failure here silently stops the masked
    // depth, so the Mapper freezes with only its first integrations and nothing
    // is logged. Throttled so a burst of bad frames is one line.
    RCLCPP_WARN_THROTTLE(
      node_->get_logger(), *node_->get_clock(), 5000, "[%s] robot_segmentation frame dropped (%s)",
      name_.c_str(), e.what());
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
  msg.header.frame_id = integration_frame(node_);
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
  const std::string & encoding, const builtin_interfaces::msg::Time & stamp)
{
  // PORT: depth_tensor_to_image_msg - the masked output keeps the SOURCE
  // encoding (16UC1 -> uint16 mm, 32FC1 -> float32 m), byte-compatible with
  // the input stream (the Python did the same).
  const std::string enc = (encoding == "16UC1") ? "16UC1" : "32FC1";
  auto msg = encode_depth(masked, h, w, enc, stamp);
  msg.header.frame_id = depth_frame_id_;
  publisher_->publish(msg);
}

}  // namespace curobo_core

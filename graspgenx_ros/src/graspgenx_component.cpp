// SPDX-License-Identifier: Apache-2.0
#include "python_backend.hpp"

#include <grasp_pose_detection_interfaces/msg/object_array.hpp>
#include <grasp_pose_detection_interfaces/srv/generate_grasps.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace graspgenx_ros
{

namespace
{

using Mat4 = std::array<double, 16>;

struct Point
{
  float x;
  float y;
  float z;
};

struct Candidate
{
  Mat4 pose;
  double score;
};

Mat4 identity()
{
  return {
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
  };
}

Mat4 multiply(const Mat4 & lhs, const Mat4 & rhs)
{
  Mat4 result{};
  for (std::size_t row = 0; row < 4; ++row) {
    for (std::size_t col = 0; col < 4; ++col) {
      double value = 0.0;
      for (std::size_t k = 0; k < 4; ++k) {
        value += lhs[row * 4 + k] * rhs[k * 4 + col];
      }
      result[row * 4 + col] = value;
    }
  }
  return result;
}

Mat4 translation(double x, double y, double z)
{
  Mat4 result = identity();
  result[3] = x;
  result[7] = y;
  result[11] = z;
  return result;
}

geometry_msgs::msg::Pose pose_from_matrix(const Mat4 & matrix)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = matrix[3];
  pose.position.y = matrix[7];
  pose.position.z = matrix[11];

  const double m00 = matrix[0];
  const double m01 = matrix[1];
  const double m02 = matrix[2];
  const double m10 = matrix[4];
  const double m11 = matrix[5];
  const double m12 = matrix[6];
  const double m20 = matrix[8];
  const double m21 = matrix[9];
  const double m22 = matrix[10];

  double qw = 1.0;
  double qx = 0.0;
  double qy = 0.0;
  double qz = 0.0;
  const double trace = m00 + m11 + m22;
  if (trace > 0.0) {
    const double scale = std::sqrt(trace + 1.0) * 2.0;
    qw = 0.25 * scale;
    qx = (m21 - m12) / scale;
    qy = (m02 - m20) / scale;
    qz = (m10 - m01) / scale;
  } else if (m00 > m11 && m00 > m22) {
    const double scale = std::sqrt(1.0 + m00 - m11 - m22) * 2.0;
    qw = (m21 - m12) / scale;
    qx = 0.25 * scale;
    qy = (m01 + m10) / scale;
    qz = (m02 + m20) / scale;
  } else if (m11 > m22) {
    const double scale = std::sqrt(1.0 + m11 - m00 - m22) * 2.0;
    qw = (m02 - m20) / scale;
    qx = (m01 + m10) / scale;
    qy = 0.25 * scale;
    qz = (m12 + m21) / scale;
  } else {
    const double scale = std::sqrt(1.0 + m22 - m00 - m11) * 2.0;
    qw = (m10 - m01) / scale;
    qx = (m02 + m20) / scale;
    qy = (m12 + m21) / scale;
    qz = 0.25 * scale;
  }

  const double norm = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
  if (norm > 0.0) {
    qw /= norm;
    qx /= norm;
    qy /= norm;
    qz /= norm;
  }
  pose.orientation.w = qw;
  pose.orientation.x = qx;
  pose.orientation.y = qy;
  pose.orientation.z = qz;
  return pose;
}

geometry_msgs::msg::PoseStamped stamped_pose(
  const Mat4 & matrix, const std::string & frame_id, const rclcpp::Time & stamp)
{
  geometry_msgs::msg::PoseStamped message;
  message.header.frame_id = frame_id;
  message.header.stamp = stamp;
  message.pose = pose_from_matrix(matrix);
  return message;
}

float read_component(
  const std::uint8_t * pointer, std::uint8_t datatype, bool big_endian)
{
  if (datatype == sensor_msgs::msg::PointField::FLOAT32) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, pointer, sizeof(bits));
    if (big_endian) {
      bits = __builtin_bswap32(bits);
    }
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }
  if (datatype == sensor_msgs::msg::PointField::FLOAT64) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, pointer, sizeof(bits));
    if (big_endian) {
      bits = __builtin_bswap64(bits);
    }
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return static_cast<float>(value);
  }
  throw std::runtime_error("PointCloud2 x/y/z fields must be float32 or float64");
}

const sensor_msgs::msg::PointField * find_field(
  const sensor_msgs::msg::PointCloud2 & cloud, const std::string & name)
{
  for (const auto & field : cloud.fields) {
    if (field.name == name) {
      return &field;
    }
  }
  return nullptr;
}

std::vector<Point> extract_points(const sensor_msgs::msg::PointCloud2 & cloud)
{
  const auto * x_field = find_field(cloud, "x");
  const auto * y_field = find_field(cloud, "y");
  const auto * z_field = find_field(cloud, "z");
  if (x_field == nullptr || y_field == nullptr || z_field == nullptr) {
    throw std::runtime_error("PointCloud2 is missing x, y or z fields");
  }
  if (cloud.point_step == 0) {
    throw std::runtime_error("PointCloud2 has point_step == 0");
  }

  const std::size_t point_count =
    static_cast<std::size_t>(cloud.width) * static_cast<std::size_t>(cloud.height);
  if (cloud.data.size() < point_count * cloud.point_step) {
    throw std::runtime_error("PointCloud2 data is shorter than width * height * point_step");
  }

  const bool big_endian = cloud.is_bigendian != 0;
  std::vector<Point> points;
  points.reserve(point_count);
  for (std::size_t i = 0; i < point_count; ++i) {
    const std::uint8_t * base = cloud.data.data() + i * cloud.point_step;
    const Point point{
      read_component(base + x_field->offset, x_field->datatype, big_endian),
      read_component(base + y_field->offset, y_field->datatype, big_endian),
      read_component(base + z_field->offset, z_field->datatype, big_endian),
    };
    if (
      std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z))
    {
      points.push_back(point);
    }
  }
  return points;
}

std::vector<std::array<float, 3>> to_float_arrays(const std::vector<Point> & points)
{
  std::vector<std::array<float, 3>> result;
  result.reserve(points.size());
  for (const auto & point : points) {
    result.push_back({point.x, point.y, point.z});
  }
  return result;
}

}  // namespace

class GraspGenXComponent : public rclcpp::Node
{
public:
  explicit GraspGenXComponent(const rclcpp::NodeOptions & options)
  : Node("graspgenx", options)
  {
    declare_parameters();

    service_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    service_ = create_service<grasp_pose_detection_interfaces::srv::GenerateGrasps>(
      "~/generate_grasps",
      std::bind(
        &GraspGenXComponent::handle_generate_grasps, this,
        std::placeholders::_1, std::placeholders::_2),
      rclcpp::ServicesQoS(), service_group_);

    rclcpp::QoS objects_qos(1);
    objects_qos.reliable().transient_local();
    std::string topic = objects_topic_;
    if (topic.empty()) {
      topic = "objects";
    }
    if (topic.front() != '/' && topic.rfind("~", 0) != 0) {
      topic = "~/" + topic;
    }
    objects_subscription_ =
      create_subscription<grasp_pose_detection_interfaces::msg::ObjectArray>(
      topic, objects_qos,
      [this](
        const grasp_pose_detection_interfaces::msg::ObjectArray::SharedPtr message)
      {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        objects_ = message;
      });

    std::string scene_topic = scene_cloud_topic_;
    if (scene_topic.empty()) {
      scene_topic = "filtered_cloud";
    }
    if (scene_topic.front() != '/' && scene_topic.rfind("~", 0) != 0) {
      scene_topic = "~/" + scene_topic;
    }
    rclcpp::QoS cloud_qos(1);
    cloud_qos.best_effort();
    scene_cloud_subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      scene_topic, cloud_qos,
      [this](const sensor_msgs::msg::PointCloud2::SharedPtr message)
      {
        std::lock_guard<std::mutex> lock(cloud_mutex_);
        scene_cloud_ = message;
      });

    RCLCPP_INFO(
      get_logger(), "GraspGenX service ready on %s, object cloud topic %s",
      service_->get_service_name(), objects_subscription_->get_topic_name());
    if (use_fast_inference_) {
      RCLCPP_WARN(
        get_logger(),
        "use_fast_inference uses fixed candidate counts (100 diffusion / 50 OBB); "
        "num_grasps is clamped to the native runtime count");
    }
    if (warmup_) {
      std::string error;
      if (!backend_.ensure_initialized(backend_settings(), error)) {
        RCLCPP_ERROR(get_logger(), "GraspGenX warmup failed: %s", error.c_str());
      } else {
        RCLCPP_INFO(
          get_logger(), "GPU memory after GraspGenX load: %s",
          backend_.memory_info().c_str());
        std::vector<std::array<float, 3>> points;
        points.reserve(216);
        for (int x = -2; x <= 3; ++x) {
          for (int y = -2; y <= 3; ++y) {
            for (int z = -2; z <= 3; ++z) {
              points.push_back(
                {0.01F * static_cast<float>(x), 0.01F * static_cast<float>(y),
                  0.01F * static_cast<float>(z)});
            }
          }
        }
        RawGrasps raw;
        std::string warning;
        if (!backend_.run(
            backend_settings(), points, {}, raw, warning, error))
        {
          RCLCPP_WARN(get_logger(), "GraspGenX warmup inference failed: %s", error.c_str());
        } else {
          RCLCPP_INFO(get_logger(), "GraspGenX warmup inference complete");
        }
      }
    }
  }

private:
  std::vector<Point> crop_object_cloud(
    const visualization_msgs::msg::Marker & obb)
  {
    sensor_msgs::msg::PointCloud2::SharedPtr scene;
    {
      std::lock_guard<std::mutex> lock(cloud_mutex_);
      scene = scene_cloud_;
    }
    if (!scene) {
      return {};
    }
    if (!obb.header.frame_id.empty() && obb.header.frame_id != frame_id_) {
      RCLCPP_WARN(
        get_logger(),
        "Object box frame_id '%s' does not match frame_id '%s'",
        obb.header.frame_id.c_str(), frame_id_.c_str());
      return {};
    }
    const auto points = extract_points(*scene);
    const double qx = obb.pose.orientation.x;
    const double qy = obb.pose.orientation.y;
    const double qz = obb.pose.orientation.z;
    const double qw = obb.pose.orientation.w;
    const double norm = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
    if (norm < 1e-9) {
      return {};
    }
    const double x = qx / norm, y = qy / norm, z = qz / norm, w = qw / norm;
    const std::array<std::array<double, 3>, 3> rotation{{
      {{1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)}},
      {{2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)}},
      {{2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)}},
    }};
    const std::array<double, 3> center{
      obb.pose.position.x, obb.pose.position.y, obb.pose.position.z};
    const double margin = 0.005;
    const std::array<double, 3> half{
      0.5 * obb.scale.x + margin,
      0.5 * obb.scale.y + margin,
      0.5 * obb.scale.z + margin};
    std::vector<Point> cropped;
    cropped.reserve(points.size() / 50);
    for (const auto & point : points) {
      const double dx = point.x - center[0];
      const double dy = point.y - center[1];
      const double dz = point.z - center[2];
      const double local[3] = {
        rotation[0][0] * dx + rotation[1][0] * dy + rotation[2][0] * dz,
        rotation[0][1] * dx + rotation[1][1] * dy + rotation[2][1] * dz,
        rotation[0][2] * dx + rotation[1][2] * dy + rotation[2][2] * dz,
      };
      if (
        std::abs(local[0]) <= half[0] && std::abs(local[1]) <= half[1] &&
        std::abs(local[2]) <= half[2])
      {
        cropped.push_back(point);
      }
    }
    RCLCPP_INFO(
      get_logger(), "Object cloud from scene: %zu of %zu point(s)",
      cropped.size(), points.size());
    return cropped;
  }

  void declare_parameters()
  {
    frame_id_ = declare_parameter<std::string>("frame_id", "base_footprint");
    objects_topic_ = declare_parameter<std::string>("objects_topic", "objects");
    scene_cloud_topic_ =
      declare_parameter<std::string>("cloud_topic", "filtered_cloud");
    checkpoint_dir_ = declare_parameter<std::string>("checkpoint_dir", "");
    gripper_name_ = declare_parameter<std::string>("gripper_name", "");
    gripper_assets_dir_ =
      declare_parameter<std::string>("gripper_assets_dir", "/opt/graspgenx/assets");

    declare_parameter<std::vector<double>>(
      "sweep_extents_open", std::vector<double>{0.08, 0.02, 0.06});
    declare_parameter<std::vector<double>>(
      "sweep_offset_open", std::vector<double>{0.0, 0.0, 0.05});
    declare_parameter<std::vector<double>>(
      "sweep_extents_mid", std::vector<double>{0.04, 0.02, 0.06});
    declare_parameter<std::vector<double>>(
      "sweep_offset_mid", std::vector<double>{0.0, 0.0, 0.05});
    sweep_extents_open_ = get_vector3("sweep_extents_open", {0.08, 0.02, 0.06});
    sweep_offset_open_ = get_vector3("sweep_offset_open", {0.0, 0.0, 0.05});
    sweep_extents_mid_ = get_vector3("sweep_extents_mid", {0.04, 0.02, 0.06});
    sweep_offset_mid_ = get_vector3("sweep_offset_mid", {0.0, 0.0, 0.05});

    gripper_type_ = declare_parameter<int>("gripper_type", 0);
    fingertip_depth_ = declare_parameter<double>("fingertip_depth", 0.0);

    declare_parameter<std::vector<double>>(
      "grasp_to_tool", std::vector<double>{
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0});
    declare_parameter<std::vector<double>>(
      "retreat_axis", std::vector<double>{-1.0, 0.0, 0.0});
    grasp_to_tool_ = get_vector16("grasp_to_tool", identity());
    retreat_axis_ = get_vector3("retreat_axis", {-1.0, 0.0, 0.0});

    planner_ = declare_parameter<std::string>("planner", "diffusion");
    score_threshold_ = declare_parameter<double>("score_threshold", -1.0);
    num_grasps_ = declare_parameter<int>("num_grasps", 100);
    max_candidates_ = declare_parameter<int>("max_candidates", 0);
    pre_grasp_dist_ = declare_parameter<double>("pre_grasp_dist", 0.1);
    retreat_dist_ = declare_parameter<double>("retreat_dist", 0.1);
    filter_collisions_ = declare_parameter<bool>("filter_collisions", false);
    collision_threshold_ = declare_parameter<double>("collision_threshold", 0.02);
    gripper_colmesh_ = declare_parameter<std::string>("gripper_colmesh", "");
    use_fast_inference_ = declare_parameter<bool>("use_fast_inference", false);
    warmup_ = declare_parameter<bool>("warmup", false);
  }

  std::array<double, 3> get_vector3(
    const std::string & name, const std::array<double, 3> & fallback)
  {
    std::vector<double> values;
    if (!get_parameter(name, values) || values.size() != 3) {
      RCLCPP_ERROR(
        get_logger(), "Parameter %s must have exactly 3 elements; using default",
        name.c_str());
      return fallback;
    }
    return {values[0], values[1], values[2]};
  }

  Mat4 get_vector16(const std::string & name, const Mat4 & fallback)
  {
    std::vector<double> values;
    if (!get_parameter(name, values) || values.size() != 16) {
      RCLCPP_ERROR(
        get_logger(), "Parameter %s must have exactly 16 elements; using identity",
        name.c_str());
      return fallback;
    }
    Mat4 result{};
    std::copy(values.begin(), values.end(), result.begin());
    return result;
  }

  BackendSettings backend_settings() const
  {
    BackendSettings settings;
    settings.checkpoint_dir = checkpoint_dir_;
    settings.gripper_name = gripper_name_;
    settings.gripper_assets_dir = gripper_assets_dir_;
    settings.sweep_extents_open = sweep_extents_open_;
    settings.sweep_offset_open = sweep_offset_open_;
    settings.sweep_extents_mid = sweep_extents_mid_;
    settings.sweep_offset_mid = sweep_offset_mid_;
    settings.gripper_type = gripper_type_;
    settings.fingertip_depth = fingertip_depth_;
    settings.use_fast_inference = use_fast_inference_;
    settings.diffusion_grasps = use_fast_inference_ ? 100 : std::max(1, num_grasps_);
    settings.obb_grasps = 50;
    settings.num_grasps = settings.diffusion_grasps;
    settings.planner = planner_;
    settings.score_threshold = score_threshold_;
    settings.filter_collisions = filter_collisions_;
    settings.collision_threshold = collision_threshold_;
    settings.gripper_colmesh = gripper_colmesh_;
    return settings;
  }

  void handle_generate_grasps(
    const std::shared_ptr<grasp_pose_detection_interfaces::srv::GenerateGrasps::Request>
    request,
    std::shared_ptr<grasp_pose_detection_interfaces::srv::GenerateGrasps::Response>
    response)
  {
    std::lock_guard<std::mutex> inference_lock(inference_mutex_);
    try {
      grasp_pose_detection_interfaces::msg::ObjectArray::SharedPtr objects;
      {
        std::lock_guard<std::mutex> cache_lock(cache_mutex_);
        objects = objects_;
      }

      if (
        !objects || request->object_index < 0 ||
        static_cast<std::size_t>(request->object_index) >= objects->objects.size())
      {
        response->success = false;
        response->message = "Invalid object index";
        return;
      }

      const auto & object =
        objects->objects[static_cast<std::size_t>(request->object_index)];
      const auto & cloud = object.cloud;
      std::vector<Point> object_points;
      if (!cloud.data.empty() && !cloud.fields.empty()) {
        if (cloud.header.frame_id != frame_id_) {
          RCLCPP_ERROR(
            get_logger(),
            "Object %d cloud frame_id '%s' does not match frame_id '%s'; "
            "refusing to treat it as %s without a TF transform",
            request->object_index, cloud.header.frame_id.c_str(),
            frame_id_.c_str(), frame_id_.c_str());
          response->success = false;
          response->message =
            "Object cloud frame_id '" + cloud.header.frame_id +
            "' does not match frame_id '" + frame_id_ + "'";
          return;
        }
        object_points = extract_points(cloud);
      }
      if (object_points.empty()) {
        object_points = crop_object_cloud(object.obb);
      }
      if (object_points.empty()) {
        response->success = false;
        response->message = "Object cloud is empty";
        return;
      }

      const BackendSettings settings = backend_settings();
      std::vector<std::array<float, 3>> scene_points;
      if (settings.filter_collisions && mesh_available()) {
        for (std::size_t i = 0; i < objects->objects.size(); ++i) {
          if (static_cast<int>(i) == request->object_index) {
            continue;
          }
          const auto & other = objects->objects[i];
          if (other.cloud.header.frame_id != frame_id_) {
            RCLCPP_WARN(
              get_logger(),
              "Skipping object %zu in collision scene: frame_id '%s' != '%s'",
              i, other.cloud.header.frame_id.c_str(), frame_id_.c_str());
            continue;
          }
          try {
            const auto points = to_float_arrays(extract_points(other.cloud));
            scene_points.insert(scene_points.end(), points.begin(), points.end());
          } catch (const std::exception & exc) {
            RCLCPP_WARN(
              get_logger(), "Skipping object %zu in collision scene: %s", i,
              exc.what());
          }
        }
      }

      std::string error;
      std::string warning;
      if (!backend_.ensure_initialized(settings, error)) {
        response->success = false;
        response->message = error;
        RCLCPP_ERROR(get_logger(), "GraspGenX initialization failed:\n%s", error.c_str());
        return;
      }
      if (!reported_memory_) {
        RCLCPP_INFO(
          get_logger(), "GPU memory after GraspGenX init: %s",
          backend_.memory_info().c_str());
        reported_memory_ = true;
      }

      RawGrasps raw;
      if (
        !backend_.run(
          settings, to_float_arrays(object_points), scene_points, raw, warning, error))
      {
        response->success = false;
        response->message = error;
        RCLCPP_ERROR(get_logger(), "GraspGenX inference failed:\n%s", error.c_str());
        return;
      }
      if (!warning.empty()) {
        RCLCPP_WARN(get_logger(), "%s", warning.c_str());
      }

      const std::size_t count = std::min(raw.poses.size(), raw.scores.size());
      std::vector<Candidate> candidates;
      candidates.reserve(count);
      for (std::size_t i = 0; i < count; ++i) {
        candidates.push_back(
          {raw.poses[i], std::isfinite(raw.scores[i]) ?
            raw.scores[i] : -std::numeric_limits<double>::infinity()});
      }
      std::sort(
        candidates.begin(), candidates.end(),
        [](const Candidate & lhs, const Candidate & rhs) {
          return lhs.score > rhs.score;
        });

      // Keep only candidates whose open finger volume actually contains
      // object points. The network predicts in free space as well; those
      // candidates close on air.
      const double half_width = std::max(0.07, 0.5 * sweep_extents_open_[0]);
      const double half_thick = std::max(0.02, 0.5 * sweep_extents_open_[1]);
      const double depth_min = 0.0;
      const double depth_max =
        std::max(0.09, sweep_offset_open_[2] + 0.5 * sweep_extents_open_[2]);
      std::vector<Candidate> inliers;
      inliers.reserve(candidates.size());
      for (const auto & candidate : candidates) {
        const Mat4 & g = candidate.pose;
        const double tx = g[3], ty = g[7], tz = g[11];
        std::size_t inside = 0;
        double min_x = 1e9, max_x = -1e9;
        for (const auto & point : object_points) {
          const double dx = point.x - tx;
          const double dy = point.y - ty;
          const double dz = point.z - tz;
          const double lx = g[0] * dx + g[4] * dy + g[8] * dz;
          const double ly = g[1] * dx + g[5] * dy + g[9] * dz;
          const double lz = g[2] * dx + g[6] * dy + g[10] * dz;
          if (
            std::abs(lx) <= half_width && std::abs(ly) <= half_thick &&
            lz >= depth_min && lz <= depth_max)
          {
            ++inside;
            min_x = std::min(min_x, lx);
            max_x = std::max(max_x, lx);
          }
        }
        const std::size_t needed = std::max<std::size_t>(
          15, object_points.size() / 50);
        const double centre = 0.5 * (min_x + max_x);
        if (
          inside >= needed && min_x <= -0.003 && max_x >= 0.003 &&
          std::abs(centre) <= 0.015)
        {
          inliers.push_back(candidate);
        }
      }
      if (!inliers.empty()) {
        candidates.swap(inliers);
      } else {
        RCLCPP_WARN(
          get_logger(),
          "No grasp has object points between the fingers; keeping %zu "
          "candidate(s)", candidates.size());
      }
      RCLCPP_INFO(
        get_logger(), "Grasp volume check kept %zu candidate(s)",
        candidates.size());

      const int requested_cap =
        request->num_grasps_selected > 0 ? request->num_grasps_selected :
        settings.num_grasps;
      const int candidate_cap =
        max_candidates_ > 0 ? std::min(requested_cap, max_candidates_) : requested_cap;
      if (candidate_cap > 0 &&
        candidates.size() > static_cast<std::size_t>(candidate_cap))
      {
        candidates.resize(static_cast<std::size_t>(candidate_cap));
      }

      double axis_x = retreat_axis_[0];
      double axis_y = retreat_axis_[1];
      double axis_z = retreat_axis_[2];
      const double axis_norm =
        std::sqrt(axis_x * axis_x + axis_y * axis_y + axis_z * axis_z);
      if (axis_norm < 1e-9) {
        axis_x = 0.0;
        axis_y = 0.0;
        axis_z = 1.0;
      } else {
        axis_x /= axis_norm;
        axis_y /= axis_norm;
        axis_z /= axis_norm;
      }

      const double pre_distance =
        request->pre_grasp_dist > 0.0F ? request->pre_grasp_dist : pre_grasp_dist_;
      const double retreat_distance =
        request->retreat_dist > 0.0F ? request->retreat_dist : retreat_dist_;
      const double grasp_width = gripper_name_.empty() ? sweep_extents_open_[0] : 0.0;
      const rclcpp::Time stamp = now();

      response->grasps.reserve(candidates.size());
      for (const auto & candidate : candidates) {
        const Mat4 grasp = multiply(candidate.pose, grasp_to_tool_);
        const Mat4 pre_grasp = multiply(
          grasp, translation(-pre_distance, 0.0, 0.0));
        const Mat4 retreat = multiply(
          grasp,
          translation(
            axis_x * retreat_distance, axis_y * retreat_distance,
            axis_z * retreat_distance));

        grasp_pose_detection_interfaces::msg::Grasp message;
        message.pre_grasp = stamped_pose(pre_grasp, frame_id_, stamp);
        message.grasp = stamped_pose(grasp, frame_id_, stamp);
        message.retreat = stamped_pose(retreat, frame_id_, stamp);
        message.score = candidate.score;
        message.is_full_antipodal = false;
        message.is_half_antipodal = false;
        message.grasp_width = grasp_width;
        response->grasps.push_back(std::move(message));
      }

      response->success = true;
      response->message = response->grasps.empty() ?
        "No grasps found" : "Generated " + std::to_string(response->grasps.size()) +
        " grasps";
    } catch (const py::error_already_set & exc) {
      py::gil_scoped_acquire gil;
      const std::string message = exc.what();
      response->success = false;
      response->message = message;
      RCLCPP_ERROR(get_logger(), "GraspGenX Python error:\n%s", message.c_str());
    } catch (const std::exception & exc) {
      response->success = false;
      response->message = exc.what();
      RCLCPP_ERROR(get_logger(), "GenerateGrasps failed: %s", exc.what());
    } catch (...) {
      response->success = false;
      response->message = "Unknown error in GenerateGrasps";
      RCLCPP_ERROR(get_logger(), "GenerateGrasps failed with an unknown error");
    }
  }

  bool mesh_available()
  {
    if (gripper_colmesh_.empty()) {
      if (!warned_missing_mesh_) {
        RCLCPP_WARN(
          get_logger(),
          "filter_collisions is true but gripper_colmesh is empty; "
          "passing an empty scene so nothing is filtered");
        warned_missing_mesh_ = true;
      }
      return false;
    }
    if (!std::filesystem::is_regular_file(gripper_colmesh_)) {
      if (!warned_missing_mesh_) {
        RCLCPP_WARN(
          get_logger(),
          "filter_collisions is true but gripper_colmesh '%s' does not exist; "
          "passing an empty scene so nothing is filtered",
          gripper_colmesh_.c_str());
        warned_missing_mesh_ = true;
      }
      return false;
    }
    return true;
  }

  std::string frame_id_;
  std::string objects_topic_;
  std::string scene_cloud_topic_;
  std::string checkpoint_dir_;
  std::string gripper_name_;
  std::string gripper_assets_dir_;
  std::array<double, 3> sweep_extents_open_{};
  std::array<double, 3> sweep_offset_open_{};
  std::array<double, 3> sweep_extents_mid_{};
  std::array<double, 3> sweep_offset_mid_{};
  int gripper_type_ = 0;
  double fingertip_depth_ = 0.0;
  Mat4 grasp_to_tool_ = identity();
  std::array<double, 3> retreat_axis_{-1.0, 0.0, 0.0};
  std::string planner_ = "diffusion";
  double score_threshold_ = -1.0;
  int num_grasps_ = 100;
  int max_candidates_ = 0;
  double pre_grasp_dist_ = 0.1;
  double retreat_dist_ = 0.1;
  bool filter_collisions_ = false;
  double collision_threshold_ = 0.02;
  std::string gripper_colmesh_;
  bool use_fast_inference_ = false;
  bool warmup_ = false;

  GraspGenXBackend backend_;
  std::mutex inference_mutex_;
  std::mutex cache_mutex_;
  std::mutex cloud_mutex_;
  grasp_pose_detection_interfaces::msg::ObjectArray::SharedPtr objects_;
  sensor_msgs::msg::PointCloud2::SharedPtr scene_cloud_;
  rclcpp::CallbackGroup::SharedPtr service_group_;
  rclcpp::Service<grasp_pose_detection_interfaces::srv::GenerateGrasps>::SharedPtr
    service_;
  rclcpp::Subscription<
    grasp_pose_detection_interfaces::msg::ObjectArray>::SharedPtr
    objects_subscription_;
  rclcpp::Subscription<
    sensor_msgs::msg::PointCloud2>::SharedPtr scene_cloud_subscription_;
  bool warned_missing_mesh_ = false;
  bool reported_memory_ = false;
};

}  // namespace graspgenx_ros

RCLCPP_COMPONENTS_REGISTER_NODE(graspgenx_ros::GraspGenXComponent)

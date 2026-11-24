// SPDX-License-Identifier: Apache-2.0
#include "python_backend.hpp"

#include <cutamp_ros_interfaces/srv/plan.hpp>
#include <curobov2_ros_interfaces/msg/goalset.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace cutamp_ros
{

namespace
{

using Mat4 = std::array<double, 16>;

Mat4 identity()
{
  return {
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
  };
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

}  // namespace

class CuTampComponent : public rclcpp::Node
{
public:
  explicit CuTampComponent(const rclcpp::NodeOptions & options)
  : Node("cutamp", options)
  {
    declare_parameters();

    service_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    service_ = create_service<cutamp_ros_interfaces::srv::Plan>(
      "~/plan",
      std::bind(
        &CuTampComponent::handle_plan, this,
        std::placeholders::_1, std::placeholders::_2),
      rclcpp::ServicesQoS(), service_group_);

    RCLCPP_INFO(
      get_logger(), "cuTAMP service ready on %s (robot config: %s)",
      service_->get_service_name(), robot_config_path().c_str());

    if (Py_IsInitialized() == 0) {
      RCLCPP_ERROR(
        get_logger(),
        "embedded Python is not initialized; the cuRobo component must be "
        "loaded before cutamp_ros::CuTampComponent. Planning requests will "
        "return an error until then.");
    } else if (warmup_) {
      std::string error;
      if (!backend_.ensure_initialized(backend_settings(), error)) {
        RCLCPP_ERROR(get_logger(), "cuTAMP warmup failed:\n%s", error.c_str());
      } else {
        RCLCPP_INFO(
          get_logger(), "cuTAMP backend loaded; GPU memory: %s",
          backend_.memory_info().c_str());
      }
    }
  }

private:
  void declare_parameters()
  {
    robot_config_path_ = declare_parameter<std::string>("robot_config_path", "");
    robot_config_file_ = declare_parameter<std::string>("robot_config_file", "");
    urdf_path_ = declare_parameter<std::string>("urdf_path", "");
    tool_frame_ = declare_parameter<std::string>("tool_frame", "");
    frame_id_ = declare_parameter<std::string>("frame_id", "");
    gripper_spheres_ = declare_parameter<std::string>("gripper_spheres", "");

    declare_parameter<std::vector<double>>(
      "tool_from_ee", std::vector<double>{
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0});
    tool_from_ee_ = get_vector16("tool_from_ee", identity());

    ik_batch_size_ = declare_parameter<int>("ik_batch_size", 1024);
    ik_num_seeds_ = declare_parameter<int>("ik_num_seeds", 12);
    self_collision_check_ = declare_parameter<bool>("self_collision_check", true);
    use_cuda_graph_ = declare_parameter<bool>("use_cuda_graph", true);
    coll_n_spheres_ = declare_parameter<int>("coll_n_spheres", 50);
    coll_sphere_radius_ = declare_parameter<double>("coll_sphere_radius", 0.005);
    device_ = declare_parameter<int>("device", 0);
    warmup_ = declare_parameter<bool>("warmup", false);
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

  std::string robot_config_path() const
  {
    return robot_config_path_.empty() ? robot_config_file_ : robot_config_path_;
  }

  RobotSettings backend_settings() const
  {
    RobotSettings settings;
    settings.robot_config_path = robot_config_path();
    settings.urdf_path = urdf_path_;
    settings.tool_frame = tool_frame_;
    settings.tool_from_ee.assign(tool_from_ee_.begin(), tool_from_ee_.end());
    settings.gripper_spheres = gripper_spheres_;
    settings.ik_batch_size = ik_batch_size_;
    settings.ik_num_seeds = ik_num_seeds_;
    settings.self_collision_check = self_collision_check_;
    settings.use_cuda_graph = use_cuda_graph_;
    settings.coll_n_spheres = coll_n_spheres_;
    settings.coll_sphere_radius = coll_sphere_radius_;
    settings.device = device_;
    return settings;
  }

  PlanRequest to_plan_request(
    const cutamp_ros_interfaces::srv::Plan::Request & request) const
  {
    PlanRequest out;
    out.objects.reserve(request.objects.size());
    for (const auto & object : request.objects) {
      SceneObjectSpec spec;
      spec.name = object.name;
      spec.type = static_cast<int>(object.type);
      spec.dims = {
        object.dimensions.x, object.dimensions.y, object.dimensions.z};
      spec.pose = {
        object.pose.position.x, object.pose.position.y, object.pose.position.z,
        object.pose.orientation.x, object.pose.orientation.y,
        object.pose.orientation.z, object.pose.orientation.w};
      out.objects.push_back(spec);
    }
    out.movable_names = request.movable_names;
    out.goal_atoms = request.goal_atoms;
    out.q_init = request.q_init;
    out.num_particles = request.num_particles;
    out.num_opt_steps = request.num_opt_steps;
    out.num_initial_plans = request.num_initial_plans;
    out.max_loop_dur = request.max_loop_dur;

    out.grasp_object_names = request.grasp_object_names;
    out.grasp_tool_poses.reserve(request.grasp_tool_poses.size());
    for (const auto & pose : request.grasp_tool_poses) {
      out.grasp_tool_poses.push_back(
        {pose.position.x, pose.position.y, pose.position.z,
          pose.orientation.x, pose.orientation.y, pose.orientation.z,
          pose.orientation.w});
    }
    out.grasp_scores = request.grasp_scores;
    return out;
  }

  void handle_plan(
    const std::shared_ptr<cutamp_ros_interfaces::srv::Plan::Request> request,
    std::shared_ptr<cutamp_ros_interfaces::srv::Plan::Response> response)
  {
    std::lock_guard<std::mutex> lock(planning_mutex_);
    response->success = false;

    if (Py_IsInitialized() == 0) {
      response->message =
        "embedded Python is not initialized; load the cuRobo component before "
        "cutamp_ros::CuTampComponent";
      RCLCPP_ERROR(get_logger(), "%s", response->message.c_str());
      return;
    }

    try {
      std::string error;
      if (!backend_.ensure_initialized(backend_settings(), error)) {
        response->message = error;
        RCLCPP_ERROR(get_logger(), "cuTAMP initialization failed:\n%s", error.c_str());
        return;
      }
      if (!reported_memory_) {
        RCLCPP_INFO(
          get_logger(), "cuTAMP backend loaded; GPU memory: %s",
          backend_.memory_info().c_str());
        reported_memory_ = true;
      }

      PlanResult result;
      if (!backend_.plan(to_plan_request(*request), result, error)) {
        response->success = false;
        response->message = result.message.empty() ? error : result.message;
        RCLCPP_ERROR(get_logger(), "cuTAMP planning failed:\n%s", response->message.c_str());
        return;
      }

      response->success = true;
      response->message = result.message;
      response->num_satisfying = result.num_satisfying;
      response->goalsets.reserve(result.poses.size());
      for (std::size_t i = 0; i < result.poses.size(); ++i) {
        curobov2_ros_interfaces::msg::Goalset goalset;
        goalset.frame_id = frame_id_;
        goalset.poses.push_back(pose_from_matrix(result.poses[i]));
        response->goalsets.push_back(std::move(goalset));
      }
      response->gripper_close.reserve(result.gripper_close.size());
      for (const int value : result.gripper_close) {
        response->gripper_close.push_back(static_cast<std::int32_t>(value));
      }
      response->action_index.reserve(result.action_index.size());
      for (const int value : result.action_index) {
        response->action_index.push_back(static_cast<std::int32_t>(value));
      }

      RCLCPP_INFO(
        get_logger(), "cuTAMP produced %zu goalsets (%d satisfying particles)",
        response->goalsets.size(), response->num_satisfying);
    } catch (const py::error_already_set & exc) {
      py::gil_scoped_acquire gil;
      response->success = false;
      response->message = exc.what();
      RCLCPP_ERROR(get_logger(), "cuTAMP Python error:\n%s", response->message.c_str());
    } catch (const std::exception & exc) {
      response->success = false;
      response->message = exc.what();
      RCLCPP_ERROR(get_logger(), "cuTAMP planning failed: %s", exc.what());
    } catch (...) {
      response->success = false;
      response->message = "unknown error in CuTampComponent::handle_plan";
      RCLCPP_ERROR(get_logger(), "%s", response->message.c_str());
    }
  }

  std::string robot_config_path_;
  std::string robot_config_file_;
  std::string urdf_path_;
  std::string tool_frame_;
  std::string frame_id_;
  std::string gripper_spheres_;
  Mat4 tool_from_ee_ = identity();
  int ik_batch_size_ = 1024;
  int ik_num_seeds_ = 12;
  bool self_collision_check_ = true;
  bool use_cuda_graph_ = true;
  int coll_n_spheres_ = 50;
  double coll_sphere_radius_ = 0.005;
  int device_ = 0;
  bool warmup_ = false;

  CuTampBackend backend_;
  std::mutex planning_mutex_;
  rclcpp::CallbackGroup::SharedPtr service_group_;
  rclcpp::Service<cutamp_ros_interfaces::srv::Plan>::SharedPtr service_;
  bool reported_memory_ = false;
};

}  // namespace cutamp_ros

RCLCPP_COMPONENTS_REGISTER_NODE(cutamp_ros::CuTampComponent)

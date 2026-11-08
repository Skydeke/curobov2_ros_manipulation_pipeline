// SPDX-License-Identifier: Apache-2.0
#include "curobo_cpp_runtime/cpp_bridge.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "curobo_robot_runtime/inverse_kinematics.h"
#include "curobo_robot_runtime/kinematics.h"
#include "curobo_robot_runtime/motion_planner.h"
#include "curobo_robot_runtime/mpc.h"
#include "curobo_robot_runtime/scene.h"
#include "curobo_robot_runtime/trajectory_optimization.h"

#include "curobo_mapper_runtime/api.h"
#include "curobo_runtime/py_bridge.hpp"

#include "curobo_cpp_runtime/artifact_bundle.hpp"
#include "curobo_cpp_runtime/detail/mini_json.hpp"
#include "curobo_cpp_runtime/native_scene.hpp"

namespace curobo_cpp_runtime
{
namespace
{

using curobo_runtime::IkOutcome;
using curobo_runtime::LinkPosesOutcome;
using curobo_runtime::PlanOutcome;
using curobo_runtime::PlanRequest;
using curobo_runtime::TensorF32;

[[noreturn]] void unsupported(const char * method)
{
  throw std::runtime_error(
          std::string(method) +
          " is not implemented by curobo_cpp_runtime (cpp_runtime API gap). "
          "Run this robot/service with curobo_runtime:=python.");
}

[[noreturn]] void fail(const char * method, const std::string & message)
{
  throw std::runtime_error(std::string(method) + ": " + message);
}

void cuda_check(const char * method, cudaError_t status)
{
  if (status != cudaSuccess) {
    fail(method, std::string("CUDA error: ") + cudaGetErrorString(status));
  }
}

class DeviceBuffer
{
public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer & operator=(const DeviceBuffer &) = delete;
  ~DeviceBuffer()
  {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  void allocate(std::size_t bytes)
  {
    if (bytes == 0) {
      data_ = nullptr;
      bytes_ = 0;
      return;
    }
    if (data_ != nullptr) {
      cudaFree(data_);
    }
    cuda_check("cudaMalloc", cudaMalloc(&data_, bytes));
    bytes_ = bytes;
  }

  void upload(const void * source, std::size_t bytes, cudaMemcpyKind kind = cudaMemcpyHostToDevice)
  {
    if (bytes == 0) {
      return;
    }
    if (data_ == nullptr || bytes > bytes_) {
      allocate(bytes);
    }
    cuda_check("cudaMemcpy", cudaMemcpy(data_, source, bytes, kind));
  }

  template<typename T>
  void upload(const std::vector<T> & values)
  {
    upload(values.data(), values.size() * sizeof(T));
  }

  void download(void * destination, std::size_t bytes) const
  {
    if (bytes == 0) {
      return;
    }
    if (data_ == nullptr || bytes > bytes_) {
      throw std::runtime_error("DeviceBuffer::download: buffer is smaller than request");
    }
    cuda_check("cudaMemcpy", cudaMemcpy(destination, data_, bytes, cudaMemcpyDeviceToHost));
  }

  template<typename T>
  void download(std::vector<T> & values) const
  {
    values.resize(bytes_ / sizeof(T));
    download(values.data(), bytes_);
  }

  void * get() {return data_;}
  const void * get() const {return data_;}
  std::size_t bytes() const {return bytes_;}

private:
  void * data_{nullptr};
  std::size_t bytes_{0};
};

std::string json_string(const json::Value & document, const std::string & key, const std::string & fallback = "")
{
  return document.string_at(key, fallback);
}

int json_int(const json::Value & document, const std::string & key, int fallback)
{
  return static_cast<int>(document.number_at(key, static_cast<double>(fallback)));
}

double json_double(const json::Value & document, const std::string & key, double fallback)
{
  return document.number_at(key, fallback);
}

std::array<float, 7> pose_from_values(const float * values)
{
  return {values[0], values[1], values[2], values[3], values[4], values[5], values[6]};
}

std::array<float, 7> pose_multiply(
  const std::array<float, 7> & a, const std::array<float, 7> & b)
{
  // a and b are [x,y,z,qw,qx,qy,qz]; returns a * b.
  const float aw = a[3], ax = a[4], ay = a[5], az = a[6];
  const float bw = b[3], bx = b[4], by = b[5], bz = b[6];
  const float w = aw * bw - ax * bx - ay * by - az * bz;
  const float x = aw * bx + ax * bw + ay * bz - az * by;
  const float y = aw * by - ax * bz + ay * bw + az * bx;
  const float z = aw * bz + ax * by - ay * bx + az * bw;
  const float tx = 2.0F * (ay * b[2] - az * b[1]);
  const float ty = 2.0F * (az * b[0] - ax * b[2]);
  const float tz = 2.0F * (ax * b[1] - ay * b[0]);
  const float rx = b[0] + aw * tx + (ay * tz - az * ty);
  const float ry = b[1] + aw * ty + (az * tx - ax * tz);
  const float rz = b[2] + aw * tz + (ax * ty - ay * tx);
  return {a[0] + rx, a[1] + ry, a[2] + rz, w, x, y, z};
}

std::array<float, 7> pose_inverse(const std::array<float, 7> & p)
{
  const float w = p[3], x = -p[4], y = -p[5], z = -p[6];
  // inverse translation = rotate(-vector) by conjugate quaternion
  const float tx = 2.0F * (y * p[2] - z * p[1]);
  const float ty = 2.0F * (z * p[0] - x * p[2]);
  const float tz = 2.0F * (x * p[1] - y * p[0]);
  const float rx = -(p[0] + w * tx + (y * tz - z * ty));
  const float ry = -(p[1] + w * ty + (z * tx - x * tz));
  const float rz = -(p[2] + w * tz + (x * ty - y * tx));
  return {rx, ry, rz, w, x, y, z};
}

void fill_pose_arrays(
  const std::vector<std::array<float, 7>> & poses,
  std::vector<float> & positions, std::vector<float> & quaternions)
{
  positions.clear();
  quaternions.clear();
  positions.reserve(poses.size() * 3);
  quaternions.reserve(poses.size() * 4);
  for (const auto & pose : poses) {
    positions.push_back(pose[0]);
    positions.push_back(pose[1]);
    positions.push_back(pose[2]);
    quaternions.push_back(pose[3]);
    quaternions.push_back(pose[4]);
    quaternions.push_back(pose[5]);
    quaternions.push_back(pose[6]);
  }
}

std::array<float, 7> pose_from_list(const std::array<double, 7> & raw)
{
  std::array<float, 7> pose{};
  for (std::size_t i = 0; i < pose.size(); ++i) {
    pose[i] = static_cast<float>(raw[i]);
  }
  return pose;
}

// -- PlanRequest waypoint-field access -------------------------------------
//
// The waypoint fields are added to curobo_runtime::PlanRequest by the Python
// bridge work (Plan.md 4.2). The detection idiom lets this package compile
// before and after that change, and accepts the two natural names the field
// may land under.
template<typename T, typename = void>
struct has_waypoints_field : std::false_type {};
template<typename T>
struct has_waypoints_field<T, std::void_t<decltype(std::declval<T>().waypoints)>> : std::true_type {};

template<typename T, typename = void>
struct has_waypoint_poses_field : std::false_type {};
template<typename T>
struct has_waypoint_poses_field<T, std::void_t<decltype(std::declval<T>().waypoint_poses)>> : std::true_type {};

template<typename T, typename = void>
struct has_tool_waypoints_field : std::false_type {};
template<typename T>
struct has_tool_waypoints_field<T, std::void_t<decltype(std::declval<T>().tool_waypoints)>> : std::true_type {};

template<typename T, typename = void>
struct has_waypoint_count_field : std::false_type {};
template<typename T>
struct has_waypoint_count_field<T, std::void_t<decltype(std::declval<T>().waypoint_count)>> : std::true_type {};

template<typename T, typename = void>
struct has_path_position_tolerance_field : std::false_type {};
template<typename T>
struct has_path_position_tolerance_field<T,
  std::void_t<decltype(std::declval<T>().path_position_tolerance)>> : std::true_type {};

template<typename T, typename = void>
struct has_path_orientation_tolerance_field : std::false_type {};
template<typename T>
struct has_path_orientation_tolerance_field<T,
  std::void_t<decltype(std::declval<T>().path_orientation_tolerance)>> : std::true_type {};

template<typename T, typename = void>
struct has_final_position_tolerance_field : std::false_type {};
template<typename T>
struct has_final_position_tolerance_field<T,
  std::void_t<decltype(std::declval<T>().final_position_tolerance)>> : std::true_type {};

template<typename T, typename = void>
struct has_final_orientation_tolerance_field : std::false_type {};
template<typename T>
struct has_final_orientation_tolerance_field<T,
  std::void_t<decltype(std::declval<T>().final_orientation_tolerance)>> : std::true_type {};

template<typename T, typename = void>
struct has_track_orientation_field : std::false_type {};
template<typename T>
struct has_track_orientation_field<T,
  std::void_t<decltype(std::declval<T>().track_orientation)>> : std::true_type {};

template<typename T, typename = void>
struct has_outer_iterations_field : std::false_type {};
template<typename T>
struct has_outer_iterations_field<T,
  std::void_t<decltype(std::declval<T>().outer_iterations)>> : std::true_type {};

template<typename T, typename = void>
struct has_waypoint_tool_frames_field : std::false_type {};
template<typename T>
struct has_waypoint_tool_frames_field<T,
  std::void_t<decltype(std::declval<T>().waypoint_tool_frames)>> : std::true_type {};

struct OrderedWaypoints
{
  bool present{false};
  bool multi_tool{false};
  int count{0};
  TensorF32 poses;                       // [K,7] for single tool, [K*T,7] for multi
  std::vector<std::string> tool_frames;
  double path_position_tolerance{0.05};
  double path_orientation_tolerance{0.261799388};
  double final_position_tolerance{0.005};
  double final_orientation_tolerance{0.05};
  bool track_orientation{true};
  int outer_iterations{4};
};

template<typename Request>
OrderedWaypoints extract_ordered_waypoints(const Request & request)
{
  OrderedWaypoints waypoints;
  if constexpr (has_tool_waypoints_field<Request>::value) {
      if constexpr (std::is_same_v<decltype(request.tool_waypoints), TensorF32>) {
          if (!request.tool_waypoints.data.empty()) {
            waypoints.present = true;
            waypoints.multi_tool = true;
            waypoints.poses = request.tool_waypoints;
          }
        }
    }
  if constexpr (has_waypoints_field<Request>::value) {
      if constexpr (std::is_same_v<decltype(request.waypoints), TensorF32>) {
          if (!waypoints.present && !request.waypoints.data.empty()) {
            waypoints.present = true;
            waypoints.poses = request.waypoints;
          }
        }
    }
  if constexpr (has_waypoint_poses_field<Request>::value) {
      if constexpr (std::is_same_v<decltype(request.waypoint_poses), TensorF32>) {
          if (!waypoints.present && !request.waypoint_poses.data.empty()) {
            waypoints.present = true;
            waypoints.poses = request.waypoint_poses;
          }
        }
    }
  if (!waypoints.present) {
    return waypoints;
  }

  if constexpr (has_waypoint_tool_frames_field<Request>::value) {
      if constexpr (std::is_same_v<decltype(request.waypoint_tool_frames),
          std::vector<std::string>>)
      {
        waypoints.tool_frames = request.waypoint_tool_frames;
      }
    }
  if (waypoints.tool_frames.empty()) {
    waypoints.tool_frames = request.tool_frames;
  }
  if constexpr (has_waypoint_count_field<Request>::value) {
      waypoints.count = static_cast<int>(request.waypoint_count);
    }
  if (waypoints.count <= 0 && waypoints.poses.shape.size() >= 2) {
    if (waypoints.multi_tool && waypoints.poses.shape.size() >= 3) {
      waypoints.count = static_cast<int>(waypoints.poses.shape[0]);
    } else if (waypoints.multi_tool) {
      const std::size_t tools = waypoints.tool_frames.empty() ? 0 :
        waypoints.tool_frames.size();
      const std::size_t rows = static_cast<std::size_t>(waypoints.poses.shape[0]);
      waypoints.count = tools == 0 ? static_cast<int>(rows) :
        static_cast<int>(rows / tools);
    } else {
      waypoints.count = static_cast<int>(waypoints.poses.shape[0]);
    }
  }
  if constexpr (has_path_position_tolerance_field<Request>::value) {
      waypoints.path_position_tolerance = static_cast<double>(request.path_position_tolerance);
    }
  if constexpr (has_path_orientation_tolerance_field<Request>::value) {
      waypoints.path_orientation_tolerance =
        static_cast<double>(request.path_orientation_tolerance);
    }
  if constexpr (has_final_position_tolerance_field<Request>::value) {
      waypoints.final_position_tolerance =
        static_cast<double>(request.final_position_tolerance);
    }
  if constexpr (has_final_orientation_tolerance_field<Request>::value) {
      waypoints.final_orientation_tolerance =
        static_cast<double>(request.final_orientation_tolerance);
    }
  if constexpr (has_track_orientation_field<Request>::value) {
      waypoints.track_orientation = request.track_orientation;
    }
  if constexpr (has_outer_iterations_field<Request>::value) {
      waypoints.outer_iterations = static_cast<int>(request.outer_iterations);
    }
  return waypoints;
}

}  // namespace

struct CppBridge::Impl
{
  std::string artifact_dir;
  std::string robot;
  std::string config_file;
  // Bundle the bridge was initialized with ("restore" lock_joints target).
  std::string home_artifact_dir;
  std::string home_config_file;
  int device_ordinal{0};
  bool python_fallback{false};

  mutable std::mutex mutex;
  bool initialized{false};
  std::string init_error;

  // Optional Python backend for methods the native API does not expose yet.
  // It is also the consistency guard: once a Python-only scene mutation runs,
  // native planning is disabled until a full native mirror exists.
  std::unique_ptr<curobo_runtime::PyBridge> python;
  curobo_runtime::RobotHandle python_handle{-1};
  bool native_scene_valid{true};
  bool python_planner_built{false};
  std::string motion_cfg_json;

  struct MapperState
  {
    CuroboMapperRuntime * runtime{nullptr};
    CuroboMapperDeviceView view{};
    int image_height{0};
    int image_width{0};
    int esdf_count{0};
    curobo::robot::ObstacleHandle voxel_handle{};
    bool voxel_added{false};
    DeviceBuffer depth;
    DeviceBuffer rgb;
    DeviceBuffer intrinsics;
    DeviceBuffer camera_positions;
    DeviceBuffer camera_quaternions;
    ~MapperState()
    {
      if (runtime != nullptr) {
        curobo_mapper_destroy(runtime);
      }
    }
  };
  std::unique_ptr<MapperState> mapper;
  std::string mapper_cfg_json;
  bool python_mapper_configured{false};

  ArtifactBundle bundle;
  ArtifactManifest manifest;

  std::vector<std::string> joint_names;
  std::vector<std::string> tool_frames;
  std::vector<std::string> link_names;
  std::vector<double> joint_lower;
  std::vector<double> joint_upper;
  std::vector<std::string> locked_joint_names;
  std::vector<double> locked_positions;
  std::vector<std::string> unpinned_lock_joint_names;

  std::unique_ptr<NativeScene> scene;
  std::unique_ptr<curobo::robot::Kinematics> kinematics;
  std::unique_ptr<curobo::robot::MotionPlanner> planner;
  std::unique_ptr<curobo::robot::InverseKinematics> ik;
  std::unique_ptr<curobo::robot::MpcController> mpc;

  std::vector<std::string> lanes;
  bool cuda_graph_reset{true};
  bool debug_cuda_graphs{false};
  std::string solve_mode;
  std::map<std::string, bool> link_collision;
  std::set<std::string> attached_objects;
  std::map<std::string, std::string> attach_links;
  struct VisualAttachment
  {
    int tool_index{0};
    std::array<float, 7> object_in_link{};
    float radius{0.0F};
  };
  std::map<std::string, VisualAttachment> visual_attachments;

  struct PlannerSettings
  {
    int num_ik_seeds{32};
    int num_trajopt_seeds{12};
    int max_attempts{2};
    int enable_graph_attempt{1};
    int max_goalset{1};
    int num_knots{16};
    int rrt_max_iterations{32};
    bool use_cuda_graph{true};
    bool check_effort_limits{false};
    double collision_activation_distance{0.025};
    double interpolation_dt{0.025};
  } planner_settings;

  struct IkSettings
  {
    int batch_size{1};
    int num_seeds{32};
    int max_goalset{1};
    bool self_collision_check{true};
    bool use_lm_seed{true};
    bool use_cuda_graph{false};
  } ik_settings;

  std::vector<float> mpc_start;
  std::vector<float> mpc_goal_state;
  std::array<float, 7> mpc_goal_pose{{0, 0, 0, 1, 0, 0, 0}};
  bool mpc_has_goal_state{false};
  bool mpc_has_goal_pose{false};
  curobo::robot::MpcResult last_mpc_result{};

  void require_initialized(const char * method) const
  {
    if (!initialized) {
      fail(method, init_error.empty() ? "bridge is not initialized" : init_error);
    }
  }

  int dof() const {return static_cast<int>(joint_names.size());}
  int num_tools() const {return static_cast<int>(tool_frames.size());}

  bool has_python() const {return python != nullptr && python_handle >= 0;}

  void require_python(const char * method) const
  {
    if (!has_python()) {
      unsupported(method);
    }
  }

  // The Python backend only needs its full motion planner when a Python-only
  // scene mutation forces a planning fallback. Building it lazily keeps a pure
  // native startup fast.
  void ensure_python_planner(const char * method)
  {
    require_python(method);
    if (python_planner_built) {
      return;
    }
    if (motion_cfg_json.empty()) {
      fail(method, "Python fallback planner config was never provided");
    }
    python->motion_planner_build(python_handle, motion_cfg_json);
    python_planner_built = true;
  }

  void mark_python_scene(const char * /*method*/)
  {
    native_scene_valid = false;
  }

  void ensure_mapper(const char * method)
  {
    if (mapper == nullptr || mapper->runtime == nullptr) {
      fail(method, "native mapper is not configured");
    }
  }

  void ensure_python_mapper(const char * method)
  {
    require_python(method);
    if (!python_mapper_configured) {
      python->configure_mapper(python_handle, mapper_cfg_json);
      python->warmup_mapper(python_handle);
      python_mapper_configured = true;
    }
  }

  std::vector<float> default_joint_vector() const
  {
    std::vector<float> q(static_cast<std::size_t>(dof()), 0.0F);
    for (std::size_t i = 0; i < manifest.home_names.size() && i < manifest.home_positions.size(); ++i) {
      const auto it = std::find(joint_names.begin(), joint_names.end(), manifest.home_names[i]);
      if (it != joint_names.end()) {
        q[static_cast<std::size_t>(std::distance(joint_names.begin(), it))] =
          static_cast<float>(manifest.home_positions[i]);
      }
    }
    return q;
  }

  std::vector<std::array<float, 7>> tool_poses_for_q(const std::vector<float> & q)
  {
    if (static_cast<int>(q.size()) != dof() || kinematics == nullptr) {
      fail("ik", "default FK state does not match the artifact");
    }
    const int tools = num_tools();
    DeviceBuffer input;
    DeviceBuffer positions;
    DeviceBuffer quaternions;
    DeviceBuffer jacobians;
    input.upload(q);
    positions.allocate(static_cast<std::size_t>(tools) * 3 * sizeof(float));
    quaternions.allocate(static_cast<std::size_t>(tools) * 4 * sizeof(float));
    jacobians.allocate(
      static_cast<std::size_t>(tools) * 6 * static_cast<std::size_t>(dof()) *
      sizeof(float));
    kinematics->forward(
      static_cast<const float *>(input.get()),
      static_cast<float *>(positions.get()), static_cast<float *>(quaternions.get()),
      static_cast<float *>(jacobians.get()), 1, 1, nullptr);
    cuda_check("ik", cudaDeviceSynchronize());
    std::vector<float> p(static_cast<std::size_t>(tools) * 3);
    std::vector<float> r(static_cast<std::size_t>(tools) * 4);
    positions.download(p);
    quaternions.download(r);
    std::vector<std::array<float, 7>> out(static_cast<std::size_t>(tools));
    for (int i = 0; i < tools; ++i) {
      out[static_cast<std::size_t>(i)] = {
        p[static_cast<std::size_t>(i) * 3 + 0],
        p[static_cast<std::size_t>(i) * 3 + 1],
        p[static_cast<std::size_t>(i) * 3 + 2],
        r[static_cast<std::size_t>(i) * 4 + 0],
        r[static_cast<std::size_t>(i) * 4 + 1],
        r[static_cast<std::size_t>(i) * 4 + 2],
        r[static_cast<std::size_t>(i) * 4 + 3]};
    }
    return out;
  }

  curobo_runtime::TensorF32 fill_native_ik_goals(
    const curobo_runtime::TensorF32 & poses,
    const std::vector<std::string> & active_frames, bool batch_rows)
  {
    const int tools = num_tools();
    if (poses.shape.size() != 2 || static_cast<int>(poses.shape[1]) != 7) {
      fail("ik", "poses must have shape [N,7]");
    }
    const int rows = static_cast<int>(poses.shape[0]);
    const int batch = batch_rows ? rows : 1;
    if (!batch_rows && rows != static_cast<int>(active_frames.size())) {
      fail("ik_multi", "tool_frames/poses length mismatch");
    }
    if (batch_rows && active_frames.size() != 1) {
      fail("ik", "single-tool IK must name exactly one tool frame");
    }

    std::vector<int> active_index(static_cast<std::size_t>(tools), -1);
    for (std::size_t row = 0; row < active_frames.size(); ++row) {
      const auto it = std::find(
        tool_frames.begin(), tool_frames.end(), active_frames[row]);
      if (it == tool_frames.end()) {
        fail("ik", "unknown tool frame '" + active_frames[row] + "'");
      }
      active_index[static_cast<std::size_t>(
        std::distance(tool_frames.begin(), it))] = static_cast<int>(row);
    }

    const auto home_poses = tool_poses_for_q(default_joint_vector());
    curobo_runtime::TensorF32 full;
    full.shape = {batch, tools, 7};
    full.data.resize(static_cast<std::size_t>(batch) * tools * 7);
    for (int b = 0; b < batch; ++b) {
      for (int tool = 0; tool < tools; ++tool) {
        const int row = active_index[static_cast<std::size_t>(tool)];
        std::array<float, 7> goal = home_poses[static_cast<std::size_t>(tool)];
        if (row >= 0) {
          // `batch_rows` means each source row is one independent IK goal, so
          // the pose for a given batch item is row `b` — NOT the tool's slot in
          // `active_frames` (always 0 for single-tool IK). Using the slot here
          // solved the whole batch against the first goal, which made every
          // result after g001 report a large pose error.
          const int source_row = batch_rows ? b : row;
          for (std::size_t k = 0; k < 7; ++k) {
            goal[k] = poses.data[static_cast<std::size_t>(source_row) * 7 + k];
          }
        }
        const std::size_t offset =
          (static_cast<std::size_t>(b) * tools + static_cast<std::size_t>(tool)) * 7;
        for (std::size_t k = 0; k < 7; ++k) {
          full.data[offset + k] = goal[k];
        }
      }
    }
    return full;
  }

  void ensure_native_voxel_grid()
  {
    if (mapper == nullptr || mapper->runtime == nullptr) {
      return;
    }
    if (mapper->voxel_added) {
      return;
    }
    const int nx = mapper->view.esdf_shape[0];
    const int ny = mapper->view.esdf_shape[1];
    const int nz = mapper->view.esdf_shape[2];
    const float voxel = mapper->view.esdf_voxel_size;
    if (nx <= 0 || ny <= 0 || nz <= 0 || voxel <= 0.0F) {
      fail("esdf_refresh", "native mapper reported an invalid ESDF shape");
    }
    if (scene == nullptr) {
      fail("esdf_refresh", "native scene is not initialized");
    }
    curobo::robot::VoxelGrid grid;
    grid.name = "mapper_esdf";
    grid.voxel_size = voxel;
    grid.dimensions = {
      static_cast<float>(nx) * voxel,
      static_cast<float>(ny) * voxel,
      static_cast<float>(nz) * voxel};
    // curobo::robot::VoxelGrid.pose is the grid centre, which is exactly the
    // AOT mapper origin. The ESDF array still starts at origin - extent/2.
    grid.pose.position = {
      mapper->view.origin[0],
      mapper->view.origin[1],
      mapper->view.origin[2]};
    grid.pose.quaternion = {1.0F, 0.0F, 0.0F, 0.0F};
    const std::size_t count =
      static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
      static_cast<std::size_t>(nz);
    if (manifest.scene_max_voxel_grids <= 0 ||
      count > static_cast<std::size_t>(manifest.scene_max_voxels_per_grid))
    {
      fail(
        "esdf_refresh",
        "artifact voxel capacity is smaller than the native mapper ESDF");
    }
    // Placeholder only; the first update below replaces it with the mapper's
    // FP16 ESDF on the same stream.
    grid.esdf.assign(count, 1.0F);
    mapper->voxel_handle = scene->scene().add_voxel_grid(grid);
    mapper->esdf_count = static_cast<int>(count);
    mapper->voxel_added = true;
  }

  void update_collision_esdf()
  {
    // The AOT ESDF's free/unknown cells are zero and its sign is unreliable,
    // so feeding it directly would turn the whole box into an obstacle. Build
    // a conservative collision field from the native mapper's observed
    // surface voxels instead: surface = 0 m, free = 5 m. The collision kernel
    // only needs the distance-to-obstacle; this is enough to reject paths
    // through observed geometry without phantom contacts in unobserved space.
    const int nx = mapper->view.esdf_shape[0];
    const int ny = mapper->view.esdf_shape[1];
    const int nz = mapper->view.esdf_shape[2];
    const float voxel = mapper->view.esdf_voxel_size;
    const std::size_t count =
      static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
      static_cast<std::size_t>(nz);
    const double half_x = 0.5 * nx * voxel;
    const double half_y = 0.5 * ny * voxel;
    const double half_z = 0.5 * nz * voxel;
    const double ox = mapper->view.origin[0] - half_x;
    const double oy = mapper->view.origin[1] - half_y;
    const double oz = mapper->view.origin[2] - half_z;

    std::vector<__half> field(count, __float2half(5.0F));
    const int capacity = 1000000;
    DeviceBuffer centers;
    DeviceBuffer pools;
    DeviceBuffer counter;
    centers.allocate(static_cast<std::size_t>(capacity) * 3 * sizeof(float));
    pools.allocate(static_cast<std::size_t>(capacity) * sizeof(int32_t));
    counter.allocate(sizeof(int32_t));
    int required = 0;
    int written = 0;
    const int rc = curobo_mapper_extract_occupied(
      mapper->runtime,
      reinterpret_cast<std::uint64_t>(centers.get()),
      reinterpret_cast<std::uint64_t>(pools.get()),
      reinterpret_cast<std::uint64_t>(counter.get()),
      capacity, 1, static_cast<float>(0.5 * voxel), 0, &required, &written);
    if (rc == 0) {
      int observed = 0;
      cuda_check(
        "esdf_refresh",
        cudaMemcpy(&observed, counter.get(), sizeof(int32_t),
          cudaMemcpyDeviceToHost));
      const int take = std::min(observed, capacity);
      std::vector<float> host_centers(static_cast<std::size_t>(take) * 3);
      if (take > 0) {
        cuda_check(
          "esdf_refresh",
          cudaMemcpy(host_centers.data(), centers.get(),
            host_centers.size() * sizeof(float), cudaMemcpyDeviceToHost));
      }
      for (int i = 0; i < take; ++i) {
        const double x = host_centers[static_cast<std::size_t>(i) * 3 + 0];
        const double y = host_centers[static_cast<std::size_t>(i) * 3 + 1];
        const double z = host_centers[static_cast<std::size_t>(i) * 3 + 2];
        const int ix = static_cast<int>(std::floor((x - ox) / voxel));
        const int iy = static_cast<int>(std::floor((y - oy) / voxel));
        const int iz = static_cast<int>(std::floor((z - oz) / voxel));
        if (ix >= 0 && ix < nx && iy >= 0 && iy < ny && iz >= 0 && iz < nz) {
          field[(static_cast<std::size_t>(ix) * ny + iy) * nz + iz] =
            __float2half(0.0F);
        }
      }
    }
    // An attached payload must not collide with its own observed voxels at
    // the pick pose. The world cuboid is disabled on attach, but the depth
    // mapper still has a surface copy of the object. Clear the payload AABB
    // (in the field only; the RViz map keeps it) so retreat/home plans see
    // the environment, not a ghost copy of the object inside the gripper.
    for (const auto & name : attached_objects) {
      const SceneObjectRecord * record = scene ? scene->object(name) : nullptr;
      if (record == nullptr) {
        continue;
      }
      const auto inv = pose_inverse({
        record->pose.position[0], record->pose.position[1],
        record->pose.position[2], record->pose.quaternion[0],
        record->pose.quaternion[1], record->pose.quaternion[2],
        record->pose.quaternion[3]});
      const double radius = (record->type == 1)
        ? static_cast<double>(record->dims[0])
        : std::sqrt(
          static_cast<double>(record->dims[0] * record->dims[0]) +
          static_cast<double>(record->dims[1] * record->dims[1]) +
          static_cast<double>(record->dims[2] * record->dims[2])) * 0.5;
      const int min_x = std::max(0, static_cast<int>(std::floor(
        (record->pose.position[0] - radius - ox) / voxel)));
      const int max_x = std::min(nx - 1, static_cast<int>(std::ceil(
        (record->pose.position[0] + radius - ox) / voxel)));
      const int min_y = std::max(0, static_cast<int>(std::floor(
        (record->pose.position[1] - radius - oy) / voxel)));
      const int max_y = std::min(ny - 1, static_cast<int>(std::ceil(
        (record->pose.position[1] + radius - oy) / voxel)));
      const int min_z = std::max(0, static_cast<int>(std::floor(
        (record->pose.position[2] - radius - oz) / voxel)));
      const int max_z = std::min(nz - 1, static_cast<int>(std::ceil(
        (record->pose.position[2] + radius - oz) / voxel)));
      const std::array<float, 3> half{
        record->dims[0] * 0.5F, record->dims[1] * 0.5F, record->dims[2] * 0.5F};
      for (int ix = min_x; ix <= max_x; ++ix) {
        for (int iy = min_y; iy <= max_y; ++iy) {
          for (int iz = min_z; iz <= max_z; ++iz) {
            const std::array<float, 7> world{
              static_cast<float>((ix + 0.5) * voxel + ox),
              static_cast<float>((iy + 0.5) * voxel + oy),
              static_cast<float>((iz + 0.5) * voxel + oz),
              1.0F, 0.0F, 0.0F, 0.0F};
            const auto local = pose_multiply(inv, world);
            if (std::abs(local[0]) <= half[0] &&
              std::abs(local[1]) <= half[1] &&
              std::abs(local[2]) <= half[2])
            {
              field[(static_cast<std::size_t>(ix) * ny + iy) * nz + iz] =
                __float2half(5.0F);
            }
          }
        }
      }
    }
    cuda_check(
      "esdf_refresh",
      cudaMemcpy(
        reinterpret_cast<void *>(mapper->view.esdf_dist_field), field.data(),
        field.size() * sizeof(__half), cudaMemcpyHostToDevice));
  }

  void update_native_voxel_grid()
  {
    ensure_native_voxel_grid();
    if (!mapper->voxel_added) {
      return;
    }
    update_collision_esdf();
    scene->scene().update_voxel_grid_cuda(
      mapper->voxel_handle, mapper->view.esdf_dist_field,
      static_cast<std::size_t>(mapper->esdf_count));
  }

  void load_artifact()
  {
    bundle = inspect_artifact_directory(artifact_dir, robot, config_file);
    if (!bundle.valid) {
      init_error = bundle.error.empty() ?
        "native artifact is invalid" : bundle.error;
      fail("init", init_error);
    }
    manifest = bundle.manifest;
    if (!manifest.robot.empty()) {
      robot = manifest.robot;
    }
    joint_names = manifest.joint_names;
    tool_frames = manifest.tool_frames;
    link_names = manifest.link_names;
    joint_lower = manifest.joint_lower;
    joint_upper = manifest.joint_upper;
    locked_joint_names = manifest.locked_joint_names;
    locked_positions = manifest.locked_positions;
    unpinned_lock_joint_names = manifest.unpinned_lock_joint_names;

    auto kinematics_ptr = std::make_unique<curobo::robot::Kinematics>(artifact_dir, device_ordinal);
    const CuroboRobotInfo info = kinematics_ptr->info();
    if (joint_names.empty()) {
      for (int index = 0; index < info.num_dof; ++index) {
        joint_names.push_back(kinematics_ptr->joint_name(index));
      }
    }
    if (tool_frames.empty()) {
      for (int index = 0; index < info.num_tool_frames; ++index) {
        tool_frames.push_back(kinematics_ptr->tool_frame_name(index));
      }
    }
    if (static_cast<int>(joint_names.size()) != info.num_dof ||
      static_cast<int>(tool_frames.size()) != info.num_tool_frames)
    {
      fail("init", "artifact manifest names do not match cpp_runtime dimensions");
    }
    kinematics = std::move(kinematics_ptr);
  }

  std::map<std::string, double> lock_map() const
  {
    std::map<std::string, double> locks;
    for (std::size_t i = 0;
      i < locked_joint_names.size() && i < locked_positions.size(); ++i)
    {
      locks[locked_joint_names[i]] = locked_positions[i];
    }
    return locks;
  }

  struct LockedArtifact
  {
    std::string directory;
    std::string config_file;
  };

  // Native artifacts bake lock_joints into the exported model, so a lock change
  // means selecting the sibling bundle exported with those values (the
  // benchmark ships one bundle per dataset: MB 0.04, MPINets 0.025). The
  // generated manifest.json records lock_joints.pinned, so the bundle is
  // identifiable without re-parsing the YAML.
  std::optional<LockedArtifact> find_artifact_for_locks(
    const std::map<std::string, double> & requested) const
  {
    const std::filesystem::path current(artifact_dir);
    const std::string arch = current.filename().string();
    const std::filesystem::path root = current.parent_path().parent_path();
    if (arch.empty() || root.empty()) {
      return std::nullopt;
    }
    std::error_code iter_error;
    for (const auto & entry : std::filesystem::directory_iterator(root, iter_error)) {
      if (iter_error) {
        break;
      }
      if (!entry.is_directory()) {
        continue;
      }
      const std::filesystem::path candidate = entry.path() / arch;
      if (!std::filesystem::is_directory(candidate)) {
        continue;
      }
      ArtifactBundle bundle_candidate;
      try {
        bundle_candidate = inspect_artifact_directory(candidate.string(), robot, "");
      } catch (const std::exception &) {
        continue;
      }
      if (!bundle_candidate.valid) {
        continue;
      }
      std::map<std::string, double> locks;
      for (std::size_t i = 0;
        i < bundle_candidate.manifest.locked_joint_names.size() &&
        i < bundle_candidate.manifest.locked_positions.size(); ++i)
      {
        locks[bundle_candidate.manifest.locked_joint_names[i]] =
          bundle_candidate.manifest.locked_positions[i];
      }
      if (locks == requested) {
        return LockedArtifact{
          candidate.string(), bundle_candidate.manifest.curobo_config};
      }
    }
    return std::nullopt;
  }

  void reload_artifact(const std::string & new_dir, const std::string & new_config)
  {
    // Everything native is per-artifact (kinematics, scene capacities, solver
    // workspaces/graphs). Drop it and rebuild from the selected bundle; the
    // caller (ConfigServices) rebuilds the planner afterwards and the ROS
    // world is re-pushed by the next scenario.
    mapper.reset();
    mpc.reset();
    planner.reset();
    ik.reset();
    kinematics.reset();
    scene.reset();
    python_planner_built = false;
    motion_cfg_json.clear();
    native_scene_valid = true;
    attached_objects.clear();
    attach_links.clear();
    visual_attachments.clear();
    artifact_dir = new_dir;
    config_file = new_config;
    init_error.clear();
    load_artifact();
    ensure_scene();
    initialized = true;
  }

  void ensure_planner(const char * method)
  {
    require_initialized(method);
    if (planner) {
      return;
    }
    curobo::robot::MotionPlanningConfig config;
    config.trajectory_optimization.device_ordinal = device_ordinal;
    config.trajectory_optimization.num_seeds = planner_settings.num_trajopt_seeds;
    config.trajectory_optimization.collision_activation_distance =
      static_cast<float>(planner_settings.collision_activation_distance);
    config.trajectory_optimization.trajectory_dt =
      static_cast<float>(planner_settings.interpolation_dt);
    config.trajectory_optimization.num_knots = planner_settings.num_knots;
    config.trajectory_optimization.check_effort_limits =
      planner_settings.check_effort_limits;
    config.inverse_kinematics.check_effort_limits =
      planner_settings.check_effort_limits;
    config.rrt.check_effort_limits = planner_settings.check_effort_limits;
    config.use_cuda_graph = planner_settings.use_cuda_graph && !debug_cuda_graphs;
    config.rrt.device_ordinal = device_ordinal;
    config.rrt.max_iterations = std::max(32, planner_settings.rrt_max_iterations);
    config.inverse_kinematics.device_ordinal = device_ordinal;
    config.inverse_kinematics.max_batch_size = planner_settings.max_goalset;
    config.inverse_kinematics.num_seeds = planner_settings.num_ik_seeds;
    config.inverse_kinematics.max_goalset = std::max(1, planner_settings.max_goalset);
    config.inverse_kinematics.use_cuda_graph = false;

    planner = std::make_unique<curobo::robot::MotionPlanner>(
      artifact_dir, config, scene ? &scene->scene() : nullptr);
    for (const auto & entry : link_collision) {
      planner->set_link_collision_enabled(entry.first, entry.second);
    }
  }

  void ensure_scene()
  {
    if (scene) {
      return;
    }
    curobo::robot::SceneConfig config;
    config.device_ordinal = device_ordinal;
    config.max_cuboids = manifest.scene_max_cuboids;
    config.max_capsules = manifest.scene_max_capsules;
    config.max_cylinders = manifest.scene_max_cylinders;
    config.max_meshes = manifest.scene_max_meshes;
    config.max_voxel_grids = manifest.scene_max_voxel_grids;
    config.max_voxels_per_grid = manifest.scene_max_voxels_per_grid;
    scene = std::make_unique<NativeScene>(config);
  }

  void ensure_ik(const char * method)
  {
    require_initialized(method);
    if (ik) {
      return;
    }
    curobo::robot::InverseKinematicsConfig config;
    config.device_ordinal = device_ordinal;
    config.max_batch_size = std::max(1, ik_settings.batch_size);
    config.num_seeds = std::max(1, ik_settings.num_seeds);
    config.max_goalset = std::max(1, ik_settings.max_goalset);
    config.check_self_collision = ik_settings.self_collision_check;
    config.use_cuda_graph = ik_settings.use_cuda_graph && !debug_cuda_graphs;
    config.lm_num_seeds = ik_settings.use_lm_seed ? config.num_seeds : 0;
    ik = std::make_unique<curobo::robot::InverseKinematics>(
      artifact_dir, config, scene ? &scene->scene() : nullptr);
    for (const auto & entry : link_collision) {
      ik->set_link_collision_enabled(entry.first, entry.second);
    }
  }

  curobo::robot::MpcController & ensure_mpc(const char * method)
  {
    ensure_planner(method);
    if (!mpc) {
      curobo::robot::MpcConfig config;
      config.trajectory_dt = static_cast<float>(planner_settings.interpolation_dt);
      config.collision_activation_distance =
        static_cast<float>(std::max(0.0, planner_settings.collision_activation_distance * 0.2));
      mpc = std::make_unique<curobo::robot::MpcController>(*planner, config);
    }
    return *mpc;
  }

  std::array<float, 7> fk_pose(const std::vector<float> & q, const char * method);
  std::array<float, 7> tool_pose(
    const std::vector<float> & q, int tool_index, const char * method);
  TensorF32 fk_tensor(const TensorF32 & q, const char * method);
  IkOutcome solve_ik(
    const TensorF32 & poses, const std::vector<std::string> & frames,
    const std::vector<float> * seed_q, int num_goalset, const char * method);
  PlanOutcome plan_cspace_impl(const PlanRequest & request);
  PlanOutcome plan_pose_impl(const PlanRequest & request);
  PlanOutcome plan_waypoints_impl(const PlanRequest & request, const OrderedWaypoints & waypoints);

  static PlanOutcome make_outcome(
    const std::vector<float> & position, const std::vector<float> & velocity,
    const std::vector<float> & acceleration, int dof, int steps, double dt,
    const std::vector<std::string> & names);
};

std::array<float, 7> CppBridge::Impl::fk_pose(
  const std::vector<float> & q, const char * method)
{
  if (kinematics == nullptr) {
    fail(method, "kinematics is not loaded");
  }
  if (static_cast<int>(q.size()) != dof()) {
    fail(method, "start pose has " + std::to_string(q.size()) + " values, expected " +
      std::to_string(dof()));
  }
  const int tools = num_tools();
  DeviceBuffer input;
  DeviceBuffer positions;
  DeviceBuffer quaternions;
  DeviceBuffer jacobians;
  input.upload(q);
  positions.allocate(static_cast<std::size_t>(tools) * 3 * sizeof(float));
  quaternions.allocate(static_cast<std::size_t>(tools) * 4 * sizeof(float));
  jacobians.allocate(
    static_cast<std::size_t>(tools) * 6 * static_cast<std::size_t>(dof()) *
    sizeof(float));
  kinematics->forward(
    static_cast<const float *>(input.get()),
    static_cast<float *>(positions.get()), static_cast<float *>(quaternions.get()),
    static_cast<float *>(jacobians.get()), 1, 1, nullptr);
  cuda_check(method, cudaDeviceSynchronize());
  std::vector<float> p(static_cast<std::size_t>(tools) * 3);
  std::vector<float> r(static_cast<std::size_t>(tools) * 4);
  positions.download(p);
  quaternions.download(r);
  std::array<float, 7> pose{};
  pose[0] = p[0];
  pose[1] = p[1];
  pose[2] = p[2];
  pose[3] = r[0];
  pose[4] = r[1];
  pose[5] = r[2];
  pose[6] = r[3];
  return pose;
}

std::array<float, 7> CppBridge::Impl::tool_pose(
  const std::vector<float> & q, int tool_index, const char * method)
{
  if (kinematics == nullptr) {
    fail(method, "kinematics is not loaded");
  }
  const int tools = num_tools();
  if (tool_index < 0 || tool_index >= tools) {
    fail(method, "tool index is outside the artifact tools");
  }
  const int dof_count = dof();
  std::vector<float> clamped = q;
  for (int i = 0; i < std::min(dof_count, static_cast<int>(clamped.size())); ++i) {
    clamped[static_cast<std::size_t>(i)] = std::clamp(
      clamped[static_cast<std::size_t>(i)],
      static_cast<float>(joint_lower[static_cast<std::size_t>(i)]),
      static_cast<float>(joint_upper[static_cast<std::size_t>(i)]));
  }
  clamped.resize(static_cast<std::size_t>(dof_count), 0.0F);
  DeviceBuffer input;
  DeviceBuffer positions;
  DeviceBuffer quaternions;
  DeviceBuffer jacobians;
  input.upload(clamped);
  positions.allocate(static_cast<std::size_t>(tools) * 3 * sizeof(float));
  quaternions.allocate(static_cast<std::size_t>(tools) * 4 * sizeof(float));
  jacobians.allocate(
    static_cast<std::size_t>(tools) * 6 * static_cast<std::size_t>(dof_count) *
    sizeof(float));
  kinematics->forward(
    static_cast<const float *>(input.get()),
    static_cast<float *>(positions.get()), static_cast<float *>(quaternions.get()),
    static_cast<float *>(jacobians.get()), 1, 1, nullptr);
  cuda_check(method, cudaDeviceSynchronize());
  std::vector<float> p(static_cast<std::size_t>(tools) * 3);
  std::vector<float> r(static_cast<std::size_t>(tools) * 4);
  positions.download(p);
  quaternions.download(r);
  const std::size_t pi = static_cast<std::size_t>(tool_index) * 3;
  const std::size_t ri = static_cast<std::size_t>(tool_index) * 4;
  return {p[pi], p[pi + 1], p[pi + 2], r[ri], r[ri + 1], r[ri + 2], r[ri + 3]};
}

curobo_runtime::TensorF32 CppBridge::Impl::fk_tensor(
  const curobo_runtime::TensorF32 & q, const char * method)
{
  if (q.shape.size() != 2 || static_cast<int>(q.shape[1]) != dof()) {
    fail(method, "q must have shape [N, dof]");
  }
  const int batch = static_cast<int>(q.shape[0]);
  const int tools = num_tools();
  DeviceBuffer input;
  DeviceBuffer positions;
  DeviceBuffer quaternions;
  DeviceBuffer jacobians;
  input.upload(q.data);
  positions.allocate(
    static_cast<std::size_t>(batch) * tools * 3 * sizeof(float));
  quaternions.allocate(
    static_cast<std::size_t>(batch) * tools * 4 * sizeof(float));
  jacobians.allocate(
    static_cast<std::size_t>(batch) * tools * 6 *
    static_cast<std::size_t>(dof()) * sizeof(float));
  kinematics->forward(
    static_cast<const float *>(input.get()),
    static_cast<float *>(positions.get()), static_cast<float *>(quaternions.get()),
    static_cast<float *>(jacobians.get()), batch, 1, nullptr);
  cuda_check(method, cudaDeviceSynchronize());
  std::vector<float> p(static_cast<std::size_t>(batch) * tools * 3);
  std::vector<float> r(static_cast<std::size_t>(batch) * tools * 4);
  positions.download(p);
  quaternions.download(r);

  TensorF32 output;
  output.shape = {batch, 1, 7};
  output.data.resize(static_cast<std::size_t>(batch) * 7);
  for (int i = 0; i < batch; ++i) {
    const std::size_t p_offset = static_cast<std::size_t>(i) * tools * 3;
    const std::size_t r_offset = static_cast<std::size_t>(i) * tools * 4;
    output.data[static_cast<std::size_t>(i) * 7 + 0] = p[p_offset + 0];
    output.data[static_cast<std::size_t>(i) * 7 + 1] = p[p_offset + 1];
    output.data[static_cast<std::size_t>(i) * 7 + 2] = p[p_offset + 2];
    output.data[static_cast<std::size_t>(i) * 7 + 3] = r[r_offset + 0];
    output.data[static_cast<std::size_t>(i) * 7 + 4] = r[r_offset + 1];
    output.data[static_cast<std::size_t>(i) * 7 + 5] = r[r_offset + 2];
    output.data[static_cast<std::size_t>(i) * 7 + 6] = r[r_offset + 3];
  }
  return output;
}

curobo_runtime::IkOutcome CppBridge::Impl::solve_ik(
  const curobo_runtime::TensorF32 & poses, const std::vector<std::string> & frames,
  const std::vector<float> * seed_q, int num_goalset, const char * method)
{
  ensure_ik(method);
  const int tools = num_tools();
  const int dof_count = dof();
  if (tools <= 0) {
    fail(method, "artifact has no tool frames");
  }

  int batch = 0;
  if (poses.shape.size() == 2) {
    if (static_cast<int>(poses.shape[1]) != 7) {
      fail(method, "poses must have shape [N,7]");
    }
    batch = static_cast<int>(poses.shape[0]);
    if (tools > 1) {
      fail(method,
        "cpp_runtime IK targets every artifact tool frame simultaneously; this artifact has " +
        std::to_string(tools) + " tool frames but the request supplied one. Use ik_multi.");
    }
  } else if (poses.shape.size() == 3) {
    if (static_cast<int>(poses.shape[2]) != 7) {
      fail(method, "poses must have shape [N,T,7]");
    }
    batch = static_cast<int>(poses.shape[0]);
    if (static_cast<int>(poses.shape[1]) != tools) {
      fail(method, "poses second dimension must match the artifact tool frames (" +
        std::to_string(tools) + ")");
    }
  } else {
    fail(method, "poses must have shape [N,7] or [N,T,7]");
  }
  if (batch <= 0) {
    fail(method, "empty IK batch");
  }

  // Map artifact tool order to the request's frames. All artifact frames must
  // be addressed; native IK always produces one goal per configured tool.
  std::vector<int> tool_order(static_cast<std::size_t>(tools));
  for (int i = 0; i < tools; ++i) {
    tool_order[static_cast<std::size_t>(i)] = i;
  }
  if (!frames.empty()) {
    if (static_cast<int>(frames.size()) != tools) {
      fail(method, "IK requires one target per artifact tool frame (" +
        std::to_string(tools) + "), got " + std::to_string(frames.size()));
    }
    std::vector<bool> used(static_cast<std::size_t>(tools), false);
    for (int slot = 0; slot < tools; ++slot) {
      const auto it = std::find(tool_frames.begin(), tool_frames.end(), frames[static_cast<std::size_t>(slot)]);
      if (it == tool_frames.end()) {
        fail(method, "unknown tool frame '" + frames[static_cast<std::size_t>(slot)] + "'");
      }
      const int index = static_cast<int>(std::distance(tool_frames.begin(), it));
      if (used[static_cast<std::size_t>(index)]) {
        fail(method, "duplicate tool frame '" + frames[static_cast<std::size_t>(slot)] + "'");
      }
      used[static_cast<std::size_t>(index)] = true;
      tool_order[static_cast<std::size_t>(slot)] = index;
    }
  }

  std::vector<float> goal_positions(
    static_cast<std::size_t>(batch) * tools * num_goalset * 3);
  std::vector<float> goal_quaternions(
    static_cast<std::size_t>(batch) * tools * num_goalset * 4);
  for (int i = 0; i < batch; ++i) {
    for (int slot = 0; slot < tools; ++slot) {
      const int source_tool = poses.shape.size() == 2 ? 0 : slot;
      const std::size_t source =
        (static_cast<std::size_t>(i) * (poses.shape.size() == 2 ? 1 : tools) +
        static_cast<std::size_t>(source_tool)) * 7;
      const int destination_tool = tool_order[static_cast<std::size_t>(slot)];
      const std::size_t destination =
        (static_cast<std::size_t>(i) * tools + static_cast<std::size_t>(destination_tool)) *
        static_cast<std::size_t>(num_goalset);
      for (int g = 0; g < num_goalset; ++g) {
        // IkSolveRequest keeps positions and quaternions in two separate
        // contiguous tensors: [batch, tool, goalset, 3] and [..., 4]. They are
        // NOT interleaved 7-float rows.
        const std::size_t value = destination + static_cast<std::size_t>(g);
        goal_positions[value * 3 + 0] = poses.data[source + 0];
        goal_positions[value * 3 + 1] = poses.data[source + 1];
        goal_positions[value * 3 + 2] = poses.data[source + 2];
        goal_quaternions[value * 4 + 0] = poses.data[source + 3];
        goal_quaternions[value * 4 + 1] = poses.data[source + 4];
        goal_quaternions[value * 4 + 2] = poses.data[source + 5];
        goal_quaternions[value * 4 + 3] = poses.data[source + 6];
      }
    }
  }

  DeviceBuffer goal_pos_device;
  DeviceBuffer goal_quat_device;
  DeviceBuffer seed_device;
  DeviceBuffer solution_device;
  DeviceBuffer success_device;
  DeviceBuffer pos_err_device;
  DeviceBuffer orient_err_device;
  DeviceBuffer collision_device;
  DeviceBuffer goalset_index_device;

  goal_pos_device.upload(goal_positions);
  goal_quat_device.upload(goal_quaternions);
  if (seed_q != nullptr) {
    if (static_cast<int>(seed_q->size()) != batch * dof_count) {
      fail(method, "IK seeds must have shape [N,dof]");
    }
    seed_device.upload(*seed_q);
  }
  solution_device.allocate(
    static_cast<std::size_t>(batch) * dof_count * sizeof(float));
  success_device.allocate(static_cast<std::size_t>(batch) * sizeof(uint8_t));
  pos_err_device.allocate(static_cast<std::size_t>(batch) * sizeof(float));
  orient_err_device.allocate(static_cast<std::size_t>(batch) * sizeof(float));
  collision_device.allocate(static_cast<std::size_t>(batch) * sizeof(float));
  if (num_goalset > 1) {
    goalset_index_device.allocate(
      static_cast<std::size_t>(batch) * tools * sizeof(int32_t));
  }

  curobo::robot::IkSolveRequest request;
  request.goal_positions = static_cast<const float *>(goal_pos_device.get());
  request.goal_quaternions = static_cast<const float *>(goal_quat_device.get());
  request.seed_positions = seed_q == nullptr ? nullptr :
    static_cast<const float *>(seed_device.get());
  request.solution = static_cast<float *>(solution_device.get());
  request.success = static_cast<uint8_t *>(success_device.get());
  request.position_error = static_cast<float *>(pos_err_device.get());
  request.orientation_error = static_cast<float *>(orient_err_device.get());
  request.collision_cost = static_cast<float *>(collision_device.get());
  request.goalset_index = num_goalset > 1 ?
    static_cast<int32_t *>(goalset_index_device.get()) : nullptr;
  request.batch_size = batch;
  request.num_goalset = num_goalset;
  request.num_solutions = 1;
  ik->solve(request);
  cuda_check(method, cudaDeviceSynchronize());

  std::vector<float> solutions(static_cast<std::size_t>(batch) * dof_count);
  std::vector<uint8_t> success(static_cast<std::size_t>(batch));
  solution_device.download(solutions);
  success_device.download(success);

  IkOutcome outcome;
  outcome.positions.shape = {batch, 1, dof_count};
  outcome.positions.data = std::move(solutions);
  outcome.valid.resize(static_cast<std::size_t>(batch));
  outcome.joint_names = joint_names;
  for (int i = 0; i < batch; ++i) {
    outcome.valid[static_cast<std::size_t>(i)] = success[static_cast<std::size_t>(i)] != 0 ? 1 : 0;
    if (outcome.valid[static_cast<std::size_t>(i)] != 0) {
      outcome.success = true;
    }
  }
  return outcome;
}

curobo_runtime::PlanOutcome CppBridge::Impl::make_outcome(
  const std::vector<float> & position, const std::vector<float> & velocity,
  const std::vector<float> & acceleration, int dof_count, int steps, double dt,
  const std::vector<std::string> & names)
{
  PlanOutcome outcome;
  outcome.positions.shape = {steps, dof_count};
  outcome.velocity.shape = {steps, dof_count};
  outcome.acceleration.shape = {steps, dof_count};
  outcome.positions.data = position;
  outcome.velocity.data = velocity;
  outcome.acceleration.data = acceleration;
  outcome.joint_names = names;
  outcome.total_time = steps > 0 ? static_cast<double>(steps - 1) * dt : 0.0;
  return outcome;
}

curobo_runtime::PlanOutcome CppBridge::Impl::plan_cspace_impl(
  const curobo_runtime::PlanRequest & request)
{
  ensure_planner("plan_cspace");
  const int dof_count = dof();
  if (static_cast<int>(request.start.size()) != dof_count) {
    fail("plan_cspace", "start has " + std::to_string(request.start.size()) +
      " values, expected " + std::to_string(dof_count));
  }
  if (static_cast<int>(request.joint_target.size()) != dof_count) {
    fail("plan_cspace", "joint_target has " + std::to_string(request.joint_target.size()) +
      " values, expected " + std::to_string(dof_count));
  }
  // Joint-state publishers carry float noise around hard limits (e.g. a
  // joint whose lower bound is exactly 0 reports -1.7e-14). The RRT terminal
  // check is strict (`value < lower`), so an otherwise valid home state was
  // rejected before the search even started. Clamp to the artifact limits at
  // the bridge boundary, where the wire values become planner values.
  std::vector<float> start = request.start;
  std::vector<float> goal = request.joint_target;
  const int limit_count = std::min(
    dof_count, static_cast<int>(std::min(joint_lower.size(), joint_upper.size())));
  for (int i = 0; i < limit_count; ++i) {
    const float lower = static_cast<float>(joint_lower[static_cast<std::size_t>(i)]);
    const float upper = static_cast<float>(joint_upper[static_cast<std::size_t>(i)]);
    start[static_cast<std::size_t>(i)] = std::clamp(start[static_cast<std::size_t>(i)], lower, upper);
    goal[static_cast<std::size_t>(i)] = std::clamp(goal[static_cast<std::size_t>(i)], lower, upper);
  }

  const auto info = planner->info();
  const int steps = info.trajectory_steps;
  const int knots = info.num_knots;

  DeviceBuffer start_device;
  DeviceBuffer goal_device;
  DeviceBuffer knots_device;
  DeviceBuffer position_device;
  DeviceBuffer velocity_device;
  DeviceBuffer acceleration_device;
  DeviceBuffer jerk_device;
  DeviceBuffer success_device;
  DeviceBuffer cost_device;
  DeviceBuffer collision_device;
  DeviceBuffer dt_device;
  DeviceBuffer rrt_device;
  DeviceBuffer rrt_iterations_device;
  DeviceBuffer selected_seed_device;

  start_device.upload(start);
  goal_device.upload(goal);
  knots_device.allocate(static_cast<std::size_t>(knots) * dof_count * sizeof(float));
  position_device.allocate(static_cast<std::size_t>(steps) * dof_count * sizeof(float));
  velocity_device.allocate(static_cast<std::size_t>(steps) * dof_count * sizeof(float));
  acceleration_device.allocate(static_cast<std::size_t>(steps) * dof_count * sizeof(float));
  jerk_device.allocate(static_cast<std::size_t>(steps) * dof_count * sizeof(float));
  success_device.allocate(sizeof(uint8_t));
  cost_device.allocate(sizeof(float));
  collision_device.allocate(sizeof(float));
  dt_device.allocate(sizeof(float));
  rrt_device.allocate(sizeof(uint8_t));
  rrt_iterations_device.allocate(sizeof(int32_t));
  selected_seed_device.allocate(sizeof(int32_t));

  curobo::robot::MotionPlanningRequest native;
  native.start_position = static_cast<const float *>(start_device.get());
  native.goal_position = static_cast<const float *>(goal_device.get());
  native.optimized_knots = static_cast<float *>(knots_device.get());
  native.trajectory_position = static_cast<float *>(position_device.get());
  native.trajectory_velocity = static_cast<float *>(velocity_device.get());
  native.trajectory_acceleration = static_cast<float *>(acceleration_device.get());
  native.trajectory_jerk = static_cast<float *>(jerk_device.get());
  native.success = static_cast<uint8_t *>(success_device.get());
  native.cost = static_cast<float *>(cost_device.get());
  native.collision_cost = static_cast<float *>(collision_device.get());
  native.trajectory_dt = static_cast<float *>(dt_device.get());
  native.used_rrt = static_cast<uint8_t *>(rrt_device.get());
  native.rrt_iterations = static_cast<int32_t *>(rrt_iterations_device.get());
  native.selected_seed = static_cast<int32_t *>(selected_seed_device.get());
  native.attempt_direct_trajectory_optimization = true;
  native.allow_rrt = request.max_attempts >= request.enable_graph_attempt;
  native.finetune_attempts = -1;
  native.finetune_dt_scale = -1.0F;
  native.outer_iterations = std::max(1, request.outer_iterations);

  const auto start_time = std::chrono::steady_clock::now();
  planner->plan(native);
  cuda_check("plan_cspace", cudaDeviceSynchronize());
  const auto end_time = std::chrono::steady_clock::now();

  uint8_t success = 0;
  float dt = 0.0F;
  float cost = 0.0F;
  int32_t selected_seed = 0;
  uint8_t used_rrt = 0;
  int32_t rrt_iterations = 0;
  success_device.download(&success, sizeof(success));
  dt_device.download(&dt, sizeof(dt));
  cost_device.download(&cost, sizeof(cost));
  selected_seed_device.download(&selected_seed, sizeof(selected_seed));
  rrt_device.download(&used_rrt, sizeof(used_rrt));
  rrt_iterations_device.download(&rrt_iterations, sizeof(rrt_iterations));

  std::vector<float> position;
  std::vector<float> velocity;
  std::vector<float> acceleration;
  position_device.download(position);
  velocity_device.download(velocity);
  acceleration_device.download(acceleration);

  PlanOutcome outcome = make_outcome(
    std::move(position), std::move(velocity), std::move(acceleration), dof_count,
    steps, dt, joint_names);
  outcome.success = success != 0;
  outcome.feasible = outcome.success;
  outcome.status = outcome.success ? "success" : "cspace planning failed";
  outcome.seed_index = selected_seed;
  outcome.solver_time_s =
    std::chrono::duration<double>(end_time - start_time).count();
  outcome.motion_time_s = (steps > 0) ? static_cast<double>(steps - 1) * dt : 0.0;
  outcome.candidates_generated = 1;
  outcome.candidates_solved = outcome.success ? 1 : 0;
  outcome.waypoints_planned = 1;
  return outcome;
}

curobo_runtime::PlanOutcome CppBridge::Impl::plan_pose_impl(
  const curobo_runtime::PlanRequest & request)
{
  ensure_planner("plan_pose");
  const int dof_count = dof();
  const int tools = num_tools();
  if (static_cast<int>(request.start.size()) != dof_count) {
    fail("plan_pose", "start has " + std::to_string(request.start.size()) +
      " values, expected " + std::to_string(dof_count));
  }
  std::vector<float> start = request.start;
  {
    const int count = std::min(
      static_cast<int>(start.size()),
      static_cast<int>(std::min(joint_lower.size(), joint_upper.size())));
    for (int i = 0; i < count; ++i) {
      const float lower = static_cast<float>(joint_lower[static_cast<std::size_t>(i)]);
      const float upper = static_cast<float>(joint_upper[static_cast<std::size_t>(i)]);
      start[static_cast<std::size_t>(i)] =
        std::clamp(start[static_cast<std::size_t>(i)], lower, upper);
    }
  }
  if (!request.trajectory_constraints.empty() ||
    !request.tool_trajectory_constraints.empty())
  {
    fail("plan_pose",
      "legacy 6-axis holds are not supported by the native planner; use ordered "
      "waypoints or run with curobo_runtime:=python");
  }
  if (request.goalset_count > 1) {
    fail("plan_pose",
      "multi-segment goalsets (goalset_count > 1) must be planned one segment at a "
      "time by the caller; cpp_runtime plan_pose handles one goalset");
  }

  struct Candidate
  {
    std::vector<std::array<float, 7>> poses;
  };
  std::vector<Candidate> candidates;

  if (!request.tool_poses.data.empty()) {
    const int tool_count = request.tool_poses.shape.size() >= 2 ?
      static_cast<int>(request.tool_poses.shape[0]) : 0;
    if (tool_count <= 0) {
      fail("plan_pose", "tool_poses must have shape [T,7]");
    }
    Candidate candidate;
    candidate.poses.reserve(static_cast<std::size_t>(tools));
    for (int tool = 0; tool < tools; ++tool) {
      candidate.poses.push_back(tool_pose(start, tool, "plan_pose"));
    }
    if (!request.tool_frames.empty() &&
      static_cast<int>(request.tool_frames.size()) != tool_count)
    {
      fail("plan_pose", "tool_frames and tool_poses must be parallel");
    }
    for (int i = 0; i < tool_count; ++i) {
      const float * row = request.tool_poses.data.data() + static_cast<std::size_t>(i) * 7;
      if (!request.tool_frames.empty()) {
        const auto it = std::find(
          tool_frames.begin(), tool_frames.end(), request.tool_frames[static_cast<std::size_t>(i)]);
        if (it == tool_frames.end()) {
          fail("plan_pose", "unknown tool frame '" +
            request.tool_frames[static_cast<std::size_t>(i)] + "'");
        }
        candidate.poses[static_cast<std::size_t>(std::distance(tool_frames.begin(), it))] =
          pose_from_values(row);
      } else if (i < tools) {
        candidate.poses[static_cast<std::size_t>(i)] = pose_from_values(row);
      }
    }
    candidates.push_back(std::move(candidate));
  } else {
    if (request.goalset.data.empty()) {
      fail("plan_pose", "request has neither goalset nor tool_poses");
    }
    const int goal_count = request.goalset.shape.size() >= 2 ?
      static_cast<int>(request.goalset.shape[0]) : 0;
    if (goal_count <= 0) {
      fail("plan_pose", "goalset must have shape [G,7]");
    }
    std::vector<std::array<float, 7>> held;
    held.reserve(static_cast<std::size_t>(tools));
    for (int tool = 0; tool < tools; ++tool) {
      held.push_back(tool_pose(start, tool, "plan_pose"));
    }
    for (int g = 0; g < goal_count; ++g) {
      Candidate candidate;
      candidate.poses = held;
      candidate.poses[0] = pose_from_values(
        request.goalset.data.data() + static_cast<std::size_t>(g) * 7);
      candidates.push_back(std::move(candidate));
    }
  }

  const auto info = planner->info();
  const int steps = info.trajectory_steps;
  const int knots = info.num_knots;
  PlanOutcome best;
  bool have_best = false;
  double best_cost = std::numeric_limits<double>::infinity();
  std::string last_status = "pose planning failed";

  for (std::size_t candidate_index = 0; candidate_index < candidates.size();
    ++candidate_index)
  {
    const auto & candidate = candidates[candidate_index];
    std::vector<float> goal_positions;
    std::vector<float> goal_quaternions;
    fill_pose_arrays(candidate.poses, goal_positions, goal_quaternions);

    DeviceBuffer start_device;
    DeviceBuffer goal_pos_device;
    DeviceBuffer goal_quat_device;
    DeviceBuffer knots_device;
    DeviceBuffer position_device;
    DeviceBuffer velocity_device;
    DeviceBuffer acceleration_device;
    DeviceBuffer jerk_device;
    DeviceBuffer success_device;
    DeviceBuffer cost_device;
    DeviceBuffer collision_device;
    DeviceBuffer dt_device;
    DeviceBuffer position_error_device;
    DeviceBuffer orientation_error_device;
    DeviceBuffer used_rrt_device;
    DeviceBuffer rrt_iterations_device;
    DeviceBuffer selected_seed_device;
    DeviceBuffer planning_attempts_device;

    start_device.upload(start);
    goal_pos_device.upload(goal_positions);
    goal_quat_device.upload(goal_quaternions);
    knots_device.allocate(static_cast<std::size_t>(knots) * dof_count * sizeof(float));
    position_device.allocate(static_cast<std::size_t>(steps) * dof_count * sizeof(float));
    velocity_device.allocate(static_cast<std::size_t>(steps) * dof_count * sizeof(float));
    acceleration_device.allocate(static_cast<std::size_t>(steps) * dof_count * sizeof(float));
    jerk_device.allocate(static_cast<std::size_t>(steps) * dof_count * sizeof(float));
    success_device.allocate(sizeof(uint8_t));
    cost_device.allocate(sizeof(float));
    collision_device.allocate(sizeof(float));
    dt_device.allocate(sizeof(float));
    position_error_device.allocate(sizeof(float));
    orientation_error_device.allocate(sizeof(float));
    used_rrt_device.allocate(sizeof(uint8_t));
    rrt_iterations_device.allocate(sizeof(int32_t));
    selected_seed_device.allocate(sizeof(int32_t));
    planning_attempts_device.allocate(sizeof(int32_t));

    curobo::robot::PoseMotionPlanningRequest native;
    native.goal_positions = static_cast<const float *>(goal_pos_device.get());
    native.goal_quaternions = static_cast<const float *>(goal_quat_device.get());
    native.current_position = static_cast<const float *>(start_device.get());
    native.optimized_knots = static_cast<float *>(knots_device.get());
    native.trajectory_position = static_cast<float *>(position_device.get());
    native.trajectory_velocity = static_cast<float *>(velocity_device.get());
    native.trajectory_acceleration = static_cast<float *>(acceleration_device.get());
    native.trajectory_jerk = static_cast<float *>(jerk_device.get());
    native.success = static_cast<uint8_t *>(success_device.get());
    native.cost = static_cast<float *>(cost_device.get());
    native.collision_cost = static_cast<float *>(collision_device.get());
    native.trajectory_dt = static_cast<float *>(dt_device.get());
    native.position_error = static_cast<float *>(position_error_device.get());
    native.orientation_error = static_cast<float *>(orientation_error_device.get());
    native.used_rrt = static_cast<uint8_t *>(used_rrt_device.get());
    native.rrt_iterations = static_cast<int32_t *>(rrt_iterations_device.get());
    native.selected_seed = static_cast<int32_t *>(selected_seed_device.get());
    native.planning_attempts = static_cast<int32_t *>(planning_attempts_device.get());
    native.max_attempts = std::max(1, request.max_attempts);
    native.enable_graph_attempt = request.enable_graph_attempt;
    native.finetune_attempts = 1;
    native.finetune_dt_scale = 0.6F;
    native.constrain_tool_path = false;
    native.outer_iterations = std::max(1, request.outer_iterations);

    const auto start_time = std::chrono::steady_clock::now();
    planner->plan_pose(native);
    cuda_check("plan_pose", cudaDeviceSynchronize());
    const auto end_time = std::chrono::steady_clock::now();

    uint8_t success = 0;
    float dt = 0.0F;
    float cost = 0.0F;
    int32_t selected_seed = 0;
    float position_error = 0.0F;
    success_device.download(&success, sizeof(success));
    dt_device.download(&dt, sizeof(dt));
    cost_device.download(&cost, sizeof(cost));
    selected_seed_device.download(&selected_seed, sizeof(selected_seed));
    position_error_device.download(&position_error, sizeof(position_error));

    if (success == 0) {
      std::ostringstream status;
      status << "pose candidate " << candidate_index << " failed";
      last_status = status.str();
      continue;
    }

    std::vector<float> position;
    std::vector<float> velocity;
    std::vector<float> acceleration;
    position_device.download(position);
    velocity_device.download(velocity);
    acceleration_device.download(acceleration);
    PlanOutcome outcome = make_outcome(
      std::move(position), std::move(velocity), std::move(acceleration), dof_count,
      steps, dt, joint_names);
    outcome.success = true;
    outcome.feasible = true;
    outcome.status = "success";
    outcome.goalset_index = static_cast<int>(candidate_index);
    outcome.seed_index = selected_seed;
    outcome.solver_time_s =
      std::chrono::duration<double>(end_time - start_time).count();
    outcome.motion_time_s = (steps > 0) ? static_cast<double>(steps - 1) * dt : 0.0;

    // The native benchmark reports the TIME-OPTIMIZED executable trajectory,
    // not the raw TrajOpt control points. plan_pose() leaves that trajectory
    // in the planner; fetch it so ROS and native report the same motion time.
    const auto timed = planner->timeopt_trajectory();
    if (timed.sample_count != nullptr) {
      int32_t timed_steps = 0;
      float timed_dt = 0.0F;
      float timed_duration = 0.0F;
      cuda_check(
        "timeopt/sample_count",
        cudaMemcpy(
          &timed_steps, timed.sample_count, sizeof(timed_steps),
          cudaMemcpyDeviceToHost));
      if (timed.dt != nullptr) {
        cuda_check(
          "timeopt/dt",
          cudaMemcpy(&timed_dt, timed.dt, sizeof(timed_dt), cudaMemcpyDeviceToHost));
      }
      if (timed.duration != nullptr) {
        cuda_check(
          "timeopt/duration",
          cudaMemcpy(
            &timed_duration, timed.duration, sizeof(timed_duration),
            cudaMemcpyDeviceToHost));
      }
      if (timed_steps > 0 && timed.position != nullptr &&
        timed.velocity != nullptr && timed.acceleration != nullptr)
      {
        const std::size_t count =
          static_cast<std::size_t>(timed_steps) * static_cast<std::size_t>(dof_count);
        std::vector<float> timed_position(count);
        std::vector<float> timed_velocity(count);
        std::vector<float> timed_acceleration(count);
        cuda_check(
          "timeopt/position",
          cudaMemcpy(
            timed_position.data(), timed.position, count * sizeof(float),
            cudaMemcpyDeviceToHost));
        cuda_check(
          "timeopt/velocity",
          cudaMemcpy(
            timed_velocity.data(), timed.velocity, count * sizeof(float),
            cudaMemcpyDeviceToHost));
        cuda_check(
          "timeopt/acceleration",
          cudaMemcpy(
            timed_acceleration.data(), timed.acceleration, count * sizeof(float),
            cudaMemcpyDeviceToHost));
        PlanOutcome timed_outcome = make_outcome(
          std::move(timed_position), std::move(timed_velocity),
          std::move(timed_acceleration), dof_count, timed_steps, timed_dt,
          joint_names);
        outcome.positions = std::move(timed_outcome.positions);
        outcome.velocity = std::move(timed_outcome.velocity);
        outcome.acceleration = std::move(timed_outcome.acceleration);
        outcome.motion_time_s =
          timed_duration > 0.0F ? static_cast<double>(timed_duration) :
          static_cast<double>(std::max(0, timed_steps - 1)) * timed_dt;
      }
    }
    outcome.position_error_mm = static_cast<double>(position_error) * 1000.0;
    outcome.candidates_generated = static_cast<std::uint32_t>(candidates.size());
    outcome.candidates_solved = 1;
    outcome.waypoints_planned = 1;
    if (!have_best || cost < best_cost) {
      best = std::move(outcome);
      best_cost = cost;
      have_best = true;
    }
  }

  if (!have_best) {
    PlanOutcome outcome;
    outcome.success = false;
    outcome.feasible = false;
    outcome.status = last_status;
    outcome.joint_names = joint_names;
    outcome.solver_time_s = 0.0;
    return outcome;
  }
  return best;
}

curobo_runtime::PlanOutcome CppBridge::Impl::plan_waypoints_impl(
  const curobo_runtime::PlanRequest & request, const OrderedWaypoints & waypoints)
{
  ensure_planner("plan_pose");
  const int dof_count = dof();
  const int tools = num_tools();
  if (static_cast<int>(request.start.size()) != dof_count) {
    fail("plan_pose", "start has " + std::to_string(request.start.size()) +
      " values, expected " + std::to_string(dof_count));
  }
  std::vector<float> start = request.start;
  {
    const int count = std::min(
      static_cast<int>(start.size()),
      static_cast<int>(std::min(joint_lower.size(), joint_upper.size())));
    for (int i = 0; i < count; ++i) {
      const float lower = static_cast<float>(joint_lower[static_cast<std::size_t>(i)]);
      const float upper = static_cast<float>(joint_upper[static_cast<std::size_t>(i)]);
      start[static_cast<std::size_t>(i)] =
        std::clamp(start[static_cast<std::size_t>(i)], lower, upper);
    }
  }
  if (waypoints.count <= 0) {
    fail("plan_pose", "waypoint_count must be positive");
  }
  if (waypoints.count > 256) {
    fail("plan_pose", "cpp_runtime supports at most 256 ordered waypoints");
  }

  std::vector<std::string> frames = waypoints.tool_frames;
  if (frames.empty()) {
    frames = request.tool_frames;
  }
  int active_slot = 0;
  std::vector<int> tool_slots;
  if (!waypoints.multi_tool && tools > 1) {
    // A single-tool ordered path on a multi-tool artifact is valid: the named
    // tool follows the path and every other tool holds its current FK pose.
    if (frames.size() > 1) {
      fail("plan_pose", "single-tool ordered waypoints must name at most one frame");
    }
    active_slot = 0;
    if (!frames.empty()) {
      const auto it = std::find(tool_frames.begin(), tool_frames.end(), frames[0]);
      if (it == tool_frames.end()) {
        fail("plan_pose", "unknown waypoint tool frame '" + frames[0] + "'");
      }
      active_slot = static_cast<int>(std::distance(tool_frames.begin(), it));
    }
    for (int i = 0; i < tools; ++i) {
      tool_slots.push_back(i);
    }
  } else {
    if (!frames.empty()) {
      if (static_cast<int>(frames.size()) != tools) {
        fail("plan_pose", "ordered waypoints require one frame per artifact tool (" +
          std::to_string(tools) + "), got " + std::to_string(frames.size()));
      }
      std::vector<bool> used(static_cast<std::size_t>(tools), false);
      for (const auto & frame : frames) {
        const auto it = std::find(tool_frames.begin(), tool_frames.end(), frame);
        if (it == tool_frames.end()) {
          fail("plan_pose", "unknown waypoint tool frame '" + frame + "'");
        }
        const int index = static_cast<int>(std::distance(tool_frames.begin(), it));
        if (used[static_cast<std::size_t>(index)]) {
          fail("plan_pose", "duplicate waypoint tool frame '" + frame + "'");
        }
        used[static_cast<std::size_t>(index)] = true;
        tool_slots.push_back(index);
      }
    } else {
      for (int i = 0; i < tools; ++i) {
        tool_slots.push_back(i);
      }
    }
  }

  const int values = static_cast<int>(waypoints.poses.data.size());
  const int expected_single = waypoints.count * 7;
  const int expected_multi = waypoints.count * tools * 7;
  if (!waypoints.multi_tool && values != expected_single) {
    fail("plan_pose", "waypoints must hold K*7 values (K=" +
      std::to_string(waypoints.count) + "), got " + std::to_string(values));
  }
  if (waypoints.multi_tool && values != expected_multi) {
    fail("plan_pose", "tool_waypoints must hold K*T*7 values (K=" +
      std::to_string(waypoints.count) + ", T=" + std::to_string(tools) +
      "), got " + std::to_string(values));
  }

  std::vector<std::array<float, 7>> held_tools;
  held_tools.reserve(static_cast<std::size_t>(tools));
  for (int tool = 0; tool < tools; ++tool) {
    held_tools.push_back(tool_pose(start, tool, "plan_pose"));
  }
  std::vector<float> positions(
    static_cast<std::size_t>(waypoints.count) * tools * 3);
  std::vector<float> quaternions(
    static_cast<std::size_t>(waypoints.count) * tools * 4);
  for (int k = 0; k < waypoints.count; ++k) {
    for (int slot = 0; slot < tools; ++slot) {
      std::array<float, 7> pose = held_tools[static_cast<std::size_t>(slot)];
      if (waypoints.multi_tool) {
        const float * row = waypoints.poses.data.data() +
          (static_cast<std::size_t>(k) * tools + static_cast<std::size_t>(slot)) * 7;
        pose = pose_from_values(row);
      } else if (slot == active_slot) {
        const float * row = waypoints.poses.data.data() +
          static_cast<std::size_t>(k) * 7;
        pose = pose_from_values(row);
      }
      const int destination_tool = tool_slots[static_cast<std::size_t>(slot)];
      const std::size_t destination =
        (static_cast<std::size_t>(k) * tools + static_cast<std::size_t>(destination_tool));
      positions[destination * 3 + 0] = pose[0];
      positions[destination * 3 + 1] = pose[1];
      positions[destination * 3 + 2] = pose[2];
      quaternions[destination * 4 + 0] = pose[3];
      quaternions[destination * 4 + 1] = pose[4];
      quaternions[destination * 4 + 2] = pose[5];
      quaternions[destination * 4 + 3] = pose[6];
    }
  }

  const auto info = planner->info();
  const int steps = info.trajectory_steps;
  const int knots = info.num_knots;
  DeviceBuffer start_device;
  DeviceBuffer waypoint_position_device;
  DeviceBuffer waypoint_quaternion_device;
  DeviceBuffer goal_pos_device;
  DeviceBuffer goal_quat_device;
  DeviceBuffer knots_device;
  DeviceBuffer position_device;
  DeviceBuffer velocity_device;
  DeviceBuffer acceleration_device;
  DeviceBuffer jerk_device;
  DeviceBuffer success_device;
  DeviceBuffer cost_device;
  DeviceBuffer collision_device;
  DeviceBuffer dt_device;
  DeviceBuffer position_error_device;
  DeviceBuffer orientation_error_device;
  DeviceBuffer used_rrt_device;
  DeviceBuffer rrt_iterations_device;
  DeviceBuffer selected_seed_device;
  DeviceBuffer planning_attempts_device;

  start_device.upload(start);
  waypoint_position_device.upload(positions);
  waypoint_quaternion_device.upload(quaternions);
  const std::size_t final_index =
    static_cast<std::size_t>(waypoints.count - 1) * tools;
  std::vector<float> final_positions(
    positions.begin() + static_cast<std::ptrdiff_t>(final_index * 3),
    positions.begin() + static_cast<std::ptrdiff_t>((final_index + tools) * 3));
  std::vector<float> final_quaternions(
    quaternions.begin() + static_cast<std::ptrdiff_t>(final_index * 4),
    quaternions.begin() + static_cast<std::ptrdiff_t>((final_index + tools) * 4));
  goal_pos_device.upload(final_positions);
  goal_quat_device.upload(final_quaternions);
  knots_device.allocate(static_cast<std::size_t>(knots) * dof_count * sizeof(float));
  position_device.allocate(static_cast<std::size_t>(steps) * dof_count * sizeof(float));
  velocity_device.allocate(static_cast<std::size_t>(steps) * dof_count * sizeof(float));
  acceleration_device.allocate(static_cast<std::size_t>(steps) * dof_count * sizeof(float));
  jerk_device.allocate(static_cast<std::size_t>(steps) * dof_count * sizeof(float));
  success_device.allocate(sizeof(uint8_t));
  cost_device.allocate(sizeof(float));
  collision_device.allocate(sizeof(float));
  dt_device.allocate(sizeof(float));
  position_error_device.allocate(sizeof(float));
  orientation_error_device.allocate(sizeof(float));
  used_rrt_device.allocate(sizeof(uint8_t));
  rrt_iterations_device.allocate(sizeof(int32_t));
  selected_seed_device.allocate(sizeof(int32_t));
  planning_attempts_device.allocate(sizeof(int32_t));

  curobo::robot::WaypointMotionPlanningRequest native;
  native.motion.goal_positions = static_cast<const float *>(goal_pos_device.get());
  native.motion.goal_quaternions = static_cast<const float *>(goal_quat_device.get());
  native.motion.current_position = static_cast<const float *>(start_device.get());
  native.motion.optimized_knots = static_cast<float *>(knots_device.get());
  native.motion.trajectory_position = static_cast<float *>(position_device.get());
  native.motion.trajectory_velocity = static_cast<float *>(velocity_device.get());
  native.motion.trajectory_acceleration = static_cast<float *>(acceleration_device.get());
  native.motion.trajectory_jerk = static_cast<float *>(jerk_device.get());
  native.motion.success = static_cast<uint8_t *>(success_device.get());
  native.motion.cost = static_cast<float *>(cost_device.get());
  native.motion.collision_cost = static_cast<float *>(collision_device.get());
  native.motion.trajectory_dt = static_cast<float *>(dt_device.get());
  native.motion.position_error = static_cast<float *>(position_error_device.get());
  native.motion.orientation_error = static_cast<float *>(orientation_error_device.get());
  native.motion.used_rrt = static_cast<uint8_t *>(used_rrt_device.get());
  native.motion.rrt_iterations = static_cast<int32_t *>(rrt_iterations_device.get());
  native.motion.selected_seed = static_cast<int32_t *>(selected_seed_device.get());
  native.motion.planning_attempts = static_cast<int32_t *>(planning_attempts_device.get());
  native.motion.max_attempts = std::max(1, request.max_attempts);
  native.motion.enable_graph_attempt = request.enable_graph_attempt;
  native.motion.finetune_attempts = 1;
  native.motion.finetune_dt_scale = 0.6F;
  native.positions = static_cast<const float *>(waypoint_position_device.get());
  native.quaternions = static_cast<const float *>(waypoint_quaternion_device.get());
  native.num_waypoints = waypoints.count;
  native.path_position_tolerance = static_cast<float>(waypoints.path_position_tolerance);
  native.path_orientation_tolerance =
    static_cast<float>(waypoints.path_orientation_tolerance);
  native.final_position_tolerance = static_cast<float>(waypoints.final_position_tolerance);
  native.final_orientation_tolerance =
    static_cast<float>(waypoints.final_orientation_tolerance);
  native.track_orientation = waypoints.track_orientation;
  native.outer_iterations = std::max(1, waypoints.outer_iterations);

  const auto start_time = std::chrono::steady_clock::now();
  planner->plan_pose_waypoints(native);
  cuda_check("plan_pose", cudaDeviceSynchronize());
  const auto end_time = std::chrono::steady_clock::now();

  uint8_t success = 0;
  float dt = 0.0F;
  float position_error = 0.0F;
  int32_t selected_seed = 0;
  success_device.download(&success, sizeof(success));
  dt_device.download(&dt, sizeof(dt));
  position_error_device.download(&position_error, sizeof(position_error));
  selected_seed_device.download(&selected_seed, sizeof(selected_seed));

  std::vector<float> position;
  std::vector<float> velocity;
  std::vector<float> acceleration;
  position_device.download(position);
  velocity_device.download(velocity);
  acceleration_device.download(acceleration);
  PlanOutcome outcome = make_outcome(
    std::move(position), std::move(velocity), std::move(acceleration), dof_count,
    steps, dt, joint_names);
  outcome.success = success != 0;
  outcome.feasible = outcome.success;
  outcome.status = outcome.success ? "success" : "waypoint planning failed";
  outcome.seed_index = selected_seed;
  outcome.solver_time_s =
    std::chrono::duration<double>(end_time - start_time).count();
  outcome.motion_time_s = (steps > 0) ? static_cast<double>(steps - 1) * dt : 0.0;
  outcome.position_error_mm = static_cast<double>(position_error) * 1000.0;
  outcome.candidates_generated = 1;
  outcome.candidates_solved = outcome.success ? 1 : 0;
  outcome.waypoints_planned = static_cast<std::uint32_t>(waypoints.count);
  outcome.waypoint_status.assign(
    static_cast<std::size_t>(waypoints.count), outcome.success ? 1 : 0);
  return outcome;
}

// ---------------------------------------------------------------------------
// CppBridge public wrappers
// ---------------------------------------------------------------------------

CppBridge::CppBridge(
  std::string artifact_directory, std::string robot_name,
  std::string robot_config_file, int device_ordinal, bool python_fallback)
: impl_(std::make_unique<Impl>())
{
  impl_->artifact_dir = std::move(artifact_directory);
  impl_->robot = std::move(robot_name);
  impl_->config_file = std::move(robot_config_file);
  impl_->device_ordinal = device_ordinal;
  impl_->python_fallback = python_fallback;
}

CppBridge::~CppBridge() = default;

const std::string & CppBridge::artifact_directory() const
{
  return impl_->artifact_dir;
}

const std::string & CppBridge::robot_name() const
{
  return impl_->robot;
}

curobo_runtime::RobotDescriptionOutcome CppBridge::robot_description(
  const std::string & robot_arg, const std::string & robot_config_file)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  curobo_runtime::RobotDescriptionOutcome outcome;
  if (!impl_->manifest.present) {
    impl_->bundle = inspect_artifact_directory(
      impl_->artifact_dir, impl_->robot, robot_config_file);
    impl_->manifest = impl_->bundle.manifest;
  }
  if (!impl_->manifest.present) {
    return outcome;
  }
  outcome.found = true;
  outcome.path = impl_->artifact_dir;
  outcome.name = impl_->manifest.robot.empty() ? robot_arg : impl_->manifest.robot;
  outcome.base_link = impl_->manifest.base_link;
  outcome.curobo_config = impl_->manifest.curobo_config;
  outcome.strategy = impl_->manifest.strategy;
  outcome.strategy_params = impl_->manifest.strategy_params;
  outcome.home_names = impl_->manifest.home_names;
  outcome.home_positions = impl_->manifest.home_positions;
  return outcome;
}

curobo_runtime::RobotHandle CppBridge::init(const std::string & cfg)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->initialized) {
    return 1;
  }
  try {
    if (!cfg.empty()) {
      const json::Value document = json::parse(cfg);
      impl_->artifact_dir = json_string(document, "artifact_dir", impl_->artifact_dir);
      impl_->robot = json_string(document, "robot", impl_->robot);
      impl_->config_file = json_string(document, "robot_config_file", impl_->config_file);
      impl_->device_ordinal = json_int(document, "device_ordinal", impl_->device_ordinal);
    }
    cuda_check("init", cudaSetDevice(impl_->device_ordinal));
    impl_->load_artifact();
    impl_->home_artifact_dir = impl_->artifact_dir;
    impl_->home_config_file = impl_->config_file;
    impl_->ensure_scene();
    if (impl_->python_fallback && std::getenv("CUROBO_DISABLE_PY_FALLBACK") == nullptr) {
      impl_->python = std::make_unique<curobo_runtime::PyBridge>();
      impl_->python_handle = impl_->python->init(cfg);
    }
    impl_->initialized = true;
    return 1;
  } catch (const std::exception & error) {
    impl_->init_error = error.what();
    throw;
  }
}

void CppBridge::shutdown(curobo_runtime::RobotHandle)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->python) {
    impl_->python->shutdown(impl_->python_handle);
    impl_->python.reset();
    impl_->python_handle = -1;
  }
  impl_->mapper.reset();
  impl_->mpc.reset();
  impl_->planner.reset();
  impl_->ik.reset();
  impl_->kinematics.reset();
  impl_->scene.reset();
  impl_->initialized = false;
}

bool CppBridge::initialized() const
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->initialized;
}

int CppBridge::interpreter_init_count() const
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->python ? impl_->python->interpreter_init_count() : 0;
}

void CppBridge::bind_lane(const std::string & name)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (std::find(impl_->lanes.begin(), impl_->lanes.end(), name) == impl_->lanes.end()) {
    impl_->lanes.push_back(name);
  }
  if (impl_->python) {
    impl_->python->bind_lane(name);
  }
}

void CppBridge::device_sync()
{
  cuda_check("device_sync", cudaDeviceSynchronize());
}

void CppBridge::sync_lane()
{
  cuda_check("sync_lane", cudaDeviceSynchronize());
}

void CppBridge::set_global_flags(bool cuda_graph_reset, bool debug_cuda_graphs)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->cuda_graph_reset = cuda_graph_reset;
  impl_->debug_cuda_graphs = debug_cuda_graphs;
  if (impl_->python) {
    impl_->python->set_global_flags(cuda_graph_reset, debug_cuda_graphs);
  }
}

std::vector<std::string> CppBridge::bound_lanes() const
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->lanes;
}

void CppBridge::prepare_solve_mode(curobo_runtime::RobotHandle, const std::string & mode)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->solve_mode = mode;
  if (!impl_->native_scene_valid) {
    impl_->ensure_python_planner("prepare_solve_mode");
    impl_->python->prepare_solve_mode(impl_->python_handle, mode);
  }
}

void CppBridge::robot_model_build(
  curobo_runtime::RobotHandle, const std::string & cfg_json)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("robot_model_build");
  if (cfg_json.empty()) {
    return;
  }
  try {
    const json::Value document = json::parse(cfg_json);
    const std::string config = json_string(document, "robot_config_file", impl_->config_file);
    if (config.empty()) {
      return;
    }
    const ArtifactBundle check = inspect_artifact_directory(
      impl_->artifact_dir, impl_->robot, config);
    if (!check.valid && check.hash_available && !check.hash_matches) {
      fail("robot_model_build", check.error);
    }
  } catch (const std::exception & error) {
    if (std::string(error.what()).find("hash") != std::string::npos) {
      throw;
    }
    // A malformed optional config is not a model-build failure; the artifact
    // already fixes the native model.
  }
  if (impl_->python) {
    impl_->python->robot_model_build(impl_->python_handle, cfg_json);
  }
}

curobo_runtime::TensorF32 CppBridge::fk(
  curobo_runtime::RobotHandle, const curobo_runtime::TensorF32 & q)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("fk");
  return impl_->fk_tensor(q, "fk");
}

curobo_runtime::LinkPosesOutcome CppBridge::link_poses(
  curobo_runtime::RobotHandle, const curobo_runtime::TensorF32 & q,
  const std::vector<std::string> & link_names)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("link_poses");
  if (q.shape.size() != 2 || static_cast<int>(q.shape[1]) != impl_->dof()) {
    fail("link_poses", "q must have shape [N,dof]");
  }
  const int batch = static_cast<int>(q.shape[0]);
  const int tools = impl_->num_tools();
  if (link_names.empty()) {
    return {};
  }
  std::vector<int> indices;
  std::vector<std::string> used_names;
  for (const auto & name : link_names) {
    const auto it = std::find(impl_->tool_frames.begin(), impl_->tool_frames.end(), name);
    if (it != impl_->tool_frames.end()) {
      indices.push_back(static_cast<int>(std::distance(impl_->tool_frames.begin(), it)));
      used_names.push_back(name);
    }
  }
  if (indices.empty()) {
    return {};
  }

  DeviceBuffer input;
  DeviceBuffer positions_device;
  DeviceBuffer quaternions_device;
  DeviceBuffer jacobians_device;
  input.upload(q.data);
  positions_device.allocate(static_cast<std::size_t>(batch) * tools * 3 * sizeof(float));
  quaternions_device.allocate(static_cast<std::size_t>(batch) * tools * 4 * sizeof(float));
  jacobians_device.allocate(
    static_cast<std::size_t>(batch) * tools * 6 * static_cast<std::size_t>(impl_->dof()) *
    sizeof(float));
  impl_->kinematics->forward(
    static_cast<const float *>(input.get()),
    static_cast<float *>(positions_device.get()),
    static_cast<float *>(quaternions_device.get()),
    static_cast<float *>(jacobians_device.get()), batch, 1, nullptr);
  cuda_check("link_poses", cudaDeviceSynchronize());
  std::vector<float> positions(static_cast<std::size_t>(batch) * tools * 3);
  std::vector<float> quaternions(static_cast<std::size_t>(batch) * tools * 4);
  positions_device.download(positions);
  quaternions_device.download(quaternions);

  LinkPosesOutcome outcome;
  outcome.link_names = used_names;
  outcome.poses.shape = {batch, static_cast<std::int64_t>(indices.size()), 7};
  outcome.poses.data.resize(
    static_cast<std::size_t>(batch) * indices.size() * 7);
  for (int i = 0; i < batch; ++i) {
    for (std::size_t slot = 0; slot < indices.size(); ++slot) {
      const std::size_t source = static_cast<std::size_t>(i) * tools +
        static_cast<std::size_t>(indices[slot]);
      const std::size_t destination =
        (static_cast<std::size_t>(i) * indices.size() + slot) * 7;
      outcome.poses.data[destination + 0] = positions[source * 3 + 0];
      outcome.poses.data[destination + 1] = positions[source * 3 + 1];
      outcome.poses.data[destination + 2] = positions[source * 3 + 2];
      outcome.poses.data[destination + 3] = quaternions[source * 4 + 0];
      outcome.poses.data[destination + 4] = quaternions[source * 4 + 1];
      outcome.poses.data[destination + 5] = quaternions[source * 4 + 2];
      outcome.poses.data[destination + 6] = quaternions[source * 4 + 3];
    }
  }
  return outcome;
}



curobo_runtime::IkOutcome CppBridge::ik(
  curobo_runtime::RobotHandle, const curobo_runtime::TensorF32 & poses,
  const std::string & tool_frame)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->native_scene_valid) {
    impl_->require_python("ik");
    return impl_->python->ik(impl_->python_handle, poses, tool_frame);
  }
  const std::string active = tool_frame.empty() ? impl_->tool_frames.front() : tool_frame;
  const auto full = impl_->fill_native_ik_goals(poses, {active}, true);
  return impl_->solve_ik(full, {}, nullptr, 1, "ik");
}

curobo_runtime::IkOutcome CppBridge::ik_multi(
  curobo_runtime::RobotHandle, const curobo_runtime::TensorF32 & poses,
  const std::vector<std::string> & tool_frames)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->native_scene_valid) {
    impl_->require_python("ik_multi");
    return impl_->python->ik_multi(impl_->python_handle, poses, tool_frames);
  }
  if (tool_frames.empty()) {
    fail("ik_multi", "no tool frames were provided");
  }
  // `ik_multi` describes ONE simultaneous multi-tool goal: rows are tools,
  // not batches.
  const auto full = impl_->fill_native_ik_goals(poses, tool_frames, false);
  return impl_->solve_ik(full, {}, nullptr, 1, "ik_multi");
}

void CppBridge::ik_config(
  curobo_runtime::RobotHandle, int batch_size, int num_seeds,
  bool self_collision_check, bool use_lm_seed, bool use_cuda_graph)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("ik_config");
  impl_->ik_settings.batch_size = std::max(1, batch_size);
  impl_->ik_settings.num_seeds = std::max(1, num_seeds);
  impl_->ik_settings.self_collision_check = self_collision_check;
  impl_->ik_settings.use_lm_seed = use_lm_seed;
  impl_->ik_settings.use_cuda_graph = use_cuda_graph;
  impl_->ik.reset();
  if (impl_->python) {
    impl_->python->ik_config(
      impl_->python_handle, batch_size, num_seeds, self_collision_check,
      use_lm_seed, use_cuda_graph);
  }
}

std::vector<std::uint8_t> CppBridge::validate(
  curobo_runtime::RobotHandle, const curobo_runtime::TensorF32 & q)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->native_scene_valid) {
    impl_->require_python("validate");
    return impl_->python->validate(impl_->python_handle, q);
  }
  impl_->ensure_planner("validate");
  if (q.shape.size() != 2 || static_cast<int>(q.shape[1]) != impl_->dof()) {
    fail("validate", "q must have shape [N,dof]");
  }
  const int batch = static_cast<int>(q.shape[0]);
  const int dof_count = impl_->dof();
  const int capacity = impl_->manifest.interpolation_buffer_size > 0 ?
    impl_->manifest.interpolation_buffer_size :
    impl_->planner->info().trajectory_steps;
  if (capacity <= 0) {
    fail("validate", "artifact has no interpolation buffer size");
  }
  const int lanes = std::max(1, impl_->planner_settings.num_trajopt_seeds);
  std::vector<std::uint8_t> valid(static_cast<std::size_t>(batch), 1);

  for (int offset = 0; offset < batch; offset += lanes) {
    const int count = std::min(lanes, batch - offset);
    std::vector<float> position(
      static_cast<std::size_t>(lanes) * capacity * dof_count, 0.0F);
    std::vector<float> velocity(position.size(), 0.0F);
    std::vector<float> acceleration(position.size(), 0.0F);
    std::vector<float> jerk(position.size(), 0.0F);
    for (int lane = 0; lane < lanes; ++lane) {
      const int source = offset + std::min(lane, count - 1);
      for (int step = 0; step < capacity; ++step) {
        const std::size_t destination =
          (static_cast<std::size_t>(lane) * capacity + static_cast<std::size_t>(step)) *
          dof_count;
        for (int joint = 0; joint < dof_count; ++joint) {
          position[destination + joint] =
            q.data[static_cast<std::size_t>(source) * dof_count + joint];
        }
      }
    }
    DeviceBuffer position_device;
    DeviceBuffer velocity_device;
    DeviceBuffer acceleration_device;
    DeviceBuffer jerk_device;
    position_device.upload(position);
    velocity_device.upload(velocity);
    acceleration_device.upload(acceleration);
    jerk_device.upload(jerk);
    curobo::robot::TrajectoryValidationRequest request;
    request.position = static_cast<const float *>(position_device.get());
    request.velocity = static_cast<const float *>(velocity_device.get());
    request.acceleration = static_cast<const float *>(acceleration_device.get());
    request.jerk = static_cast<const float *>(jerk_device.get());
    request.num_samples = capacity;
    const int32_t * violations = impl_->planner->validate(request);
    cuda_check("validate", cudaDeviceSynchronize());
    std::vector<int32_t> host(static_cast<std::size_t>(lanes));
    cuda_check(
      "validate",
      cudaMemcpy(host.data(), violations, host.size() * sizeof(int32_t),
        cudaMemcpyDeviceToHost));
    for (int lane = 0; lane < count; ++lane) {
      valid[static_cast<std::size_t>(offset + lane)] =
        host[static_cast<std::size_t>(lane)] == 0 ? 1 : 0;
    }
  }
  return valid;
}

std::vector<float> CppBridge::collision_distance(
  curobo_runtime::RobotHandle, const curobo_runtime::TensorF32 & q)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->native_scene_valid) {
    impl_->require_python("collision_distance");
    return impl_->python->collision_distance(impl_->python_handle, q);
  }
  unsupported("collision_distance");
}

curobo_runtime::CollisionAttribution CppBridge::collision_attribution(
  curobo_runtime::RobotHandle, const curobo_runtime::TensorF32 & q)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->native_scene_valid) {
    impl_->require_python("collision_attribution");
    return impl_->python->collision_attribution(impl_->python_handle, q);
  }
  unsupported("collision_attribution");
}

curobo_runtime::CacheBucketCounts CppBridge::collision_cache_bucket_counts(
  curobo_runtime::RobotHandle)
{
  unsupported("collision_cache_bucket_counts");
}

curobo_runtime::MpcCostBreakdown CppBridge::mpc_cost_breakdown(
  curobo_runtime::RobotHandle)
{
  curobo_runtime::MpcCostBreakdown breakdown;
  return breakdown;
}

void CppBridge::torch_synchronize()
{
  cuda_check("torch_synchronize", cudaDeviceSynchronize());
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->python) {
    impl_->python->torch_synchronize();
  }
}

curobo_runtime::PlanImageOutcome CppBridge::render_plan_image(
  const curobo_runtime::TensorF32 &, const curobo_runtime::TensorF32 &,
  const curobo_runtime::TensorF32 &, const std::vector<std::string> &, double)
{
  unsupported("render_plan_image");
}

curobo_runtime::SceneVersion CppBridge::scene_set(
  curobo_runtime::RobotHandle, const std::string & objects_json, bool clear_first)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("scene_set");
  impl_->ensure_scene();
  const auto version = impl_->scene->set_objects_json(objects_json, clear_first);
  if (clear_first && impl_->mapper && impl_->mapper->voxel_added) {
    // Scene::clear() also drops the voxel layer; re-attach the mapper grid from
    // its stable device ESDF buffer.
    impl_->mapper->voxel_added = false;
    impl_->update_native_voxel_grid();
  }
  if (impl_->scene) {
    for (const auto & name : impl_->attached_objects) {
      impl_->scene->set_object_enabled(name, false);
    }
  }
  if (impl_->python) {
    impl_->python->scene_set(impl_->python_handle, objects_json, clear_first);
  }
  return version;
}

std::string CppBridge::scene_get(curobo_runtime::RobotHandle)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("scene_get");
  if (!impl_->native_scene_valid) {
    impl_->require_python("scene_get");
    return impl_->python->scene_get(impl_->python_handle);
  }
  impl_->ensure_scene();
  return impl_->scene->get_objects_json();
}

curobo_runtime::JointLimitsOutcome CppBridge::joint_limits(curobo_runtime::RobotHandle)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("joint_limits");
  if (static_cast<int>(impl_->joint_lower.size()) != impl_->dof() ||
    static_cast<int>(impl_->joint_upper.size()) != impl_->dof())
  {
    fail("joint_limits",
      "artifact manifest has no complete joint_limits.lower/upper arrays");
  }
  curobo_runtime::JointLimitsOutcome outcome;
  outcome.joint_names = impl_->joint_names;
  outcome.positions.shape = {2, impl_->dof()};
  outcome.positions.data.resize(static_cast<std::size_t>(2 * impl_->dof()));
  for (int i = 0; i < impl_->dof(); ++i) {
    outcome.positions.data[static_cast<std::size_t>(i)] =
      static_cast<float>(impl_->joint_lower[static_cast<std::size_t>(i)]);
    outcome.positions.data[static_cast<std::size_t>(impl_->dof() + i)] =
      static_cast<float>(impl_->joint_upper[static_cast<std::size_t>(i)]);
  }
  outcome.locked_joint_names = impl_->locked_joint_names;
  outcome.locked_positions = impl_->locked_positions;
  outcome.unpinned_lock_joint_names = impl_->unpinned_lock_joint_names;
  return outcome;
}

void CppBridge::attach_object(
  curobo_runtime::RobotHandle, const std::string & name, const std::vector<float> & q)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("attach_object");
  if (name.empty()) {
    return;
  }
  impl_->ensure_scene();
  const SceneObjectRecord * record = impl_->scene->object(name);
  if (record == nullptr) {
    fail("attach_object", "unknown scene object '" + name + "'");
  }
  impl_->ensure_planner("attach_object");
  // The artifact exposes one attachment slot per arm; choose the slot whose
  // tool is closest to the object at the attach configuration. This mirrors
  // the Python bridge's nearest-attach-link choice without a mesh fit.
  const int tools = impl_->num_tools();
  int best_tool = 0;
  float best_distance = std::numeric_limits<float>::infinity();
  std::vector<std::array<float, 7>> tool_poses;
  tool_poses.reserve(static_cast<std::size_t>(tools));
  for (int t = 0; t < tools; ++t) {
    const auto pose = impl_->tool_pose(q, t, "attach_object");
    tool_poses.push_back(pose);
    const float dx = pose[0] - record->pose.position[0];
    const float dy = pose[1] - record->pose.position[1];
    const float dz = pose[2] - record->pose.position[2];
    const float distance = dx * dx + dy * dy + dz * dz;
    if (distance < best_distance) {
      best_distance = distance;
      best_tool = t;
    }
  }
  std::string link = best_tool == 0 ? "attached_object_left" :
    (best_tool == 1 ? "attached_object_right" :
    impl_->tool_frames[static_cast<std::size_t>(best_tool)]);
  if (std::find(impl_->link_names.begin(), impl_->link_names.end(), link) ==
    impl_->link_names.end())
  {
    link = impl_->tool_frames[static_cast<std::size_t>(best_tool)];
  }

  curobo::robot::SphereAttachment attachment;
  attachment.link_name = link;
  const auto tool_pose = tool_poses[static_cast<std::size_t>(best_tool)];
  const std::array<float, 7> object_pose{
    record->pose.position[0], record->pose.position[1], record->pose.position[2],
    record->pose.quaternion[0], record->pose.quaternion[1],
    record->pose.quaternion[2], record->pose.quaternion[3]};
  const auto relative = pose_multiply(pose_inverse(tool_pose), object_pose);
  attachment.object_in_link.position = {relative[0], relative[1], relative[2]};
  attachment.object_in_link.quaternion = {
    relative[3], relative[4], relative[5], relative[6]};
  // One enclosing sphere is conservative and keeps the attachment within the
  // per-link slot budget the artifact reserves for a grasp payload.
  curobo::robot::CollisionSphere sphere;
  sphere.center = {0.0F, 0.0F, 0.0F};
  if (record->type == 1) {  // sphere
    sphere.radius = std::max(0.005F, record->dims[0]);
  } else {
    // Inscribed sphere: never protrudes outside the object's own AABB, so it
    // cannot dip through the support plane at the grasp pose.
    sphere.radius = std::max(
      0.005F, 0.5F * std::min(
        record->dims[0], std::min(record->dims[1], record->dims[2])));
  }
  attachment.spheres.push_back(sphere);
  if (impl_->scene) {
    // The grasped obstacle stays in the scene list (so detach can restore it)
    // but must not collide with the attachment model or the gripper at the
    // start state. Python disables it on the solver's scene checker too.
    impl_->scene->set_object_enabled(name, false);
  }
  impl_->planner->attach_object(attachment);
  if (impl_->ik) {
    impl_->ik->attach_object(attachment);
  }
  impl_->attached_objects.insert(name);
  impl_->attach_links[name] = link;
  impl_->visual_attachments[name] = Impl::VisualAttachment{
    best_tool, relative, sphere.radius};
}

std::string CppBridge::detach_object(curobo_runtime::RobotHandle)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->attached_objects.empty()) {
    return {};
  }
  impl_->ensure_planner("detach_object");
  for (const auto & entry : impl_->attach_links) {
    impl_->planner->detach_object(entry.second);
    if (impl_->ik) {
      impl_->ik->detach_object(entry.second);
    }
  }
  if (impl_->scene) {
    for (const auto & name : impl_->attached_objects) {
      impl_->scene->set_object_enabled(name, true);
    }
  }
  std::string detached = *impl_->attached_objects.begin();
  impl_->attached_objects.clear();
  impl_->attach_links.clear();
  impl_->visual_attachments.clear();
  return detached;
}

curobo_runtime::SphereSetOutcome CppBridge::collision_spheres_with_attached(
  curobo_runtime::RobotHandle, const std::vector<float> & q)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_python("collision_spheres_with_attached");
  auto outcome = impl_->python->collision_spheres_with_attached(impl_->python_handle, q);
  if (impl_->visual_attachments.empty() || impl_->kinematics == nullptr ||
    static_cast<int>(q.size()) != impl_->dof() ||
    outcome.spheres.shape.size() != 2 || outcome.spheres.shape[1] != 4)
  {
    return outcome;
  }
  const std::size_t base_rows = static_cast<std::size_t>(outcome.spheres.shape[0]);
  for (const auto & entry : impl_->visual_attachments) {
    const auto & attachment = entry.second;
    const auto tool = impl_->tool_pose(q, attachment.tool_index, "collision_spheres");
    const auto world = pose_multiply(tool, attachment.object_in_link);
    outcome.spheres.data.push_back(world[0]);
    outcome.spheres.data.push_back(world[1]);
    outcome.spheres.data.push_back(world[2]);
    outcome.spheres.data.push_back(attachment.radius);
    outcome.attached.push_back(1);
    if (outcome.attached_local.shape.size() == 2 &&
      outcome.attached_local.shape[1] == 4)
    {
      outcome.attached_local.data.push_back(0.0F);
      outcome.attached_local.data.push_back(0.0F);
      outcome.attached_local.data.push_back(0.0F);
      outcome.attached_local.data.push_back(0.0F);
      outcome.attached_local.shape[0] =
        static_cast<std::int64_t>(outcome.attached_local.shape[0] + 1);
    }
  }
  outcome.spheres.shape[0] = static_cast<std::int64_t>(
    base_rows + impl_->visual_attachments.size());
  return outcome;
}

std::vector<std::string> CppBridge::tool_frames(curobo_runtime::RobotHandle)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("tool_frames");
  return impl_->tool_frames;
}

curobo_runtime::LinkCollisionOutcome CppBridge::set_link_collision(
  curobo_runtime::RobotHandle, const std::vector<std::string> & link_names, bool enabled)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("set_link_collision");
  curobo_runtime::LinkCollisionOutcome outcome;
  for (const auto & name : link_names) {
    if (std::find(impl_->link_names.begin(), impl_->link_names.end(), name) ==
      impl_->link_names.end())
    {
      outcome.unknown.push_back(name);
      continue;
    }
    impl_->link_collision[name] = enabled;
    if (impl_->planner) {
      impl_->planner->set_link_collision_enabled(name, enabled);
    }
    if (impl_->ik) {
      impl_->ik->set_link_collision_enabled(name, enabled);
    }
    outcome.applied.push_back(name);
  }
  if (impl_->python) {
    impl_->python->set_link_collision(impl_->python_handle, link_names, enabled);
  }
  return outcome;
}

void CppBridge::motion_planner_build(
  curobo_runtime::RobotHandle, const std::string & cfg_json)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("motion_planner_build");
  if (!cfg_json.empty()) {
    try {
      const json::Value document = json::parse(cfg_json);
      impl_->planner_settings.num_ik_seeds =
        json_int(document, "num_ik_seeds", impl_->planner_settings.num_ik_seeds);
      impl_->planner_settings.num_trajopt_seeds =
        json_int(document, "num_trajopt_seeds", impl_->planner_settings.num_trajopt_seeds);
      impl_->planner_settings.rrt_max_iterations =
        json_int(document, "rrt_max_iterations", impl_->planner_settings.rrt_max_iterations);
      impl_->planner_settings.max_attempts =
        json_int(document, "max_attempts", impl_->planner_settings.max_attempts);
      impl_->planner_settings.enable_graph_attempt =
        json_int(document, "enable_graph_attempt", impl_->planner_settings.enable_graph_attempt);
      impl_->planner_settings.max_goalset =
        json_int(document, "max_goalset", impl_->planner_settings.max_goalset);
      impl_->planner_settings.use_cuda_graph =
        document.bool_at("use_cuda_graph", impl_->planner_settings.use_cuda_graph);
      impl_->planner_settings.check_effort_limits =
        document.bool_at("check_effort_limits", impl_->planner_settings.check_effort_limits);
      impl_->planner_settings.collision_activation_distance =
        json_double(
        document, "collision_activation_distance",
        impl_->planner_settings.collision_activation_distance);
      impl_->planner_settings.interpolation_dt =
        json_double(document, "interpolation_dt", impl_->planner_settings.interpolation_dt);
      if (const json::Value * cache = document.find_path("collision_cache.voxel.dims")) {
        (void)cache;
      }
    } catch (const std::exception &) {
      // The cfg is a hint for native construction; unknown keys are ignored.
    }
  }
  impl_->motion_cfg_json = cfg_json;
  impl_->python_planner_built = false;
  impl_->planner.reset();
  impl_->mpc.reset();
  impl_->ensure_planner("motion_planner_build");
}

curobo_runtime::PlanOutcome CppBridge::plan_pose(
  curobo_runtime::RobotHandle, const curobo_runtime::PlanRequest & request)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("plan_pose");
  if (!impl_->native_scene_valid) {
    impl_->ensure_python_planner("plan_pose");
    return impl_->python->plan_pose(impl_->python_handle, request);
  }
  // Per-tool pose goals are not ordered waypoint paths. Route them straight to
  // the pose planner; the waypoint extractor probes optional fields on the
  // request and must not shadow a populated `tool_poses` goal.
  if (!request.tool_poses.data.empty()) {
    return impl_->plan_pose_impl(request);
  }
  // Ordered waypoint / Cartesian paths have a native implementation.
  const OrderedWaypoints waypoints = extract_ordered_waypoints(request);
  if (waypoints.present) {
    // Native is the only ordered-waypoint path. Do not fall back to the
    // Python planner here: its Torch/Dynamo recompile can block the service
    // for minutes during a task.
    return impl_->plan_waypoints_impl(request, waypoints);
  }
  // The native bridge implements this plan mode. Use it directly; the Python
  // planner is only the fallback for a missing native artifact (handled at
  // the top of this function), never a runtime recompile path.
  return impl_->plan_pose_impl(request);
}

curobo_runtime::PlanOutcome CppBridge::plan_cspace(
  curobo_runtime::RobotHandle, const curobo_runtime::PlanRequest & request)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->native_scene_valid) {
    impl_->ensure_python_planner("plan_cspace");
    return impl_->python->plan_cspace(impl_->python_handle, request);
  }
  return impl_->plan_cspace_impl(request);
}

void CppBridge::reset_seed(curobo_runtime::RobotHandle)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("reset_seed");
  if (!impl_->native_scene_valid) {
    impl_->require_python("reset_seed");
    impl_->python->reset_seed(impl_->python_handle);
    return;
  }
  // MotionPlanner resets its RRT seed at the start of every plan call
  // (cpp_runtime/robot/src/motion_planner_runtime.cpp). There is no public
  // standalone reset to call here, so this is intentionally a no-op.
}

void CppBridge::update_link_inertial(
  curobo_runtime::RobotHandle, const std::string & link_name, double mass)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("update_link_inertial");
  impl_->ensure_planner("update_link_inertial");
  curobo::robot::LinkInertialUpdate update;
  update.link_name = link_name;
  update.mass = static_cast<float>(mass);
  impl_->planner->update_link_inertial(update);
}

void CppBridge::robot_model_rebuild_locks(
  curobo_runtime::RobotHandle, const std::string & lock_json)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("robot_model_rebuild_locks");

  std::map<std::string, double> requested;
  if (!lock_json.empty() && lock_json != "{}") {
    const json::Value document = json::parse(lock_json);
    if (!document.is_object()) {
      fail("robot_model_rebuild_locks", "lock_joints override must be a JSON object");
    }
    for (const auto & [name, value] : document.as_object()) {
      if (!value.is_number()) {
        fail("robot_model_rebuild_locks",
          "lock_joints value for '" + name + "' is not a number");
      }
      requested[name] = value.as_double();
    }
  }

  // Keep the Python fallback consistent when it is enabled. The native model
  // itself is reloaded below.
  if (impl_->python) {
    impl_->python->robot_model_rebuild_locks(impl_->python_handle, lock_json);
  }

  // Empty override = restore the launch-time YAML locks. The home artifact was
  // exported with exactly those values, so switch back to it (no-op when the
  // bridge never left it).
  if (requested.empty()) {
    if (impl_->artifact_dir == impl_->home_artifact_dir) {
      return;
    }
    impl_->reload_artifact(impl_->home_artifact_dir, impl_->home_config_file);
    std::fprintf(
      stderr, "[curobo_cpp_runtime] restored launch artifact '%s' for lock_joints {}\n",
      impl_->home_artifact_dir.c_str());
    return;
  }

  if (requested == impl_->lock_map()) {
    return;
  }

  const auto selected = impl_->find_artifact_for_locks(requested);
  if (!selected) {
    std::ostringstream message;
    message << "no pre-exported artifact bundle under '"
            << std::filesystem::path(impl_->artifact_dir).parent_path().parent_path().string()
            << "' matches lock_joints " << lock_json
            << "; re-export the robot with those lock values";
    fail("robot_model_rebuild_locks", message.str());
  }
  impl_->reload_artifact(selected->directory, selected->config_file);
  std::fprintf(
    stderr, "[curobo_cpp_runtime] reloaded native artifact '%s' for lock_joints %s\n",
    selected->directory.c_str(), lock_json.c_str());
}

void CppBridge::robot_model_update_lock_joints(
  curobo_runtime::RobotHandle, const std::string & lock_json)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_python("robot_model_update_lock_joints");
  impl_->python->robot_model_update_lock_joints(impl_->python_handle, lock_json);
  // See robot_model_rebuild_locks: the native model is frozen at export. Do not
  // invalidate the native scene solely for a lock update; that would force
  // every OpenArm finger update onto the Python planner.
}

void CppBridge::configure_mapper(
  curobo_runtime::RobotHandle, const std::string & cfg_json)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("configure_mapper");
  // Headless benchmarks have no depth camera and explicitly do not want a
  // perception voxel layer in the collision world.
  if (std::getenv("CUROBO_DISABLE_NATIVE_MAPPER") != nullptr) {
    return;
  }
  impl_->ensure_scene();

  impl_->mapper_cfg_json = cfg_json;
  const std::string mapper_dir = impl_->artifact_dir + "/mapper";
  if (!std::filesystem::is_regular_file(mapper_dir + "/mapper_aot_manifest.json")) {
    // Hybrid fallback: no native mapper artifact, keep the Python mapper and
    // force planning onto Python so the ESDF and planner share one scene.
    impl_->require_python("configure_mapper");
    impl_->python->configure_mapper(impl_->python_handle, cfg_json);
    impl_->mark_python_scene("configure_mapper");
    return;
  }

  auto state = std::make_unique<Impl::MapperState>();
  if (curobo_mapper_create(
      mapper_dir.c_str(), impl_->device_ordinal, &state->runtime) != 0)
  {
    fail("configure_mapper", curobo_mapper_last_error());
  }
  if (curobo_mapper_get_device_view(state->runtime, &state->view) != 0) {
    fail("configure_mapper", curobo_mapper_last_error());
  }

  // The AOT mapper artifact fixes the image shape. A camera whose resolution
  // differs would integrate zero frames, so this is fatal rather than a
  // warning: re-export the mapper for the deployed camera instead of silently
  // running with no ESDF.
  int image_height = 0;
  int image_width = 0;
  if (curobo_mapper_config_int(state->runtime, "image_height", &image_height) != 0 ||
    curobo_mapper_config_int(state->runtime, "image_width", &image_width) != 0 ||
    image_height <= 0 || image_width <= 0)
  {
    fail("configure_mapper", "native mapper artifact has no image dimensions");
  }
  state->image_height = image_height;
  state->image_width = image_width;
  int requested_height = 0;
  int requested_width = 0;
  try {
    const json::Value document = json::parse(cfg_json);
    requested_height = json_int(document, "image_height", 0);
    requested_width = json_int(document, "image_width", 0);
  } catch (const std::exception &) {
    // No parseable camera hint: the artifact dimensions remain authoritative.
  }
  if (requested_height > 0 && requested_width > 0 &&
    (requested_height != image_height || requested_width != image_width))
  {
    fail(
      "configure_mapper",
      "camera image is " + std::to_string(requested_width) + "x" +
      std::to_string(requested_height) + " but the native mapper artifact is " +
      std::to_string(image_width) + "x" + std::to_string(image_height) +
      "; re-export the mapper with the camera resolution");
  }
  const std::size_t pixels =
    static_cast<std::size_t>(image_height) * static_cast<std::size_t>(image_width);
  state->depth.allocate(pixels * sizeof(float));
  state->rgb.allocate(pixels * 3 * sizeof(std::uint8_t));
  state->intrinsics.allocate(9 * sizeof(float));
  state->camera_positions.allocate(3 * sizeof(float));
  state->camera_quaternions.allocate(4 * sizeof(float));
  std::vector<std::uint8_t> black(pixels * 3, 0);
  state->rgb.upload(black);
  // The AOT artifact is exported from mapper.yaml, which does not carry the
  // node's time-decay knob; it therefore defaults to 1.0 (never decay). Apply
  // the live value here so stale observations fade exactly as they do in the
  // Python mapper.
  double decay_factor = 1.0;
  double minimum_tsdf_weight = 0.1;
  try {
    const json::Value document = json::parse(cfg_json);
    decay_factor = json_double(document, "decay_factor", decay_factor);
    minimum_tsdf_weight = json_double(
      document, "minimum_tsdf_weight", minimum_tsdf_weight);
  } catch (const std::exception &) {
  }
  if (curobo_mapper_set_tsdf_runtime_parameters(
      state->runtime, static_cast<float>(decay_factor),
      static_cast<float>(minimum_tsdf_weight)) != 0)
  {
    fail("configure_mapper", curobo_mapper_last_error());
  }
  impl_->mapper = std::move(state);
}

void CppBridge::warmup_mapper(curobo_runtime::RobotHandle)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("warmup_mapper");
  if (std::getenv("CUROBO_DISABLE_NATIVE_MAPPER") != nullptr) {
    return;
  }
  if (impl_->mapper == nullptr || !impl_->native_scene_valid) {
    impl_->ensure_python_mapper("warmup_mapper");
    impl_->mark_python_scene("warmup_mapper");
    return;
  }
  if (curobo_mapper_compute_esdf(
      impl_->mapper->runtime, impl_->mapper->view.esdf_dist_field, 0) != 0)
  {
    fail("warmup_mapper", curobo_mapper_last_error());
  }
  cuda_check("warmup_mapper", cudaDeviceSynchronize());
  impl_->update_native_voxel_grid();
}

bool CppBridge::esdf_refresh(curobo_runtime::RobotHandle)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("esdf_refresh");
  if (std::getenv("CUROBO_DISABLE_NATIVE_MAPPER") != nullptr) {
    return true;
  }
  if (impl_->mapper == nullptr || !impl_->native_scene_valid) {
    impl_->ensure_python_mapper("esdf_refresh");
    const bool updated = impl_->python->esdf_refresh(impl_->python_handle);
    impl_->mark_python_scene("esdf_refresh");
    return updated;
  }
  if (curobo_mapper_compute_esdf(
      impl_->mapper->runtime, impl_->mapper->view.esdf_dist_field, 0) != 0)
  {
    fail("esdf_refresh", curobo_mapper_last_error());
  }
  // The mapper enqueues its ESDF kernels on the default stream while the
  // planner replays CUDA graphs on a private non-blocking stream. Without this
  // hand-off the first plan can capture/replay before the ESDF is complete,
  // which showed up as RRT rejecting both terminal states.
  cuda_check("esdf_refresh", cudaDeviceSynchronize());
  impl_->update_native_voxel_grid();
  return true;
}

bool CppBridge::world_push(curobo_runtime::RobotHandle)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("world_push");
  if (impl_->mapper == nullptr || !impl_->native_scene_valid) {
    impl_->ensure_python_mapper("world_push");
    return impl_->python->world_push(impl_->python_handle);
  }
  // The native Scene is the live collision world; there is no Python-style
  // copy step to push. esdf_refresh already updated the voxel layer in place.
  return true;
}

curobo_runtime::VoxelGrid CppBridge::voxel_grid(curobo_runtime::RobotHandle)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("voxel_grid");
  if (std::getenv("CUROBO_DISABLE_NATIVE_MAPPER") != nullptr) {
    return curobo_runtime::VoxelGrid{};
  }
  if (impl_->mapper == nullptr || !impl_->native_scene_valid) {
    impl_->ensure_python_mapper("voxel_grid");
    return impl_->python->voxel_grid(impl_->python_handle);
  }
  const int nx = impl_->mapper->view.esdf_shape[0];
  const int ny = impl_->mapper->view.esdf_shape[1];
  const int nz = impl_->mapper->view.esdf_shape[2];
  const std::size_t count =
    static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
    static_cast<std::size_t>(nz);
  curobo_runtime::VoxelGrid grid;
  grid.shape = {nx, ny, nz};
  grid.resolution = impl_->mapper->view.esdf_voxel_size;
  // The AOT mapper calls the grid centre `origin`; the ROS VoxelGrid message
  // publishes the minimum corner so the dense reconstruction covers the whole
  // box (the same convention the node uses for mapper_grid_center).
  const double half_x = 0.5 * nx * grid.resolution;
  const double half_y = 0.5 * ny * grid.resolution;
  const double half_z = 0.5 * nz * grid.resolution;
  grid.origin = {
    impl_->mapper->view.origin[0] - half_x,
    impl_->mapper->view.origin[1] - half_y,
    impl_->mapper->view.origin[2] - half_z};
  grid.data.resize(count, 0);
  // The native AOT ESDF has no reliable sign and its unallocated cells are
  // zero, so thresholding the ESDF painted the whole free/unknown box. Extract
  // the observed TSDF surface band instead (surface_only=1 uses |TSDF| and is
  // independent of the broken sign), which matches what the sensor saw.
  const int capacity = 1000000;
  DeviceBuffer centers;
  DeviceBuffer pools;
  DeviceBuffer counter;
  centers.allocate(static_cast<std::size_t>(capacity) * 3 * sizeof(float));
  pools.allocate(static_cast<std::size_t>(capacity) * sizeof(int32_t));
  counter.allocate(sizeof(int32_t));
  for (int attempt = 0; attempt < 2; ++attempt) {
    // Attempt 0: solid occupied volume (Python parity). Attempt 1: observed
    // surface band, used only if the AOT artifact's TSDF sign is inverted.
    const int surface_only = attempt == 0 ? 0 : 1;
    const float threshold = attempt == 0 ? 0.0F :
      static_cast<float>(0.5 * grid.resolution);
    int required = 0;
    int written = 0;
    const int rc = curobo_mapper_extract_occupied(
      impl_->mapper->runtime,
      reinterpret_cast<std::uint64_t>(centers.get()),
      reinterpret_cast<std::uint64_t>(pools.get()),
      reinterpret_cast<std::uint64_t>(counter.get()),
      capacity, surface_only, threshold, 0, &required, &written);
    if (rc != 0) {
      break;
    }
    int count_host = 0;
    cuda_check(
      "voxel_grid",
      cudaMemcpy(&count_host, counter.get(), sizeof(int32_t),
        cudaMemcpyDeviceToHost));
    if (count_host <= 0) {
      continue;
    }
    const int take = std::min(count_host, capacity);
    std::vector<float> center_host(static_cast<std::size_t>(take) * 3);
    cuda_check(
      "voxel_grid",
      cudaMemcpy(center_host.data(), centers.get(),
        center_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < take; ++i) {
      const float x = center_host[static_cast<std::size_t>(i) * 3 + 0];
      const float y = center_host[static_cast<std::size_t>(i) * 3 + 1];
      const float z = center_host[static_cast<std::size_t>(i) * 3 + 2];
      const int ix = static_cast<int>(std::floor((x - grid.origin[0]) / grid.resolution));
      const int iy = static_cast<int>(std::floor((y - grid.origin[1]) / grid.resolution));
      const int iz = static_cast<int>(std::floor((z - grid.origin[2]) / grid.resolution));
      if (ix >= 0 && ix < nx && iy >= 0 && iy < ny && iz >= 0 && iz < nz) {
        grid.data[(static_cast<std::size_t>(ix) * ny + iy) * nz + iz] = 1;
      }
    }
    break;
  }
  // Python's voxel_grid also rasterizes the analytic scene primitives. Do the
  // same here so the native map shows the table and grasp objects even when a
  // face is not currently hit by the depth camera. Attached payloads are
  // skipped: they live on the robot now, not at their old world pose.
  if (impl_->scene) {
    for (const auto & entry : impl_->scene->objects()) {
      const std::string & name = entry.first;
      if (impl_->attached_objects.count(name) != 0) {
        continue;
      }
      const auto & record = entry.second;
      const auto inv = pose_inverse({
        record.pose.position[0], record.pose.position[1], record.pose.position[2],
        record.pose.quaternion[0], record.pose.quaternion[1],
        record.pose.quaternion[2], record.pose.quaternion[3]});
      const double radius = (record.type == 1)
        ? static_cast<double>(record.dims[0])
        : std::sqrt(
          static_cast<double>(record.dims[0] * record.dims[0]) +
          static_cast<double>(record.dims[1] * record.dims[1]) +
          static_cast<double>(record.dims[2] * record.dims[2])) * 0.5;
      const int min_x = std::max(0, static_cast<int>(std::floor(
        (record.pose.position[0] - radius - grid.origin[0]) / grid.resolution)));
      const int max_x = std::min(nx - 1, static_cast<int>(std::ceil(
        (record.pose.position[0] + radius - grid.origin[0]) / grid.resolution)));
      const int min_y = std::max(0, static_cast<int>(std::floor(
        (record.pose.position[1] - radius - grid.origin[1]) / grid.resolution)));
      const int max_y = std::min(ny - 1, static_cast<int>(std::ceil(
        (record.pose.position[1] + radius - grid.origin[1]) / grid.resolution)));
      const int min_z = std::max(0, static_cast<int>(std::floor(
        (record.pose.position[2] - radius - grid.origin[2]) / grid.resolution)));
      const int max_z = std::min(nz - 1, static_cast<int>(std::ceil(
        (record.pose.position[2] + radius - grid.origin[2]) / grid.resolution)));
      const std::array<float, 3> half{
        record.dims[0] * 0.5F, record.dims[1] * 0.5F, record.dims[2] * 0.5F};
      for (int ix = min_x; ix <= max_x; ++ix) {
        for (int iy = min_y; iy <= max_y; ++iy) {
          for (int iz = min_z; iz <= max_z; ++iz) {
            const std::array<float, 7> world{
              static_cast<float>((ix + 0.5) * grid.resolution + grid.origin[0]),
              static_cast<float>((iy + 0.5) * grid.resolution + grid.origin[1]),
              static_cast<float>((iz + 0.5) * grid.resolution + grid.origin[2]),
              1.0F, 0.0F, 0.0F, 0.0F};
            const auto local = pose_multiply(inv, world);
            bool inside = false;
            if (record.type == 1) {
              const double r = record.dims[0];
              inside = local[0] * local[0] + local[1] * local[1] +
                local[2] * local[2] <= r * r;
            } else if (record.type == 2 || record.type == 3) {
              inside = std::abs(local[0]) <= half[0] &&
                std::abs(local[1]) <= half[1] &&
                std::abs(local[2]) <= half[2];
            } else {
              inside = std::abs(local[0]) <= half[0] &&
                std::abs(local[1]) <= half[1] &&
                std::abs(local[2]) <= half[2];
            }
            if (inside) {
              grid.data[
                (static_cast<std::size_t>(ix) * ny + iy) * nz + iz] = 1;
            }
          }
        }
      }
    }
  }
  return grid;
}

int CppBridge::clear_voxel_map(curobo_runtime::RobotHandle)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("clear_voxel_map");
  if (impl_->mapper == nullptr || !impl_->native_scene_valid) {
    impl_->ensure_python_mapper("clear_voxel_map");
    const int cleared = impl_->python->clear_voxel_map(impl_->python_handle);
    impl_->mark_python_scene("clear_voxel_map");
    return cleared;
  }
  if (curobo_mapper_reset(impl_->mapper->runtime, 0) != 0) {
    fail("clear_voxel_map", curobo_mapper_last_error());
  }
  if (curobo_mapper_compute_esdf(
      impl_->mapper->runtime, impl_->mapper->view.esdf_dist_field, 0) != 0)
  {
    fail("clear_voxel_map", curobo_mapper_last_error());
  }
  cuda_check("clear_voxel_map", cudaDeviceSynchronize());
  impl_->update_native_voxel_grid();
  if (impl_->python && impl_->python_mapper_configured) {
    try {
      impl_->python->clear_voxel_map(impl_->python_handle);
    } catch (const std::exception &) {
    }
  }
  return impl_->mapper->esdf_count;
}

int CppBridge::clear_voxel_region(
  curobo_runtime::RobotHandle, const std::array<double, 3> & bounds_min,
  const std::array<double, 3> & bounds_max)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->mapper == nullptr || !impl_->native_scene_valid) {
    impl_->ensure_python_mapper("clear_voxel_region");
    const int cleared = impl_->python->clear_voxel_region(
      impl_->python_handle, bounds_min, bounds_max);
    impl_->mark_python_scene("clear_voxel_region");
    return cleared;
  }
  unsupported("clear_voxel_region");
}

void CppBridge::depth_integrate(
  curobo_runtime::RobotHandle, const curobo_runtime::TensorF32 & depth,
  const curobo_runtime::TensorF32 & intrinsics,
  const std::array<double, 7> & camera_pose)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("depth_integrate");
  if (std::getenv("CUROBO_DISABLE_NATIVE_MAPPER") != nullptr) {
    return;
  }
  if (impl_->mapper == nullptr || !impl_->native_scene_valid) {
    impl_->ensure_python_mapper("depth_integrate");
    impl_->python->depth_integrate(
      impl_->python_handle, depth, intrinsics, camera_pose);
    impl_->mark_python_scene("depth_integrate");
    return;
  }
  if (depth.shape.size() < 2) {
    fail("depth_integrate", "depth must have shape [H,W] or [1,H,W]");
  }
  const int height = static_cast<int>(depth.shape[depth.shape.size() - 2]);
  const int width = static_cast<int>(depth.shape[depth.shape.size() - 1]);
  if (height != impl_->mapper->image_height || width != impl_->mapper->image_width) {
    fail("depth_integrate", "depth image shape does not match the mapper artifact");
  }
  std::vector<float> intr(9, 0.0F);
  for (std::size_t i = 0; i < intr.size() && i < intrinsics.data.size(); ++i) {
    intr[i] = intrinsics.data[i];
  }
  impl_->mapper->depth.upload(depth.data);
  impl_->mapper->intrinsics.upload(intr);
  const std::array<float, 3> position{
    static_cast<float>(camera_pose[0]),
    static_cast<float>(camera_pose[1]),
    static_cast<float>(camera_pose[2])};
  const std::array<float, 4> quaternion{
    static_cast<float>(camera_pose[3]),
    static_cast<float>(camera_pose[4]),
    static_cast<float>(camera_pose[5]),
    static_cast<float>(camera_pose[6])};
  impl_->mapper->camera_positions.upload(
    position.data(), position.size() * sizeof(float));
  impl_->mapper->camera_quaternions.upload(
    quaternion.data(), quaternion.size() * sizeof(float));
  CuroboMapperFrame frame{};
  frame.depth_images = reinterpret_cast<std::uint64_t>(impl_->mapper->depth.get());
  frame.rgb_images = reinterpret_cast<std::uint64_t>(impl_->mapper->rgb.get());
  frame.intrinsics = reinterpret_cast<std::uint64_t>(impl_->mapper->intrinsics.get());
  frame.cam_positions =
    reinterpret_cast<std::uint64_t>(impl_->mapper->camera_positions.get());
  frame.cam_quaternions =
    reinterpret_cast<std::uint64_t>(impl_->mapper->camera_quaternions.get());
  frame.stream = 0;
  if (curobo_mapper_integrate(impl_->mapper->runtime, &frame) != 0) {
    fail("depth_integrate", curobo_mapper_last_error());
  }
}

void CppBridge::lidar_integrate(
  curobo_runtime::RobotHandle, const curobo_runtime::TensorF32 & range_image,
  const curobo_runtime::TensorF32 & poses, const curobo_runtime::TensorF32 & valid_range,
  const curobo_runtime::TensorF32 & elevation_range)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->ensure_python_mapper("lidar_integrate");
  impl_->python->lidar_integrate(
    impl_->python_handle, range_image, poses, valid_range, elevation_range);
  impl_->mark_python_scene("lidar_integrate");
}

curobo_runtime::TensorF32 CppBridge::robot_segment(
  curobo_runtime::RobotHandle, const curobo_runtime::TensorF32 & depth,
  const curobo_runtime::TensorF32 & intrinsics_meta,
  const std::array<double, 7> & cam_pose, const std::vector<float> & q,
  const std::string & masks_json)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_python("robot_segment");
  return impl_->python->robot_segment(
    impl_->python_handle, depth, intrinsics_meta, cam_pose, q, masks_json);
}

void CppBridge::mpc_build(
  curobo_runtime::RobotHandle, const std::string & cfg_json)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("mpc_build");
  if (!cfg_json.empty()) {
    try {
      const json::Value document = json::parse(cfg_json);
      impl_->planner_settings.interpolation_dt =
        json_double(document, "trajectory_dt", impl_->planner_settings.interpolation_dt);
    } catch (const std::exception &) {
    }
  }
  impl_->mpc.reset();
  impl_->ensure_mpc("mpc_build");
}

curobo_runtime::MpcGoalOutcome CppBridge::mpc_setup(
  curobo_runtime::RobotHandle, const std::vector<float> & start_q)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("mpc_setup");
  if (static_cast<int>(start_q.size()) != impl_->dof()) {
    fail("mpc_setup", "start state size does not match the artifact dof");
  }
  impl_->mpc_start = start_q;
  curobo_runtime::MpcGoalOutcome outcome;
  outcome.applied = true;
  try {
    const auto pose = impl_->fk_pose(start_q, "mpc_setup");
    outcome.has_ee = true;
    outcome.ee_position = {pose[0], pose[1], pose[2]};
  } catch (const std::exception &) {
    outcome.has_ee = false;
  }
  return outcome;
}

curobo_runtime::MpcStepOutcome CppBridge::mpc_step(
  curobo_runtime::RobotHandle, const curobo_runtime::ReactiveState & state, bool)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("mpc_step");
  auto & mpc = impl_->ensure_mpc("mpc_step");
  const int dof_count = impl_->dof();
  if (static_cast<int>(state.position.size()) != dof_count) {
    fail("mpc_step", "state.position size does not match the artifact dof");
  }
  std::vector<float> velocity(dof_count, 0.0F);
  std::vector<float> acceleration(dof_count, 0.0F);
  if (static_cast<int>(state.velocity.size()) == dof_count) {
    velocity = state.velocity;
  }
  if (static_cast<int>(state.acceleration.size()) == dof_count) {
    acceleration = state.acceleration;
  }
  std::vector<float> jerk(dof_count, 0.0F);
  DeviceBuffer state_device;
  DeviceBuffer velocity_device;
  DeviceBuffer acceleration_device;
  DeviceBuffer jerk_device;
  state_device.upload(state.position);
  velocity_device.upload(velocity);
  acceleration_device.upload(acceleration);
  jerk_device.upload(jerk);
  impl_->last_mpc_result = mpc.step(
    static_cast<const float *>(state_device.get()),
    static_cast<const float *>(velocity_device.get()),
    static_cast<const float *>(acceleration_device.get()),
    static_cast<const float *>(jerk_device.get()), 0);
  cuda_check("mpc_step", cudaDeviceSynchronize());

  curobo_runtime::MpcStepOutcome outcome;
  outcome.action.has_action = impl_->last_mpc_result.feasible;
  if (impl_->last_mpc_result.position != nullptr) {
    std::vector<float> position(static_cast<std::size_t>(dof_count));
    cuda_check(
      "mpc_step",
      cudaMemcpy(
        position.data(), impl_->last_mpc_result.position,
        position.size() * sizeof(float), cudaMemcpyDeviceToHost));
    outcome.action.position.shape = {1, dof_count};
    outcome.action.position.data = position;
  }
  if (impl_->last_mpc_result.velocity != nullptr) {
    std::vector<float> values(static_cast<std::size_t>(dof_count));
    cuda_check(
      "mpc_step",
      cudaMemcpy(
        values.data(), impl_->last_mpc_result.velocity,
        values.size() * sizeof(float), cudaMemcpyDeviceToHost));
    outcome.action.velocity.shape = {1, dof_count};
    outcome.action.velocity.data = values;
  }
  if (impl_->last_mpc_result.acceleration != nullptr) {
    std::vector<float> values(static_cast<std::size_t>(dof_count));
    cuda_check(
      "mpc_step",
      cudaMemcpy(
        values.data(), impl_->last_mpc_result.acceleration,
        values.size() * sizeof(float), cudaMemcpyDeviceToHost));
    outcome.action.acceleration.shape = {1, dof_count};
    outcome.action.acceleration.data = values;
  }
  try {
    const auto pose = impl_->fk_pose(state.position, "mpc_step");
    outcome.has_ee = true;
    outcome.ee_position = {pose[0], pose[1], pose[2]};
  } catch (const std::exception &) {
  }
  return outcome;
}

bool CppBridge::mpc_update_goal_pose(
  curobo_runtime::RobotHandle, const std::array<double, 7> & goal_raw, bool)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("mpc_update_goal_pose");
  auto & mpc = impl_->ensure_mpc("mpc_update_goal_pose");
  if (impl_->mpc_start.empty()) {
    fail("mpc_update_goal_pose", "mpc_setup must be called before applying a goal");
  }
  const int tools = impl_->num_tools();
  const auto goal = pose_from_list(goal_raw);
  const auto held = impl_->fk_pose(impl_->mpc_start, "mpc_update_goal_pose");
  std::vector<float> positions;
  std::vector<float> quaternions;
  std::vector<std::array<float, 7>> poses(static_cast<std::size_t>(tools), held);
  poses[0] = goal;
  fill_pose_arrays(poses, positions, quaternions);

  const auto info = impl_->planner->info();
  const int knots = info.num_knots;
  std::vector<float> seed(static_cast<std::size_t>(knots) * impl_->dof());
  for (int k = 0; k < knots; ++k) {
    for (int j = 0; j < impl_->dof(); ++j) {
      seed[static_cast<std::size_t>(k * impl_->dof() + j)] =
        impl_->mpc_start[static_cast<std::size_t>(j)];
    }
  }
  DeviceBuffer position_device;
  DeviceBuffer quaternion_device;
  DeviceBuffer seed_device;
  position_device.upload(positions);
  quaternion_device.upload(quaternions);
  seed_device.upload(seed);
  curobo::robot::MpcCartesianConfig config;
  config.constrain_path = false;
  mpc.reset_cartesian(
    static_cast<const float *>(position_device.get()),
    static_cast<const float *>(quaternion_device.get()),
    static_cast<const float *>(seed_device.get()), config);
  cuda_check("mpc_update_goal_pose", cudaDeviceSynchronize());
  impl_->mpc_goal_pose = goal;
  impl_->mpc_has_goal_pose = true;
  impl_->mpc_has_goal_state = false;
  return true;
}

void CppBridge::mpc_update_goal_state(
  curobo_runtime::RobotHandle, const std::vector<float> & q)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("mpc_update_goal_state");
  auto & mpc = impl_->ensure_mpc("mpc_update_goal_state");
  if (static_cast<int>(q.size()) != impl_->dof()) {
    fail("mpc_update_goal_state", "goal state size does not match the artifact dof");
  }
  if (impl_->mpc_start.empty()) {
    fail("mpc_update_goal_state", "mpc_setup must be called before applying a goal");
  }
  const auto info = impl_->planner->info();
  const int knots = info.num_knots;
  std::vector<float> seed(static_cast<std::size_t>(knots) * impl_->dof());
  for (int k = 0; k < knots; ++k) {
    for (int j = 0; j < impl_->dof(); ++j) {
      seed[static_cast<std::size_t>(k * impl_->dof() + j)] =
        impl_->mpc_start[static_cast<std::size_t>(j)];
    }
  }
  DeviceBuffer goal_device;
  DeviceBuffer seed_device;
  goal_device.upload(q);
  seed_device.upload(seed);
  mpc.reset(
    static_cast<const float *>(goal_device.get()),
    static_cast<const float *>(seed_device.get()));
  cuda_check("mpc_update_goal_state", cudaDeviceSynchronize());
  impl_->mpc_goal_state = q;
  impl_->mpc_has_goal_state = true;
  impl_->mpc_has_goal_pose = false;
}

void CppBridge::mpc_reseed(
  curobo_runtime::RobotHandle, const std::vector<float> & q)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("mpc_reseed");
  if (static_cast<int>(q.size()) != impl_->dof()) {
    fail("mpc_reseed", "seed state size does not match the artifact dof");
  }
  impl_->mpc_start = q;
  if (impl_->mpc_has_goal_state) {
    // Reset is the native way to replace the fingerprint seed.
    auto & mpc = impl_->ensure_mpc("mpc_reseed");
    const auto info = impl_->planner->info();
    std::vector<float> seed(
      static_cast<std::size_t>(info.num_knots) * static_cast<std::size_t>(impl_->dof()));
    for (int k = 0; k < info.num_knots; ++k) {
      for (int j = 0; j < impl_->dof(); ++j) {
        seed[static_cast<std::size_t>(k * impl_->dof() + j)] =
          q[static_cast<std::size_t>(j)];
      }
    }
    DeviceBuffer goal_device;
    DeviceBuffer seed_device;
    goal_device.upload(impl_->mpc_goal_state);
    seed_device.upload(seed);
    mpc.reset(
      static_cast<const float *>(goal_device.get()),
      static_cast<const float *>(seed_device.get()));
    cuda_check("mpc_reseed", cudaDeviceSynchronize());
  }
}

curobo_runtime::IkOutcome CppBridge::mpc_solve_goal_state(
  curobo_runtime::RobotHandle, const std::array<double, 7> & goal_raw,
  const std::vector<float> & seed_q)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const int tools = impl_->num_tools();
  if (tools > 1) {
    fail("mpc_solve_goal_state",
      "cpp_runtime MPC goal IK requires one target per tool frame");
  }
  TensorF32 poses;
  poses.shape = {1, 7};
  poses.data = {
    static_cast<float>(goal_raw[0]), static_cast<float>(goal_raw[1]),
    static_cast<float>(goal_raw[2]), static_cast<float>(goal_raw[3]),
    static_cast<float>(goal_raw[4]), static_cast<float>(goal_raw[5]),
    static_cast<float>(goal_raw[6])};
  return impl_->solve_ik(poses, {}, &seed_q, 1, "mpc_solve_goal_state");
}

void CppBridge::mpc_load_scene(
  curobo_runtime::RobotHandle, const std::string &)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("mpc_load_scene");
  // The native MPC shares MotionPlanner's live Scene; callers still update the
  // scene through scene_set. There is no separate world copy to load.
}

std::string CppBridge::mpc_metrics(curobo_runtime::RobotHandle)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  std::ostringstream out;
  out << "{\"feasible\":" << (impl_->last_mpc_result.feasible ? "true" : "false")
      << ",\"cost\":" << impl_->last_mpc_result.cost
      << ",\"collision_cost\":" << impl_->last_mpc_result.collision_cost
      << ",\"trajectory_dt\":" << impl_->last_mpc_result.trajectory_dt
      << ",\"selected_candidate\":" << impl_->last_mpc_result.selected_candidate
      << "}";
  return out.str();
}

void CppBridge::retarget_build(
  curobo_runtime::RobotHandle, const std::string &)
{
  unsupported("retarget_build");
}

void CppBridge::retarget_reset(curobo_runtime::RobotHandle)
{
  unsupported("retarget_reset");
}

curobo_runtime::MpcStepOutcome CppBridge::retarget_step(
  curobo_runtime::RobotHandle, const std::array<double, 7> &)
{
  unsupported("retarget_step");
}

void CppBridge::retarget_load_scene(
  curobo_runtime::RobotHandle, const std::string &)
{
  unsupported("retarget_load_scene");
}

}  // namespace curobo_cpp_runtime

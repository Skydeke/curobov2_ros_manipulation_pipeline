// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "curobo_robot_runtime/inverse_kinematics.h"
#include "curobo_robot_runtime/kinematics.h"
#include "curobo_robot_runtime/motion_planner.h"
#include "curobo_robot_runtime/mpc.h"
#include "curobo_robot_runtime/scene.h"
#include "curobo_robot_runtime/trajectory_optimization.h"
#include "curobo_mapper_runtime/api.h"
#include "curobo_runtime/py_bridge.hpp"

#include "curobo_cpp_runtime/artifact_bundle.hpp"
#include "curobo_cpp_runtime/cpp_bridge.hpp"
#include "curobo_cpp_runtime/detail/mini_json.hpp"
#include "curobo_cpp_runtime/native_scene.hpp"

#include "detail/device_buffer.hpp"
#include "detail/errors.hpp"
#include "detail/json_helpers.hpp"
#include "detail/pose_math.hpp"
#include "detail/sphere_fit.hpp"
#include "detail/waypoints.hpp"

namespace curobo_cpp_runtime
{

using curobo_runtime::IkOutcome;
using curobo_runtime::LinkPosesOutcome;
using curobo_runtime::PlanOutcome;
using curobo_runtime::PlanRequest;
using curobo_runtime::TensorF32;
using detail::cuda_check;
using detail::DeviceBuffer;
using detail::OrderedWaypoints;
using detail::extract_ordered_waypoints;
using detail::fail;
using detail::fill_pose_arrays;
using detail::fit_surface_spheres;
using detail::json_double;
using detail::json_int;
using detail::json_string;
using detail::pose_from_values;
using detail::pose_inverse;
using detail::pose_multiply;
using detail::unsupported;

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
  // Per-object collision layers used by the extended SetLinkCollision path.
  // Objects start on the common layer 1; an allowance moves one object to a
  // private bit and clears that bit on the allowed links only.
  std::map<std::string, std::uint32_t> object_layer_bits;
  std::map<std::string, std::uint32_t> link_blocked_bits;
  std::uint32_t next_object_layer_bit{2U};

  struct MapperState
  {
    CuroboMapperRuntime * runtime{nullptr};
    CuroboMapperDeviceView view{};
    int image_height{0};
    int image_width{0};
    int esdf_count{0};
    curobo::robot::ObstacleHandle voxel_handle{};
    bool voxel_added{false};
    std::array<double, 3> frame_position{{0.0, 0.0, 0.0}};
    std::array<double, 4> frame_quaternion{{1.0, 0.0, 0.0, 0.0}};
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
  //: Range-width fallback for continuous joints (the artifact does not record
  //: the URDF joint type). Their limits are a coordinate window, never a stop.
  std::vector<std::uint8_t> joint_continuous;
  std::vector<std::string> locked_joint_names;
  std::vector<double> locked_positions;
  std::vector<std::string> unpinned_lock_joint_names;

  std::unique_ptr<NativeScene> scene;
  std::unique_ptr<curobo::robot::Kinematics> kinematics;
  std::unique_ptr<curobo::robot::MotionPlanner> planner;
  std::unique_ptr<curobo::robot::InverseKinematics> ik;
  std::unique_ptr<curobo::robot::MpcController> mpc;

  // Native collision-sphere model copied from the artifact's IK manifest. This
  // is the exact sphere list / pair list / padding the native solvers use; the
  // Python fallback is only consulted when an older artifact lacks it.
  bool native_sphere_model_ready{false};
  curobo::robot::SphereModel native_sphere_model;
  std::map<std::string, std::vector<int>> native_sphere_slots;
  std::vector<std::string> native_sphere_link_names;
  DeviceBuffer native_sphere_q;
  DeviceBuffer native_sphere_output;

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
    // The SAME spheres the planner collides with, in the object frame. One
    // ball for the whole payload hid every corner the surface fit covers.
    std::vector<curobo::robot::CollisionSphere> spheres;
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
  bool mpc_accept_pending{false};
  int mpc_accept_knots{0};
  // Sample interval of the native MPC trajectory. The mpc_build config carries
  // the node's mpc_step_dt; keep it separate from planner_settings so selecting
  // MPC does not retime the open-loop planners when the user switches back.
  float mpc_trajectory_dt{0.0F};
  // Servo command interval. The optimized trajectory starts at the measured
  // state, so command sample 0 would hold the robot in place; the executor must
  // advance to the sample that covers this interval.
  float mpc_command_interval{0.0F};

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

  // Grid centre + orientation in the planner base frame: the AOT mapper origin
  // (a mapper-frame point) mapped through the mapper frame's pose.
  curobo::robot::Pose mapper_grid_pose() const
  {
    curobo::robot::Pose pose;
    const auto & q = mapper->frame_quaternion;
    const auto & t = mapper->frame_position;
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    const double vx = mapper->view.origin[0];
    const double vy = mapper->view.origin[1];
    const double vz = mapper->view.origin[2];
    const double tx = 2.0 * (y * vz - z * vy);
    const double ty = 2.0 * (z * vx - x * vz);
    const double tz = 2.0 * (x * vy - y * vx);
    pose.position = {
      static_cast<float>(vx + w * tx + (y * tz - z * ty) + t[0]),
      static_cast<float>(vy + w * ty + (z * tx - x * tz) + t[1]),
      static_cast<float>(vz + w * tz + (x * ty - y * tx) + t[2])};
    pose.quaternion = {
      static_cast<float>(w), static_cast<float>(x),
      static_cast<float>(y), static_cast<float>(z)};
    return pose;
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
    grid.pose = mapper_grid_pose();
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
    constexpr double kTwoPi = 6.283185307179586;
    joint_continuous.assign(joint_names.size(), 0);
    for (std::size_t i = 0; i < joint_names.size() &&
      i < joint_lower.size() && i < joint_upper.size(); ++i)
    {
      joint_continuous[i] =
        (std::isfinite(joint_lower[i]) && std::isfinite(joint_upper[i]) &&
        (joint_upper[i] - joint_lower[i]) >= (kTwoPi - 1e-6)) ? 1 : 0;
    }
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
    try {
      auto sphere_model = kinematics->sphere_model();
      if (sphere_model.num_spheres > 0 &&
        sphere_model.link_spheres.size() ==
        static_cast<std::size_t>(sphere_model.num_spheres) * 4 &&
        sphere_model.link_sphere_map.size() ==
        static_cast<std::size_t>(sphere_model.num_spheres))
      {
        native_sphere_link_names.assign(
          static_cast<std::size_t>(sphere_model.num_spheres), std::string());
        for (int i = 0; i < sphere_model.num_spheres; ++i) {
          const int link_index = sphere_model.link_sphere_map[static_cast<std::size_t>(i)];
          if (link_index >= 0 && static_cast<std::size_t>(link_index) < link_names.size()) {
            native_sphere_link_names[static_cast<std::size_t>(i)] =
              link_names[static_cast<std::size_t>(link_index)];
            native_sphere_slots[link_names[static_cast<std::size_t>(link_index)]].push_back(i);
          }
        }
        native_sphere_model = std::move(sphere_model);
        native_sphere_model_ready = true;
      }
    } catch (const std::exception &) {
      // Artifact predates the native sphere API; the Python fallback still
      // provides the RViz model.
      native_sphere_model_ready = false;
    }
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
    mapper.reset();
    mpc.reset();
    planner.reset();
    ik.reset();
    kinematics.reset();
    scene.reset();
    native_sphere_model_ready = false;
    native_sphere_model = curobo::robot::SphereModel{};
    native_sphere_slots.clear();
    native_sphere_link_names.clear();
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
    // Ignore sub-micrometre numerical overshoot at joint limits; a measured
    // joint can sit at -2e-13 for a 0.0 bound and must not invalidate the
    // start state.
    config.trajectory_optimization.constraint_tolerance = 1e-4F;
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
      if (mpc_trajectory_dt > 0.0F) {
        config.trajectory_dt = mpc_trajectory_dt;
      }
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

}  // namespace curobo_cpp_runtime

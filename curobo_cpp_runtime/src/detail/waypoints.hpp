// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "curobo_runtime/bridge.hpp"

namespace curobo_cpp_runtime
{
namespace detail
{

template<typename T, typename = void>
struct has_waypoints_field : std::false_type {};
template<typename T>
struct has_waypoints_field<T, std::void_t<decltype(std::declval<T>().waypoints)>>
  : std::true_type {};

template<typename T, typename = void>
struct has_waypoint_poses_field : std::false_type {};
template<typename T>
struct has_waypoint_poses_field<T,
  std::void_t<decltype(std::declval<T>().waypoint_poses)>> : std::true_type {};

template<typename T, typename = void>
struct has_tool_waypoints_field : std::false_type {};
template<typename T>
struct has_tool_waypoints_field<T,
  std::void_t<decltype(std::declval<T>().tool_waypoints)>> : std::true_type {};

template<typename T, typename = void>
struct has_waypoint_count_field : std::false_type {};
template<typename T>
struct has_waypoint_count_field<T,
  std::void_t<decltype(std::declval<T>().waypoint_count)>> : std::true_type {};

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
  curobo_runtime::TensorF32 poses;  // [K,7] single tool, [K*T,7] multi tool
  std::vector<std::string> tool_frames;
  double path_position_tolerance{0.05};
  double path_orientation_tolerance{0.261799388};
  double final_position_tolerance{0.005};
  double final_orientation_tolerance{0.05};
  bool track_orientation{true};
  int outer_iterations{4};
};

// Field precedence for poses: tool_waypoints > waypoints > waypoint_poses.
template<typename Request>
OrderedWaypoints extract_ordered_waypoints(const Request & request)
{
  OrderedWaypoints waypoints;
  if constexpr (has_tool_waypoints_field<Request>::value) {
    if constexpr (std::is_same_v<decltype(request.tool_waypoints),
        curobo_runtime::TensorF32>)
    {
      if (!request.tool_waypoints.data.empty()) {
        waypoints.present = true;
        waypoints.multi_tool = true;
        waypoints.poses = request.tool_waypoints;
      }
    }
  }
  if constexpr (has_waypoints_field<Request>::value) {
    if constexpr (std::is_same_v<decltype(request.waypoints),
        curobo_runtime::TensorF32>)
    {
      if (!waypoints.present && !request.waypoints.data.empty()) {
        waypoints.present = true;
        waypoints.poses = request.waypoints;
      }
    }
  }
  if constexpr (has_waypoint_poses_field<Request>::value) {
    if constexpr (std::is_same_v<decltype(request.waypoint_poses),
        curobo_runtime::TensorF32>)
    {
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
    waypoints.path_position_tolerance =
      static_cast<double>(request.path_position_tolerance);
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

}  // namespace detail
}  // namespace curobo_cpp_runtime

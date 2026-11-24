// SPDX-License-Identifier: Apache-2.0
#ifndef CUTAMP_ROS__PYTHON_BACKEND_HPP_
#define CUTAMP_ROS__PYTHON_BACKEND_HPP_

// cuTAMP is Python. This header owns all pybind11 use for the component.
// It never constructs an interpreter: the cuRobo PyBridge owns the single
// process-wide CPython and must be loaded before this component.

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace cutamp_ros
{
namespace py = pybind11;

// Robot-side data. These values are never hardcoded for a specific robot; the
// ROS node fills them from its parameters.
struct RobotSettings
{
  std::string robot_config_path;
  std::string urdf_path;
  std::string tool_frame;
  std::vector<double> tool_from_ee;      // 16 row-major values
  std::string gripper_spheres;           // path to an (n, 4) .npy in tool frame
  int ik_batch_size = 1024;
  int ik_num_seeds = 12;
  bool self_collision_check = true;
  bool use_cuda_graph = true;
  int coll_n_spheres = 50;
  double coll_sphere_radius = 0.005;
  int device = 0;
};

struct SceneObjectSpec
{
  std::string name;
  int type = 0;
  std::array<double, 3> dims{0.0, 0.0, 0.0};
  std::array<double, 7> pose{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};  // xyz qx qy qz qw
};

struct PlanRequest
{
  std::vector<SceneObjectSpec> objects;
  std::vector<std::string> movable_names;
  std::vector<std::string> goal_atoms;
  std::vector<double> q_init;
  int num_particles = 0;
  int num_opt_steps = 0;
  int num_initial_plans = 0;
  double max_loop_dur = 0.0;
  // Optional external 6-DOF grasp candidates, parallel arrays. Poses are the
  // arm tool frame in the robot base frame.
  std::vector<std::string> grasp_object_names;
  std::vector<std::array<double, 7>> grasp_tool_poses;  // xyz qx qy qz qw
  std::vector<double> grasp_scores;
};

struct PlanResult
{
  bool success = false;
  std::string message;
  // One row-major 4x4 tool pose per goalset (the bridge emits exactly one
  // pose per ordered segment).
  std::vector<std::array<double, 16>> poses;
  std::vector<int> gripper_close;
  std::vector<int> action_index;
  int num_satisfying = 0;
};

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#endif
class CuTampBackend
{
public:
  CuTampBackend() = default;

  CuTampBackend(const CuTampBackend &) = delete;
  CuTampBackend & operator=(const CuTampBackend &) = delete;

  ~CuTampBackend()
  {
    reset();
  }

  bool ensure_initialized(const RobotSettings & settings, std::string & error)
  {
    if (Py_IsInitialized() == 0) {
      error =
        "embedded Python is not initialized; load the cuRobo component before "
        "cutamp_ros::CuTampComponent";
      return false;
    }

    py::gil_scoped_acquire gil;
    if (initialized_) {
      return true;
    }

    try {
      bridge_ = py::module_::import("cutamp_ros.bridge");
      bridge_.attr("ensure_initialized")(to_py_settings(settings));
      initialized_ = true;
      return true;
    } catch (const py::error_already_set & exc) {
      error = exc.what();
    } catch (const std::exception & exc) {
      error = exc.what();
    }

    clear_python_objects();
    return false;
  }

  bool plan(
    const PlanRequest & request,
    PlanResult & out,
    std::string & error)
  {
    if (Py_IsInitialized() == 0) {
      error =
        "embedded Python is not initialized; load the cuRobo component before "
        "cutamp_ros::CuTampComponent";
      return false;
    }
    if (!initialized_) {
      error = "cuTAMP backend has not been initialized";
      return false;
    }

    py::gil_scoped_acquire gil;
    try {
      py::dict response = bridge_.attr("plan")(to_py_request(request)).cast<py::dict>();

      out = PlanResult();
      out.success = response["success"].cast<bool>();
      out.message = response["message"].cast<std::string>();
      out.num_satisfying = response["num_satisfying"].cast<int>();

      py::object goalsets = response["goalsets"];
      for (py::handle goalset : goalsets) {
        std::array<double, 16> matrix{};
        std::size_t index = 0;
        for (py::handle value : goalset) {
          if (index < matrix.size()) {
            matrix[index] = value.cast<double>();
          }
          ++index;
        }
        if (index != matrix.size()) {
          error = "cuTAMP bridge returned a goalset without 16 pose values";
          out = PlanResult();
          return false;
        }
        out.poses.push_back(matrix);
      }

      out.gripper_close = response["gripper_close"].cast<std::vector<int>>();
      out.action_index = response["action_index"].cast<std::vector<int>>();

      if (out.gripper_close.size() != out.poses.size() ||
        out.action_index.size() != out.poses.size())
      {
        error = "cuTAMP bridge returned parallel arrays of different lengths";
        out = PlanResult();
        return false;
      }
      return out.success;
    } catch (const py::error_already_set & exc) {
      error = exc.what();
    } catch (const std::exception & exc) {
      error = exc.what();
    }
    out = PlanResult();
    return false;
  }

  std::string memory_info() const
  {
    if (Py_IsInitialized() == 0) {
      return "python not initialized";
    }
    py::gil_scoped_acquire gil;
    try {
      py::tuple info = py::module_::import("torch").attr("cuda").attr("mem_get_info")();
      const double free_mb = py::cast<double>(info[0]) / (1024.0 * 1024.0);
      const double total_mb = py::cast<double>(info[1]) / (1024.0 * 1024.0);
      return std::to_string(static_cast<int>(free_mb)) + " MiB free / " +
             std::to_string(static_cast<int>(total_mb)) + " MiB total";
    } catch (const py::error_already_set &) {
      return "unavailable";
    } catch (const std::exception &) {
      return "unavailable";
    }
  }

private:
  static py::dict to_py_settings(const RobotSettings & settings)
  {
    py::dict out;
    out["robot_config_path"] = settings.robot_config_path;
    out["urdf_path"] = settings.urdf_path;
    out["tool_frame"] = settings.tool_frame;
    out["tool_from_ee"] = settings.tool_from_ee;
    out["gripper_spheres"] = settings.gripper_spheres;
    out["ik_batch_size"] = settings.ik_batch_size;
    out["ik_num_seeds"] = settings.ik_num_seeds;
    out["self_collision_check"] = settings.self_collision_check;
    out["use_cuda_graph"] = settings.use_cuda_graph;
    out["coll_n_spheres"] = settings.coll_n_spheres;
    out["coll_sphere_radius"] = settings.coll_sphere_radius;
    out["device"] = settings.device;
    return out;
  }

  static py::dict to_py_request(const PlanRequest & request)
  {
    py::dict out;
    py::list objects;
    for (const auto & object : request.objects) {
      py::dict entry;
      entry["name"] = object.name;
      entry["type"] = object.type;
      entry["dims"] = py::make_tuple(object.dims[0], object.dims[1], object.dims[2]);
      entry["pose"] = py::make_tuple(
        object.pose[0], object.pose[1], object.pose[2], object.pose[3],
        object.pose[4], object.pose[5], object.pose[6]);
      objects.append(entry);
    }
    out["objects"] = objects;
    out["movable_names"] = request.movable_names;
    out["goal_atoms"] = request.goal_atoms;
    out["q_init"] = request.q_init;
    out["num_particles"] = request.num_particles;
    out["num_opt_steps"] = request.num_opt_steps;
    out["num_initial_plans"] = request.num_initial_plans;
    out["max_loop_dur"] = request.max_loop_dur;

    py::list grasp_tool_poses;
    for (const auto & pose : request.grasp_tool_poses) {
      grasp_tool_poses.append(py::make_tuple(
        pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6]));
    }
    out["grasp_object_names"] = request.grasp_object_names;
    out["grasp_tool_poses"] = grasp_tool_poses;
    out["grasp_scores"] = request.grasp_scores;
    return out;
  }

  void clear_python_objects()
  {
    bridge_ = py::object();
    initialized_ = false;
  }

  void reset()
  {
    if (Py_IsInitialized() == 0) {
      return;
    }
    py::gil_scoped_acquire gil;
    clear_python_objects();
  }

  py::object bridge_;
  bool initialized_ = false;
};
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

}  // namespace cutamp_ros

#endif  // CUTAMP_ROS__PYTHON_BACKEND_HPP_

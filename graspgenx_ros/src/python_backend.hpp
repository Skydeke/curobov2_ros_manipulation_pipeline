// SPDX-License-Identifier: Apache-2.0
#ifndef GRASPGENX_ROS__PYTHON_BACKEND_HPP_
#define GRASPGENX_ROS__PYTHON_BACKEND_HPP_

// GraspGenX is Python.  This header owns all pybind11 use for the component.
// It never constructs an interpreter: the cuRobo PyBridge owns the single
// process-wide CPython and must be loaded before this component.

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <array>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace graspgenx_ros
{
namespace py = pybind11;

struct BackendSettings
{
  std::string checkpoint_dir;
  std::string gripper_name;
  std::string gripper_assets_dir;

  std::array<double, 3> sweep_extents_open{0.08, 0.02, 0.06};
  std::array<double, 3> sweep_offset_open{0.0, 0.0, 0.05};
  std::array<double, 3> sweep_extents_mid{0.04, 0.02, 0.06};
  std::array<double, 3> sweep_offset_mid{0.0, 0.0, 0.05};
  int gripper_type = 0;
  double fingertip_depth = 0.0;

  bool use_fast_inference = false;
  int diffusion_grasps = 100;
  int obb_grasps = 50;

  std::string planner = "diffusion";
  double score_threshold = -1.0;
  int num_grasps = 100;

  bool filter_collisions = false;
  double collision_threshold = 0.02;
  std::string gripper_colmesh;
  int min_free_mb = 0;
};

struct RawGrasps
{
  std::vector<std::array<double, 16>> poses;
  std::vector<double> scores;
};

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#endif
class GraspGenXBackend
{
public:
  GraspGenXBackend() = default;

  GraspGenXBackend(const GraspGenXBackend &) = delete;
  GraspGenXBackend & operator=(const GraspGenXBackend &) = delete;

  ~GraspGenXBackend()
  {
    reset();
  }

  bool ensure_initialized(const BackendSettings & settings, std::string & error)
  {
    if (Py_IsInitialized() == 0) {
      error =
        "embedded Python is not initialized; load the cuRobo component before "
        "graspgenx_ros::GraspGenXComponent";
      return false;
    }

    py::gil_scoped_acquire gil;
    if (initialized_) {
      return true;
    }

    try {
      std::filesystem::path checkpoint_root(settings.checkpoint_dir);
      std::filesystem::path gen_dir;
      std::filesystem::path dis_dir;

      if (!settings.checkpoint_dir.empty()) {
        if (
          std::filesystem::is_directory(checkpoint_root / "gen") &&
          std::filesystem::is_directory(checkpoint_root / "dis"))
        {
          gen_dir = checkpoint_root / "gen";
          dis_dir = checkpoint_root / "dis";
        } else if (
          checkpoint_root.filename() == "gen" &&
          std::filesystem::is_directory(checkpoint_root.parent_path() / "dis"))
        {
          gen_dir = checkpoint_root;
          dis_dir = checkpoint_root.parent_path() / "dis";
        }
      }

      if (gen_dir.empty() || dis_dir.empty()) {
        error =
          "checkpoint_dir must be a checkpoint root containing gen/ and dis/ "
          "(or point directly at gen/ with a sibling dis/)";
        return false;
      }

      py::dict env = py::module_::import("os").attr("environ");
      env[py::str("GRASPGENX_CHECKPOINT_DIR")] = py::str(settings.checkpoint_dir);
      env[py::str("GRASPGENX_GRIPPER_CFG_DIR")] = py::str(
        settings.gripper_assets_dir.empty() ? "/opt/graspgenx" :
        settings.gripper_assets_dir);

      py::object load_model_cfg =
        py::module_::import("graspgenx.utils.checkpoint_io").attr("load_model_cfg");
      py::object grasp_server = py::module_::import("graspgenx.grasp_server");      py::object sampler_class = grasp_server.attr("GraspGenXSampler");
      py::object load_model = grasp_server.attr("load_grasp_gen_model");
      py::object run_planner =
        py::module_::import("graspgenx.samplers").attr("run_planner_on_object");
      py::object sweep_params_class =
        py::module_::import("graspgenx.serving.types").attr("SweepVolumeParams");

      cfg_ = load_model_cfg(gen_dir.string(), dis_dir.string());
      model_ = load_model(cfg_);

      if (!settings.gripper_name.empty()) {
        sampler_ = sampler_class(
          cfg_, settings.gripper_name,
          py::arg("assets_dir") = settings.gripper_assets_dir,
          py::arg("use_fast_inference") = settings.use_fast_inference,
          py::arg("diffusion_grasps") = settings.diffusion_grasps,
          py::arg("obb_grasps") = settings.obb_grasps,
          py::arg("model") = model_);
      } else {
        py::object depth = settings.fingertip_depth > 0.0 ?
          py::cast(settings.fingertip_depth) : py::none();
        py::object params = sweep_params_class(
          py::arg("extents_open") = to_numpy(settings.sweep_extents_open),
          py::arg("offset_open") = to_numpy(settings.sweep_offset_open),
          py::arg("extents_mid") = to_numpy(settings.sweep_extents_mid),
          py::arg("offset_mid") = to_numpy(settings.sweep_offset_mid),
          py::arg("gripper_type") = settings.gripper_type,
          py::arg("fingertip_depth") = depth);
        sampler_ = sampler_class.attr("from_sweep_volume")(
          cfg_, params,
          py::arg("model") = model_,
          py::arg("use_fast_inference") = settings.use_fast_inference,
          py::arg("diffusion_grasps") = settings.diffusion_grasps,
          py::arg("obb_grasps") = settings.obb_grasps);
      }

      run_planner_ = run_planner;
      if (settings.filter_collisions) {
        try {
          filter_colliding_ =
            py::module_::import("graspgenx.utils.collision_filter").attr(
            "filter_colliding_grasps");
        } catch (const py::error_already_set &) {
          filter_colliding_ = py::object();
        }
      }
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

  bool run(
    const BackendSettings & settings,
    const std::vector<std::array<float, 3>> & object_points,
    const std::vector<std::array<float, 3>> & scene_points,
    RawGrasps & out,
    std::string & warning,
    std::string & error)
  {    if (Py_IsInitialized() == 0) {
      error =
        "embedded Python is not initialized; load the cuRobo component before "
        "graspgenx_ros::GraspGenXComponent";
      return false;
    }
    if (!initialized_) {
      error = "GraspGenX backend has not been initialized";
      return false;
    }

    py::gil_scoped_acquire gil;
    try {
      py::array_t<float> object_array = to_numpy(object_points);
      py::tuple result = py::cast<py::tuple>(
        run_planner_(
          object_array, sampler_,
          py::arg("planner") = settings.planner,
          py::arg("grasp_threshold") = settings.score_threshold,
          py::arg("num_grasps") = settings.num_grasps,
          py::arg("topk_num_grasps") = settings.num_grasps));

      auto grasps = py::array_t<
        float, py::array::c_style | py::array::forcecast>::ensure(result[0]);
      auto scores = py::array_t<
        float, py::array::c_style | py::array::forcecast>::ensure(result[1]);
      if (!grasps || !scores) {
        error = "GraspGenX returned arrays with unsupported dtypes";
        return false;
      }

      py::buffer_info grasp_info = grasps.request();
      py::buffer_info score_info = scores.request();
      if (
        grasp_info.ndim != 3 || grasp_info.shape[1] != 4 ||
        grasp_info.shape[2] != 4 || score_info.ndim != 1 ||
        score_info.shape[0] != grasp_info.shape[0])
      {
        error = "GraspGenX returned grasps with an unexpected shape";
        return false;
      }

      const auto count = static_cast<std::size_t>(grasp_info.shape[0]);
      const float * grasp_data = static_cast<const float *>(grasp_info.ptr);
      const float * score_data = static_cast<const float *>(score_info.ptr);

      std::vector<bool> keep(count, true);
      if (settings.filter_collisions) {
        py::array_t<float> scene_array = to_numpy(
          settings.gripper_colmesh.empty() ? std::vector<std::array<float, 3>>{} :
          scene_points);
        py::object mesh = py::none();
        py::object surface_points = py::none();

        if (!settings.gripper_colmesh.empty() && !scene_points.empty()) {
          try {
            py::object path = py::module_::import("pathlib").attr("Path")(
              settings.gripper_colmesh);
            const std::string suffix =
              std::filesystem::path(settings.gripper_colmesh).extension().string();
            if (suffix == ".npy") {
              surface_points = py::module_::import("numpy").attr("load")(path);
            } else {
              mesh = py::module_::import("trimesh").attr("load")(
                path, py::arg("force") = "mesh");
            }
          } catch (const py::error_already_set & exc) {
            warning += std::string("failed to load gripper_colmesh, not filtering: ") +
              exc.what() + "\n";
            scene_array = to_numpy(std::vector<std::array<float, 3>>{});
          }
        }

        if (!filter_colliding_) {
          warning += "collision filter is unavailable; keeping all grasps\n";
        } else {
          try {
            auto mask = py::array_t<
              uint8_t, py::array::c_style | py::array::forcecast>::ensure(
              filter_colliding_(
                scene_array, grasps,
                py::arg("gripper_collision_mesh") = mesh,
                py::arg("gripper_surface_points") = surface_points,
                py::arg("collision_threshold") = settings.collision_threshold));
            if (!mask || mask.size() != static_cast<py::ssize_t>(count)) {
              warning += "collision filter returned an unexpected mask; "
                "keeping all grasps\n";
            } else {
              const uint8_t * mask_data = mask.data();
              for (std::size_t i = 0; i < count; ++i) {
                keep[i] = mask_data[i] != 0;
              }
            }
          } catch (const py::error_already_set & exc) {
            warning += std::string("collision filter failed, keeping all grasps: ") +
              exc.what() + "\n";
          }
        }
      }

      out.poses.clear();
      out.scores.clear();
      out.poses.reserve(count);
      out.scores.reserve(count);
      for (std::size_t i = 0; i < count; ++i) {
        if (!keep[i]) {
          continue;
        }
        std::array<double, 16> pose{};
        const float * src = grasp_data + i * 16;
        for (std::size_t k = 0; k < 16; ++k) {
          pose[k] = static_cast<double>(src[k]);
        }
        out.poses.push_back(pose);
        out.scores.push_back(static_cast<double>(score_data[i]));
      }
      return true;
    } catch (const py::error_already_set & exc) {
      error = exc.what();
    } catch (const std::exception & exc) {
      error = exc.what();
    }
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
      std::ostringstream out;
      out << static_cast<int>(free_mb) << " MiB free / "
          << static_cast<int>(total_mb) << " MiB total";
      return out.str();
    } catch (const py::error_already_set &) {
      return "unavailable";
    } catch (const std::exception &) {
      return "unavailable";
    }
  }

private:
  static py::array_t<float> to_numpy(const std::array<double, 3> & values)
  {
    py::array_t<float> array(3);
    float * data = array.mutable_data();
    for (std::size_t i = 0; i < 3; ++i) {
      data[i] = static_cast<float>(values[i]);
    }
    return array;
  }

  static py::array_t<float> to_numpy(
    const std::vector<std::array<float, 3>> & points)
  {
    py::array_t<float> array(
      std::vector<py::ssize_t>{
        static_cast<py::ssize_t>(points.size()), 3});
    float * data = array.mutable_data();
    for (std::size_t i = 0; i < points.size(); ++i) {
      data[i * 3 + 0] = points[i][0];
      data[i * 3 + 1] = points[i][1];
      data[i * 3 + 2] = points[i][2];
    }
    return array;
  }

  void clear_python_objects()
  {
    filter_colliding_ = py::object();
    run_planner_ = py::object();
    sampler_ = py::object();
    model_ = py::object();
    cfg_ = py::object();
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

  py::object cfg_;
  py::object model_;
  py::object sampler_;
  py::object run_planner_;
  py::object filter_colliding_;
  bool initialized_ = false;
};
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

}  // namespace graspgenx_ros

#endif  // GRASPGENX_ROS__PYTHON_BACKEND_HPP_

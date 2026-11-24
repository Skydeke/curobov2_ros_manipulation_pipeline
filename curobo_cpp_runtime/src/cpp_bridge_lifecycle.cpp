// SPDX-License-Identifier: Apache-2.0
#include "cpp_bridge_impl.hpp"

namespace curobo_cpp_runtime
{

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
  outcome.named_config_names = impl_->manifest.named_config_names;
  outcome.named_config_joint_names = impl_->manifest.named_config_joint_names;
  outcome.named_config_positions = impl_->manifest.named_config_positions;
  outcome.named_config_sizes = impl_->manifest.named_config_sizes;
  outcome.gripper_names = impl_->manifest.gripper_names;
  outcome.gripper_types = impl_->manifest.gripper_types;
  outcome.gripper_controllers = impl_->manifest.gripper_controllers;
  outcome.gripper_action_ns = impl_->manifest.gripper_action_ns;
  outcome.gripper_joint_lists = impl_->manifest.gripper_joint_lists;
  outcome.gripper_command_joints = impl_->manifest.gripper_command_joints;
  outcome.gripper_open_positions = impl_->manifest.gripper_open_positions;
  outcome.gripper_close_positions = impl_->manifest.gripper_close_positions;
  outcome.gripper_max_efforts = impl_->manifest.gripper_max_efforts;
  outcome.gripper_time_from_start = impl_->manifest.gripper_time_from_start;
  if (impl_->has_python()) {
    try {
      const auto descriptor =
        impl_->python->robot_description(robot_arg, robot_config_file);
      outcome.continuous_joint_names = descriptor.continuous_joint_names;
    } catch (const std::exception &) {
      // Keep the artifact-only outcome; joint_limits() then falls back to the
      // range heuristic (fixed-base arms without an explicit override).
    }
  }
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
  const std::vector<float> transforms =
    impl_->python->robot_model_fixed_transforms(impl_->python_handle);
  if (!transforms.empty()) {
    curobo_robot_runtime::update_artifact_fixed_transforms(
      impl_->artifact_dir, impl_->device_ordinal, transforms);
  }
  // See robot_model_rebuild_locks: the native model is frozen at export. Do not
  // invalidate the native scene solely for a lock update; that would force
  // every OpenArm finger update onto the Python planner.
}
}  // namespace curobo_cpp_runtime

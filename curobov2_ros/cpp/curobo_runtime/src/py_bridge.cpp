// SPDX-License-Identifier: Apache-2.0
// PyBridge: embedded CPython + the ROS-free `curobo_bridge` module.
// pybind11 lives only in this translation unit of this package.
//
// C1: there is ONE interpreter per process. It is owned here, created once and
// deliberately never finalised (a leaked process-global): finalising CPython at
// static-destruction time is unsafe, and a second interpreter is forbidden.
#include "curobo_runtime/py_bridge.hpp"

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifndef _WIN32
#include <dlfcn.h>
#endif

#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace curobo_runtime
{

namespace
{

// The process-global interpreter state. Leaked on purpose.
struct InterpreterState
{
  std::unique_ptr<py::scoped_interpreter> interp;      // created once
  std::unique_ptr<py::gil_scoped_release> release;     // created once, after setup
  bool imports_ok = false;
  int init_count = 0;
  std::string site_packages;
};

InterpreterState & interpreter_state()
{
  static InterpreterState * state = new InterpreterState();   // never destroyed
  return *state;
}

std::mutex & global_mutex()
{
  static std::mutex * m = new std::mutex();
  return *m;
}

void ensure_python_symbols_global()
{
#ifndef _WIN32
#ifdef CUROBO_PYTHON_LIB_NAME
  (void)dlopen(
    (std::string(CUROBO_PYTHON_LIB_DIR) + "/" + CUROBO_PYTHON_LIB_NAME).c_str(),
    RTLD_NOW | RTLD_GLOBAL);
  (void)dlopen(CUROBO_PYTHON_LIB_NAME, RTLD_NOW | RTLD_GLOBAL);
#endif
#endif
}

bool module_importable(const std::string & name)
{
  try {
    py::module_::import(name.c_str());
    return true;
  } catch (const py::error_already_set &) {
    return false;
  }
}

}  // namespace

struct PyBridge::Impl
{
  bool initialized = false;
  RobotHandle handle = -1;
  std::mutex mutex;
  std::vector<std::string> lanes;
};

PyBridge::PyBridge() : impl_(std::make_unique<Impl>()) {}
PyBridge::~PyBridge() = default;

// PORT: curobo_bridge:init (design bridge_api.md section 1)
RobotHandle PyBridge::init(const std::string & cfg)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->initialized) {
    return impl_->handle;
  }

  InterpreterState & state = interpreter_state();
  {
    std::lock_guard<std::mutex> glock(global_mutex());
    if (!state.interp) {
      ensure_python_symbols_global();
      state.interp = std::make_unique<py::scoped_interpreter>();
      state.init_count = 1;
    }
    {
      py::gil_scoped_acquire gil;
      if (!cfg.empty()) {
        state.site_packages = cfg;
        py::object sys = py::module_::import("sys");
        sys.attr("path").attr("insert")(0, cfg);
      }
      if (!module_importable("numpy")) {
        throw BridgeError("config", "numpy is not importable inside the embedded interpreter");
      }
      if (!module_importable("torch")) {
        throw BridgeError("config", "torch is not importable inside the embedded interpreter");
      }
      // curobo_bridge is our own module; it may not exist yet (M1 skeleton).
      try {
        py::module_::import("curobo_bridge");
      } catch (const py::error_already_set &) {
        // Imported lazily when it is installed; not fatal at M1.
      }
      // Build the registry entry the rest of the bridge indexes by handle
      // (PORT: curobo_bridge:init). `cfg` is a JSON object string; the
      // site-packages prepend happens above only when it is not JSON.
      py::object bridge_init = py::module_::import("curobo_bridge").attr("init");
      py::object py_cfg;
      {
        py::object json = py::module_::import("json");
        try {
          py_cfg = json.attr("loads")(cfg.empty() ? std::string("{}") : cfg);
        } catch (const py::error_already_set &) {
          py_cfg = py::dict();
        }
      }
      impl_->handle = bridge_init(py_cfg).cast<RobotHandle>();
      state.imports_ok = true;
    }
    if (!state.release) {
      // After setup, release the GIL so lane threads can take it (design 5).
      state.release = std::make_unique<py::gil_scoped_release>();
    }
  }

  impl_->initialized = true;
  return impl_->handle;
}

void PyBridge::shutdown(RobotHandle)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->initialized = false;
}

bool PyBridge::initialized() const
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->initialized;
}

int PyBridge::interpreter_init_count() const
{
  return interpreter_state().init_count;
}

bool PyBridge::interpreter_running() const
{
  return Py_IsInitialized() != 0;
}

bool PyBridge::module_available(const std::string & name) const
{
  if (!Py_IsInitialized()) {
    return false;
  }
  py::gil_scoped_acquire gil;
  py::object sys_modules = py::module_::import("sys").attr("modules");
  return !sys_modules.attr("get")(name).is_none();
}

void PyBridge::bind_lane(const std::string & name)
{
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->lanes.push_back(name);
  }
  if (!Py_IsInitialized()) {
    return;
  }
  py::gil_scoped_acquire gil;
  try {
    py::module_::import("curobo_bridge").attr("bind_lane")(name);
  } catch (const py::error_already_set &) {
    // curobo_bridge not installed yet; the stream binding is a no-op.
  }
}

void PyBridge::device_sync()
{
  if (!Py_IsInitialized()) {
    return;
  }
  py::gil_scoped_acquire gil;
  try {
    py::object torch = py::module_::import("torch");
    if (torch.attr("cuda").attr("is_available")().cast<bool>() &&
      torch.attr("cuda").attr("is_initialized")().cast<bool>())
    {
      torch.attr("cuda").attr("synchronize")();
    }
  } catch (const py::error_already_set &) {
  }
}

void PyBridge::sync_lane()
{
  device_sync();
}

void PyBridge::set_global_flags(bool cuda_graph_reset, bool debug_cuda_graphs)
{
  if (!Py_IsInitialized()) {
    return;
  }
  py::gil_scoped_acquire gil;
  try {
    py::module_::import("curobo_bridge").attr("set_global_flags")(
      cuda_graph_reset, debug_cuda_graphs);
  } catch (const py::error_already_set &) {
  }
}

std::vector<std::string> PyBridge::bound_lanes() const
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->lanes;
}

namespace
{
using F32Array = py::array_t<float, py::array::c_style | py::array::forcecast>;

TensorF32 array_to_tensor(py::handle obj)
{
  F32Array arr = F32Array::ensure(obj);
  if (!arr) {
    throw BridgeError("programming", "bridge returned a non-float32 array");
  }
  py::buffer_info info = arr.request();
  TensorF32 out;
  for (auto dim : info.shape) {
    out.shape.push_back(static_cast<std::int64_t>(dim));
  }
  const float * ptr = static_cast<const float *>(info.ptr);
  out.data.assign(ptr, ptr + info.size);
  return out;
}

F32Array tensor_to_array(const TensorF32 & tensor)
{
  // Size the array from the ACTUAL data: shape and data can disagree for
  // dynamically-shaped tensors, and a prod(shape)-sized allocation with a
  // larger memcpy would corrupt the host heap.
  std::vector<py::ssize_t> shape;
  std::size_t prod = 1;
  for (auto d : tensor.shape) {
    shape.push_back(static_cast<py::ssize_t>(d));
    prod *= static_cast<std::size_t>(d > 0 ? d : 0);
  }
  if (shape.empty() || prod != tensor.data.size()) {
    shape.assign(1, static_cast<py::ssize_t>(tensor.data.size()));
  }
  F32Array arr(shape);
  if (!tensor.data.empty()) {
    std::memcpy(arr.mutable_data(), tensor.data.data(), tensor.data.size() * sizeof(float));
  }
  return arr;
}

py::module_ bridge_module()
{
  return py::module_::import("curobo_bridge");
}
}  // namespace

// PORT: robot_description.load_robot_description
RobotDescriptionOutcome PyBridge::robot_description(
  const std::string & robot_arg, const std::string & robot_config_file)
{
  py::gil_scoped_acquire gil;
  try {
    py::object result = bridge_module().attr("robot_description")(
      robot_arg, robot_config_file);
    RobotDescriptionOutcome out;
    out.found = result.attr("__getitem__")("found").cast<bool>();
    if (!out.found) {
      return out;
    }
    out.path = result.attr("__getitem__")("path").cast<std::string>();
    out.name = result.attr("__getitem__")("name").cast<std::string>();
    out.base_link = result.attr("__getitem__")("base_link").cast<std::string>();
    out.curobo_config = result.attr("__getitem__")("curobo_config").cast<std::string>();
    out.strategy = result.attr("__getitem__")("strategy").cast<std::string>();
    out.strategy_params =
      result.attr("__getitem__")("strategy_params").cast<std::map<std::string, std::string>>();
    out.home_names =
      result.attr("__getitem__")("home_names").cast<std::vector<std::string>>();
    out.home_positions =
      result.attr("__getitem__")("home_positions").cast<std::vector<double>>();
    return out;
  } catch (const py::error_already_set & e) {
    throw BridgeError("config", e.what());
  }
}

// PORT: robot_model_manager.RobotModelManager.__init__
void PyBridge::robot_model_build(RobotHandle handle, const std::string & cfg_json)
{
  py::gil_scoped_acquire gil;
  try {
    bridge_module().attr("robot_model_build")(handle, cfg_json);
  } catch (const py::error_already_set & e) {
    throw BridgeError("config", e.what());
  }
}

// PORT: fk_services._compute_poses
TensorF32 PyBridge::fk(RobotHandle handle, const TensorF32 & q)
{
  py::gil_scoped_acquire gil;
  try {
    return array_to_tensor(bridge_module().attr("fk")(handle, tensor_to_array(q)));
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: ik_services._solve
IkOutcome PyBridge::ik(RobotHandle handle, const TensorF32 & poses)
{
  py::gil_scoped_acquire gil;
  try {
    py::object result = bridge_module().attr("ik")(handle, tensor_to_array(poses));
    IkOutcome outcome;
    outcome.success = result.attr("__getitem__")("success").cast<bool>();
    outcome.positions = array_to_tensor(result.attr("__getitem__")("positions"));
    for (auto v : result.attr("__getitem__")("valid").cast<std::vector<int>>()) {
      outcome.valid.push_back(static_cast<std::uint8_t>(v != 0));
    }
    outcome.joint_names =
      result.attr("__getitem__")("joint_names").cast<std::vector<std::string>>();
    return outcome;
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: config_wrapper_ik.ConfigWrapperIK.build
void PyBridge::ik_config(
  RobotHandle handle, int batch_size, int num_seeds, bool self_collision_check,
  bool use_lm_seed, bool use_cuda_graph)
{
  py::gil_scoped_acquire gil;
  try {
    bridge_module().attr("ik_config")(
      handle, batch_size, num_seeds, self_collision_check, use_lm_seed,
      use_cuda_graph);
  } catch (const py::error_already_set & e) {
    throw BridgeError("config", e.what());
  }
}

// PORT: fk_services._validate
std::vector<std::uint8_t> PyBridge::validate(RobotHandle handle, const TensorF32 & q){
  py::gil_scoped_acquire gil;
  try {
    std::vector<std::uint8_t> out;
    for (bool v : bridge_module().attr("validate")(handle, tensor_to_array(q)).cast<std::vector<bool>>()) {
      out.push_back(v ? 1 : 0);
    }
    return out;
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: collision_distance._compute_sphere_distance
std::vector<float> PyBridge::collision_distance(RobotHandle handle, const TensorF32 & q)
{
  py::gil_scoped_acquire gil;
  try {
    return bridge_module().attr("collision_distance")(handle, tensor_to_array(q))
           .cast<std::vector<float>>();
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: collision_distance._attributed_collisions / _attributed_self_collisions
// / _attributed_joint_limit_violations -> StateCollisions + the failure log.
CollisionAttribution PyBridge::collision_attribution(
  RobotHandle handle, const TensorF32 & q)
{
  py::gil_scoped_acquire gil;
  try {
    py::object result =
      bridge_module().attr("collision_attribution")(handle, tensor_to_array(q));
    CollisionAttribution out;
    for (const auto & row : result.attr("__getitem__")("world")
         .cast<std::vector<py::dict>>())
    {
      WorldContact c;
      c.link = row["link"].cast<std::string>();
      c.obstacle = row["obstacle"].cast<std::string>();
      c.depth = row["depth"].cast<double>();
      out.world_contacts.push_back(std::move(c));
    }
    for (const auto & row : result.attr("__getitem__")("self")
         .cast<std::vector<py::dict>>())
    {
      SelfContact c;
      c.link_a = row["link"].cast<std::string>();
      c.link_b = row["link_b"].cast<std::string>();
      c.depth = row["depth"].cast<double>();
      out.self_contacts.push_back(std::move(c));
    }
    for (const auto & row : result.attr("__getitem__")("cspace")
         .cast<std::vector<py::dict>>())
    {
      CspaceViolation c;
      c.joint = row["joint"].cast<std::string>();
      c.value = row["value"].cast<double>();
      c.lower = row["lower"].cast<double>();
      c.upper = row["upper"].cast<double>();
      out.cspace_violations.push_back(std::move(c));
    }
    return out;
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: mpc_planner.MPCController._cost_breakdown
MpcCostBreakdown PyBridge::mpc_cost_breakdown(RobotHandle handle)
{
  py::gil_scoped_acquire gil;
  try {
    py::object result = bridge_module().attr("mpc_cost_breakdown")(handle);
    MpcCostBreakdown out;
    auto blob = result.cast<py::dict>();
    if (blob.size() == 0) {
      return out;
    }
    out.has_value = true;
    const auto get = [&blob](const char * key) -> double {
        if (!blob.contains(key)) {
          return 0.0;
        }
        return blob[key].cast<double>();
      };
    out.cost_tool_pose_pos = get("cost_tool_pose_pos");
    out.cost_tool_pose_orient = get("cost_tool_pose_orient");
    out.cost_cspace = get("cost_cspace");
    out.con_self_collision = get("con_self_collision");
    out.con_scene_collision = get("con_scene_collision");
    out.con_cspace_bound = get("con_cspace_bound");
    return out;
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: plan_plot.render_plan_plot (single_planner._publish_plan_image)
PlanImageOutcome PyBridge::render_plan_image(
  const TensorF32 & position, const TensorF32 & velocity,
  const TensorF32 & acceleration, const std::vector<std::string> & names, double dt)
{
  py::gil_scoped_acquire gil;
  try {
    py::object result = bridge_module().attr("render_plan_image")(
      tensor_to_array(position), tensor_to_array(velocity),
      tensor_to_array(acceleration), py::cast(names), dt);
    PlanImageOutcome out;
    if (result.is_none()) {
      return out;
    }
    auto arr = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>::ensure(
      result);
    if (!arr) {
      return out;
    }
    py::buffer_info info = arr.request();
    if (info.ndim < 2) {
      return out;
    }
    out.has_value = true;
    out.height = static_cast<int>(info.shape[0]);
    out.width = static_cast<int>(info.shape[1]);
    out.channels = info.ndim >= 3 ? static_cast<int>(info.shape[2]) : 1;
    const auto * ptr = static_cast<const std::uint8_t *>(info.ptr);
    out.data.assign(ptr, ptr + info.size);
    return out;
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: unified_planner_node.torch_sync_enabled
void PyBridge::torch_synchronize()
{
  py::gil_scoped_acquire gil;
  try {
    bridge_module().attr("torch_synchronize")();
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: obstacle_manager._solver_bucket_counts
CacheBucketCounts PyBridge::collision_cache_bucket_counts(RobotHandle handle)
{
  py::gil_scoped_acquire gil;
  try {
    py::object result = bridge_module().attr("collision_cache_bucket_counts")(handle);
    CacheBucketCounts counts;
    counts.cuboid = result.attr("__getitem__")("cuboid").cast<int>();
    counts.mesh = result.attr("__getitem__")("mesh").cast<int>();
    return counts;
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: obstacle_manager scene ops
SceneVersion PyBridge::scene_set(
  RobotHandle handle, const std::string & objects_json, bool clear_first)
{
  py::gil_scoped_acquire gil;
  try {
    return bridge_module().attr("scene_set")(handle, objects_json, clear_first).cast<SceneVersion>();
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

std::string PyBridge::scene_get(RobotHandle handle)
{
  py::gil_scoped_acquire gil;
  try {
    return bridge_module().attr("scene_get")(handle).cast<std::string>();
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: ros_service_manager._callback_get_joint_info
JointLimitsOutcome PyBridge::joint_limits(RobotHandle handle){
  py::gil_scoped_acquire gil;
  try {
    py::object result = bridge_module().attr("joint_limits")(handle);
    JointLimitsOutcome outcome;
    outcome.positions = array_to_tensor(result.attr("__getitem__")("positions"));
    outcome.joint_names =
      result.attr("__getitem__")("joint_names").cast<std::vector<std::string>>();
    outcome.locked_joint_names =
      result.attr("__getitem__")("locked_joint_names").cast<std::vector<std::string>>();
    outcome.locked_positions =
      result.attr("__getitem__")("locked_positions").cast<std::vector<double>>();
    outcome.unpinned_lock_joint_names =
      result.attr("__getitem__")("unpinned_lock_joint_names")
        .cast<std::vector<std::string>>();
    return outcome;
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: attachment_services._fit_and_attach (+ disable bookkeeping)
void PyBridge::attach_object(
  RobotHandle handle, const std::string & name, const std::vector<float> & q)
{
  py::gil_scoped_acquire gil;
  try {
    bridge_module().attr("attach_object")(handle, name, q);
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: attachment_services.detach
std::string PyBridge::detach_object(RobotHandle handle)
{
  py::gil_scoped_acquire gil;
  try {
    return bridge_module().attr("detach_object")(handle).cast<std::string>();
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

namespace
{
PlanOutcome dict_to_plan_outcome(py::object result)
{
  PlanOutcome out;
  out.success = result.attr("__getitem__")("success").cast<bool>();
  out.feasible = result.attr("__getitem__")("feasible").cast<bool>();
  out.status = result.attr("__getitem__")("status").cast<std::string>();
  out.goalset_index = result.attr("__getitem__")("goalset_index").cast<int>();
  out.seed_index = result.attr("__getitem__")("seed_index").cast<int>();
  out.total_time = result.attr("__getitem__")("total_time").cast<double>();
  out.joint_names = result.attr("__getitem__")("joint_names").cast<std::vector<std::string>>();
  out.problems = result.attr("__getitem__")("problems").cast<std::uint32_t>();
  out.candidates_generated =
    result.attr("__getitem__")("candidates_generated").cast<std::uint32_t>();
  out.candidates_solved = result.attr("__getitem__")("candidates_solved").cast<std::uint32_t>();
  out.candidates_pruned = result.attr("__getitem__")("candidates_pruned").cast<std::uint32_t>();
  out.waypoints_planned = result.attr("__getitem__")("waypoints_planned").cast<std::uint32_t>();
  out.solver_time_s = result.attr("__getitem__")("solver_time_s").cast<double>();
  out.motion_time_s = result.attr("__getitem__")("motion_time_s").cast<double>();
  out.position_error_mm = result.attr("__getitem__")("position_error_mm").cast<double>();
  for (auto v : result.attr("__getitem__")("waypoint_status").cast<std::vector<int>>()) {
    out.waypoint_status.push_back(static_cast<std::int16_t>(v));
  }
  // PORT: PlanningStats.considered (single_planner._segment_considered_rows).
  // Optional: tolerate a bridge build that predates the key (a stale
  // curobo_bridge must not turn every plan into a KeyError).
  py::object considered = py::none();
  if (result.attr("__contains__")("considered").cast<bool>()) {
    considered = result.attr("__getitem__")("considered");
  }
  if (!considered.is_none()) {
    for (py::handle row : considered) {
      ConsideredRow r;
      r.problem = row.attr("__getitem__")("problem").cast<std::uint32_t>();
      r.segment = row.attr("__getitem__")("segment").cast<std::uint32_t>();
      r.goalset_candidate =
        row.attr("__getitem__")("goalset_candidate").cast<std::uint32_t>();
      r.seed = row.attr("__getitem__")("seed").cast<std::uint32_t>();
      r.success = row.attr("__getitem__")("success").cast<bool>();
      r.cost = row.attr("__getitem__")("cost").cast<double>();
      r.waypoint_cost = row.attr("__getitem__")("waypoint_cost").cast<double>();
      r.path_length = row.attr("__getitem__")("path_length").cast<double>();
      r.clearance = row.attr("__getitem__")("clearance").cast<double>();
      r.max_waypoint_error =
        row.attr("__getitem__")("max_waypoint_error").cast<double>();
      r.solve_time = row.attr("__getitem__")("solve_time").cast<double>();
      out.considered.push_back(r);
    }
  }
  if (!result.attr("__getitem__")("positions").is_none()) {
    out.positions = array_to_tensor(result.attr("__getitem__")("positions"));
  }
  if (!result.attr("__getitem__")("velocity").is_none()) {
    out.velocity = array_to_tensor(result.attr("__getitem__")("velocity"));
  }
  if (!result.attr("__getitem__")("acceleration").is_none()) {
    out.acceleration = array_to_tensor(result.attr("__getitem__")("acceleration"));
  }
  return out;
}

py::dict plan_request_to_dict(const PlanRequest & request)
{
  py::dict out;
  out["start"] = request.start;
  out["joint_target"] = request.joint_target;
  out["max_attempts"] = request.max_attempts;
  out["enable_graph_attempt"] = request.enable_graph_attempt;
  if (!request.trajectory_constraints.empty()) {
    out["trajectory_constraints"] = request.trajectory_constraints;
  }
  if (!request.goalset.shape.empty()) {
    out["goalset"] = tensor_to_array(request.goalset);
  }
  return out;
}
}  // namespace

// PORT: config_wrapper_motion.set_motion_gen_config
void PyBridge::motion_planner_build(RobotHandle handle, const std::string & cfg_json)
{
  py::gil_scoped_acquire gil;
  try {
    bridge_module().attr("motion_planner_build")(handle, cfg_json);
  } catch (const py::error_already_set & e) {
    throw BridgeError("config", e.what());
  }
}

// PORT: single_planner._plan_trajectory (plan_pose)
PlanOutcome PyBridge::plan_pose(RobotHandle handle, const PlanRequest & request)
{
  py::gil_scoped_acquire gil;
  try {
    return dict_to_plan_outcome(
      bridge_module().attr("plan_pose")(handle, plan_request_to_dict(request)));
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: MotionPlanner.reset_seed
void PyBridge::reset_seed(RobotHandle handle)
{
  py::gil_scoped_acquire gil;
  try {
    bridge_module().attr("reset_seed")(handle);
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: robot_model_manager.set_lock_joints
void PyBridge::robot_model_rebuild_locks(RobotHandle handle, const std::string & lock_json)
{
  py::gil_scoped_acquire gil;
  try {
    bridge_module().attr("robot_model_rebuild_locks")(handle, lock_json);
  } catch (const py::error_already_set & e) {
    throw BridgeError("config", e.what());
  }
}

// PORT: robot_model_manager.update_lock_joints
void PyBridge::robot_model_update_lock_joints(
  RobotHandle handle, const std::string & lock_json)
{
  py::gil_scoped_acquire gil;
  try {
    bridge_module().attr("robot_model_update_lock_joints")(handle, lock_json);
  } catch (const py::error_already_set & e) {
    throw BridgeError("config", e.what());
  }
}

// PORT: obstacle_manager._build_mapper
void PyBridge::configure_mapper(RobotHandle handle, const std::string & cfg_json)
{
  py::gil_scoped_acquire gil;
  try {
    bridge_module().attr("configure_mapper")(handle, cfg_json);
  } catch (const py::error_already_set & e) {
    throw BridgeError("config", e.what());
  }
}

// Force the Mapper's Warp kernels to compile/load now (see IBridge doc).
void PyBridge::warmup_mapper(RobotHandle handle)
{
  py::gil_scoped_acquire gil;
  try {
    bridge_module().attr("warmup_mapper")(handle);
  } catch (const py::error_already_set & e) {
    throw BridgeError("config", e.what());
  }
}

// PORT: obstacle_manager.refresh_esdf
bool PyBridge::esdf_refresh(RobotHandle handle)
{
  py::gil_scoped_acquire gil;
  try {
    return bridge_module().attr("esdf_refresh")(handle).cast<bool>();
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: unified_planner_node.update_all_solvers_world
bool PyBridge::world_push(RobotHandle handle)
{
  py::gil_scoped_acquire gil;
  try {
    return bridge_module().attr("world_push")(handle).cast<bool>();
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// Reset + re-capture the trajopt CUDA graphs when the solve mode changes
// (pose<->cspace), so a stale graph is never replayed.
void PyBridge::prepare_solve_mode(RobotHandle handle, const std::string & mode)
{
  py::gil_scoped_acquire gil;
  try {
    bridge_module().attr("prepare_solve_mode")(handle, mode);
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: obstacle_manager.get_voxel_grid
VoxelGrid PyBridge::voxel_grid(RobotHandle handle)
{
  py::gil_scoped_acquire gil;
  try {
    py::object result = bridge_module().attr("voxel_grid")(handle);
    VoxelGrid grid;
    grid.shape = result.attr("__getitem__")("shape").cast<std::vector<std::int64_t>>();
    grid.resolution = result.attr("__getitem__")("resolution").cast<double>();
    // origin is a 3-vector; assign explicitly.
    auto origin = result.attr("__getitem__")("origin").cast<std::vector<double>>();
    for (std::size_t i = 0; i < 3 && i < origin.size(); ++i) {
      grid.origin[i] = origin[i];
    }
    auto data = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>::ensure(
      result.attr("__getitem__")("data"));
    if (data) {
      py::buffer_info info = data.request();
      const auto * ptr = static_cast<const std::uint8_t *>(info.ptr);
      grid.data.assign(ptr, ptr + info.size);
    }
    return grid;
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: obstacle_manager.clear_dynamic_voxels
int PyBridge::clear_voxel_map(RobotHandle handle)
{
  py::gil_scoped_acquire gil;
  try {
    return bridge_module().attr("clear_voxel_map")(handle).cast<int>();
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: obstacle_manager.clear_voxel_region
int PyBridge::clear_voxel_region(
  RobotHandle handle, const std::array<double, 3> & bounds_min,
  const std::array<double, 3> & bounds_max)
{
  py::gil_scoped_acquire gil;
  try {
    return bridge_module().attr("clear_voxel_region")(
      handle, bounds_min, bounds_max).cast<int>();
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: cameras/camera_depth_map_strategy._integrate_once
void PyBridge::depth_integrate(
  RobotHandle handle, const TensorF32 & depth, const TensorF32 & intrinsics,
  const std::array<double, 7> & camera_pose)
{
  py::gil_scoped_acquire gil;
  try {
    bridge_module().attr("depth_integrate")(
      handle, tensor_to_array(depth), tensor_to_array(intrinsics), camera_pose);
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: lasers/laser_context._build_and_integrate
void PyBridge::lidar_integrate(
  RobotHandle handle, const TensorF32 & range_image, const TensorF32 & poses,
  const TensorF32 & valid_range, const TensorF32 & elevation_range)
{
  py::gil_scoped_acquire gil;
  try {
    bridge_module().attr("lidar_integrate")(
      handle, tensor_to_array(range_image), tensor_to_array(poses),
      tensor_to_array(valid_range), tensor_to_array(elevation_range));
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: camera_robot_segmentation_strategy._mask_depth_image
TensorF32 PyBridge::robot_segment(
  RobotHandle handle, const TensorF32 & depth, const TensorF32 & intrinsics_meta,
  const std::array<double, 7> & cam_pose, const std::vector<float> & q,
  const std::string & masks_json)
{
  py::gil_scoped_acquire gil;
  try {
    return array_to_tensor(bridge_module().attr("robot_segment")(
      handle, tensor_to_array(depth), tensor_to_array(intrinsics_meta), cam_pose, q,
      masks_json));
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: joint_space_planner._plan_trajectory (plan_cspace)
PlanOutcome PyBridge::plan_cspace(RobotHandle handle, const PlanRequest & request)
{
  py::gil_scoped_acquire gil;
  try {
    return dict_to_plan_outcome(
      bridge_module().attr("plan_cspace")(handle, plan_request_to_dict(request)));
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: robot_model_manager.get_collision_spheres_with_attached
SphereSetOutcome PyBridge::collision_spheres_with_attached(
  RobotHandle handle, const std::vector<float> & q)
{
  py::gil_scoped_acquire gil;
  try {
    py::array_t<float> q_arr(static_cast<py::ssize_t>(q.size()));
    std::memcpy(q_arr.mutable_data(), q.data(), q.size() * sizeof(float));
    py::object result = bridge_module().attr("collision_spheres_with_attached")(handle, q_arr);
    SphereSetOutcome out;
    out.spheres = array_to_tensor(result.attr("__getitem__")("spheres"));
    for (bool v : result.attr("__getitem__")("attached").cast<std::vector<bool>>()) {
      out.attached.push_back(v ? 1 : 0);
    }
    // Optional link-relative payload spheres (newer Python bridge). Absent in
    // an older bridge: leave them empty and the publisher falls back to the
    // base-frame spheres.
    try {
      out.attached_local = array_to_tensor(result.attr("__getitem__")("attached_local"));
    } catch (const py::error_already_set &) {
      PyErr_Clear();
      out.attached_local = TensorF32{};
    }
    try {
      out.attach_frame =
        result.attr("__getitem__")("attach_frame").cast<std::string>();
    } catch (const py::error_already_set &) {
      PyErr_Clear();
      out.attach_frame.clear();
    }
    return out;
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: the kinematics model's tool frames
std::vector<std::string> PyBridge::tool_frames(RobotHandle handle)
{
  py::gil_scoped_acquire gil;
  try {
    return bridge_module().attr("tool_frames")(handle).cast<std::vector<std::string>>();
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: robot_model_manager.set_link_collision
LinkCollisionOutcome PyBridge::set_link_collision(
  RobotHandle handle, const std::vector<std::string> & link_names, bool enabled)
{
  py::gil_scoped_acquire gil;
  try {
    py::object result = bridge_module().attr("set_link_collision")(
      handle, link_names, enabled);
    LinkCollisionOutcome out;
    out.applied = result.attr("__getitem__")("applied").cast<std::vector<std::string>>();
    out.unknown = result.attr("__getitem__")("unknown").cast<std::vector<std::string>>();
    return out;
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// ---- M5: reactive control (MPC) --------------------------------------------
namespace
{
py::array_t<float> flat_f32(const std::vector<float> & values)
{
  py::array_t<float> arr(static_cast<py::ssize_t>(values.size()));
  std::memcpy(arr.mutable_data(), values.data(), values.size() * sizeof(float));
  return arr;
}

ReactiveAction dict_to_reactive_action(py::handle obj)
{
  ReactiveAction out;
  if (obj.is_none()) {
    return out;
  }
  out.has_action = true;
  out.position = array_to_tensor(obj.attr("__getitem__")("position"));
  py::object vel = obj.attr("__getitem__")("velocity");
  if (!vel.is_none()) {
    out.velocity = array_to_tensor(vel);
  }
  py::object acc = obj.attr("__getitem__")("acceleration");
  if (!acc.is_none()) {
    out.acceleration = array_to_tensor(acc);
  }
  return out;
}
}  // namespace

// PORT: mpc_planner.MPCController.build_solver
void PyBridge::mpc_build(RobotHandle handle, const std::string & cfg_json)
{
  py::gil_scoped_acquire gil;
  try {
    bridge_module().attr("mpc_build")(handle, cfg_json);
  } catch (const py::error_already_set & e) {
    throw BridgeError("config", e.what());
  }
}

// PORT: mpc_planner.MPCController.setup + _apply_goal
MpcGoalOutcome PyBridge::mpc_setup(RobotHandle handle, const std::vector<float> & start_q)
{
  py::gil_scoped_acquire gil;
  try {
    py::object result = bridge_module().attr("mpc_setup")(handle, flat_f32(start_q));
    MpcGoalOutcome out;
    py::object ee = result.attr("__getitem__")("ee_position");
    if (!ee.is_none()) {
      auto values = ee.cast<std::vector<double>>();
      if (values.size() >= 3) {
        out.has_ee = true;
        out.ee_position = {values[0], values[1], values[2]};
      }
    }
    return out;
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: mpc_planner.MPCController.step / step_paced
MpcStepOutcome PyBridge::mpc_step(
  RobotHandle handle, const ReactiveState & state, bool paced)
{
  py::gil_scoped_acquire gil;
  try {
    py::dict state_dict;
    state_dict["position"] = flat_f32(state.position);
    if (!state.velocity.empty()) {
      state_dict["velocity"] = flat_f32(state.velocity);
    }
    if (!state.acceleration.empty()) {
      state_dict["acceleration"] = flat_f32(state.acceleration);
    }
    py::object result = bridge_module().attr("mpc_step")(handle, state_dict, paced);
    MpcStepOutcome out;
    out.action = dict_to_reactive_action(result.attr("__getitem__")("action"));
    py::object horizon = result.attr("__getitem__")("horizon_position");
    if (!horizon.is_none()) {
      out.horizon_position = array_to_tensor(horizon);
    }
    py::object ee = result.attr("__getitem__")("ee_position");
    if (!ee.is_none()) {
      auto values = ee.cast<std::vector<double>>();
      if (values.size() >= 3) {
        out.has_ee = true;
        out.ee_position = {values[0], values[1], values[2]};
      }
    }
    // PORT: mpc_planner._publish_costs pose_* fields (None -> 0.0).
    py::object pose_pos = result.attr("__getitem__")("pose_pos_err_m");
    py::object pose_rot = result.attr("__getitem__")("pose_rot_err_rad");
    if (!pose_pos.is_none() || !pose_rot.is_none()) {
      out.has_pose_err = true;
      out.pose_pos_err_m = pose_pos.is_none() ? 0.0 : pose_pos.cast<double>();
      out.pose_rot_err_rad = pose_rot.is_none() ? 0.0 : pose_rot.cast<double>();
    }
    return out;
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: mpc_planner.MPCController._apply_goal (Cartesian branch)
bool PyBridge::mpc_update_goal_pose(
  RobotHandle handle, const std::array<double, 7> & goal_raw, bool run_ik)
{
  py::gil_scoped_acquire gil;
  try {
    return bridge_module().attr("mpc_update_goal_pose")(
      handle, std::vector<double>(goal_raw.begin(), goal_raw.end()), run_ik).cast<bool>();
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: mpc_planner.MPCController._apply_goal (joint branch)
void PyBridge::mpc_update_goal_state(RobotHandle handle, const std::vector<float> & q)
{
  py::gil_scoped_acquire gil;
  try {
    bridge_module().attr("mpc_update_goal_state")(handle, flat_f32(q));
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: mpc_planner.MPCController._reseed_toward_goal
void PyBridge::mpc_reseed(RobotHandle handle, const std::vector<float> & q)
{
  py::gil_scoped_acquire gil;
  try {
    bridge_module().attr("mpc_reseed")(handle, flat_f32(q));
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: mpc_planner.MPCController._solve_goal_state
IkOutcome PyBridge::mpc_solve_goal_state(
  RobotHandle handle, const std::array<double, 7> & goal_raw,
  const std::vector<float> & seed_q)
{
  py::gil_scoped_acquire gil;
  try {
    py::object result = bridge_module().attr("mpc_solve_goal_state")(
      handle, std::vector<double>(goal_raw.begin(), goal_raw.end()), flat_f32(seed_q));
    IkOutcome out;
    out.success = result.attr("__getitem__")("success").cast<bool>();
    py::object sol = result.attr("__getitem__")("solution");
    if (!sol.is_none()) {
      out.positions = array_to_tensor(sol);
      out.valid = {1};
    }
    return out;
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: mpc_planner.MPCController.update_world
void PyBridge::mpc_load_scene(RobotHandle handle, const std::string & scene_json)
{
  py::gil_scoped_acquire gil;
  try {
    bridge_module().attr("mpc_load_scene")(handle, scene_json);
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// PORT: mpc_planner.MPCController._cost_breakdown
std::string PyBridge::mpc_metrics(RobotHandle handle)
{
  py::gil_scoped_acquire gil;
  try {
    return bridge_module().attr("mpc_metrics")(handle).cast<std::string>();
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

// ---- M5: reactive control (retarget / teleop) ------------------------------
void PyBridge::retarget_build(RobotHandle handle, const std::string & cfg_json)
{
  py::gil_scoped_acquire gil;
  try {
    bridge_module().attr("retarget_build")(handle, cfg_json);
  } catch (const py::error_already_set & e) {
    throw BridgeError("config", e.what());
  }
}

void PyBridge::retarget_reset(RobotHandle handle)
{
  py::gil_scoped_acquire gil;
  try {
    bridge_module().attr("retarget_reset")(handle);
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

MpcStepOutcome PyBridge::retarget_step(
  RobotHandle handle, const std::array<double, 7> & goal_raw)
{
  py::gil_scoped_acquire gil;
  try {
    py::object result = bridge_module().attr("retarget_step")(
      handle, std::vector<double>(goal_raw.begin(), goal_raw.end()));
    MpcStepOutcome out;
    out.action = dict_to_reactive_action(result.attr("__getitem__")("action"));
    py::object ee = result.attr("__getitem__")("ee_position");
    if (!ee.is_none()) {
      auto values = ee.cast<std::vector<double>>();
      if (values.size() >= 3) {
        out.has_ee = true;
        out.ee_position = {values[0], values[1], values[2]};
      }
    }
    return out;
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

void PyBridge::retarget_load_scene(RobotHandle handle, const std::string & scene_json)
{
  py::gil_scoped_acquire gil;
  try {
    bridge_module().attr("retarget_load_scene")(handle, scene_json);
  } catch (const py::error_already_set & e) {
    throw BridgeError("programming", e.what());
  }
}

}  // namespace curobo_runtime

// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/core/fk_services.py.
#include "curobo_core/core/fk_services.hpp"
#include "curobo_core/core/gpu_runtime.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>

#include "curobo_core/convert.hpp"
#include "curobo_runtime/runtime.hpp"

namespace curobo_core
{

namespace
{
// PORT: fk_services._on_lane - run on lane_kin (PRIO_KIN, 120 s) unless this
// thread already is the lane worker. Shared with IKServices, so it is what
// serialises FK against IK (and therefore against itself).
//
// The barrier is owned here, never by the caller: a lane job already runs inside
// barrier.shared() (GpuLane::run_job), so a caller that pre-acquired a read hold
// would then block on the lane while a waiting capture (writer) blocks on that
// read hold - a three-way deadlock. Only the no-lane (standalone/test) path
// needs the barrier taken explicitly, like the Python _on_lane fallback.
template<class F>
auto on_kin_lane(curobo_runtime::GpuLane * lane, F && fn) -> decltype(fn())
{
  if (lane == nullptr) {
    auto gpu_guard = curobo_runtime::Runtime::instance().barrier().shared();
    return fn();
  }
  if (lane->owns_current_thread()) {
    return fn();
  }
  curobo_runtime::SubmitOptions opts;
  opts.prio = curobo_runtime::PRIO_KIN;
  return lane->call_for(opts, 120.0, std::forward<F>(fn));
}
}  // namespace

FKServices::FKServices(
  rclcpp::Node * node, curobo_runtime::IBridge * bridge,
  curobo_runtime::RobotHandle handle, std::vector<std::string> joint_names,
  int cache_cuboid, int cache_mesh)
: node_(node), bridge_(bridge), handle_(handle),
  joint_names_(std::move(joint_names)), cache_cuboid_(cache_cuboid), cache_mesh_(cache_mesh)
{
  const std::string name = node_->get_name();
  warmup_srv_ = node_->create_service<curobov2_ros_interfaces::srv::WarmupFK>(
    name + "/warmup_fk",
    std::bind(&FKServices::handle_warmup_fk, this, std::placeholders::_1, std::placeholders::_2));
  fk_srv_ = node_->create_service<curobov2_ros_interfaces::srv::Fk>(
    name + "/fk",
    std::bind(&FKServices::handle_fk, this, std::placeholders::_1, std::placeholders::_2));
  fk_batch_srv_ = node_->create_service<curobov2_ros_interfaces::srv::FkBatch>(
    name + "/fk_batch",
    std::bind(&FKServices::handle_fk_batch, this, std::placeholders::_1, std::placeholders::_2));

  RCLCPP_INFO(
    node_->get_logger(),
    "FKServices registered (model allocated by the startup warmup, or on demand "
    "via /warmup_fk when warmup_fk:=false)");
}

// PORT: fk_services.FKServices.warmup
std::string FKServices::warmup(int batch_size)
{
  batch_size = std::max(1, batch_size);
  if (initialized_) {
    return "FK model already ready (batch_size=" + std::to_string(batch_size) + ")";
  }
  init(batch_size);
  return "FK model ready (batch_size=" + std::to_string(batch_size) + ")";
}

// PORT: fk_services.FKServices._init
void FKServices::init(int batch_size)
{
  RCLCPP_INFO(node_->get_logger(), "Initializing FK model (batch_size=%d)...", batch_size);

  // The shared Kinematics is owned by the bridge (never a second model here);
  // the advisory collision validator is the bridge's `validate`.
  (void)cache_cuboid_;
  (void)cache_mesh_;

  // PORT: fk_services._init assigns the (shared) model BEFORE the warmup batch,
  // so the internal `not initialized` guard does not fire and the warmup batch
  // actually runs.
  initialized_ = true;
  fk_batch_ = batch_size;

  // Warmup batch: one FK call over `batch_size` rows.
  std::vector<std::vector<double>> qs(
    static_cast<std::size_t>(batch_size),
    std::vector<double>(joint_names_.size(), 0.0));
  std::vector<curobo_runtime::TensorF32> poses;
  {
    // PORT: fk_services._gpu_guard - `with self._gpu_guard(): self._init(...)`
    // (blocking gpu_lock). Model construction + the warmup batch are eager CUDA
    // ops that must not overlap a CUDA-graph capture.
    //
    // Order matters: enter the `kin` lane FIRST and take the exclusive barrier
    // *inside* the lane job (like IKServices::warmup). Taking the barrier before
    // submitting makes GpuLane::submit_opts reject the wait on lane 'kin' with
    // DeadlockRisk. compute_poses() is already running on the lane thread here,
    // so its own on_kin_lane() re-enters inline.
    on_kin_lane(lane_, [&]() {
      auto gpu_guard = curobo_runtime::Runtime::instance().barrier().exclusive();
      compute_poses(qs, poses);
    });
  }

  RCLCPP_INFO(node_->get_logger(), "FK model ready");
}

// PORT: fk_services.FKServices._warmup_fk_callback
void FKServices::handle_warmup_fk(
  const std::shared_ptr<curobov2_ros_interfaces::srv::WarmupFK::Request> request,
  std::shared_ptr<curobov2_ros_interfaces::srv::WarmupFK::Response> response)
{
  try {
    response->message = warmup(static_cast<int>(request->batch_size));
    response->success = true;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(node_->get_logger(), "FK warmup failed: %s", e.what());
    response->success = false;
    response->message = e.what();
  }
}

// PORT: fk_services.FKServices._positions_for_model
std::vector<double> FKServices::positions_for_model(
  const std::vector<double> & positions, const std::vector<std::string> & names) const
{
  if (names.size() == positions.size() && !names.empty()) {
    std::vector<double> out;
    out.reserve(joint_names_.size());
    bool ok = true;
    for (const std::string & want : joint_names_) {
      auto it = std::find(names.begin(), names.end(), want);
      if (it == names.end()) {
        ok = false;
        break;
      }
      out.push_back(positions[static_cast<std::size_t>(it - names.begin())]);
    }
    if (ok) {
      return out;
    }
  }
  const std::size_t dof = joint_names_.size();
  return std::vector<double>(positions.begin(), positions.begin() +
           static_cast<std::ptrdiff_t>(std::min(dof, positions.size())));
}

// PORT: fk_services.FKServices._compute_poses
bool FKServices::compute_poses(
  const std::vector<std::vector<double>> & qs,
  std::vector<curobo_runtime::TensorF32> & poses) const
{
  if (qs.empty()) {
    RCLCPP_ERROR(node_->get_logger(), "FK: empty joint state list");
    return false;
  }
  if (!initialized_) {
    RCLCPP_ERROR(node_->get_logger(), "FK not initialized. Call warmup_fk first.");
    return false;
  }
  const std::size_t dof = joint_names_.size();
  curobo_runtime::TensorF32 q;
  q.shape = {static_cast<std::int64_t>(qs.size()), static_cast<std::int64_t>(dof)};
  q.data.reserve(qs.size() * dof);
  for (const auto & row : qs) {
    for (std::size_t j = 0; j < dof; ++j) {
      q.data.push_back(static_cast<float>(j < row.size() ? row[j] : 0.0));
    }
  }
  // PORT: fk_services._on_lane - FK shares lane_kin with IK, so the two never
  // issue kinematics kernels concurrently.
  const curobo_runtime::TensorF32 out = on_kin_lane(
    lane_, [&]() {return bridge_->fk(handle_, q);});
  poses.clear();
  const std::size_t n = qs.size();
  for (std::size_t i = 0; i < n; ++i) {
    curobo_runtime::TensorF32 one;
    one.shape = {1, 1, 7};
    one.data.assign(out.data.begin() + static_cast<std::ptrdiff_t>(i * 7),
      out.data.begin() + static_cast<std::ptrdiff_t>((i + 1) * 7));
    poses.push_back(std::move(one));
  }
  return true;
}

// PORT: fk_services.FKServices._validate
std::vector<std::uint8_t> FKServices::validate(
  const std::vector<std::vector<double>> & qs) const
{
  const std::size_t dof = joint_names_.size();
  curobo_runtime::TensorF32 q;
  q.shape = {static_cast<std::int64_t>(qs.size()), static_cast<std::int64_t>(dof)};
  for (const auto & row : qs) {
    for (std::size_t j = 0; j < dof; ++j) {
      q.data.push_back(static_cast<float>(j < row.size() ? row[j] : 0.0));
    }
  }
  try {
    return on_kin_lane(lane_, [&]() {return bridge_->validate(handle_, q);});
  } catch (const std::exception & e) {
    RCLCPP_ERROR(node_->get_logger(), "FK collision validation failed: %s", e.what());
    return std::vector<std::uint8_t>(qs.size(), 1);
  }
}

namespace
{
std::vector<std::vector<double>> joint_rows(
  const std::vector<sensor_msgs::msg::JointState> & states,
  const FKServices & services)
{
  std::vector<std::vector<double>> rows;
  rows.reserve(states.size());
  for (const auto & state : states) {
    rows.push_back(services.positions_for_model(state.position, state.name));
  }
  return rows;
}
}  // namespace

// PORT: fk_services.FKServices._fk_callback
void FKServices::handle_fk(
  const std::shared_ptr<curobov2_ros_interfaces::srv::Fk::Request> request,
  std::shared_ptr<curobov2_ros_interfaces::srv::Fk::Response> response)
{
  if (!initialized_) {
    RCLCPP_ERROR(node_->get_logger(), "FK not initialized. Call warmup_fk first.");
    return;
  }
  if (request->joint_states.empty()) {
    RCLCPP_ERROR(node_->get_logger(), "FK: no joint states provided");
    return;
  }
  const auto qs = joint_rows(request->joint_states, *this);

  std::vector<curobo_runtime::TensorF32> poses;
  bool ok = false;
  std::vector<std::uint8_t> valid;
  try {
    // PORT: fk_services._on_lane - FK jobs run on the shared `kin` lane; the
    // lane job itself holds barrier.shared() (GpuLane::run_job), so compute_poses
    // and validate must not pre-acquire the barrier here - doing so would block
    // on the lane while holding a read hold a waiting capture cannot get past.
    ok = compute_poses(qs, poses);
    if (ok) {
      valid = validate(qs);
    }
  } catch (const std::exception & exc) {
    RCLCPP_WARN_THROTTLE(
      node_->get_logger(), *node_->get_clock(), 5000,
      "FK failed, returning failure (GPU fault?): %s", exc.what());
    handle_gpu_fault(node_, "running FK", exc.what());
    return;
  }
  if (!ok) {
    return;
  }
  for (const auto & one : poses) {
    Pose7 pose;
    for (int i = 0; i < 7; ++i) {
      pose[i] = one.data[static_cast<std::size_t>(i)];
    }
    response->poses.push_back(to_ros_pose(pose));
  }
  for (std::uint8_t v : valid) {
    std_msgs::msg::Bool b;
    b.data = v != 0;
    response->poses_valid.push_back(b);
  }
}

// PORT: fk_services.FKServices._fk_batch_callback
void FKServices::handle_fk_batch(
  const std::shared_ptr<curobov2_ros_interfaces::srv::FkBatch::Request> request,
  std::shared_ptr<curobov2_ros_interfaces::srv::FkBatch::Response> response)
{
  if (!initialized_) {
    RCLCPP_ERROR(node_->get_logger(), "FK not initialized. Call warmup_fk first.");
    response->success = false;
    response->error_msg.data = "FK not initialized. Call warmup_fk first.";
    return;
  }
  if (request->joint_states.empty()) {
    RCLCPP_ERROR(node_->get_logger(), "FK batch: no joint states provided");
    response->success = false;
    response->error_msg.data = "FK batch: no joint states provided";
    return;
  }
  auto qs = joint_rows(request->joint_states, *this);
  const std::size_t n = qs.size();
  if (fk_batch_ > 0 && n > 0 && static_cast<int>(n) < fk_batch_) {
    for (int i = static_cast<int>(n); i < fk_batch_; ++i) {
      qs.push_back(qs.back());
    }
  } else if (fk_batch_ > 0 && static_cast<int>(n) > fk_batch_) {
    RCLCPP_WARN_THROTTLE(
      node_->get_logger(), *node_->get_clock(), 60000,
      "FK batch size %zu exceeds warmup width %d: GPU buffers realloc", n, fk_batch_);
  }

  std::vector<curobo_runtime::TensorF32> poses;
  std::vector<std::uint8_t> valid;
  bool ok = false;
  try {
    // PORT: fk_services._on_lane - shared `kin` lane, see handle_fk: the lane
    // job holds the shared barrier itself, so do not pre-acquire it here.
    ok = compute_poses(qs, poses);
    if (ok) {
      valid = validate(qs);
    }
  } catch (const std::exception & exc) {
    RCLCPP_WARN_THROTTLE(
      node_->get_logger(), *node_->get_clock(), 5000,
      "FK batch failed, returning failure (GPU fault?): %s", exc.what());
    handle_gpu_fault(node_, "running FK batch", exc.what());
    response->success = false;
    response->error_msg.data = std::string("FK batch failed: ") + exc.what();
    return;
  }
  if (!ok) {
    response->success = false;
    response->error_msg.data = "FK batch solve failed";
    return;
  }
  // Trim padding rows: only the first n entries belong to the request.
  for (std::size_t i = 0; i < n && i < poses.size(); ++i) {
    Pose7 pose;
    for (int j = 0; j < 7; ++j) {
      pose[j] = poses[i].data[static_cast<std::size_t>(j)];
    }
    response->poses.push_back(to_ros_pose(pose));
  }
  for (std::size_t i = 0; i < n && i < valid.size(); ++i) {
    std_msgs::msg::Bool b;
    b.data = valid[i] != 0;
    response->poses_valid.push_back(b);
  }
  response->success = true;
}

}  // namespace curobo_core

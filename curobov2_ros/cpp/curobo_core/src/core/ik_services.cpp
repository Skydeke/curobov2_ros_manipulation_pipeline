// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/core/ik_services.py.
#include "curobo_core/core/ik_services.hpp"
#include "curobo_core/core/gpu_runtime.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <utility>

#include "curobo_runtime/runtime.hpp"

namespace curobo_core
{

namespace
{
// PORT: ik_services._on_lane - run a GPU job on lane_kin (PRIO_KIN, the Python
// _LANE_TIMEOUT_S budget) unless this thread already IS the lane worker. The
// lane's single worker thread is what serialises IK/FK against each other; the
// port ran them inline on the caller's thread.
template<class F>
auto on_kin_lane(curobo_runtime::GpuLane * lane, F && fn) -> decltype(fn())
{
  if (lane == nullptr || lane->owns_current_thread()) {
    return fn();
  }
  curobo_runtime::SubmitOptions opts;
  opts.prio = curobo_runtime::PRIO_KIN;
  return lane->call_for(opts, 120.0, std::forward<F>(fn));
}
// PORT: frame_utils.resolve_poses_to_base - the server owns the transform. TF is
// not ported yet (M3), so only the empty/base-identity case resolves here; any
// other frame errors exactly like the Python path with no TF buffer.
std::vector<Pose7> resolve_poses_to_base(
  const std::vector<Pose7> & poses, const std::string & frame_id,
  const std::string & base_link)
{
  if (frame_id.empty() || frame_id == base_link) {
    return poses;
  }
  throw std::runtime_error(
    "cannot transform '" + frame_id + "' -> '" + base_link + "': no TF buffer");
}

curobo_runtime::TensorF32 poses_to_tensor(const std::vector<Pose7> & poses)
{
  curobo_runtime::TensorF32 tensor;
  tensor.shape = {static_cast<std::int64_t>(poses.size()), 7};
  tensor.data.reserve(poses.size() * 7);
  for (const auto & pose : poses) {
    for (int i = 0; i < 7; ++i) {
      tensor.data.push_back(static_cast<float>(pose[i]));
    }
  }
  return tensor;
}
}  // namespace

IKServices::IKServices(
  rclcpp::Node * node, curobo_runtime::IBridge * bridge,
  curobo_runtime::RobotHandle handle, std::string base_link, int num_ik_seeds)
: node_(node), bridge_(bridge), handle_(handle), base_link_(std::move(base_link)),
  num_ik_seeds_(num_ik_seeds)
{
  const std::string name = node_->get_name();
  warmup_srv_ = node_->create_service<curobov2_ros_interfaces::srv::WarmupIK>(
    name + "/warmup_ik",
    std::bind(&IKServices::handle_warmup_ik, this, std::placeholders::_1, std::placeholders::_2));
  ik_srv_ = node_->create_service<curobov2_ros_interfaces::srv::Ik>(
    name + "/ik",
    std::bind(&IKServices::handle_ik, this, std::placeholders::_1, std::placeholders::_2));
  ik_batch_srv_ = node_->create_service<curobov2_ros_interfaces::srv::IkBatch>(
    name + "/ik_batch",
    std::bind(&IKServices::handle_ik_batch, this, std::placeholders::_1, std::placeholders::_2));
  update_ik_config_srv_ = node_->create_service<std_srvs::srv::Trigger>(
    name + "/update_ik_config",
    std::bind(
      &IKServices::handle_update_ik_config, this,
      std::placeholders::_1, std::placeholders::_2));

  RCLCPP_INFO(
    node_->get_logger(),
    "IKServices registered (solver allocated by the startup warmup, or on demand "
    "via /warmup_ik when warmup_ik:=false)");
}

// PORT: ik_services.IKServices.warmup
std::string IKServices::warmup(int batch_size, int num_seeds)
{
  batch_size = std::max(1, batch_size);
  if (num_seeds < 0) {
    num_seeds = resolve_num_seeds();
  }
  if (ready_ && !needs_rebuild(batch_size, num_seeds)) {
    return "IK solver already ready (batch_size=" + std::to_string(batch_size) + ")";
  }
  // PORT: config_wrapper_ik.ConfigWrapperIK.build - resolve the tunables from
  // the node params at build time; they bake into the solver buffers.
  const bool self_collision_check = resolve_collision_check();
  const bool use_lm_seed = resolve_use_lm_seed();
  const bool use_cuda_graph = resolve_use_cuda_graph();
  RCLCPP_INFO(
    node_->get_logger(),
    "IK solver envelope: use_cuda_graph=%s, num_ik_seeds=%d, collision_check=%s, "
    "use_lm_seed=%s, batch_size=%d",
    use_cuda_graph ? "True" : "False", num_seeds,
    self_collision_check ? "True" : "False", use_lm_seed ? "True" : "False",
    batch_size);
  {
    // PORT: ik_services._gpu_guard + config_wrapper_ik.set_ik_config - building
    // the solver (Re)CAPTURES its CUDA graph, so it takes the exclusive
    // gpu_lock. Unguarded it could capture while the plan/camera lane was
    // capturing, invalidating one of them.
    on_kin_lane(lane_, [&]() {
        auto gpu_guard = curobo_runtime::Runtime::instance().barrier().exclusive();
        bridge_->ik_config(
          handle_, batch_size, num_seeds, self_collision_check, use_lm_seed,
          use_cuda_graph);
      });
  }
  // PORT: config_wrapper_ik build - the torch_sync CPU/GPU ordering bridge.
  if (node_ != nullptr && node_->has_parameter("torch_sync") &&
    node_->get_parameter("torch_sync").as_bool())
  {
    bridge_->torch_synchronize();
  }
  ik_batch_size_ = batch_size;
  ik_num_seeds_ = num_seeds;
  ik_collision_check_ = self_collision_check;
  ik_use_lm_seed_ = use_lm_seed;
  ik_use_cuda_graph_ = use_cuda_graph;
  ready_ = true;
  return "IK solver ready (batch_size=" + std::to_string(batch_size) + ")";
}

// PORT: config_wrapper_ik.ConfigWrapperIK._resolve_num_ik_seeds
int IKServices::resolve_num_seeds() const
{
  if (node_ != nullptr && node_->has_parameter("num_ik_seeds")) {
    return std::max(
      1, static_cast<int>(node_->get_parameter("num_ik_seeds").as_int()));
  }
  return num_ik_seeds_;
}

// PORT: config_wrapper_ik.ConfigWrapperIK._resolve_collision_check
bool IKServices::resolve_collision_check() const
{
  if (node_ != nullptr && node_->has_parameter("ik_collision_check")) {
    return node_->get_parameter("ik_collision_check").as_bool();
  }
  return true;
}

// PORT: config_wrapper_ik.ConfigWrapperIK._resolve_use_lm_seed
bool IKServices::resolve_use_lm_seed() const
{
  if (node_ != nullptr && node_->has_parameter("ik_use_lm_seed")) {
    return node_->get_parameter("ik_use_lm_seed").as_bool();
  }
  return true;
}

// PORT: config_wrapper.resolve_use_cuda_graph - env var wins, then the ROS
// param, then True.
bool IKServices::resolve_use_cuda_graph() const
{
  // PORT: config_wrapper.resolve_use_cuda_graph (shared helper).
  return curobo_core::resolve_use_cuda_graph(node_);
}

// PORT: config_wrapper_ik.ConfigWrapperIK.needs_rebuild
bool IKServices::needs_rebuild(int batch_size, int num_seeds) const
{
  return batch_size != ik_batch_size_ || num_seeds != ik_num_seeds_ ||
         resolve_collision_check() != ik_collision_check_ ||
         resolve_use_lm_seed() != ik_use_lm_seed_ ||
         resolve_use_cuda_graph() != ik_use_cuda_graph_;
}

// PORT: ik_services.IKServices._warmup_ik_callback
void IKServices::handle_warmup_ik(
  const std::shared_ptr<curobov2_ros_interfaces::srv::WarmupIK::Request> request,
  std::shared_ptr<curobov2_ros_interfaces::srv::WarmupIK::Response> response)
{
  try {
    response->message = warmup(static_cast<int>(request->batch_size));
    response->success = true;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(node_->get_logger(), "IK warmup failed: %s", e.what());
    response->success = false;
    response->message = e.what();
  }
}

// PORT: ik_services.IKServices._solve / _solve_locked
bool IKServices::solve(
  const std::vector<Pose7> & poses, curobo_runtime::IkOutcome & outcome)
{
  if (poses.empty()) {
    return false;
  }
  std::lock_guard<std::mutex> lock(solve_lock_);
  try {
    // PORT: ik_services.IKServices._solve - rebuild lazily when the request
    // batch no longer matches the live solver (needs_rebuild -> build). Without
    // this a batch larger than the warmed solver (e.g. a reachability grid)
    // solves against a too-small buffer and every cell fails. Also picks up a
    // `ros2 param set num_ik_seeds` / `ik_collision_check` flip, like the Python
    // `needs_rebuild(n, num_seeds)` check.
    const int requested = static_cast<int>(poses.size());
    if (!ready_ || needs_rebuild(requested, resolve_num_seeds())) {
      warmup(requested, -1);
    }
    // PORT: ik_services._on_lane - every solve ran on lane_kin inside
    // barrier.shared(); the port called the bridge bare, so a solve could issue
    // kernels while another thread captured a graph.
    auto gpu_guard = curobo_runtime::Runtime::instance().barrier().shared();
    outcome = on_kin_lane(lane_, [&]() {
        return bridge_->ik(handle_, poses_to_tensor(poses));
      });
    // PORT: ik_services._solve_call - the torch_sync CPU/GPU ordering bridge.
    if (node_ != nullptr && node_->has_parameter("torch_sync") &&
      node_->get_parameter("torch_sync").as_bool())
    {
      bridge_->torch_synchronize();
    }
    return true;
  } catch (const std::exception & e) {
    // PORT: a lane/solve fatal CUDA error must stop the node, not be
    // reported as an ordinary failure from a dead context.
    handle_gpu_fault(node_, "solving IK", e.what());
    RCLCPP_ERROR(node_->get_logger(), "IK solve failed: %s", e.what());
    return false;
  }
}

// PORT: ik_services.IKServices._ik_callback
void IKServices::handle_ik(
  const std::shared_ptr<curobov2_ros_interfaces::srv::Ik::Request> request,
  std::shared_ptr<curobov2_ros_interfaces::srv::Ik::Response> response)
{
  if (!ready_) {
    response->success = false;
    response->error_msg.data = "IK not initialized. Call warmup_ik first.";
    return;
  }
  Pose7 pose = from_ros_pose(request->pose);
  std::vector<Pose7> poses;
  try {
    poses = resolve_poses_to_base({pose}, request->frame_id, base_link_);
  } catch (const std::runtime_error & e) {
    response->success = false;
    response->error_msg.data = e.what();
    return;
  }

  curobo_runtime::IkOutcome outcome;
  try {
    if (!solve(poses, outcome)) {
      response->success = false;
      response->error_msg.data = "IK solve failed";
      return;
    }
  } catch (const std::exception & exc) {
    RCLCPP_WARN_THROTTLE(
      node_->get_logger(), *node_->get_clock(), 5000,
      "IK failed, returning failure (GPU fault?): %s", exc.what());
    response->success = false;
    response->error_msg.data = std::string("IK failed: ") + exc.what();
    return;
  }

  sensor_msgs::msg::JointState js;
  const std::size_t dof = outcome.positions.shape.size() > 2 ?
    static_cast<std::size_t>(outcome.positions.shape[2]) : 0;
  for (std::size_t j = 0; j < dof; ++j) {
    js.position.push_back(outcome.positions.data[j]);
  }
  std_msgs::msg::Bool valid;
  valid.data = !outcome.valid.empty() && outcome.valid[0] != 0;
  response->joint_states = js;
  response->joint_states_valid = valid;
  response->success = true;
}

// PORT: ik_services.IKServices._ik_batch_callback
void IKServices::handle_ik_batch(
  const std::shared_ptr<curobov2_ros_interfaces::srv::IkBatch::Request> request,
  std::shared_ptr<curobov2_ros_interfaces::srv::IkBatch::Response> response)
{
  if (!ready_) {
    response->success = false;
    response->error_msg.data = "IK not initialized. Call warmup_ik first.";
    return;
  }
  std::vector<Pose7> poses;
  poses.reserve(request->poses.size());
  for (const auto & msg : request->poses) {
    poses.push_back(from_ros_pose(msg));
  }
  try {
    poses = resolve_poses_to_base(poses, request->frame_id, base_link_);
  } catch (const std::runtime_error & e) {
    response->success = false;
    response->error_msg.data = e.what();
    return;
  }

  curobo_runtime::IkOutcome outcome;
  try {
    if (!solve(poses, outcome)) {
      response->success = false;
      response->error_msg.data = "IK batch solve failed";
      return;
    }
  } catch (const std::exception & exc) {
    RCLCPP_WARN_THROTTLE(
      node_->get_logger(), *node_->get_clock(), 5000,
      "IK batch failed, returning failure (GPU fault?): %s", exc.what());
    response->success = false;
    response->error_msg.data = std::string("IK batch failed: ") + exc.what();
    return;
  }

  const std::size_t dof = outcome.positions.shape.size() > 2 ?
    static_cast<std::size_t>(outcome.positions.shape[2]) : 0;
  for (std::size_t i = 0; i < poses.size(); ++i) {   // padded rows ignored
    sensor_msgs::msg::JointState js;
    for (std::size_t j = 0; j < dof; ++j) {
      const std::size_t idx = i * dof + j;
      js.position.push_back(idx < outcome.positions.data.size() ?
        outcome.positions.data[idx] : 0.0f);
    }
    std_msgs::msg::Bool valid;
    valid.data = i < outcome.valid.size() && outcome.valid[i] != 0;
    response->joint_states.push_back(js);
    response->joint_states_valid.push_back(valid);
  }
  response->success = true;
}

// PORT: ik_services.IKServices.solve_poses
bool IKServices::solve_poses(
  const std::vector<Pose7> & poses, std::vector<std::vector<float>> & positions,
  std::vector<std::uint8_t> & flags, std::vector<std::string> & joint_names)
{
  curobo_runtime::IkOutcome outcome;
  if (!solve(poses, outcome)) {
    return false;
  }
  const std::size_t dof = outcome.positions.shape.size() > 2 ?
    static_cast<std::size_t>(outcome.positions.shape[2]) : 0;
  positions.clear();
  flags = outcome.valid;
  joint_names = outcome.joint_names;
  for (std::size_t i = 0; i < poses.size(); ++i) {
    std::vector<float> row;
    for (std::size_t j = 0; j < dof; ++j) {
      const std::size_t idx = i * dof + j;
      row.push_back(idx < outcome.positions.data.size() ? outcome.positions.data[idx] : 0.0f);
    }
    positions.push_back(std::move(row));
  }
  return true;
}

// PORT: config_wrapper_ik.ConfigWrapperIK.set_ik_config
void IKServices::handle_update_ik_config(
  const std::shared_ptr<std_srvs::srv::Trigger::Request>,
  std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
  try {
    // (Re)build the solver from the current params (rebuilds the CUDA graph).
    warmup(std::max(1, ik_batch_size_), -1);
    response->success = true;
    response->message = "IK solver config set";
  } catch (const std::exception & e) {
    response->success = false;
    response->message = std::string("update_ik_config failed: ") + e.what();
  }
}

}  // namespace curobo_core

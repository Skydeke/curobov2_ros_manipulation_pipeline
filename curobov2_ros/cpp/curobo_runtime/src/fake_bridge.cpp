// SPDX-License-Identifier: Apache-2.0
#include "curobo_runtime/fake_bridge.hpp"

#include <cmath>
#include <string>

namespace curobo_runtime
{

RobotDescriptionOutcome FakeBridge::robot_description(
  const std::string &, const std::string &)
{
  // Test double: no descriptor (the node keeps its parameter defaults).
  return RobotDescriptionOutcome{};
}

RobotHandle FakeBridge::init(const std::string & cfg)
{
  std::lock_guard<std::mutex> lk(mutex_);
  init_cfg_ = cfg;
  interpreter_init_count_ += 1;
  initialized_ = true;
  return 1;
}

void FakeBridge::shutdown(RobotHandle handle)
{
  std::lock_guard<std::mutex> lk(mutex_);
  initialized_ = false;
  shutdown_handle_ = handle;
}

bool FakeBridge::initialized() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return initialized_;
}

int FakeBridge::interpreter_init_count() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return interpreter_init_count_;
}

void FakeBridge::bind_lane(const std::string & name)
{
  std::lock_guard<std::mutex> lk(mutex_);
  bound_lanes_.push_back(name);
}

void FakeBridge::device_sync()
{
  std::lock_guard<std::mutex> lk(mutex_);
  device_sync_count_ += 1;
}

void FakeBridge::sync_lane()
{
  std::lock_guard<std::mutex> lk(mutex_);
  sync_lane_count_ += 1;
}

void FakeBridge::set_global_flags(bool cuda_graph_reset, bool debug_cuda_graphs)
{
  std::lock_guard<std::mutex> lk(mutex_);
  global_flags_ = cuda_graph_reset && debug_cuda_graphs;
}

std::vector<std::string> FakeBridge::bound_lanes() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return bound_lanes_;
}

int FakeBridge::depth_integrates() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return depth_integrates_;
}

int FakeBridge::device_sync_count() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return device_sync_count_;
}

int FakeBridge::sync_lane_count() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return sync_lane_count_;
}

std::string FakeBridge::init_cfg() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return init_cfg_;
}

bool FakeBridge::global_flags() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return global_flags_;
}

RobotHandle FakeBridge::shutdown_handle() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return shutdown_handle_;
}

// ---- M2 ------------------------------------------------------------------
// The fake is deterministic and CPU-only: FK is a fixed function of q, IK
// validity is "position inside the unit sphere", validate is all-True, scene
// ops just track a version. Tests assert these exact values, so a C++/bridge
// marshalling bug shows up without a GPU.

void FakeBridge::robot_model_build(RobotHandle, const std::string & cfg_json)
{
  std::lock_guard<std::mutex> lk(mutex_);
  model_cfg_ = cfg_json;
}

TensorF32 FakeBridge::fk(RobotHandle, const TensorF32 & q)
{
  std::lock_guard<std::mutex> lk(mutex_);
  const std::int64_t n = q.shape.empty() ? 0 : q.shape[0];
  const std::int64_t dof = q.shape.size() > 1 ? q.shape[1] : 0;
  TensorF32 out;
  out.shape = {n, 1, 7};
  out.data.assign(static_cast<std::size_t>(n) * 7, 0.0f);
  for (std::int64_t i = 0; i < n; ++i) {
    const float * row = q.data.data() + static_cast<std::size_t>(i) * dof;
    float * dst = out.data.data() + static_cast<std::size_t>(i) * 7;
    // [x,y,z,qw,qx,qy,qz]; identity quaternion unless dof >= 6.
    dst[0] = row[0];
    dst[1] = dof > 1 ? row[1] : 0.0f;
    dst[2] = dof > 2 ? row[2] : 0.0f;
    dst[3] = 1.0f;
    dst[4] = dof > 3 ? row[3] : 0.0f;
    dst[5] = dof > 4 ? row[4] : 0.0f;
    dst[6] = dof > 5 ? row[5] : 0.0f;
  }
  return out;
}

IkOutcome FakeBridge::ik(RobotHandle, const TensorF32 & poses)
{
  std::lock_guard<std::mutex> lk(mutex_);
  const std::int64_t n = poses.shape.empty() ? 0 : poses.shape[0];
  IkOutcome out;
  out.positions.shape = {n, 1, static_cast<std::int64_t>(dof_)};
  out.positions.data.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(dof_), 0.0f);
  out.valid.resize(static_cast<std::size_t>(n), 0);
  for (std::int64_t i = 0; i < n; ++i) {
    const float * row = poses.data.data() + static_cast<std::size_t>(i) * 7;
    const double norm = std::sqrt(
      static_cast<double>(row[0]) * row[0] + static_cast<double>(row[1]) * row[1] +
      static_cast<double>(row[2]) * row[2]);
    out.valid[static_cast<std::size_t>(i)] = norm <= 1.0 ? 1 : 0;
    out.success = out.success || norm <= 1.0;
  }
  for (int j = 0; j < dof_; ++j) {
    out.joint_names.push_back("dof_" + std::to_string(j));
  }
  return out;
}

std::vector<std::uint8_t> FakeBridge::validate(RobotHandle, const TensorF32 & q)
{
  std::lock_guard<std::mutex> lk(mutex_);
  const std::int64_t n = q.shape.empty() ? 0 : q.shape[0];
  return std::vector<std::uint8_t>(static_cast<std::size_t>(n), 1);
}

std::vector<float> FakeBridge::collision_distance(RobotHandle, const TensorF32 & q)
{
  std::lock_guard<std::mutex> lk(mutex_);
  // One distance per element, equal to the joint value: order-preserving and
  // trivially assertable.
  return std::vector<float>(q.data.begin(), q.data.end());
}

CollisionAttribution FakeBridge::collision_attribution(RobotHandle, const TensorF32 &)
{
  std::lock_guard<std::mutex> lk(mutex_);
  // No attributed contacts by default; tests that need them use the PyBridge.
  return CollisionAttribution{};
}

CacheBucketCounts FakeBridge::collision_cache_bucket_counts(RobotHandle)
{
  std::lock_guard<std::mutex> lk(mutex_);
  return CacheBucketCounts{};
}

MpcCostBreakdown FakeBridge::mpc_cost_breakdown(RobotHandle)
{
  std::lock_guard<std::mutex> lk(mutex_);
  return MpcCostBreakdown{};
}

void FakeBridge::torch_synchronize() {}

PlanImageOutcome FakeBridge::render_plan_image(
  const TensorF32 &, const TensorF32 &, const TensorF32 &,
  const std::vector<std::string> &, double)
{
  return PlanImageOutcome{};
}

SceneVersion FakeBridge::scene_set(
  RobotHandle, const std::string & objects_json, bool clear_first)
{
  std::lock_guard<std::mutex> lk(mutex_);
  scene_version_ += 1;
  scene_json_ = objects_json;
  last_clear_first_ = clear_first;
  return scene_version_;
}

std::string FakeBridge::scene_get(RobotHandle)
{
  std::lock_guard<std::mutex> lk(mutex_);
  return scene_json_;
}

JointLimitsOutcome FakeBridge::joint_limits(RobotHandle)
{
  std::lock_guard<std::mutex> lk(mutex_);
  JointLimitsOutcome out;
  out.positions.shape = {2, dof_};
  out.positions.data.resize(static_cast<std::size_t>(dof_) * 2);
  for (int j = 0; j < dof_; ++j) {
    out.positions.data[static_cast<std::size_t>(j)] = -1.0f;                       // lower
    out.positions.data[static_cast<std::size_t>(dof_) + static_cast<std::size_t>(j)] = 1.0f;  // upper
    out.joint_names.push_back("dof_" + std::to_string(j));
  }
  return out;
}

void FakeBridge::set_dof(int dof)
{
  std::lock_guard<std::mutex> lk(mutex_);
  dof_ = dof;
}

void FakeBridge::ik_config(RobotHandle, int batch_size, int num_seeds)
{
  std::lock_guard<std::mutex> lk(mutex_);
  ik_batch_ = batch_size;
  ik_num_seeds_ = num_seeds;
}

// PORT: attachment_services.attach / detach (test double: tracks the name).
void FakeBridge::attach_object(
  RobotHandle, const std::string & name, const std::vector<float> &)
{
  std::lock_guard<std::mutex> lk(mutex_);
  attached_ = name;
}

std::string FakeBridge::detach_object(RobotHandle)
{
  std::lock_guard<std::mutex> lk(mutex_);
  const std::string released = attached_;
  attached_.clear();
  return released;
}

// M3: the fake planner is deterministic. plan_cspace succeeds when the target is
// within 10 rad of the start (else "no solution"); plan_pose succeeds when the
// goal pose is inside the unit sphere (mirroring the IK double). The plan is a
// straight line of 3 waypoints.
void FakeBridge::motion_planner_build(RobotHandle, const std::string & cfg_json)
{
  std::lock_guard<std::mutex> lk(mutex_);
  planner_cfg_ = cfg_json;
}

namespace
{
PlanOutcome line_plan(const std::vector<float> & start, const std::vector<float> & target)
{
  PlanOutcome out;
  out.positions.shape = {3, static_cast<std::int64_t>(start.size())};
  for (int t = 0; t < 3; ++t) {
    const float w = static_cast<float>(t) / 2.0f;
    for (std::size_t j = 0; j < start.size(); ++j) {
      out.positions.data.push_back(start[j] + w * (target[j] - start[j]));
    }
  }
  out.success = true;
  out.feasible = true;
  out.status = "success";
  out.total_time = 0.001;
  return out;
}
}  // namespace

PlanOutcome FakeBridge::plan_cspace(RobotHandle, const PlanRequest & request)
{
  PlanOutcome out;
  const auto & target = request.joint_target;
  if (target.size() != request.start.size() || target.empty()) {
    out.status = "no solution";
    return out;
  }
  float d = 0.0f;
  for (std::size_t j = 0; j < target.size(); ++j) {
    d += std::fabs(target[j] - request.start[j]);
  }
  if (d > 10.0f) {
    out.status = "no solution";
    return out;
  }
  return line_plan(request.start, target);
}

void FakeBridge::reset_seed(RobotHandle)
{
  std::lock_guard<std::mutex> lk(mutex_);
  seed_resets_ += 1;
}

void FakeBridge::robot_model_rebuild_locks(RobotHandle, const std::string & lock_json)
{
  std::lock_guard<std::mutex> lk(mutex_);
  model_lock_json_ = lock_json;
}

// M4 perception doubles: a fixed 4x4x4 grid whose cell (i,j,k) is occupied when
// (i+j+k) is even, so the occupied-index arithmetic is easy to assert.
void FakeBridge::configure_mapper(RobotHandle, const std::string &) {}

bool FakeBridge::esdf_refresh(RobotHandle)
{
  std::lock_guard<std::mutex> lk(mutex_);
  esdf_refreshes_ += 1;
  return true;
}

VoxelGrid FakeBridge::voxel_grid(RobotHandle)
{
  std::lock_guard<std::mutex> lk(mutex_);
  if (has_voxel_grid_override_) {
    return voxel_grid_override_;
  }
  VoxelGrid grid;
  grid.shape = {4, 4, 4};
  grid.resolution = 0.05;
  grid.origin = {-0.1, -0.1, -0.1};
  grid.data.resize(64, 0);
  for (std::int64_t i = 0; i < 4; ++i) {
    for (std::int64_t j = 0; j < 4; ++j) {
      for (std::int64_t k = 0; k < 4; ++k) {
        if ((i + j + k) % 2 == 0) {
          grid.data[static_cast<std::size_t>((i * 4 + j) * 4 + k)] = 1;
        }
      }
    }
  }
  return grid;
}

bool FakeBridge::world_push(RobotHandle)
{
  return true;
}

int FakeBridge::clear_voxel_map(RobotHandle)
{
  std::lock_guard<std::mutex> lk(mutex_);
  return 1;
}

int FakeBridge::clear_voxel_region(
  RobotHandle, const std::array<double, 3> &, const std::array<double, 3> &)
{
  std::lock_guard<std::mutex> lk(mutex_);
  return 1;
}

void FakeBridge::depth_integrate(
  RobotHandle, const TensorF32 &, const TensorF32 &, const std::array<double, 7> &)
{
  std::lock_guard<std::mutex> lk(mutex_);
  depth_integrates_ += 1;
}

void FakeBridge::lidar_integrate(
  RobotHandle, const TensorF32 &, const TensorF32 &, const TensorF32 &, const TensorF32 &)
{
  std::lock_guard<std::mutex> lk(mutex_);
  lidar_integrates_ += 1;
}

int FakeBridge::lidar_integrates() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return lidar_integrates_;
}

SphereSetOutcome FakeBridge::collision_spheres_with_attached(
  RobotHandle, const std::vector<float> & q)
{
  // Test double: two spheres (one robot, one attached), radius 0.05 / -100.
  SphereSetOutcome out;
  out.spheres.shape = {2, 4};
  const float x = q.empty() ? 0.0f : q[0];
  out.spheres.data = {x, 0.0f, 0.0f, 0.05f, 0.0f, 0.0f, 0.0f, -100.0f};
  out.attached = {0, 1};
  return out;
}

std::vector<std::string> FakeBridge::tool_frames(RobotHandle)
{
  return {"tool0"};
}

LinkCollisionOutcome FakeBridge::set_link_collision(
  RobotHandle, const std::vector<std::string> & link_names, bool)
{
  // Test double: every named link is known.
  LinkCollisionOutcome out;
  out.applied = link_names;
  return out;
}

TensorF32 FakeBridge::robot_segment(
  RobotHandle, const TensorF32 & depth, const TensorF32 &, const std::array<double, 7> &,
  const std::vector<float> &, const std::string &)
{
  // Test double: identity mask (returns the depth unchanged).
  return depth;
}

PlanOutcome FakeBridge::plan_pose(RobotHandle, const PlanRequest & request)
{
  PlanOutcome out;
  if (request.goalset.shape.empty() || request.goalset.shape[0] == 0) {
    out.status = "no solution";
    return out;
  }
  const float * row = request.goalset.data.data();
  const double norm = std::sqrt(
    static_cast<double>(row[0]) * row[0] + static_cast<double>(row[1]) * row[1] +
    static_cast<double>(row[2]) * row[2]);
  if (norm > 1.0) {
    out.status = "no solution";
    return out;
  }
  // Goal joint row: use the pose xyz as a stand-in target for the line plan.
  std::vector<float> target = request.start;
  for (std::size_t j = 0; j < target.size() && j < 3; ++j) {
    target[j] = row[j];
  }
  return line_plan(request.start, target);
}

int FakeBridge::dof() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return dof_;
}

std::string FakeBridge::model_cfg() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return model_cfg_;
}

SceneVersion FakeBridge::scene_version() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return scene_version_;
}

bool FakeBridge::last_clear_first() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return last_clear_first_;
}

// ---- M5: reactive control (deterministic first-order servo double) ----------
void FakeBridge::mpc_build(RobotHandle, const std::string & cfg_json)
{
  std::lock_guard<std::mutex> lk(mutex_);
  mpc_cfg_ = cfg_json;
  mpc_builds_ += 1;
}

MpcGoalOutcome FakeBridge::mpc_setup(
  RobotHandle, const std::vector<float> & start_q)
{
  std::lock_guard<std::mutex> lk(mutex_);
  MpcGoalOutcome out;
  out.applied = true;
  out.has_ee = start_q.size() >= 3;
  if (out.has_ee) {
    out.ee_position = {start_q[0], start_q[1], start_q[2]};
  }
  return out;
}

MpcStepOutcome FakeBridge::mpc_step(RobotHandle, const ReactiveState & state, bool paced)
{
  std::lock_guard<std::mutex> lk(mutex_);
  mpc_steps_ += 1;
  if (paced) {
    mpc_paced_steps_ += 1;
  }
  MpcStepOutcome out;
  out.has_ee = state.position.size() >= 3;
  if (out.has_ee) {
    out.ee_position = {state.position[0], state.position[1], state.position[2]};
  }
  // First-order pull toward the goal on the first three joints (converges so the
  // loop's on_target signal can be exercised).
  std::vector<float> next = state.position;
  if (next.size() >= 3) {
    next[0] += 0.5f * (static_cast<float>(mpc_goal_[0]) - next[0]);
    next[1] += 0.5f * (static_cast<float>(mpc_goal_[1]) - next[1]);
    next[2] += 0.5f * (static_cast<float>(mpc_goal_[2]) - next[2]);
  }
  const std::int64_t rows = paced ? 2 : 1;
  out.action.position.shape = {rows, static_cast<std::int64_t>(next.size())};
  out.action.velocity.shape = {rows, static_cast<std::int64_t>(next.size())};
  out.action.acceleration.shape = {rows, static_cast<std::int64_t>(next.size())};
  for (std::int64_t r = 0; r < rows; ++r) {
    out.action.position.data.insert(
      out.action.position.data.end(), next.begin(), next.end());
    out.action.velocity.data.insert(
      out.action.velocity.data.end(), next.size(), 0.0f);
    out.action.acceleration.data.insert(
      out.action.acceleration.data.end(), next.size(), 0.0f);
  }
  out.action.has_action = true;
  return out;
}

bool FakeBridge::mpc_update_goal_pose(
  RobotHandle, const std::array<double, 7> & goal_raw, bool)
{
  std::lock_guard<std::mutex> lk(mutex_);
  mpc_goal_ = goal_raw;
  mpc_goal_set_ = true;
  return true;
}

void FakeBridge::mpc_update_goal_state(RobotHandle, const std::vector<float> &) {}

void FakeBridge::mpc_reseed(RobotHandle, const std::vector<float> &)
{
  std::lock_guard<std::mutex> lk(mutex_);
  mpc_reseeds_ += 1;
}

IkOutcome FakeBridge::mpc_solve_goal_state(
  RobotHandle, const std::array<double, 7> &, const std::vector<float> & seed_q)
{
  IkOutcome out;
  out.success = true;
  out.positions.shape = {1, 1, static_cast<std::int64_t>(seed_q.size())};
  out.positions.data = seed_q;
  out.valid = {1};
  return out;
}

void FakeBridge::mpc_load_scene(RobotHandle, const std::string &) {}

std::string FakeBridge::mpc_metrics(RobotHandle)
{
  return "{}";
}

void FakeBridge::retarget_build(RobotHandle, const std::string & cfg_json)
{
  std::lock_guard<std::mutex> lk(mutex_);
  mpc_cfg_ = cfg_json;
}

void FakeBridge::retarget_reset(RobotHandle)
{
  std::lock_guard<std::mutex> lk(mutex_);
}

MpcStepOutcome FakeBridge::retarget_step(
  RobotHandle, const std::array<double, 7> & goal_raw)
{
  std::lock_guard<std::mutex> lk(mutex_);
  retarget_steps_ += 1;
  MpcStepOutcome out;
  out.action.position.shape = {1, 7};
  out.action.velocity.shape = {1, 7};
  out.action.acceleration.shape = {1, 7};
  out.action.position.data = {
    static_cast<float>(goal_raw[0]), static_cast<float>(goal_raw[1]),
    static_cast<float>(goal_raw[2]), 0.0f, 0.0f, 0.0f, 0.0f};
  out.action.velocity.data.assign(7, 0.0f);
  out.action.acceleration.data.assign(7, 0.0f);
  out.action.has_action = true;
  out.has_ee = true;
  out.ee_position = {goal_raw[0], goal_raw[1], goal_raw[2]};
  return out;
}

void FakeBridge::retarget_load_scene(RobotHandle, const std::string &) {}

int FakeBridge::mpc_builds() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return mpc_builds_;
}

int FakeBridge::mpc_steps() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return mpc_steps_;
}

int FakeBridge::mpc_paced_steps() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return mpc_paced_steps_;
}

int FakeBridge::retarget_steps() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return retarget_steps_;
}

int FakeBridge::mpc_reseeds() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return mpc_reseeds_;
}

std::string FakeBridge::mpc_cfg() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return mpc_cfg_;
}

}  // namespace curobo_runtime

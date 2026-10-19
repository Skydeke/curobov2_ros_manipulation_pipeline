// SPDX-License-Identifier: Apache-2.0
// Deterministic bridge with no interpreter and no GPU, for unit tests
// (design section 14.2). Records the calls so control flow can be asserted.
#ifndef CUROBO_RUNTIME__FAKE_BRIDGE_HPP_
#define CUROBO_RUNTIME__FAKE_BRIDGE_HPP_

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "curobo_runtime/bridge.hpp"

namespace curobo_runtime
{

class FakeBridge : public IBridge
{
public:
  RobotDescriptionOutcome robot_description(
    const std::string & robot_arg, const std::string & robot_config_file) override;
  RobotHandle init(const std::string & cfg) override;
  void shutdown(RobotHandle handle) override;
  bool initialized() const override;
  int interpreter_init_count() const override;

  void bind_lane(const std::string & name) override;
  void device_sync() override;
  void sync_lane() override;
  void set_global_flags(bool cuda_graph_reset, bool debug_cuda_graphs) override;
  std::vector<std::string> bound_lanes() const override;

  // ---- M2 ----
  void robot_model_build(RobotHandle handle, const std::string & cfg_json) override;
  TensorF32 fk(RobotHandle handle, const TensorF32 & q) override;
  IkOutcome ik(RobotHandle handle, const TensorF32 & poses) override;
  void ik_config(
    RobotHandle handle, int batch_size, int num_seeds, bool self_collision_check,
    bool use_lm_seed, bool use_cuda_graph) override;
  std::vector<std::uint8_t> validate(RobotHandle handle, const TensorF32 & q) override;
  std::vector<float> collision_distance(RobotHandle handle, const TensorF32 & q) override;
  CollisionAttribution collision_attribution(
    RobotHandle handle, const TensorF32 & q) override;
  CacheBucketCounts collision_cache_bucket_counts(RobotHandle handle) override;
  MpcCostBreakdown mpc_cost_breakdown(RobotHandle handle) override;
  void torch_synchronize() override;
  PlanImageOutcome render_plan_image(
    const TensorF32 & position, const TensorF32 & velocity,
    const TensorF32 & acceleration, const std::vector<std::string> & names,
    double dt) override;
  SceneVersion scene_set(
    RobotHandle handle, const std::string & objects_json, bool clear_first) override;
  std::string scene_get(RobotHandle handle) override;
  JointLimitsOutcome joint_limits(RobotHandle handle) override;
  void attach_object(
    RobotHandle handle, const std::string & name, const std::vector<float> & q) override;
  std::string detach_object(RobotHandle handle) override;
  SphereSetOutcome collision_spheres_with_attached(
    RobotHandle handle, const std::vector<float> & q) override;
  std::vector<std::string> tool_frames(RobotHandle handle) override;
  LinkCollisionOutcome set_link_collision(
    RobotHandle handle, const std::vector<std::string> & link_names, bool enabled) override;

  // M3
  void motion_planner_build(RobotHandle handle, const std::string & cfg_json) override;
  PlanOutcome plan_pose(RobotHandle handle, const PlanRequest & request) override;
  PlanOutcome plan_cspace(RobotHandle handle, const PlanRequest & request) override;
  void reset_seed(RobotHandle handle) override;
  void robot_model_rebuild_locks(RobotHandle handle, const std::string & lock_json) override;
  void configure_mapper(RobotHandle handle, const std::string & cfg_json) override;
  bool esdf_refresh(RobotHandle handle) override;
  bool world_push(RobotHandle handle) override;
  VoxelGrid voxel_grid(RobotHandle handle) override;
  // Test hook: install the grid voxel_grid() returns (parity tests replay a
  // recorded oracle grid through the analytic rasterization).
  void set_voxel_grid(const VoxelGrid & grid)
  {
    std::lock_guard<std::mutex> lk(mutex_);
    voxel_grid_override_ = grid;
    has_voxel_grid_override_ = true;
  }
  int clear_voxel_map(RobotHandle handle) override;
  int clear_voxel_region(
    RobotHandle handle, const std::array<double, 3> & bounds_min,
    const std::array<double, 3> & bounds_max) override;
  void depth_integrate(
    RobotHandle handle, const TensorF32 & depth, const TensorF32 & intrinsics,
    const std::array<double, 7> & camera_pose) override;
  void lidar_integrate(
    RobotHandle handle, const TensorF32 & range_image, const TensorF32 & poses,
    const TensorF32 & valid_range, const TensorF32 & elevation_range) override;
  TensorF32 robot_segment(
    RobotHandle handle, const TensorF32 & depth, const TensorF32 & intrinsics_meta,
    const std::array<double, 7> & cam_pose, const std::vector<float> & q,
    const std::string & masks_json) override;

  // M5
  void mpc_build(RobotHandle handle, const std::string & cfg_json) override;
  MpcGoalOutcome mpc_setup(
    RobotHandle handle, const std::vector<float> & start_q) override;
  MpcStepOutcome mpc_step(
    RobotHandle handle, const ReactiveState & state, bool paced) override;
  bool mpc_update_goal_pose(
    RobotHandle handle, const std::array<double, 7> & goal_raw, bool run_ik) override;
  void mpc_update_goal_state(RobotHandle handle, const std::vector<float> & q) override;
  void mpc_reseed(RobotHandle handle, const std::vector<float> & q) override;
  IkOutcome mpc_solve_goal_state(
    RobotHandle handle, const std::array<double, 7> & goal_raw,
    const std::vector<float> & seed_q) override;
  void mpc_load_scene(RobotHandle handle, const std::string & scene_json) override;
  std::string mpc_metrics(RobotHandle handle) override;
  void retarget_build(RobotHandle handle, const std::string & cfg_json) override;
  void retarget_reset(RobotHandle handle) override;
  MpcStepOutcome retarget_step(
    RobotHandle handle, const std::array<double, 7> & goal_raw) override;
  void retarget_load_scene(RobotHandle handle, const std::string & scene_json) override;

  // Introspection for tests.
  int depth_integrates() const;
  int lidar_integrates() const;
  int device_sync_count() const;
  int sync_lane_count() const;
  std::string init_cfg() const;
  bool global_flags() const;                 // true if set_global_flags(true, true) seen
  RobotHandle shutdown_handle() const;
  // Test double knobs (M2).
  void set_dof(int dof);                     // default 7
  int dof() const;
  std::string model_cfg() const;
  SceneVersion scene_version() const;
  bool last_clear_first() const;
  // M5 test hooks.
  int mpc_builds() const;
  int mpc_steps() const;
  int mpc_paced_steps() const;
  int mpc_reseeds() const;
  int retarget_steps() const;
  std::string mpc_cfg() const;
  // IK test hooks (the standalone IK solver build: num_seeds/flags are baked in,
  // so the port must forward what ConfigWrapperIK.build resolved).
  int ik_builds() const;
  int ik_batch() const;
  int ik_num_seeds() const;
  bool ik_self_collision_check() const;
  bool ik_use_lm_seed() const;
  bool ik_use_cuda_graph() const;

private:
  mutable std::mutex mutex_;
  bool initialized_ = false;
  int interpreter_init_count_ = 0;
  int device_sync_count_ = 0;
  int sync_lane_count_ = 0;
  bool global_flags_ = false;
  std::string init_cfg_;
  std::vector<std::string> bound_lanes_;
  RobotHandle shutdown_handle_ = -1;
  int dof_ = 7;
  std::string model_cfg_;
  SceneVersion scene_version_ = 0;
  std::string scene_json_ = "[]";
  bool last_clear_first_ = false;
  VoxelGrid voxel_grid_override_;
  bool has_voxel_grid_override_ = false;
  // M5 test-double state.
  std::string mpc_cfg_;
  std::array<double, 7> mpc_goal_{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
  bool mpc_goal_set_ = false;
  int mpc_builds_ = 0;
  int mpc_steps_ = 0;
  int mpc_paced_steps_ = 0;
  int retarget_steps_ = 0;
  int mpc_reseeds_ = 0;
  int ik_batch_ = 0;
  int ik_num_seeds_ = 0;
  int ik_builds_ = 0;
  bool ik_self_collision_check_ = true;
  bool ik_use_lm_seed_ = true;
  bool ik_use_cuda_graph_ = true;
  std::string attached_;
  std::string planner_cfg_;
  int seed_resets_ = 0;
  int esdf_refreshes_ = 0;
  int depth_integrates_ = 0;
  int lidar_integrates_ = 0;
  std::string model_lock_json_;
};

}  // namespace curobo_runtime

#endif  // CUROBO_RUNTIME__FAKE_BRIDGE_HPP_

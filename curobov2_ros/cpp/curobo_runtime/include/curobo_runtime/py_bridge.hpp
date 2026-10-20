// SPDX-License-Identifier: Apache-2.0
// PyBridge: the real bridge. Owns the embedded CPython interpreter and forwards
// to the ROS-free Python module `curobo_bridge` (design sections 5, 6).
//
// pybind11 is confined to this package (rule 11.3); this header exposes no
// pybind types (pimpl).
#ifndef CUROBO_RUNTIME__PY_BRIDGE_HPP_
#define CUROBO_RUNTIME__PY_BRIDGE_HPP_

#include <memory>
#include <string>
#include <vector>

#include "curobo_runtime/bridge.hpp"

namespace curobo_runtime
{

class PyBridge : public IBridge
{
public:
  PyBridge();
  ~PyBridge() override;

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
  void prepare_solve_mode(RobotHandle handle, const std::string & mode) override;

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

  void motion_planner_build(RobotHandle handle, const std::string & cfg_json) override;
  PlanOutcome plan_pose(RobotHandle handle, const PlanRequest & request) override;
  PlanOutcome plan_cspace(RobotHandle handle, const PlanRequest & request) override;
  void reset_seed(RobotHandle handle) override;
  void robot_model_rebuild_locks(RobotHandle handle, const std::string & lock_json) override;
  void configure_mapper(RobotHandle handle, const std::string & cfg_json) override;
  void warmup_mapper(RobotHandle handle) override;
  bool esdf_refresh(RobotHandle handle) override;
  bool world_push(RobotHandle handle) override;
  VoxelGrid voxel_grid(RobotHandle handle) override;
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

  // Test/introspection: interpreter state and imported modules.
  bool interpreter_running() const;
  bool module_available(const std::string & name) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace curobo_runtime

#endif  // CUROBO_RUNTIME__PY_BRIDGE_HPP_

// SPDX-License-Identifier: Apache-2.0
// CppBridge: the cpp_runtime implementation of curobo_runtime::IBridge.
//
// It is installed in Runtime before curobo_core::UnifiedPlannerNode is
// constructed, so the ROS surface (services, actions, parameters) is identical
// to the Python backend. With `auto`, the bridge also owns a Python fallback:
// native methods are used while the native scene is an exact mirror; methods
// that only the Python backend implements switch the planner to Python so the
// two scenes cannot silently diverge. Explicit `cpp` keeps the fallback off and
// throws for unsupported methods.
#ifndef CUROBO_CPP_RUNTIME__NATIVE_BRIDGE_HPP_
#define CUROBO_CPP_RUNTIME__NATIVE_BRIDGE_HPP_

#include <memory>
#include <string>

#include "curobo_runtime/bridge.hpp"

namespace curobo_cpp_runtime
{

class CppBridge : public curobo_runtime::IBridge
{
public:
  CppBridge(
    std::string artifact_directory,
    std::string robot_name = "",
    std::string robot_config_file = "",
    int device_ordinal = 0,
    bool python_fallback = false);
  ~CppBridge() override;

  CppBridge(const CppBridge &) = delete;
  CppBridge & operator=(const CppBridge &) = delete;

  const std::string & artifact_directory() const;
  const std::string & robot_name() const;

  // ---- lifecycle / lanes ----
  curobo_runtime::RobotDescriptionOutcome robot_description(
    const std::string & robot_arg, const std::string & robot_config_file) override;
  curobo_runtime::RobotHandle init(const std::string & cfg) override;
  void shutdown(curobo_runtime::RobotHandle handle) override;
  bool initialized() const override;
  int interpreter_init_count() const override;
  void bind_lane(const std::string & name) override;
  void device_sync() override;
  void sync_lane() override;
  void set_global_flags(bool cuda_graph_reset, bool debug_cuda_graphs) override;
  std::vector<std::string> bound_lanes() const override;
  void prepare_solve_mode(curobo_runtime::RobotHandle handle, const std::string & mode) override;

  // ---- robot model / FK / IK ----
  void robot_model_build(curobo_runtime::RobotHandle handle, const std::string & cfg_json) override;
  curobo_runtime::TensorF32 fk(
    curobo_runtime::RobotHandle handle, const curobo_runtime::TensorF32 & q) override;
  curobo_runtime::LinkPosesOutcome link_poses(
    curobo_runtime::RobotHandle handle, const curobo_runtime::TensorF32 & q,
    const std::vector<std::string> & link_names) override;
  curobo_runtime::IkOutcome ik(
    curobo_runtime::RobotHandle handle, const curobo_runtime::TensorF32 & poses,
    const std::string & tool_frame = "") override;
  curobo_runtime::IkOutcome ik_multi(
    curobo_runtime::RobotHandle handle, const curobo_runtime::TensorF32 & poses,
    const std::vector<std::string> & tool_frames) override;
  void ik_config(
    curobo_runtime::RobotHandle handle, int batch_size, int num_seeds,
    bool self_collision_check, bool use_lm_seed, bool use_cuda_graph) override;
  std::vector<std::uint8_t> validate(
    curobo_runtime::RobotHandle handle, const curobo_runtime::TensorF32 & q) override;
  std::vector<float> collision_distance(
    curobo_runtime::RobotHandle handle, const curobo_runtime::TensorF32 & q) override;
  curobo_runtime::CollisionAttribution collision_attribution(
    curobo_runtime::RobotHandle handle, const curobo_runtime::TensorF32 & q) override;
  curobo_runtime::CacheBucketCounts collision_cache_bucket_counts(
    curobo_runtime::RobotHandle handle) override;
  curobo_runtime::MpcCostBreakdown mpc_cost_breakdown(
    curobo_runtime::RobotHandle handle) override;
  void torch_synchronize() override;
  curobo_runtime::PlanImageOutcome render_plan_image(
    const curobo_runtime::TensorF32 & position, const curobo_runtime::TensorF32 & velocity,
    const curobo_runtime::TensorF32 & acceleration,
    const std::vector<std::string> & names, double dt) override;

  // ---- scene / introspection ----
  curobo_runtime::SceneVersion scene_set(
    curobo_runtime::RobotHandle handle, const std::string & objects_json,
    bool clear_first) override;
  std::string scene_get(curobo_runtime::RobotHandle handle) override;
  curobo_runtime::JointLimitsOutcome joint_limits(
    curobo_runtime::RobotHandle handle) override;
  void attach_object(
    curobo_runtime::RobotHandle handle, const std::string & name,
    const std::vector<float> & q) override;
  std::string detach_object(curobo_runtime::RobotHandle handle) override;
  curobo_runtime::SphereSetOutcome collision_spheres_with_attached(
    curobo_runtime::RobotHandle handle, const std::vector<float> & q) override;
  std::vector<std::string> tool_frames(curobo_runtime::RobotHandle handle) override;
  curobo_runtime::LinkCollisionOutcome set_link_collision(
    curobo_runtime::RobotHandle handle, const std::vector<std::string> & link_names,
    bool enabled) override;

  // ---- planning ----
  void motion_planner_build(
    curobo_runtime::RobotHandle handle, const std::string & cfg_json) override;
  void update_link_inertial(
    curobo_runtime::RobotHandle handle, const std::string & link_name,
    double mass) override;
  curobo_runtime::PlanOutcome plan_pose(
    curobo_runtime::RobotHandle handle, const curobo_runtime::PlanRequest & request) override;
  curobo_runtime::PlanOutcome plan_cspace(
    curobo_runtime::RobotHandle handle, const curobo_runtime::PlanRequest & request) override;
  void reset_seed(curobo_runtime::RobotHandle handle) override;
  void robot_model_rebuild_locks(
    curobo_runtime::RobotHandle handle, const std::string & lock_json) override;
  void robot_model_update_lock_joints(
    curobo_runtime::RobotHandle handle, const std::string & lock_json) override;

  // ---- perception (kept on Python for now) ----
  void configure_mapper(
    curobo_runtime::RobotHandle handle, const std::string & cfg_json) override;
  void warmup_mapper(curobo_runtime::RobotHandle handle) override;
  bool esdf_refresh(curobo_runtime::RobotHandle handle) override;
  bool world_push(curobo_runtime::RobotHandle handle) override;
  curobo_runtime::VoxelGrid voxel_grid(curobo_runtime::RobotHandle handle) override;
  int clear_voxel_map(curobo_runtime::RobotHandle handle) override;
  int clear_voxel_region(
    curobo_runtime::RobotHandle handle, const std::array<double, 3> & bounds_min,
    const std::array<double, 3> & bounds_max) override;
  void depth_integrate(
    curobo_runtime::RobotHandle handle, const curobo_runtime::TensorF32 & depth,
    const curobo_runtime::TensorF32 & intrinsics,
    const std::array<double, 7> & camera_pose) override;
  void lidar_integrate(
    curobo_runtime::RobotHandle handle, const curobo_runtime::TensorF32 & range_image,
    const curobo_runtime::TensorF32 & poses, const curobo_runtime::TensorF32 & valid_range,
    const curobo_runtime::TensorF32 & elevation_range) override;
  curobo_runtime::TensorF32 robot_segment(
    curobo_runtime::RobotHandle handle, const curobo_runtime::TensorF32 & depth,
    const curobo_runtime::TensorF32 & intrinsics_meta,
    const std::array<double, 7> & cam_pose, const std::vector<float> & q,
    const std::string & masks_json) override;

  // ---- MPC ----
  void mpc_build(curobo_runtime::RobotHandle handle, const std::string & cfg_json) override;
  curobo_runtime::MpcGoalOutcome mpc_setup(
    curobo_runtime::RobotHandle handle, const std::vector<float> & start_q) override;
  curobo_runtime::MpcStepOutcome mpc_step(
    curobo_runtime::RobotHandle handle, const curobo_runtime::ReactiveState & state,
    bool paced) override;
  bool mpc_update_goal_pose(
    curobo_runtime::RobotHandle handle, const std::array<double, 7> & goal_raw,
    bool run_ik) override;
  void mpc_update_goal_state(
    curobo_runtime::RobotHandle handle, const std::vector<float> & q) override;
  void mpc_reseed(
    curobo_runtime::RobotHandle handle, const std::vector<float> & q) override;
  curobo_runtime::IkOutcome mpc_solve_goal_state(
    curobo_runtime::RobotHandle handle, const std::array<double, 7> & goal_raw,
    const std::vector<float> & seed_q) override;
  void mpc_load_scene(
    curobo_runtime::RobotHandle handle, const std::string & scene_json) override;
  std::string mpc_metrics(curobo_runtime::RobotHandle handle) override;
  void retarget_build(curobo_runtime::RobotHandle handle, const std::string & cfg_json) override;
  void retarget_reset(curobo_runtime::RobotHandle handle) override;
  curobo_runtime::MpcStepOutcome retarget_step(
    curobo_runtime::RobotHandle handle, const std::array<double, 7> & goal_raw) override;
  void retarget_load_scene(
    curobo_runtime::RobotHandle handle, const std::string & scene_json) override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace curobo_cpp_runtime

#endif  // CUROBO_CPP_RUNTIME__NATIVE_BRIDGE_HPP_

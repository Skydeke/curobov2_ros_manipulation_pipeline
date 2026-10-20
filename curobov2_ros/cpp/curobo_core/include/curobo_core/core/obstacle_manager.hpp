// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/core/obstacle_manager.py
//
// The scene bookkeeping half of ObstacleManager: the name registry, the
// per-type buckets, spec validation, versioning/epoch and the exact service
// messages. The cuRobo scene itself is built by the bridge (scene_set); this
// class never touches cuRobo (design section 3).
#ifndef CUROBO_CORE__CORE__OBSTACLE_MANAGER_HPP_
#define CUROBO_CORE__CORE__OBSTACLE_MANAGER_HPP_

#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "curobo_core/convert.hpp"
#include "curobo_runtime/bridge.hpp"

namespace curobo_core
{

// SceneObject.msg / AddObject.srv type constants.
inline constexpr int CUBOID = 0;
inline constexpr int SPHERE = 1;
inline constexpr int CAPSULE = 2;
inline constexpr int CYLINDER = 3;
inline constexpr int MESH = 4;

// A scene object as the scene layer sees it (mirrors the msg field set).
struct SceneObjectSpec
{
  int type = CUBOID;
  std::string name;
  std::string mesh_file_path;
  std::vector<std::array<double, 3>> vertices;   // inline MESH
  std::vector<int> triangles;                    // inline MESH (flat)
  Pose7 pose = Pose7::Zero();                    // [x,y,z,qw,qx,qy,qz]
  std::array<double, 3> dims{0.0, 0.0, 0.0};
  std::array<double, 4> color{0.0, 0.0, 0.0, 0.0};
};

// The structured get_scene_objects view (parallel arrays).
struct SceneView
{
  std::vector<std::string> names;
  std::vector<int> types;
  std::vector<Pose7> poses;
  std::vector<std::array<double, 3>> dimensions;
  std::vector<std::array<double, 4>> colors;
  std::vector<std::string> mesh_file_paths;
  std::vector<bool> attached;
};

class ObstacleManager
{
public:
  ObstacleManager(curobo_runtime::IBridge * bridge, curobo_runtime::RobotHandle handle);

  // PORT: obstacle_manager.add_object
  bool add_object(const SceneObjectSpec & spec, std::string & message);
  // PORT: obstacle_manager.set_objects
  bool set_objects(
    const std::vector<SceneObjectSpec> & specs, bool clear_first, std::string & message);
  // PORT: obstacle_manager.remove_object
  bool remove_object(const std::string & name, std::string & message);
  // PORT: obstacle_manager.remove_all_objects
  bool remove_all_objects(std::string & message);
  // PORT: obstacle_manager.get_obstacles
  std::string get_obstacles() const;
  // PORT: obstacle_manager.get_scene_objects
  SceneView get_scene_objects() const;

  // PORT: obstacle_manager.refresh_esdf (mapper.compute_esdf + scene.voxel).
  bool refresh_esdf();
  // PORT: obstacle_manager.get_voxel_grid (dense occupied grid).
  curobo_runtime::VoxelGrid get_voxel_grid();
  // PORT: obstacle_manager.clear_dynamic_voxels / clear_voxel_region.
  int clear_dynamic_voxels();
  int clear_voxel_region(
    const std::array<double, 3> & bounds_min, const std::array<double, 3> & bounds_max);
  // PORT: obstacle_manager.publish_sparse_voxel_grid (occupied linear indices,
  // row-major [X][Y][Z]: (x*sy + y)*sz + z).
  static std::vector<std::int32_t> occupied_indices(const curobo_runtime::VoxelGrid & grid);

  std::int64_t perception_epoch() const {return perception_epoch_;}

  // PORT: unified_planner_node.should_refresh_world - ESDF freshness gate.
  // Analytic content is pushed eagerly at mutation time; this is purely about
  // ESDF staleness (so skipping it can never plan against a stale object set).
  bool should_refresh_world(bool force) const;
  // Record a successful ESDF rebuild + push (updates the gate state).
  void mark_world_pushed();
  // A push FAILED: the solvers may be blind to the current perception. Force the
  // next plan/servo refresh to rebuild and push again instead of treating a
  // blind solver as fresh (it must never plan through un-seen camera voxels).
  void invalidate_world() {last_pushed_epoch_ = -1;}
  // PORT: camera_depth_map_strategy's `node.perception_epoch += 1` on integrate.
  void bump_perception_epoch() {epoch_ += 1;}
  // `esdf_refresh_min_interval` param (open-loop refresh throttle).
  void set_esdf_refresh_min_interval(double seconds)
  {esdf_refresh_min_interval_ = seconds;}

  // PORT: obstacle_manager._validate_spec (public for testing / set_objects).
  std::string validate_spec(
    const SceneObjectSpec & spec, const std::set<std::string> & taken) const;

  // The analytic scene as the bridge payload (PORT: obstacle_manager.get_scene).
  std::string scene_json() const;

  const std::vector<std::string> & obstacle_names() const {return order_;}
  std::int64_t epoch() const {return epoch_;}
  std::int64_t scene_version() const {return scene_version_;}
  bool empty() const {return order_.empty();}

  // Attach bookkeeping (PORT: AttachmentServices / attached_name).
  void set_attached(const std::string & name) {attached_name_ = name;}
  void clear_attached() {attached_name_.clear();}
  const std::string & attached_name() const {return attached_name_;}

  int bucket_count(const std::string & bucket) const;

private:
  void append_spec(const SceneObjectSpec & spec);        // PORT: _append_spec
  void clear_all();                                      // PORT: _clear_all
  void notify_world_changed();                           // PORT: _notify_world_changed

  curobo_runtime::IBridge * bridge_;
  curobo_runtime::RobotHandle handle_;
  std::vector<std::string> order_;                       // insertion order
  std::map<std::string, SceneObjectSpec> objects_;
  std::map<std::string, int> bucket_counts_;
  std::string attached_name_;
  std::int64_t epoch_ = 0;
  std::int64_t perception_epoch_ = 0;
  std::int64_t scene_version_ = -1;
  // PORT: unified_planner_node._last_pushed_epoch / _last_esdf_push_time /
  // _esdf_refresh_min_interval (should_refresh_world gate state).
  std::int64_t last_pushed_epoch_ = -1;
  double last_push_time_ = 0.0;
  double esdf_refresh_min_interval_ = 0.0;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__CORE__OBSTACLE_MANAGER_HPP_

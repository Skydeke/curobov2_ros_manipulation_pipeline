// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of the scene bookkeeping in curobov2_ros/core/obstacle_manager.py.
#include "curobo_core/core/obstacle_manager.hpp"
#include "curobo_runtime/runtime.hpp"

#include "curobo_core/core/frame_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace curobo_core
{

namespace
{
const char * kBuckets[] = {"cuboid", "capsule", "cylinder", "sphere", "mesh"};

std::string bucket_of(int type)
{
  switch (type) {
    case CUBOID: return "cuboid";
    case CAPSULE: return "capsule";
    case CYLINDER: return "cylinder";
    case SPHERE: return "sphere";
    case MESH: return "mesh";
    default: return "";
  }
}

std::string json_escape(const std::string & value)
{
  std::string out;
  out.reserve(value.size() + 2);
  for (char c : value) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out += c;
    }
  }
  return out;
}

void append_num(std::ostringstream & out, double value)
{
  out << value;
}
}  // namespace

ObstacleManager::ObstacleManager(
  curobo_runtime::IBridge * bridge, curobo_runtime::RobotHandle handle)
: bridge_(bridge), handle_(handle)
{
  for (const char * bucket : kBuckets) {
    bucket_counts_[bucket] = 0;
  }
}

int ObstacleManager::bucket_count(const std::string & bucket) const
{
  auto it = bucket_counts_.find(bucket);
  return it == bucket_counts_.end() ? 0 : it->second;
}

// PORT: obstacle_manager._validate_spec
std::string ObstacleManager::validate_spec(
  const SceneObjectSpec & spec, const std::set<std::string> & taken) const
{
  if (taken.count(spec.name) != 0u) {
    return "Object with name \"" + spec.name + "\" already exists";
  }
  if (spec.dims[0] <= 0.0 || spec.dims[1] <= 0.0 || spec.dims[2] <= 0.0) {
    return "Object dimensions must be positive";
  }
  if (spec.type == MESH) {
    const std::size_t n_v = spec.vertices.size();
    const std::size_t n_t = spec.triangles.size();
    if (n_v > 0 || n_t > 0) {
      if (n_v == 0 || n_t == 0) {
        return "Inline MESH requires both vertices and triangles";
      }
      if (n_t % 3 != 0) {
        return "Inline MESH triangles must be a flat index buffer with a multiple-of-3 length";
      }
    } else if (spec.mesh_file_path.empty()) {
      return "Mesh file not found: " + spec.mesh_file_path;
    }
  } else if (spec.type != CUBOID && spec.type != SPHERE &&
    spec.type != CAPSULE && spec.type != CYLINDER)
  {
    return "Object type \"" + std::to_string(spec.type) + "\" not recognized";
  }
  return {};
}

// PORT: obstacle_manager._append_spec (bookkeeping half; the bridge builds the
// cuRobo primitive on scene_set)
void ObstacleManager::append_spec(const SceneObjectSpec & spec)
{
  const std::string bucket = bucket_of(spec.type);
  if (bucket.empty()) {
    throw std::invalid_argument(
      "Object type \"" + std::to_string(spec.type) + "\" not recognized");
  }
  objects_[spec.name] = spec;
  order_.push_back(spec.name);
  bucket_counts_[bucket] += 1;
}

// PORT: obstacle_manager._clear_all
void ObstacleManager::clear_all()
{
  for (const char * bucket : kBuckets) {
    bucket_counts_[bucket] = 0;
  }
  objects_.clear();
  order_.clear();
  attached_name_.clear();
}

std::string ObstacleManager::scene_json() const
{
  std::ostringstream out;
  out << "[";
  bool first = true;
  for (const std::string & name : order_) {
    const SceneObjectSpec & spec = objects_.at(name);
    if (!first) {
      out << ",";
    }
    first = false;
    out << "{\"name\":\"" << json_escape(spec.name) << "\",\"type\":" << spec.type
        << ",\"mesh_file_path\":\"" << json_escape(spec.mesh_file_path) << "\""
        << ",\"pose\":[";
    for (int i = 0; i < 7; ++i) {
      if (i != 0) {
        out << ",";
      }
      append_num(out, spec.pose[i]);
    }
    out << "],\"dims\":[" << spec.dims[0] << "," << spec.dims[1] << "," << spec.dims[2] << "]"
        << ",\"color\":[" << spec.color[0] << "," << spec.color[1] << ","
        << spec.color[2] << "," << spec.color[3] << "]";
    if (spec.type == MESH && (!spec.vertices.empty() || !spec.triangles.empty())) {
      out << ",\"vertices\":[";
      for (std::size_t i = 0; i < spec.vertices.size(); ++i) {
        if (i != 0) {
          out << ",";
        }
        out << "[" << spec.vertices[i][0] << "," << spec.vertices[i][1] << ","
            << spec.vertices[i][2] << "]";
      }
      out << "],\"triangles\":[";
      for (std::size_t i = 0; i < spec.triangles.size(); ++i) {
        if (i != 0) {
          out << ",";
        }
        out << spec.triangles[i];
      }
      out << "]";
    }
    out << "}";
  }
  out << "]";
  return out.str();
}

// PORT: obstacle_manager._notify_world_changed (analytic scene pushed eagerly;
// the ESDF freshness gate lives in should_refresh_world)
void ObstacleManager::notify_world_changed()
{
  scene_version_ = bridge_->scene_set(handle_, scene_json(), true);
  epoch_ += 1;
  // PORT: obstacle_manager.set_world_update_callback ->
  // unified_planner_node.update_all_solvers_world(obstacle_manager.get_scene())
  // (node line ~1084): every scene MUTATION pushes the analytic world to the
  // solvers immediately. That is exactly what lets should_refresh_world skip a
  // refresh on its min-interval throttle without ever leaving a stale object
  // set - the gate is about ESDF/perception staleness ONLY. Without this push
  // the solvers keep the PREVIOUS scene until the throttle opens, so a burst of
  // mutations plans against the wrong world: the benchmark sets one world per
  // problem ~0.7 s apart against the default 2 s interval, so about a third of
  // the problems were planned against their predecessor's obstacles.
  // Held under the exclusive GPU barrier: the push is a CUDA op and mutations
  // can arrive from service threads (the reference takes gpu_lock here too).
  //
  // This pushes the analytic layer but is NOT an ESDF/perception refresh, so it
  // must NOT mark the perception world as pushed: `epoch_` is bumped by depth
  // integrates too, and the gate is what decides whether a later plan rebuilds
  // the ESDF. Marking here would make should_refresh_world skip that rebuild, so
  // a plan issued after an analytic mutation would collide against a scene whose
  // camera voxels are stale (or absent) and solve straight through them.
  try {
    auto gpu_guard = curobo_runtime::Runtime::instance().barrier().exclusive();
    if (!bridge_->world_push(handle_)) {
      // The mutation must still land, but the solvers are now on a world the
      // gate must not consider fresh.
      invalidate_world();
    }
  } catch (const std::exception &) {
    // A push must never break the mutation that triggered it; retry next plan.
    invalidate_world();
  }
}

// PORT: obstacle_manager.add_object
bool ObstacleManager::add_object(const SceneObjectSpec & spec, std::string & message)
{
  std::set<std::string> taken(order_.begin(), order_.end());
  const std::string err = validate_spec(spec, taken);
  if (!err.empty()) {
    message = err;
    return false;
  }
  try {
    append_spec(spec);
  } catch (const std::exception & e) {
    message = std::string("Failed to add obstacle: ") + e.what();
    return false;
  }
  message = "Object '" + spec.name + "' added (" +
    std::to_string(bucket_count("cuboid")) + " cuboids, " +
    std::to_string(bucket_count("mesh")) + " meshes)";
  notify_world_changed();
  return true;
}

// PORT: obstacle_manager.set_objects
bool ObstacleManager::set_objects(
  const std::vector<SceneObjectSpec> & specs, bool clear_first, std::string & message)
{
  std::set<std::string> taken;
  if (!clear_first) {
    taken.insert(order_.begin(), order_.end());
  }
  for (const auto & spec : specs) {
    const std::string err = validate_spec(spec, taken);
    if (!err.empty()) {
      message = "Invalid object '" + spec.name + "': " + err;
      return false;
    }
    taken.insert(spec.name);
  }
  try {
    if (clear_first) {
      clear_all();
    }
    for (const auto & spec : specs) {
      append_spec(spec);
    }
  } catch (const std::exception & e) {
    message = std::string("Failed to set objects: ") + e.what();
    return false;
  }
  message = "Scene set to " + std::to_string(specs.size()) + " objects (" +
    std::to_string(bucket_count("cuboid")) + " cuboids, " +
    std::to_string(bucket_count("mesh")) + " meshes)";
  notify_world_changed();
  return true;
}

// PORT: obstacle_manager.remove_object
bool ObstacleManager::remove_object(const std::string & name, std::string & message)
{
  auto it = objects_.find(name);
  if (it == objects_.end()) {
    message = "Object '" + name + "' not found";
    return false;
  }
  const std::string bucket = bucket_of(it->second.type);
  objects_.erase(it);
  order_.erase(std::remove(order_.begin(), order_.end(), name), order_.end());
  bucket_counts_[bucket] -= 1;
  if (attached_name_ == name) {
    attached_name_.clear();
  }
  message = "Object '" + name + "' removed from " + bucket + "s";
  notify_world_changed();
  return true;
}

// PORT: obstacle_manager.remove_all_objects
bool ObstacleManager::remove_all_objects(std::string & message)
{
  int total = 0;
  for (const char * bucket : kBuckets) {
    total += bucket_count(bucket);
  }
  clear_all();
  message = "All " + std::to_string(total) + " obstacles removed";
  notify_world_changed();
  return true;
}

// PORT: obstacle_manager.refresh_esdf
bool ObstacleManager::refresh_esdf()
{
  const bool refreshed = bridge_->esdf_refresh(handle_);
  if (refreshed) {
    perception_epoch_ += 1;
  }
  return refreshed;
}

namespace
{
double steady_now()
{
  return std::chrono::duration<double>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

// PORT: unified_planner_node.should_refresh_world - ESDF freshness gate. The
// reactive servo passes force=true (refresh whenever the epoch advanced); the
// open-loop paths share the min-interval throttle. `last_pushed_epoch_ < 0`
// always refreshes so the first request builds the baseline. Without this gate
// the servo loop re-runs compute_esdf + world_push EVERY
// perception_refresh_period steps even with no new depth, which churns the GPU
// buffers (observed: heap corruption / malloc abort under the 10-minute soak).
bool ObstacleManager::should_refresh_world(bool force) const
{
  if (last_pushed_epoch_ < 0) {
    return true;
  }
  if (force) {
    return epoch_ != last_pushed_epoch_;
  }
  if (epoch_ == last_pushed_epoch_) {
    return false;
  }
  return (steady_now() - last_push_time_) >= esdf_refresh_min_interval_;
}

void ObstacleManager::mark_world_pushed()
{
  last_pushed_epoch_ = epoch_;
  last_push_time_ = steady_now();
}

// PORT: obstacle_manager.get_voxel_grid. The dense grid (mapper perception OR
// the GPU-rasterized analytic primitives) is computed by the bridge
// (curobo_bridge.voxel_grid -> _compute_dense_voxel_grid); the C++ layer only
// shapes it into the response.
curobo_runtime::VoxelGrid ObstacleManager::get_voxel_grid()
{
  return bridge_->voxel_grid(handle_);
}

// PORT: obstacle_manager.clear_dynamic_voxels
int ObstacleManager::clear_dynamic_voxels()
{
  return bridge_->clear_voxel_map(handle_);
}

// PORT: obstacle_manager.clear_voxel_region
int ObstacleManager::clear_voxel_region(
  const std::array<double, 3> & bounds_min, const std::array<double, 3> & bounds_max)
{
  return bridge_->clear_voxel_region(handle_, bounds_min, bounds_max);
}

// PORT: obstacle_manager.publish_sparse_voxel_grid - occupied linear indices,
// row-major [X][Y][Z].
std::vector<std::int32_t> ObstacleManager::occupied_indices(
  const curobo_runtime::VoxelGrid & grid)
{
  std::vector<std::int32_t> out;
  if (grid.shape.size() != 3) {
    return out;
  }
  const std::int64_t sx = grid.shape[0];
  const std::int64_t sy = grid.shape[1];
  const std::int64_t sz = grid.shape[2];
  for (std::int64_t x = 0; x < sx; ++x) {
    for (std::int64_t y = 0; y < sy; ++y) {
      for (std::int64_t z = 0; z < sz; ++z) {
        const std::size_t idx = static_cast<std::size_t>((x * sy + y) * sz + z);
        if (idx < grid.data.size() && grid.data[idx] != 0) {
          out.push_back(static_cast<std::int32_t>(idx));
        }
      }
    }
  }
  return out;
}

// PORT: obstacle_manager.get_obstacles
std::string ObstacleManager::get_obstacles() const
{
  std::ostringstream out;
  for (const std::string & name : order_) {
    out << name << "\n";
  }
  return out.str();
}

// PORT: obstacle_manager.get_scene_objects
SceneView ObstacleManager::get_scene_objects() const
{
  SceneView view;
  for (const std::string & name : order_) {
    const SceneObjectSpec & spec = objects_.at(name);
    view.names.push_back(spec.name);
    view.types.push_back(spec.type);
    view.poses.push_back(spec.pose);
    if (spec.type == SPHERE) {
      view.dimensions.push_back({spec.dims[0], spec.dims[0], spec.dims[0]});
    } else {
      view.dimensions.push_back(spec.dims);
    }
    view.colors.push_back(spec.color);
    view.mesh_file_paths.push_back(spec.type == MESH ? spec.mesh_file_path : "");
    view.attached.push_back(spec.name == attached_name_);
  }
  return view;
}

}  // namespace curobo_core

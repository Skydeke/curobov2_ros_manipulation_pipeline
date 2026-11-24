// SPDX-License-Identifier: Apache-2.0
#include "cpp_bridge_impl.hpp"

namespace curobo_cpp_runtime
{

void CppBridge::configure_mapper(
  curobo_runtime::RobotHandle, const std::string & cfg_json)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("configure_mapper");
  // Headless benchmarks have no depth camera and explicitly do not want a
  // perception voxel layer in the collision world.
  if (std::getenv("CUROBO_DISABLE_NATIVE_MAPPER") != nullptr) {
    return;
  }
  impl_->ensure_scene();

  impl_->mapper_cfg_json = cfg_json;
  const std::string mapper_dir = impl_->artifact_dir + "/mapper";
  if (!std::filesystem::is_regular_file(mapper_dir + "/mapper_aot_manifest.json")) {
    // Hybrid fallback: no native mapper artifact, keep the Python mapper and
    // force planning onto Python so the ESDF and planner share one scene.
    impl_->require_python("configure_mapper");
    impl_->python->configure_mapper(impl_->python_handle, cfg_json);
    impl_->mark_python_scene("configure_mapper");
    return;
  }

  auto state = std::make_unique<Impl::MapperState>();
  if (curobo_mapper_create(
      mapper_dir.c_str(), impl_->device_ordinal, &state->runtime) != 0)
  {
    fail("configure_mapper", curobo_mapper_last_error());
  }
  if (curobo_mapper_get_device_view(state->runtime, &state->view) != 0) {
    fail("configure_mapper", curobo_mapper_last_error());
  }

  int image_height = 0;
  int image_width = 0;
  if (curobo_mapper_config_int(state->runtime, "image_height", &image_height) != 0 ||
    curobo_mapper_config_int(state->runtime, "image_width", &image_width) != 0 ||
    image_height <= 0 || image_width <= 0)
  {
    fail("configure_mapper", "native mapper artifact has no image dimensions");
  }
  state->image_height = image_height;
  state->image_width = image_width;
  int requested_height = 0;
  int requested_width = 0;
  try {
    const json::Value document = json::parse(cfg_json);
    requested_height = json_int(document, "image_height", 0);
    requested_width = json_int(document, "image_width", 0);
  } catch (const std::exception &) {
    // No parseable camera hint: the artifact dimensions remain authoritative.
  }
  if (requested_height > 0 && requested_width > 0 &&
    (requested_height != image_height || requested_width != image_width))
  {
    fail(
      "configure_mapper",
      "camera image is " + std::to_string(requested_width) + "x" +
      std::to_string(requested_height) + " but the native mapper artifact is " +
      std::to_string(image_width) + "x" + std::to_string(image_height) +
      "; re-export the mapper with the camera resolution");
  }
  const std::size_t pixels =
    static_cast<std::size_t>(image_height) * static_cast<std::size_t>(image_width);
  state->depth.allocate(pixels * sizeof(float));
  state->rgb.allocate(pixels * 3 * sizeof(std::uint8_t));
  state->intrinsics.allocate(9 * sizeof(float));
  state->camera_positions.allocate(3 * sizeof(float));
  state->camera_quaternions.allocate(4 * sizeof(float));
  std::vector<std::uint8_t> black(pixels * 3, 0);
  state->rgb.upload(black);
  double decay_factor = 1.0;
  double minimum_tsdf_weight = 0.1;
  try {
    const json::Value document = json::parse(cfg_json);
    decay_factor = json_double(document, "decay_factor", decay_factor);
    minimum_tsdf_weight = json_double(
      document, "minimum_tsdf_weight", minimum_tsdf_weight);
  } catch (const std::exception &) {
  }
  if (curobo_mapper_set_tsdf_runtime_parameters(
      state->runtime, static_cast<float>(decay_factor),
      static_cast<float>(minimum_tsdf_weight)) != 0)
  {
    fail("configure_mapper", curobo_mapper_last_error());
  }
  impl_->mapper = std::move(state);
}

void CppBridge::warmup_mapper(curobo_runtime::RobotHandle)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("warmup_mapper");
  if (std::getenv("CUROBO_DISABLE_NATIVE_MAPPER") != nullptr) {
    return;
  }
  if (impl_->mapper == nullptr || !impl_->native_scene_valid) {
    impl_->ensure_python_mapper("warmup_mapper");
    impl_->mark_python_scene("warmup_mapper");
    return;
  }
  if (curobo_mapper_compute_esdf(
      impl_->mapper->runtime, impl_->mapper->view.esdf_dist_field, 0) != 0)
  {
    fail("warmup_mapper", curobo_mapper_last_error());
  }
  cuda_check("warmup_mapper", cudaDeviceSynchronize());
  impl_->update_native_voxel_grid();
}

bool CppBridge::esdf_refresh(curobo_runtime::RobotHandle)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("esdf_refresh");
  if (std::getenv("CUROBO_DISABLE_NATIVE_MAPPER") != nullptr) {
    return true;
  }
  if (impl_->mapper == nullptr || !impl_->native_scene_valid) {
    impl_->ensure_python_mapper("esdf_refresh");
    const bool updated = impl_->python->esdf_refresh(impl_->python_handle);
    impl_->mark_python_scene("esdf_refresh");
    return updated;
  }
  if (curobo_mapper_compute_esdf(
      impl_->mapper->runtime, impl_->mapper->view.esdf_dist_field, 0) != 0)
  {
    fail("esdf_refresh", curobo_mapper_last_error());
  }
  cuda_check("esdf_refresh", cudaDeviceSynchronize());
  impl_->update_native_voxel_grid();
  return true;
}

bool CppBridge::world_push(curobo_runtime::RobotHandle)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("world_push");
  if (impl_->mapper == nullptr || !impl_->native_scene_valid) {
    impl_->ensure_python_mapper("world_push");
    return impl_->python->world_push(impl_->python_handle);
  }
  // The native Scene is the live collision world; there is no Python-style
  // copy step to push. esdf_refresh already updated the voxel layer in place.
  return true;
}

void CppBridge::set_esdf_pose(
  curobo_runtime::RobotHandle, const std::array<double, 3> & position,
  const std::array<double, 4> & quaternion_wxyz)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("set_esdf_pose");
  if (impl_->mapper == nullptr || !impl_->native_scene_valid) {
    return;
  }
  impl_->mapper->frame_position = position;
  impl_->mapper->frame_quaternion = quaternion_wxyz;
  if (impl_->mapper->voxel_added && impl_->scene != nullptr) {
    impl_->scene->scene().update_obstacle_pose(
      impl_->mapper->voxel_handle, impl_->mapper_grid_pose());
  }
}

curobo_runtime::VoxelGrid CppBridge::voxel_grid(curobo_runtime::RobotHandle)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("voxel_grid");
  if (std::getenv("CUROBO_DISABLE_NATIVE_MAPPER") != nullptr) {
    return curobo_runtime::VoxelGrid{};
  }
  if (impl_->mapper == nullptr || !impl_->native_scene_valid) {
    impl_->ensure_python_mapper("voxel_grid");
    return impl_->python->voxel_grid(impl_->python_handle);
  }
  const int nx = impl_->mapper->view.esdf_shape[0];
  const int ny = impl_->mapper->view.esdf_shape[1];
  const int nz = impl_->mapper->view.esdf_shape[2];
  const std::size_t count =
    static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
    static_cast<std::size_t>(nz);
  curobo_runtime::VoxelGrid grid;
  grid.shape = {nx, ny, nz};
  grid.resolution = impl_->mapper->view.esdf_voxel_size;
  // The AOT mapper calls the grid centre `origin`; the ROS VoxelGrid message
  // publishes the minimum corner so the dense reconstruction covers the whole
  // box (the same convention the node uses for mapper_grid_center).
  const double half_x = 0.5 * nx * grid.resolution;
  const double half_y = 0.5 * ny * grid.resolution;
  const double half_z = 0.5 * nz * grid.resolution;
  grid.origin = {
    impl_->mapper->view.origin[0] - half_x,
    impl_->mapper->view.origin[1] - half_y,
    impl_->mapper->view.origin[2] - half_z};
  grid.data.resize(count, 0);
  const int capacity = 1000000;
  DeviceBuffer centers;
  DeviceBuffer pools;
  DeviceBuffer counter;
  centers.allocate(static_cast<std::size_t>(capacity) * 3 * sizeof(float));
  pools.allocate(static_cast<std::size_t>(capacity) * sizeof(int32_t));
  counter.allocate(sizeof(int32_t));
  for (int attempt = 0; attempt < 2; ++attempt) {
    // Attempt 0: solid occupied volume (Python parity). Attempt 1: observed
    // surface band, used only if the AOT artifact's TSDF sign is inverted.
    const int surface_only = attempt == 0 ? 0 : 1;
    const float threshold = attempt == 0 ? 0.0F :
      static_cast<float>(0.5 * grid.resolution);
    int required = 0;
    int written = 0;
    const int rc = curobo_mapper_extract_occupied(
      impl_->mapper->runtime,
      reinterpret_cast<std::uint64_t>(centers.get()),
      reinterpret_cast<std::uint64_t>(pools.get()),
      reinterpret_cast<std::uint64_t>(counter.get()),
      capacity, surface_only, threshold, 0, &required, &written);
    if (rc != 0) {
      break;
    }
    int count_host = 0;
    cuda_check(
      "voxel_grid",
      cudaMemcpy(&count_host, counter.get(), sizeof(int32_t),
        cudaMemcpyDeviceToHost));
    if (count_host <= 0) {
      continue;
    }
    const int take = std::min(count_host, capacity);
    std::vector<float> center_host(static_cast<std::size_t>(take) * 3);
    cuda_check(
      "voxel_grid",
      cudaMemcpy(center_host.data(), centers.get(),
        center_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < take; ++i) {
      const float x = center_host[static_cast<std::size_t>(i) * 3 + 0];
      const float y = center_host[static_cast<std::size_t>(i) * 3 + 1];
      const float z = center_host[static_cast<std::size_t>(i) * 3 + 2];
      const int ix = static_cast<int>(std::floor((x - grid.origin[0]) / grid.resolution));
      const int iy = static_cast<int>(std::floor((y - grid.origin[1]) / grid.resolution));
      const int iz = static_cast<int>(std::floor((z - grid.origin[2]) / grid.resolution));
      if (ix >= 0 && ix < nx && iy >= 0 && iy < ny && iz >= 0 && iz < nz) {
        grid.data[(static_cast<std::size_t>(ix) * ny + iy) * nz + iz] = 1;
      }
    }
    break;
  }
  return grid;
}

int CppBridge::clear_voxel_map(curobo_runtime::RobotHandle)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("clear_voxel_map");
  if (impl_->mapper == nullptr || !impl_->native_scene_valid) {
    impl_->ensure_python_mapper("clear_voxel_map");
    const int cleared = impl_->python->clear_voxel_map(impl_->python_handle);
    impl_->mark_python_scene("clear_voxel_map");
    return cleared;
  }
  if (curobo_mapper_reset(impl_->mapper->runtime, 0) != 0) {
    fail("clear_voxel_map", curobo_mapper_last_error());
  }
  if (curobo_mapper_compute_esdf(
      impl_->mapper->runtime, impl_->mapper->view.esdf_dist_field, 0) != 0)
  {
    fail("clear_voxel_map", curobo_mapper_last_error());
  }
  cuda_check("clear_voxel_map", cudaDeviceSynchronize());
  impl_->update_native_voxel_grid();
  if (impl_->python && impl_->python_mapper_configured) {
    try {
      impl_->python->clear_voxel_map(impl_->python_handle);
    } catch (const std::exception &) {
    }
  }
  return impl_->mapper->esdf_count;
}

int CppBridge::clear_voxel_region(
  curobo_runtime::RobotHandle, const std::array<double, 3> & bounds_min,
  const std::array<double, 3> & bounds_max)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->mapper == nullptr || !impl_->native_scene_valid) {
    impl_->ensure_python_mapper("clear_voxel_region");
    const int cleared = impl_->python->clear_voxel_region(
      impl_->python_handle, bounds_min, bounds_max);
    impl_->mark_python_scene("clear_voxel_region");
    return cleared;
  }
  unsupported("clear_voxel_region");
}

void CppBridge::depth_integrate(
  curobo_runtime::RobotHandle, const curobo_runtime::TensorF32 & depth,
  const curobo_runtime::TensorF32 & intrinsics,
  const std::array<double, 7> & camera_pose)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("depth_integrate");
  if (std::getenv("CUROBO_DISABLE_NATIVE_MAPPER") != nullptr) {
    return;
  }
  if (impl_->mapper == nullptr || !impl_->native_scene_valid) {
    impl_->ensure_python_mapper("depth_integrate");
    impl_->python->depth_integrate(
      impl_->python_handle, depth, intrinsics, camera_pose);
    impl_->mark_python_scene("depth_integrate");
    return;
  }
  if (depth.shape.size() < 2) {
    fail("depth_integrate", "depth must have shape [H,W] or [1,H,W]");
  }
  const int height = static_cast<int>(depth.shape[depth.shape.size() - 2]);
  const int width = static_cast<int>(depth.shape[depth.shape.size() - 1]);
  if (height != impl_->mapper->image_height || width != impl_->mapper->image_width) {
    fail("depth_integrate", "depth image shape does not match the mapper artifact");
  }
  std::vector<float> intr(9, 0.0F);
  for (std::size_t i = 0; i < intr.size() && i < intrinsics.data.size(); ++i) {
    intr[i] = intrinsics.data[i];
  }
  impl_->mapper->depth.upload(depth.data);
  impl_->mapper->intrinsics.upload(intr);
  const std::array<float, 3> position{
    static_cast<float>(camera_pose[0]),
    static_cast<float>(camera_pose[1]),
    static_cast<float>(camera_pose[2])};
  const std::array<float, 4> quaternion{
    static_cast<float>(camera_pose[3]),
    static_cast<float>(camera_pose[4]),
    static_cast<float>(camera_pose[5]),
    static_cast<float>(camera_pose[6])};
  impl_->mapper->camera_positions.upload(
    position.data(), position.size() * sizeof(float));
  impl_->mapper->camera_quaternions.upload(
    quaternion.data(), quaternion.size() * sizeof(float));
  CuroboMapperFrame frame{};
  frame.depth_images = reinterpret_cast<std::uint64_t>(impl_->mapper->depth.get());
  frame.rgb_images = reinterpret_cast<std::uint64_t>(impl_->mapper->rgb.get());
  frame.intrinsics = reinterpret_cast<std::uint64_t>(impl_->mapper->intrinsics.get());
  frame.cam_positions =
    reinterpret_cast<std::uint64_t>(impl_->mapper->camera_positions.get());
  frame.cam_quaternions =
    reinterpret_cast<std::uint64_t>(impl_->mapper->camera_quaternions.get());
  frame.stream = 0;
  if (curobo_mapper_integrate(impl_->mapper->runtime, &frame) != 0) {
    fail("depth_integrate", curobo_mapper_last_error());
  }
}

void CppBridge::lidar_integrate(
  curobo_runtime::RobotHandle, const curobo_runtime::TensorF32 & range_image,
  const curobo_runtime::TensorF32 & poses, const curobo_runtime::TensorF32 & valid_range,
  const curobo_runtime::TensorF32 & elevation_range)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->ensure_python_mapper("lidar_integrate");
  impl_->python->lidar_integrate(
    impl_->python_handle, range_image, poses, valid_range, elevation_range);
  impl_->mark_python_scene("lidar_integrate");
}

curobo_runtime::TensorF32 CppBridge::robot_segment(
  curobo_runtime::RobotHandle, const curobo_runtime::TensorF32 & depth,
  const curobo_runtime::TensorF32 & intrinsics_meta,
  const std::array<double, 7> & cam_pose, const std::vector<float> & q,
  const std::string & masks_json)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_python("robot_segment");
  return impl_->python->robot_segment(
    impl_->python_handle, depth, intrinsics_meta, cam_pose, q, masks_json);
}
}  // namespace curobo_cpp_runtime

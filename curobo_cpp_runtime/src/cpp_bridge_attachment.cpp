// SPDX-License-Identifier: Apache-2.0
#include "cpp_bridge_impl.hpp"

namespace curobo_cpp_runtime
{

void CppBridge::attach_object(
  curobo_runtime::RobotHandle, const std::string & name, const std::vector<float> & q)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("attach_object");
  if (name.empty()) {
    return;
  }
  impl_->ensure_scene();
  const SceneObjectRecord * record = impl_->scene->object(name);
  if (record == nullptr) {
    fail("attach_object", "unknown scene object '" + name + "'");
  }
  impl_->ensure_planner("attach_object");
  // The artifact exposes one attachment slot per arm; choose the slot whose
  // tool is closest to the object at the attach configuration. This mirrors
  // the Python bridge's nearest-attach-link choice without a mesh fit.
  const int tools = impl_->num_tools();
  int best_tool = 0;
  float best_distance = std::numeric_limits<float>::infinity();
  std::vector<std::array<float, 7>> tool_poses;
  tool_poses.reserve(static_cast<std::size_t>(tools));
  for (int t = 0; t < tools; ++t) {
    const auto pose = impl_->tool_pose(q, t, "attach_object");
    tool_poses.push_back(pose);
    const float dx = pose[0] - record->pose.position[0];
    const float dy = pose[1] - record->pose.position[1];
    const float dz = pose[2] - record->pose.position[2];
    const float distance = dx * dx + dy * dy + dz * dz;
    if (distance < best_distance) {
      best_distance = distance;
      best_tool = t;
    }
  }
  const auto has_link = [this](const std::string & candidate) {
      return std::find(impl_->link_names.begin(), impl_->link_names.end(),
               candidate) != impl_->link_names.end();
    };
  std::string link;
  if (best_tool == 0) {
    link = "attached_object_left";
  } else if (best_tool == 1) {
    link = "attached_object_right";
  } else {
    link = impl_->tool_frames[static_cast<std::size_t>(best_tool)];
  }
  if (!has_link(link)) {
    for (const char * candidate : {"attached_object", "attached_object_left",
             "attached_object_right"}) {
      if (has_link(candidate)) {
        link = candidate;
        break;
      }
    }
  }
  if (!has_link(link)) {
    link = impl_->tool_frames[static_cast<std::size_t>(best_tool)];
  }

  curobo::robot::SphereAttachment attachment;
  attachment.link_name = link;
  const auto tool_pose = tool_poses[static_cast<std::size_t>(best_tool)];
  const std::array<float, 7> object_pose{
    record->pose.position[0], record->pose.position[1], record->pose.position[2],
    record->pose.quaternion[0], record->pose.quaternion[1],
    record->pose.quaternion[2], record->pose.quaternion[3]};
  const auto relative = pose_multiply(pose_inverse(tool_pose), object_pose);
  attachment.object_in_link.position = {relative[0], relative[1], relative[2]};
  attachment.object_in_link.quaternion = {
    relative[3], relative[4], relative[5], relative[6]};
  int attach_capacity = 0;
  {
    const auto slots_it = impl_->native_sphere_slots.find(link);
    if (slots_it != impl_->native_sphere_slots.end()) {
      attach_capacity = static_cast<int>(slots_it->second.size());
    }
  }
  attachment.spheres = fit_surface_spheres(
    record->type, record->dims, attach_capacity);
  if (impl_->scene) {
    // The grasped obstacle stays in the scene list (so detach can restore it)
    // but must not collide with the attachment model or the gripper at the
    // start state. Python disables it on the solver's scene checker too.
    impl_->scene->set_object_enabled(name, false);
  }
  impl_->planner->attach_object(attachment);
  if (impl_->ik) {
    impl_->ik->attach_object(attachment);
  }
  impl_->attached_objects.insert(name);
  impl_->attach_links[name] = link;
  impl_->visual_attachments[name] = Impl::VisualAttachment{
    best_tool, relative, attachment.spheres};
  // Diagnostic: where the payload actually lands in the world, so a retreat
  // that cannot find a path can be checked against the table plane and the
  // grasp pose instead of guessed at.
  {
    const auto tool = tool_poses[static_cast<std::size_t>(best_tool)];
    const auto object_world = pose_multiply(tool, relative);
    const float w = object_world[3], x = object_world[4], y = object_world[5],
      zq = object_world[6];
    float zmin = std::numeric_limits<float>::infinity();
    float zmax = -std::numeric_limits<float>::infinity();
    for (const auto & sphere : attachment.spheres) {
      const float vx = sphere.center[0], vy = sphere.center[1],
        vz = sphere.center[2];
      const float tx = 2.0F * (y * vz - zq * vy);
      const float ty = 2.0F * (zq * vx - x * vz);
      const float tz = 2.0F * (x * vy - y * vx);
      const float cz = vz + w * tz + (x * ty - y * tx) + object_world[2];
      zmin = std::min(zmin, cz - sphere.radius);
      zmax = std::max(zmax, cz + sphere.radius);
    }
    std::fprintf(
      stderr,
      "[attach] '%s' -> '%s': object world z=%.4f dims=(%.3f,%.3f,%.3f) "
      "%zu spheres, payload world z [%.4f, %.4f]\n",
      name.c_str(), link.c_str(), object_world[2],
      record->dims[0], record->dims[1], record->dims[2],
      attachment.spheres.size(), zmin, zmax);
  }
}

std::string CppBridge::detach_object(curobo_runtime::RobotHandle)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->attached_objects.empty()) {
    return {};
  }
  impl_->ensure_planner("detach_object");
  for (const auto & entry : impl_->attach_links) {
    impl_->planner->detach_object(entry.second);
    if (impl_->ik) {
      impl_->ik->detach_object(entry.second);
    }
  }
  if (impl_->scene) {
    for (const auto & name : impl_->attached_objects) {
      impl_->scene->set_object_enabled(name, true);
    }
  }
  std::string detached = *impl_->attached_objects.begin();
  impl_->attached_objects.clear();
  impl_->attach_links.clear();
  impl_->visual_attachments.clear();
  return detached;
}

curobo_runtime::SphereSetOutcome CppBridge::collision_spheres_with_attached(
  curobo_runtime::RobotHandle, const std::vector<float> & q)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->native_sphere_model_ready && impl_->kinematics != nullptr &&
    static_cast<int>(q.size()) == impl_->dof())
  {
    const auto & model = impl_->native_sphere_model;
    const std::size_t n = static_cast<std::size_t>(model.num_spheres);
    impl_->native_sphere_q.upload(q.data(), q.size() * sizeof(float));
    const std::size_t sphere_bytes = n * 4 * sizeof(float);
    if (impl_->native_sphere_output.bytes() < sphere_bytes) {
      impl_->native_sphere_output.allocate(sphere_bytes);
    }
    impl_->kinematics->forward_spheres(
      static_cast<const float *>(impl_->native_sphere_q.get()),
      static_cast<float *>(impl_->native_sphere_output.get()), 1, 1, nullptr);
    cuda_check("collision_spheres", cudaDeviceSynchronize());

    curobo_runtime::SphereSetOutcome outcome;
    outcome.spheres.shape = {static_cast<std::int64_t>(n), 4};
    outcome.spheres.data.resize(n * 4);
    impl_->native_sphere_output.download(
      outcome.spheres.data.data(), sphere_bytes);
    outcome.self_padding.shape = {static_cast<std::int64_t>(n), 1};
    outcome.self_padding.data = model.self_collision_padding;
    if (outcome.self_padding.data.size() != n) {
      outcome.self_padding.data.assign(n, 0.0F);
    }
    outcome.self_distance.shape = {static_cast<std::int64_t>(n), 1};
    outcome.self_distance.data.assign(
      n, -std::numeric_limits<float>::infinity());
    outcome.link_names = impl_->native_sphere_link_names;
    outcome.attached.assign(n, 0);

    // Disabled links are stored as radius -100 by cuRobo. Mirror the bridge's
    // link_collision map so a toggled link disappears from RViz immediately.
    for (std::size_t i = 0; i < n; ++i) {
      const auto & link = outcome.link_names[i];
      const auto state = impl_->link_collision.find(link);
      if (state != impl_->link_collision.end() && !state->second) {
        outcome.spheres.data[i * 4 + 3] = -100.0F;
      }
    }

    const auto rotate_point = [](const std::array<float, 7> & pose,
                                 const std::array<float, 3> & v) {
        // Poses here are [x,y,z,qw,qx,qy,qz] (w FIRST), matching pose_multiply /
        // pose_inverse and Pose::quaternion.
        const float w = pose[3], x = pose[4], y = pose[5], z = pose[6];
        const float tx = 2.0F * (y * v[2] - z * v[1]);
        const float ty = 2.0F * (z * v[0] - x * v[2]);
        const float tz = 2.0F * (x * v[1] - y * v[0]);
        return std::array<float, 3>{
          v[0] + w * tx + (y * tz - z * ty) + pose[0],
          v[1] + w * ty + (z * tx - x * tz) + pose[1],
          v[2] + w * tz + (x * ty - y * tx) + pose[2]};
      };
    std::set<int> used_slots;
    for (const auto & entry : impl_->visual_attachments) {
      const auto link_it = impl_->attach_links.find(entry.first);
      if (link_it == impl_->attach_links.end()) {
        continue;
      }
      const auto slots_it = impl_->native_sphere_slots.find(link_it->second);
      if (slots_it == impl_->native_sphere_slots.end()) {
        continue;
      }
      const auto & attachment = entry.second;
      const auto tool = impl_->tool_pose(q, attachment.tool_index, "collision_spheres");
      const auto world = pose_multiply(tool, attachment.object_in_link);
      const auto & slots = slots_it->second;
      for (std::size_t k = 0; k < slots.size(); ++k) {
        const std::size_t slot = static_cast<std::size_t>(slots[k]);
        if (slot >= n) {
          continue;
        }
        used_slots.insert(slots[k]);
        const std::size_t base = slot * 4;
        if (k < attachment.spheres.size()) {
          const auto & sphere = attachment.spheres[k];
          const auto center = rotate_point(world, sphere.center);
          outcome.spheres.data[base + 0] = center[0];
          outcome.spheres.data[base + 1] = center[1];
          outcome.spheres.data[base + 2] = center[2];
          outcome.spheres.data[base + 3] = sphere.radius;
          outcome.attached[slot] = 1;
        } else {
          outcome.spheres.data[base + 3] = -100.0F;
        }
      }
    }

    // Exact self-collision pair distances, same tensors as the native kernel.
    const auto & pairs = model.self_collision_pairs;
    const auto & padding = outcome.self_padding.data;
    if (pairs.size() >= 2 && padding.size() == n) {
      for (std::size_t p = 0; p + 1 < pairs.size(); p += 2) {
        const int a = pairs[p];
        const int b = pairs[p + 1];
        if (a < 0 || b < 0 || static_cast<std::size_t>(a) >= n ||
          static_cast<std::size_t>(b) >= n)
        {
          continue;
        }
        const auto ia = static_cast<std::size_t>(a);
        const auto ib = static_cast<std::size_t>(b);
        const float radius_a = outcome.spheres.data[ia * 4 + 3];
        const float radius_b = outcome.spheres.data[ib * 4 + 3];
        if (radius_a < 0.0F || radius_b < 0.0F) {
          continue;
        }
        const float dx = outcome.spheres.data[ia * 4 + 0] -
          outcome.spheres.data[ib * 4 + 0];
        const float dy = outcome.spheres.data[ia * 4 + 1] -
          outcome.spheres.data[ib * 4 + 1];
        const float dz = outcome.spheres.data[ia * 4 + 2] -
          outcome.spheres.data[ib * 4 + 2];
        const float centre = std::sqrt(dx * dx + dy * dy + dz * dz);
        const float penetration =
          (radius_a + padding[ia]) + (radius_b + padding[ib]) - centre;
        outcome.self_distance.data[ia] =
          std::max(outcome.self_distance.data[ia], penetration);
        outcome.self_distance.data[ib] =
          std::max(outcome.self_distance.data[ib], penetration);
        if (penetration > 0.0F) {
          curobo_runtime::SelfPairContact contact;
          contact.sphere_a = a;
          contact.sphere_b = b;
          contact.depth = penetration;
          outcome.self_contacts.push_back(contact);
        }
      }
    }

    // World distance has no public native query yet; keep the Python bridge's
    // scene query when it is available, and leave the array empty otherwise.
    if (impl_->python) {
      try {
        curobo_runtime::TensorF32 q_tensor;
        q_tensor.shape = {1, static_cast<std::int64_t>(q.size())};
        q_tensor.data = q;
        const auto distances =
          impl_->python->collision_distance(impl_->python_handle, q_tensor);
        if (distances.size() == n) {
          outcome.world_distance.shape = {static_cast<std::int64_t>(n), 1};
          outcome.world_distance.data = distances;
          // The Python diagnostic scene still contains the attached object's
          // world copy (native attach disables it only in the native scene).
          // Do not report the payload as penetrating itself.
          for (const int slot : used_slots) {
            if (slot >= 0 && static_cast<std::size_t>(slot) < n) {
              outcome.world_distance.data[static_cast<std::size_t>(slot)] =
                std::numeric_limits<float>::quiet_NaN();
            }
          }
        }
      } catch (const std::exception &) {
        outcome.world_distance = curobo_runtime::TensorF32{};
      }
    }
    return outcome;
  }

  impl_->require_python("collision_spheres_with_attached");
  auto outcome = impl_->python->collision_spheres_with_attached(impl_->python_handle, q);
  if (impl_->visual_attachments.empty() || impl_->kinematics == nullptr ||
    static_cast<int>(q.size()) != impl_->dof() ||
    outcome.spheres.shape.size() != 2 || outcome.spheres.shape[1] != 4)
  {
    return outcome;
  }
  const std::size_t base_rows = static_cast<std::size_t>(outcome.spheres.shape[0]);
  const auto has_per_row = [base_rows](const TensorF32 & tensor) {
      return tensor.shape.size() == 2 && tensor.shape[1] == 1 &&
             tensor.data.size() == base_rows;
    };
  const bool has_padding = has_per_row(outcome.self_padding);
  const bool has_self_distance = has_per_row(outcome.self_distance);
  const bool has_world_distance = has_per_row(outcome.world_distance);
  const bool has_link_names = outcome.link_names.size() == base_rows;
  const auto rotate_point = [](const std::array<float, 7> & pose,
                               const std::array<float, 3> & v) {
      const float w = pose[3], x = pose[4], y = pose[5], z = pose[6];
      const float tx = 2.0F * (y * v[2] - z * v[1]);
      const float ty = 2.0F * (z * v[0] - x * v[2]);
      const float tz = 2.0F * (x * v[1] - y * v[0]);
      return std::array<float, 3>{
        v[0] + w * tx + (y * tz - z * ty) + pose[0],
        v[1] + w * ty + (z * tx - x * tz) + pose[1],
        v[2] + w * tz + (x * ty - y * tx) + pose[2]};
    };
  for (const auto & entry : impl_->visual_attachments) {
    const auto & attachment = entry.second;
    const auto tool = impl_->tool_pose(q, attachment.tool_index, "collision_spheres");
    const auto world = pose_multiply(tool, attachment.object_in_link);
    for (const auto & sphere : attachment.spheres) {
      const auto center = rotate_point(world, sphere.center);
      outcome.spheres.data.push_back(center[0]);
      outcome.spheres.data.push_back(center[1]);
      outcome.spheres.data.push_back(center[2]);
      outcome.spheres.data.push_back(sphere.radius);
      outcome.attached.push_back(1);
      if (has_padding) {
        outcome.self_padding.data.push_back(0.0F);
      }
      if (has_self_distance) {
        // Not a member of the embedded model's pair list: no self constraint.
        outcome.self_distance.data.push_back(
          -std::numeric_limits<float>::infinity());
      }
      if (has_world_distance) {
        // The world copy is disabled while attached; the diagnostic scene may
        // still contain it. Leave this row explicitly unknown rather than
        // reporting the payload as penetrating itself.
        outcome.world_distance.data.push_back(
          std::numeric_limits<float>::quiet_NaN());
      }
      if (has_link_names) {
        const auto link = impl_->attach_links.find(entry.first);
        outcome.link_names.push_back(
          link == impl_->attach_links.end() ? std::string() : link->second);
      }
      if (outcome.attached_local.shape.size() == 2 &&
        outcome.attached_local.shape[1] == 4)
      {
        const auto local = rotate_point(attachment.object_in_link, sphere.center);
        outcome.attached_local.data.push_back(local[0]);
        outcome.attached_local.data.push_back(local[1]);
        outcome.attached_local.data.push_back(local[2]);
        outcome.attached_local.data.push_back(sphere.radius);
        outcome.attached_local.shape[0] =
          static_cast<std::int64_t>(outcome.attached_local.shape[0] + 1);
      }
    }
  }
  const std::size_t rows = static_cast<std::size_t>(outcome.spheres.data.size() / 4);
  if (has_padding) {
    outcome.self_padding.shape[0] = static_cast<std::int64_t>(rows);
  }
  if (has_self_distance) {
    outcome.self_distance.shape[0] = static_cast<std::int64_t>(rows);
  }
  if (has_world_distance) {
    outcome.world_distance.shape[0] = static_cast<std::int64_t>(rows);
  }
  outcome.spheres.shape[0] = static_cast<std::int64_t>(rows);
  return outcome;
}
}  // namespace curobo_cpp_runtime

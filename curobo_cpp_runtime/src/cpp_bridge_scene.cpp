// SPDX-License-Identifier: Apache-2.0
#include "cpp_bridge_impl.hpp"

namespace curobo_cpp_runtime
{

curobo_runtime::SceneVersion CppBridge::scene_set(
  curobo_runtime::RobotHandle, const std::string & objects_json, bool clear_first)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("scene_set");
  impl_->ensure_scene();
  const auto version = impl_->scene->set_objects_json(objects_json, clear_first);
  if (clear_first) {
    // A new world starts with every link on the full layer mask. Restore any
    // per-object allowance bits left by the previous scene before dropping
    // the bookkeeping.
    for (const auto & entry : impl_->link_blocked_bits) {
      if (impl_->planner) {
        impl_->planner->set_link_collision_mask(entry.first, 0xFFFFFFFFU);
      }
      if (impl_->ik) {
        impl_->ik->set_link_collision_mask(entry.first, 0xFFFFFFFFU);
      }
    }
    impl_->object_layer_bits.clear();
    impl_->link_blocked_bits.clear();
    impl_->next_object_layer_bit = 2U;
  }
  if (clear_first && impl_->mapper && impl_->mapper->voxel_added) {
    // Scene::clear() also drops the voxel layer; re-attach the mapper grid from
    // its stable device ESDF buffer.
    impl_->mapper->voxel_added = false;
    impl_->update_native_voxel_grid();
  }
  if (impl_->scene) {
    for (const auto & name : impl_->attached_objects) {
      impl_->scene->set_object_enabled(name, false);
    }
  }
  if (impl_->python) {
    impl_->python->scene_set(impl_->python_handle, objects_json, clear_first);
  }
  return version;
}

std::string CppBridge::scene_get(curobo_runtime::RobotHandle)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("scene_get");
  if (!impl_->native_scene_valid) {
    impl_->require_python("scene_get");
    return impl_->python->scene_get(impl_->python_handle);
  }
  impl_->ensure_scene();
  return impl_->scene->get_objects_json();
}

curobo_runtime::LinkCollisionOutcome CppBridge::set_link_collision(
  curobo_runtime::RobotHandle, const std::vector<std::string> & link_names, bool enabled)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("set_link_collision");
  curobo_runtime::LinkCollisionOutcome outcome;
  for (const auto & name : link_names) {
    if (std::find(impl_->link_names.begin(), impl_->link_names.end(), name) ==
      impl_->link_names.end())
    {
      outcome.unknown.push_back(name);
      continue;
    }
    impl_->link_collision[name] = enabled;
    if (impl_->planner) {
      impl_->planner->set_link_collision_enabled(name, enabled);
    }
    if (impl_->ik) {
      impl_->ik->set_link_collision_enabled(name, enabled);
    }
    outcome.applied.push_back(name);
  }
  if (impl_->python) {
    impl_->python->set_link_collision(impl_->python_handle, link_names, enabled);
  }
  return outcome;
}

curobo_runtime::LinkCollisionOutcome CppBridge::set_object_link_collision(
  curobo_runtime::RobotHandle, const std::string & object_name,
  const std::vector<std::string> & link_names, bool enabled)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->require_initialized("set_object_link_collision");
  curobo_runtime::LinkCollisionOutcome outcome;

  const auto * record = impl_->scene ? impl_->scene->object(object_name) : nullptr;
  if (record == nullptr) {
    outcome.unknown = link_names;
    if (impl_->python) {
      try {
        impl_->python->set_object_link_collision(
          impl_->python_handle, object_name, link_names, enabled);
      } catch (const std::exception &) {
      }
    }
    return outcome;
  }

  const std::uint32_t all_bits = 0xFFFFFFFFU;
  auto apply_masks = [&](const std::vector<std::string> & links, std::uint32_t blocked_add,
      std::uint32_t blocked_remove) {
      for (const auto & link : links) {
        if (std::find(impl_->link_names.begin(), impl_->link_names.end(), link) ==
          impl_->link_names.end())
        {
          outcome.unknown.push_back(link);
          continue;
        }
        std::uint32_t & blocked = impl_->link_blocked_bits[link];
        blocked |= blocked_add;
        blocked &= ~blocked_remove;
        const std::uint32_t mask = all_bits & ~blocked;
        if (impl_->planner) {
          impl_->planner->set_link_collision_mask(link, mask);
        }
        if (impl_->ik) {
          impl_->ik->set_link_collision_mask(link, mask);
        }
        outcome.applied.push_back(link);
      }
    };

  auto it = impl_->object_layer_bits.find(object_name);
  if (!enabled) {
    std::uint32_t bit = 0U;
    if (it == impl_->object_layer_bits.end()) {
      bit = impl_->next_object_layer_bit;
      if (bit == 0U || bit == 1U) {
        outcome.unknown = link_names;
        return outcome;
      }
      impl_->next_object_layer_bit <<= 1U;
      impl_->object_layer_bits[object_name] = bit;
    } else {
      bit = it->second;
    }
    impl_->scene->scene().set_obstacle_layers(record->handle, bit);
    apply_masks(link_names, bit, 0U);
  } else if (it != impl_->object_layer_bits.end()) {
    const std::uint32_t bit = it->second;
    impl_->scene->scene().set_obstacle_layers(record->handle, 1U);
    apply_masks(link_names, 0U, bit);
  } else {
    // Already enabled. Still report the links so a repeated call is a no-op
    // success rather than an empty failure.
    for (const auto & link : link_names) {
      if (std::find(impl_->link_names.begin(), impl_->link_names.end(), link) ==
        impl_->link_names.end())
      {
        outcome.unknown.push_back(link);
      } else {
        outcome.applied.push_back(link);
      }
    }
  }

  if (impl_->python) {
    try {
      impl_->python->set_object_link_collision(
        impl_->python_handle, object_name, link_names, enabled);
    } catch (const std::exception &) {
    }
  }
  return outcome;
}
}  // namespace curobo_cpp_runtime

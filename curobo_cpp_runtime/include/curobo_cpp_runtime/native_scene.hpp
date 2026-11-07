// SPDX-License-Identifier: Apache-2.0
// Small JSON -> native curobo::robot::Scene adapter.
//
// The ROS side has always handed the bridge SceneObject-shaped JSON (see
// obstacle_manager.scene_json). This adapter keeps that boundary and only
// chooses native primitives; it does not parse YAML.
#ifndef CUROBO_CPP_RUNTIME__NATIVE_SCENE_HPP_
#define CUROBO_CPP_RUNTIME__NATIVE_SCENE_HPP_

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "curobo_robot_runtime/scene.h"

namespace curobo_cpp_runtime
{

struct SceneObjectRecord
{
  int type{0};
  curobo::robot::Pose pose{};
  std::array<float, 3> dims{1.0F, 1.0F, 1.0F};
  // Stable handle into NativeScene::scene(). Used to disable a grasped object
  // while it is represented by the attachment model.
  curobo::robot::ObstacleHandle handle{};
};

class NativeScene
{
public:
  explicit NativeScene(const curobo::robot::SceneConfig & config = {});

  NativeScene(const NativeScene &) = delete;
  NativeScene & operator=(const NativeScene &) = delete;

  curobo::robot::Scene & scene() {return scene_;}
  const curobo::robot::Scene & scene() const {return scene_;}

  // Replace/append from obstacle_manager's scene JSON. Returns the scene
  // version (monotonic, like the Python bridge).
  std::int64_t set_objects_json(const std::string & objects_json, bool clear_first);

  // {"version":N,"names":[...]} - the subset the ROS get_scene_objects path
  // consumes. The full primitive descriptions stay in ObstacleManager.
  std::string get_objects_json() const;

  std::int64_t version() const {return version_;}
  const std::vector<std::string> & names() const {return names_;}
  const std::map<std::string, SceneObjectRecord> & objects() const {return objects_;}
  const SceneObjectRecord * object(const std::string & name) const
  {
    const auto it = objects_.find(name);
    return it == objects_.end() ? nullptr : &it->second;
  }

  // Enable/disable one object in the live collision world without removing it
  // from the scene list. Returns false when the name is unknown.
  bool set_object_enabled(const std::string & name, bool enabled);

private:
  curobo::robot::Scene scene_;
  std::int64_t version_{0};
  std::vector<std::string> names_;
  std::map<std::string, SceneObjectRecord> objects_;
};

}  // namespace curobo_cpp_runtime

#endif  // CUROBO_CPP_RUNTIME__NATIVE_SCENE_HPP_

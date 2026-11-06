// SPDX-License-Identifier: Apache-2.0
// Small JSON -> native curobo::robot::Scene adapter.
//
// The ROS side has always handed the bridge SceneObject-shaped JSON (see
// obstacle_manager.scene_json). This adapter keeps that boundary and only
// chooses native primitives; it does not parse YAML.
#ifndef CUROBO_CPP_RUNTIME__NATIVE_SCENE_HPP_
#define CUROBO_CPP_RUNTIME__NATIVE_SCENE_HPP_

#include <cstdint>
#include <string>
#include <vector>

#include "curobo_robot_runtime/scene.h"

namespace curobo_cpp_runtime
{

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

private:
  curobo::robot::Scene scene_;
  std::int64_t version_{0};
  std::vector<std::string> names_;
};

}  // namespace curobo_cpp_runtime

#endif  // CUROBO_CPP_RUNTIME__NATIVE_SCENE_HPP_

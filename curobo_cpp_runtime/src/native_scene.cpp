// SPDX-License-Identifier: Apache-2.0
#include "curobo_cpp_runtime/native_scene.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "curobo_cpp_runtime/detail/mini_json.hpp"

namespace curobo_cpp_runtime
{
namespace
{

curobo::robot::Pose to_pose(const json::Value * value, const std::string & object)
{
  curobo::robot::Pose pose;
  if (value == nullptr || !value->is_array() || value->as_array().size() < 7) {
    throw std::runtime_error(
            "native scene_set: object '" + object +
            "' is missing a [x,y,z,qw,qx,qy,qz] pose");
  }
  for (int i = 0; i < 3; ++i) {
    pose.position[static_cast<std::size_t>(i)] =
      static_cast<float>(value->as_array()[static_cast<std::size_t>(i)].as_double());
  }
  for (int i = 0; i < 4; ++i) {
    pose.quaternion[static_cast<std::size_t>(i)] =
      static_cast<float>(value->as_array()[static_cast<std::size_t>(3 + i)].as_double());
  }
  return pose;
}

std::array<float, 3> to_dims(const json::Value * value)
{
  std::array<float, 3> dims{1.0F, 1.0F, 1.0F};
  if (value != nullptr && value->is_array()) {
    for (std::size_t i = 0; i < 3 && i < value->as_array().size(); ++i) {
      dims[i] = static_cast<float>(value->as_array()[i].as_double(1.0));
    }
  }
  return dims;
}

std::string object_name(const json::Value & object, std::size_t index)
{
  const std::string name = object.string_at("name");
  return name.empty() ? "object_" + std::to_string(index) : name;
}

curobo::robot::ObstacleHandle add_inline_mesh(
  curobo::robot::Scene & scene, const json::Value & object, const std::string & name,
  const curobo::robot::Pose & pose, const std::array<float, 3> & dims)
{
  const json::Value * vertices = object.find("vertices");
  const json::Value * triangles = object.find("triangles");
  if (vertices == nullptr || triangles == nullptr ||
    !vertices->is_array() || !triangles->is_array())
  {
    const std::string path = object.string_at("mesh_file_path");
    if (!path.empty()) {
      throw std::runtime_error(
              "native scene_set: mesh file loading is not supported by "
              "cpp_runtime (object '" + name + "', file '" + path + "')");
    }
    throw std::runtime_error(
            "native scene_set: mesh object '" + name +
            "' has neither inline vertices/triangles nor a supported file");
  }

  curobo::robot::Mesh mesh;
  mesh.name = name;
  mesh.pose = pose;
  const bool has_scale = std::abs(dims[0] - 1.0F) > 1.0e-6F ||
    std::abs(dims[1] - 1.0F) > 1.0e-6F || std::abs(dims[2] - 1.0F) > 1.0e-6F;
  for (const auto & vertex : vertices->as_array()) {
    if (!vertex.is_array() || vertex.as_array().size() < 3) {
      throw std::runtime_error(
              "native scene_set: mesh object '" + name + "' has a malformed vertex");
    }
    std::array<float, 3> value{};
    for (std::size_t i = 0; i < 3; ++i) {
      value[i] = static_cast<float>(vertex.as_array()[i].as_double());
      if (has_scale) {
        value[i] *= dims[i];
      }
    }
    mesh.vertices.push_back(value);
  }
  const auto & indices = triangles->as_array();
  if (indices.size() % 3 != 0) {
    throw std::runtime_error(
            "native scene_set: mesh object '" + name +
            "' triangle indices are not a multiple of three");
  }
  for (std::size_t i = 0; i + 2 < indices.size(); i += 3) {
    mesh.triangles.push_back(
      {static_cast<std::int32_t>(indices[i].as_double()),
        static_cast<std::int32_t>(indices[i + 1].as_double()),
        static_cast<std::int32_t>(indices[i + 2].as_double())});
  }
  return scene.add_mesh(mesh);
}

}  // namespace

NativeScene::NativeScene(const curobo::robot::SceneConfig & config)
: scene_(config)
{
}

std::int64_t NativeScene::set_objects_json(
  const std::string & objects_json, bool clear_first)
{
  const json::Value document = json::parse(objects_json);
  if (!document.is_array()) {
    throw std::runtime_error("native scene_set: the object list must be a JSON array");
  }

  if (clear_first) {
    scene_.clear();
    names_.clear();
    objects_.clear();
  }

  std::size_t index = 0;
  for (const auto & object : document.as_array()) {
    if (!object.is_object()) {
      throw std::runtime_error("native scene_set: every scene object must be a JSON object");
    }
    const std::string name = object_name(object, index++);
    const int type = static_cast<int>(object.number_at("type"));
    const curobo::robot::Pose pose = to_pose(object.find("pose"), name);
    const std::array<float, 3> dims = to_dims(object.find("dims"));
    curobo::robot::ObstacleHandle handle;

    switch (type) {
      case 0: {  // CUBOID
          curobo::robot::Cuboid cuboid;
          cuboid.name = name;
          cuboid.pose = pose;
          cuboid.dimensions = dims;
          handle = scene_.add_cuboid(cuboid);
          break;
        }
      case 1: {  // SPHERE
          curobo::robot::Sphere sphere;
          sphere.name = name;
          sphere.pose = pose;
          sphere.radius = dims[0];
          handle = scene_.add_sphere(sphere);
          break;
        }
      case 2: {  // CAPSULE
          curobo::robot::Capsule capsule;
          capsule.name = name;
          capsule.pose = pose;
          capsule.base = {0.0F, 0.0F, 0.0F};
          capsule.tip = {0.0F, 0.0F, dims[1]};
          capsule.radius = dims[0];
          handle = scene_.add_capsule(capsule);
          break;
        }
      case 3: {  // CYLINDER
          curobo::robot::Cylinder cylinder;
          cylinder.name = name;
          cylinder.pose = pose;
          cylinder.radius = dims[0];
          cylinder.height = dims[1];
          handle = scene_.add_cylinder(cylinder);
          break;
        }
      case 4:  // MESH
        handle = add_inline_mesh(scene_, object, name, pose, dims);
        break;
      default:
        throw std::runtime_error(
                "native scene_set: object '" + name + "' has unsupported type " +
                std::to_string(type));
    }
    names_.push_back(name);
    SceneObjectRecord record;
    record.type = type;
    record.pose = pose;
    record.dims = dims;
    record.handle = handle;
    objects_[name] = record;
  }

  ++version_;
  return version_;
}

bool NativeScene::set_object_enabled(const std::string & name, bool enabled)
{
  const auto it = objects_.find(name);
  if (it == objects_.end() || !it->second.handle.valid()) {
    return false;
  }
  scene_.set_obstacle_enabled(it->second.handle, enabled);
  return true;
}

std::string NativeScene::get_objects_json() const
{
  std::ostringstream out;
  out << "{\"version\":" << version_ << ",\"names\":[";
  for (std::size_t i = 0; i < names_.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    out << json::quote(names_[i]);
  }
  out << "]}";
  return out.str();
}

}  // namespace curobo_cpp_runtime

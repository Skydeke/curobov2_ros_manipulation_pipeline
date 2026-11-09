#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <functional>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <OgreQuaternion.h>
#include <OgreSceneNode.h>
#include <OgreVector3.h>

#include <rclcpp/exceptions.hpp>
#include <rclcpp/parameter_client.hpp>
#include <rclcpp/time.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/frame_manager_iface.hpp>
#include <rviz_common/logging.hpp>
#include <rviz_common/panel_dock_widget.hpp>
#include <rviz_common/properties/parse_color.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <rviz_common/validate_floats.hpp>
#include <rviz_common/window_manager_interface.hpp>
#include <rviz_default_plugins/robot/robot_link.hpp>
#include <rviz_rendering/objects/shape.hpp>

#include <curobov2_ros_rviz/curobo_link_updater.hpp>
#include <curobov2_ros_rviz/curobo_planning_display.hpp>

namespace
{

bool validateFloats(const visualization_msgs::msg::InteractiveMarker & msg)
{
  bool valid = rviz_common::validateFloats(msg.pose);
  valid = valid && rviz_common::validateFloats(msg.scale);
  for (const auto & control : msg.controls) {
    valid = valid && rviz_common::validateFloats(control.orientation);
    for (const auto & marker : control.markers) {
      valid = valid && rviz_common::validateFloats(marker.pose);
      valid = valid && rviz_common::validateFloats(marker.scale);
      valid = valid && rviz_common::validateFloats(marker.color);
      valid = valid && rviz_common::validateFloats(marker.points);
    }
  }
  return valid;
}

std::vector<std::string> splitToolFrames(const std::string & text)
{
  std::vector<std::string> frames;
  std::string current;
  for (char c : text) {
    if (c == ',' || c == ';' || std::isspace(static_cast<unsigned char>(c))) {
      if (!current.empty()) {
        frames.push_back(current);
        current.clear();
      }
    } else {
      current.push_back(c);
    }
  }
  if (!current.empty()) {
    frames.push_back(current);
  }
  return frames;
}

}  // namespace

namespace curobov2_ros_rviz
{

void CuroboPlanningDisplay::updateCollisionTopic()
{
  spheres_subscription_.reset();
  {
    std::lock_guard<std::mutex> lock(spheres_mutex_);
    spheres_msg_new_.reset();
  }
  spheres_got_message_ = false;
  sphere_active_count_ = 0;
  live_spheres_.clear();
  playback_spheres_.clear();
  for (auto & sv : sphere_pool_) {
    if (sv.node) {
      sv.node->setVisible(false);
    }
  }

  if (!collision_topic_property_->getStdString().empty()) {
    auto ros_node_abstraction = context_->getRosNodeAbstraction().lock();
    if (!ros_node_abstraction) {
      return;
    }
    spheres_subscription_ = ros_node_abstraction->get_raw_node()
      ->create_subscription<visualization_msgs::msg::MarkerArray>(
      collision_topic_property_->getStdString(), 10,
      std::bind(&CuroboPlanningDisplay::onCollisionSpheresMessage, this,
        std::placeholders::_1));
  }
}

void CuroboPlanningDisplay::onCollisionSpheresMessage(
  visualization_msgs::msg::MarkerArray::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(spheres_mutex_);
  spheres_msg_new_ = msg;
}

void CuroboPlanningDisplay::updateSceneTopic()
{
  scene_subscription_.reset();
  {
    std::lock_guard<std::mutex> lock(scene_mutex_);
    scene_msg_new_.reset();
  }
  scene_got_message_ = false;
  scene_active_count_ = 0;
  live_scene_objects_.clear();
  for (auto & sv : scene_pool_) {
    if (sv.node) {
      sv.node->setVisible(false);
    }
  }

  if (scene_topic_property_ && !scene_topic_property_->getStdString().empty()) {
    auto ros_node_abstraction = context_->getRosNodeAbstraction().lock();
    if (!ros_node_abstraction) {
      return;
    }
    scene_subscription_ = ros_node_abstraction->get_raw_node()
      ->create_subscription<visualization_msgs::msg::MarkerArray>(
      scene_topic_property_->getStdString(), 10,
      std::bind(&CuroboPlanningDisplay::onSceneObjectsMessage, this,
        std::placeholders::_1));
  }
}

void CuroboPlanningDisplay::onSceneObjectsMessage(
  visualization_msgs::msg::MarkerArray::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(scene_mutex_);
  scene_msg_new_ = msg;
}

void CuroboPlanningDisplay::updateShowAttached()
{
  updateSpheresVisibility();
}

void CuroboPlanningDisplay::updateShowCollisionSpheres()
{
  updateSpheresVisibility();
}

void CuroboPlanningDisplay::updateShowSceneObjects()
{
  updateSceneObjectsVisibility();
}

void CuroboPlanningDisplay::updateSpheresVisibility()
{
  const bool show_attached = show_attached_property_->getBool();
  const bool show_collision = show_collision_spheres_property_->getBool();
  const size_t active = std::min(sphere_active_count_, sphere_pool_.size());
  for (size_t i = 0; i < active; ++i) {
    if (sphere_pool_[i].node) {
      sphere_pool_[i].node->setVisible(
        sphere_pool_[i].attached ? show_attached : show_collision);
    }
  }
  for (size_t i = active; i < sphere_pool_.size(); ++i) {
    if (sphere_pool_[i].node) {
      sphere_pool_[i].node->setVisible(false);
    }
  }
}

void CuroboPlanningDisplay::applySpheres(const std::vector<SphereDatum> & data)
{
  while (sphere_pool_.size() < data.size()) {
    SphereVisual sv;
    sv.node = scene_node_->createChildSceneNode();
    sv.shape = std::make_shared<rviz_rendering::Shape>(
      rviz_rendering::Shape::Sphere, scene_manager_, sv.node);
    sv.node->setVisible(false);
    sphere_pool_.push_back(std::move(sv));
  }
  for (size_t i = 0; i < data.size(); ++i) {
    SphereVisual & sv = sphere_pool_[i];
    const SphereDatum & datum = data[i];
    sv.shape->setScale(Ogre::Vector3(datum.sx, datum.sy, datum.sz));
    sv.shape->setColor(datum.r, datum.g, datum.b, datum.a);
    sv.attached = datum.attached;
  }
  for (size_t i = data.size(); i < sphere_pool_.size(); ++i) {
    if (sphere_pool_[i].node) {
      sphere_pool_[i].node->setVisible(false);
    }
  }
  sphere_active_count_ = data.size();
  updateSpheresVisibility();
}

void CuroboPlanningDisplay::syncSpheres()
{
  visualization_msgs::msg::MarkerArray::ConstSharedPtr msg;
  {
    std::lock_guard<std::mutex> lock(spheres_mutex_);
    if (!spheres_msg_new_) {
      return;
    }
    msg = spheres_msg_new_;
    spheres_msg_new_.reset();
  }
  spheres_got_message_ = true;
  deleteStatus("Collision Spheres");

  size_t want = 0;
  for (const auto & marker : msg->markers) {
    if (marker.action == visualization_msgs::msg::Marker::ADD &&
      marker.type == visualization_msgs::msg::Marker::SPHERE)
    {
      ++want;
    }
  }
  while (sphere_pool_.size() < want) {
    SphereVisual sv;
    sv.node = scene_node_->createChildSceneNode();
    sv.shape = std::make_shared<rviz_rendering::Shape>(
      rviz_rendering::Shape::Sphere, scene_manager_, sv.node);
    sv.node->setVisible(false);
    sphere_pool_.push_back(std::move(sv));
  }

  std::vector<SphereDatum> live;
  for (const auto & marker : msg->markers) {
    const bool attached = marker.ns == "attached_spheres";
    const bool collision = marker.ns == "collision_spheres";
    if (marker.action != visualization_msgs::msg::Marker::ADD ||
      marker.type != visualization_msgs::msg::Marker::SPHERE ||
      (!attached && !collision))
    {
      continue;
    }
    SphereDatum datum;
    datum.local = Ogre::Vector3(
      static_cast<float>(marker.pose.position.x),
      static_cast<float>(marker.pose.position.y),
      static_cast<float>(marker.pose.position.z));
    datum.frame = marker.header.frame_id;
    datum.sx = static_cast<float>(marker.scale.x);
    datum.sy = static_cast<float>(marker.scale.y);
    datum.sz = static_cast<float>(marker.scale.z);
    datum.r = marker.color.r;
    datum.g = marker.color.g;
    datum.b = marker.color.b;
    datum.a = marker.color.a;
    datum.attached = attached;
    live.push_back(datum);
  }

  live_spheres_ = live;
  const bool has_attached = std::any_of(
    live_spheres_.begin(), live_spheres_.end(),
    [](const SphereDatum & datum) {return datum.attached;});
  if (!animating_ || has_attached) {
    playback_spheres_ = live_spheres_;
  }
  applySpheres(playback_spheres_);
  poseSpheres();
}

void CuroboPlanningDisplay::poseSpheres()
{
  const size_t active = std::min(sphere_active_count_, sphere_pool_.size());
  Ogre::Vector3 offset = Ogre::Vector3::ZERO;
  Ogre::Quaternion rotation = Ogre::Quaternion::IDENTITY;
  std::string resolved_frame;
  for (size_t i = 0; i < active; ++i) {
    SphereVisual & sv = sphere_pool_[i];
    if (!sv.node || i >= playback_spheres_.size()) {
      continue;
    }
    const SphereDatum & datum = playback_spheres_[i];
    // Only attached spheres are link-relative and ride the trajectory ghost.
    // Robot collision spheres are already expressed in the live base frame, so
    // they must follow TF even while a plan is displayed.
    const auto fk = datum.attached ? current_transforms_.find(datum.frame) :
      current_transforms_.end();
    if (fk != current_transforms_.end()) {
      const Eigen::Vector3d point = fk->second * Eigen::Vector3d(
        datum.local.x, datum.local.y, datum.local.z);
      sv.node->setPosition(robot_root_offset_ + robot_root_rotation_ *
        Ogre::Vector3(
          static_cast<float>(point.x()), static_cast<float>(point.y()),
          static_cast<float>(point.z())));
      continue;
    }
    auto * frame_manager = context_ ? context_->getFrameManager() : nullptr;
    if (datum.frame.empty() || frame_manager == nullptr ||
      datum.frame == frame_manager->getFixedFrame())
    {
      sv.node->setPosition(datum.local);
      continue;
    }
    if (datum.frame != resolved_frame) {
      rclcpp::Time stamp(0, 0, RCL_ROS_TIME);
      if (!frame_manager->getTransform(datum.frame, stamp, offset, rotation)) {
        setMissingTransformToFixedFrame(datum.frame);
        sv.node->setVisible(false);
        continue;
      }
      setTransformOk();
      resolved_frame = datum.frame;
    }
    sv.node->setPosition(rotation * datum.local + offset);
  }
}

// ============================================================================
// World scene-object markers
// ============================================================================

void CuroboPlanningDisplay::syncSceneObjects()
{
  visualization_msgs::msg::MarkerArray::ConstSharedPtr msg;
  {
    std::lock_guard<std::mutex> lock(scene_mutex_);
    if (!scene_msg_new_) {
      return;
    }
    msg = scene_msg_new_;
    scene_msg_new_.reset();
  }
  scene_got_message_ = true;
  deleteStatus("Scene Objects");

  std::vector<SceneDatum> live;
  for (const auto & marker : msg->markers) {
    if (marker.action == visualization_msgs::msg::Marker::DELETEALL) {
      live.clear();
      continue;
    }
    if (marker.action != visualization_msgs::msg::Marker::ADD) {
      continue;
    }
    // Cube/Sphere/Cylinder have exact rviz_rendering::Shape equivalents.
    // Capsules are published as cylinders by the scene publisher.
    if (marker.type != visualization_msgs::msg::Marker::CUBE &&
      marker.type != visualization_msgs::msg::Marker::SPHERE &&
      marker.type != visualization_msgs::msg::Marker::CYLINDER)
    {
      continue;
    }
    SceneDatum datum;
    datum.type = marker.type;
    datum.local = Ogre::Vector3(
      static_cast<float>(marker.pose.position.x),
      static_cast<float>(marker.pose.position.y),
      static_cast<float>(marker.pose.position.z));
    datum.orientation = Ogre::Quaternion(
      static_cast<Ogre::Real>(marker.pose.orientation.w),
      static_cast<Ogre::Real>(marker.pose.orientation.x),
      static_cast<Ogre::Real>(marker.pose.orientation.y),
      static_cast<Ogre::Real>(marker.pose.orientation.z));
    datum.frame = marker.header.frame_id;
    datum.sx = static_cast<float>(marker.scale.x);
    datum.sy = static_cast<float>(marker.scale.y);
    datum.sz = static_cast<float>(marker.scale.z);
    datum.r = marker.color.r;
    datum.g = marker.color.g;
    datum.b = marker.color.b;
    datum.a = marker.color.a;
    live.push_back(datum);
  }

  live_scene_objects_ = live;
  applySceneObjects(live_scene_objects_);
  poseSceneObjects();
}

void CuroboPlanningDisplay::applySceneObjects(const std::vector<SceneDatum> & data)
{
  while (scene_pool_.size() < data.size()) {
    SceneVisual sv;
    sv.node = scene_node_->createChildSceneNode();
    sv.node->setVisible(false);
    scene_pool_.push_back(std::move(sv));
  }
  for (size_t i = 0; i < data.size(); ++i) {
    SceneVisual & sv = scene_pool_[i];
    const SceneDatum & datum = data[i];
    if (!sv.node) {
      continue;
    }
    if (sv.marker_type != datum.type) {
      // Shape type is fixed at construction, so recreate only when the marker
      // changes primitive type (rare). The parent node is retained.
      sv.shape.reset();
      rviz_rendering::Shape::Type type = rviz_rendering::Shape::Cube;
      if (datum.type == visualization_msgs::msg::Marker::SPHERE) {
        type = rviz_rendering::Shape::Sphere;
      } else if (datum.type == visualization_msgs::msg::Marker::CYLINDER) {
        type = rviz_rendering::Shape::Cylinder;
      }
      sv.shape = std::make_shared<rviz_rendering::Shape>(type, scene_manager_, sv.node);
      sv.marker_type = datum.type;
    }
    if (sv.shape) {
      sv.shape->setScale(Ogre::Vector3(datum.sx, datum.sy, datum.sz));
      sv.shape->setColor(datum.r, datum.g, datum.b, datum.a);
    }
  }
  for (size_t i = data.size(); i < scene_pool_.size(); ++i) {
    if (scene_pool_[i].node) {
      scene_pool_[i].node->setVisible(false);
    }
  }
  scene_active_count_ = data.size();
  updateSceneObjectsVisibility();
}

void CuroboPlanningDisplay::updateSceneObjectsVisibility()
{
  const bool show = show_scene_objects_property_ &&
    show_scene_objects_property_->getBool();
  const size_t active = std::min(scene_active_count_, scene_pool_.size());
  for (size_t i = 0; i < active; ++i) {
    if (scene_pool_[i].node) {
      scene_pool_[i].node->setVisible(show);
    }
  }
  for (size_t i = active; i < scene_pool_.size(); ++i) {
    if (scene_pool_[i].node) {
      scene_pool_[i].node->setVisible(false);
    }
  }
}

void CuroboPlanningDisplay::poseSceneObjects()
{
  const size_t active = std::min(scene_active_count_, scene_pool_.size());
  Ogre::Vector3 offset = Ogre::Vector3::ZERO;
  Ogre::Quaternion rotation = Ogre::Quaternion::IDENTITY;
  std::string resolved_frame;
  for (size_t i = 0; i < active; ++i) {
    SceneVisual & sv = scene_pool_[i];
    if (!sv.node || i >= live_scene_objects_.size()) {
      continue;
    }
    const SceneDatum & datum = live_scene_objects_[i];
    Ogre::Quaternion node_rotation = Ogre::Quaternion::IDENTITY;
    auto * frame_manager = context_ ? context_->getFrameManager() : nullptr;
    if (datum.frame.empty() || frame_manager == nullptr ||
      datum.frame == frame_manager->getFixedFrame())
    {
      sv.node->setPosition(datum.local);
    } else {
      if (datum.frame != resolved_frame) {
        rclcpp::Time stamp(0, 0, RCL_ROS_TIME);
        if (!frame_manager->getTransform(datum.frame, stamp, offset, rotation)) {
          setMissingTransformToFixedFrame(datum.frame);
          sv.node->setVisible(false);
          continue;
        }
        setTransformOk();
        resolved_frame = datum.frame;
      }
      sv.node->setPosition(rotation * datum.local + offset);
      node_rotation = rotation;
    }
    sv.node->setOrientation(node_rotation);
    if (sv.shape) {
      sv.shape->setOrientation(datum.orientation);
    }
  }
}

// ============================================================================
// Trail
// ============================================================================

void CuroboPlanningDisplay::rebuildTrail()
{
  std::vector<int> wanted;
  if (show_trail_property_->getBool() && trajectory_robot_ && !waypoints_.empty()) {
    const int step = std::max(1, trail_step_size_property_->getInt());
    const int count = static_cast<int>(waypoints_.size());
    for (int i = 0; i < count; i += step) {
      wanted.push_back(i);
    }
  }

  const float alpha = alpha_property_->getFloat() * 0.4f;
  while (trail_robots_.size() < wanted.size()) {
    TrailRobot trail;
    trail.node = scene_node_->createChildSceneNode();
    // Apply the current base->fixed offset at creation. updateTrajectoryRoot()
    // has already run this frame, so a new trail node would otherwise render
    // once at the base-frame origin and visibly jump into place next frame.
    trail.node->setPosition(robot_root_offset_);
    trail.node->setOrientation(robot_root_rotation_);
    trail.robot = std::make_unique<rviz_default_plugins::robot::Robot>(
      trail.node, context_, "Trail", nullptr);
    trail.robot->load(*urdf_model_, true, false, false, false);
    trail.robot->setVisualVisible(true);
    trail.robot->setCollisionVisible(false);
    trail.robot->setAlpha(alpha);
    trail.robot->setVisible(false);
    trail_robots_.push_back(std::move(trail));
  }

  for (size_t i = 0; i < trail_robots_.size(); ++i) {
    TrailRobot & trail = trail_robots_[i];
    if (i < wanted.size()) {
      trail.waypoint_index = wanted[i];
      CuroboLinkUpdater updater(waypoints_[wanted[i]].link_transforms);
      trail.robot->update(updater);
      trail.robot->setAlpha(alpha);
    } else {
      trail.waypoint_index = -1;
      trail.robot->setVisible(false);
    }
  }
  updateTrailVisibility();
}

void CuroboPlanningDisplay::clearTrailRobots()
{
  for (auto & trail : trail_robots_) {
    trail.robot.reset();
    if (trail.node) {
      scene_node_->removeAndDestroyChild(trail.node);
      trail.node = nullptr;
    }
  }
  trail_robots_.clear();
}

void CuroboPlanningDisplay::updateTrailVisibility()
{
  const bool show = show_trail_property_->getBool() && animating_;
  for (auto & trail : trail_robots_) {
    trail.robot->setVisible(show && trail.waypoint_index <= current_state_);
  }
}

// ============================================================================
// Targets
// ============================================================================


}  // namespace curobov2_ros_rviz

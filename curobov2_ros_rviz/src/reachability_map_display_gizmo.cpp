#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>

#include <OgreColourValue.h>
#include <OgreQuaternion.h>
#include <OgreSceneNode.h>
#include <OgreVector3.h>

#include <rclcpp/time.hpp>
#include <rclcpp/exceptions.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/logging.hpp>
#include <rviz_common/properties/parse_color.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <rviz_common/validate_floats.hpp>

#include <curobov2_ros_rviz/curobo_link_updater.hpp>
#include <curobov2_ros_rviz/reachability_map_display.hpp>

namespace
{

/// All floats finite? Same checks as rviz's InteractiveMarkerDisplay.
bool validateFloats(const visualization_msgs::msg::InteractiveMarker & msg)
{
  bool valid = true;
  valid = valid && rviz_common::validateFloats(msg.pose);
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

}  // namespace

namespace curobov2_ros_rviz
{

void ReachabilityMapDisplay::createGizmo()
{
  auto ros_node_abstraction = context_->getRosNodeAbstraction().lock();
  if (!ros_node_abstraction) {
    return;
  }
  auto node = ros_node_abstraction->get_raw_node();

  if (gizmo_frame_.empty()) {
    gizmo_frame_ = context_->getFrameManager()->getFixedFrame();
    if (gizmo_frame_.empty()) {
      gizmo_frame_ = "base_link";
    }
  }

  if (!gizmo_server_) {
    gizmo_server_ = std::make_shared<interactive_markers::InteractiveMarkerServer>(
      gizmo_name_, node.get());
    RCLCPP_INFO(
      node->get_logger(),
      "Reachability plane gizmo on /%s/* (rendered via embedded "
      "interactive-marker client).", gizmo_name_.c_str());
  }
  makeGizmoMarker(planePoseFromProperties());
  gizmo_server_->applyChanges();
  connectGizmoClient();
}

void ReachabilityMapDisplay::destroyGizmo()
{
  disconnectGizmoClient();
  if (gizmo_server_) {
    gizmo_server_->clear();
    gizmo_server_->applyChanges();
  }
  gizmo_server_.reset();
}

void ReachabilityMapDisplay::connectGizmoClient()
{
  if (!gizmo_client_ || gizmo_client_connected_) {
    return;
  }
  gizmo_client_->connect(gizmo_name_);
  gizmo_client_connected_ = true;
}

void ReachabilityMapDisplay::disconnectGizmoClient()
{
  if (gizmo_client_) {
    gizmo_client_->disconnect();
  }
  gizmo_client_connected_ = false;
  eraseAllGizmoMarkers();
}

void ReachabilityMapDisplay::makeGizmoMarker(const geometry_msgs::msg::Pose & pose)
{
  if (!gizmo_server_) {
    return;
  }

  gizmo_body_size_x_ = plane_size_x_->getFloat();
  gizmo_body_size_y_ = plane_size_y_->getFloat();

  visualization_msgs::msg::InteractiveMarker int_marker;
  int_marker.header.frame_id = gizmo_frame_;
  // Frame-locked marker (header.stamp == 0) so RViz follows TF every frame;
  // a timestamped marker would freeze the reference transform at creation.
  // See CuroboPlanningDisplay::insertGizmoMarker for the full rationale.
  int_marker.pose = pose;
  int_marker.scale = 1.0;
  int_marker.name = gizmo_name_;
  int_marker.description = "Reachability plane (6-DOF)";

  // Semi-transparent body that tracks the plane geometry.
  visualization_msgs::msg::Marker body;
  body.type = visualization_msgs::msg::Marker::CUBE;
  body.pose.orientation.w = 1.0;
  body.scale.x = static_cast<float>(gizmo_body_size_x_);
  body.scale.y = static_cast<float>(gizmo_body_size_y_);
  body.scale.z = 0.004f;
  body.color.r = 0.75f;
  body.color.g = 0.75f;
  body.color.b = 0.75f;
  body.color.a = 0.35f;
  visualization_msgs::msg::InteractiveMarkerControl body_control;
  body_control.always_visible = true;
  body_control.markers.push_back(body);
  int_marker.controls.push_back(body_control);

  // 6-DOF controls (rotate/move per axis), same scheme as ArrowInteraction.
  const struct { float wx, wy, wz; const char * rotate; const char * move; } axes[] = {
    {1.0f, 0.0f, 0.0f, "rotate_x", "move_x"},
    {0.0f, 1.0f, 0.0f, "rotate_y", "move_y"},
    {0.0f, 0.0f, 1.0f, "rotate_z", "move_z"},
  };
  for (const auto & a : axes) {
    visualization_msgs::msg::InteractiveMarkerControl rotate;
    rotate.orientation.w = 1.0;
    rotate.orientation.x = a.wx;
    rotate.orientation.y = a.wy;
    rotate.orientation.z = a.wz;
    rotate.name = a.rotate;
    rotate.interaction_mode = visualization_msgs::msg::InteractiveMarkerControl::ROTATE_AXIS;
    int_marker.controls.push_back(rotate);

    visualization_msgs::msg::InteractiveMarkerControl move;
    move.orientation.w = 1.0;
    move.orientation.x = a.wx;
    move.orientation.y = a.wy;
    move.orientation.z = a.wz;
    move.name = a.move;
    move.interaction_mode = visualization_msgs::msg::InteractiveMarkerControl::MOVE_AXIS;
    int_marker.controls.push_back(move);
  }

  gizmo_server_->insert(
    int_marker,
    std::bind(
      &ReachabilityMapDisplay::gizmoFeedback, this, std::placeholders::_1));
}

void ReachabilityMapDisplay::gizmoFeedback(
  const visualization_msgs::msg::InteractiveMarkerFeedback::ConstSharedPtr & feedback)
{
  if (!feedback) {
    return;
  }
  if (feedback->event_type !=
      visualization_msgs::msg::InteractiveMarkerFeedback::POSE_UPDATE) {
    return;
  }
  std::lock_guard<std::mutex> lock(gizmo_mutex_);
  gizmo_feedback_pose_ = feedback->pose;
  gizmo_feedback_dirty_ = true;
}

void ReachabilityMapDisplay::syncGizmoToProperties()
{
  geometry_msgs::msg::Pose pose;
  {
    std::lock_guard<std::mutex> lock(gizmo_mutex_);
    if (!gizmo_feedback_dirty_) {
      return;
    }
    pose = gizmo_feedback_pose_;
    gizmo_feedback_dirty_ = false;
  }

  // Write the dragged pose back, quaternion straight through (no RPY round
  // trip). Suppress the property->gizmo echo in updatePlane().
  applying_gizmo_ = true;
  plane_position_x_->setFloat(pose.position.x);
  plane_position_y_->setFloat(pose.position.y);
  plane_position_z_->setFloat(pose.position.z);
  plane_orientation_w_->setFloat(pose.orientation.w);
  plane_orientation_x_->setFloat(pose.orientation.x);
  plane_orientation_y_->setFloat(pose.orientation.y);
  plane_orientation_z_->setFloat(pose.orientation.z);
  applying_gizmo_ = false;
}

void ReachabilityMapDisplay::syncPropertiesToGizmo()
{
  gizmo_sync_pending_ = false;
  if (!gizmo_server_) {
    return;
  }

  // A size change reshapes the body cube, which needs a re-insert.
  const double sx = plane_size_x_->getFloat();
  const double sy = plane_size_y_->getFloat();
  if (sx != gizmo_body_size_x_ || sy != gizmo_body_size_y_) {
    gizmo_server_->clear();
    makeGizmoMarker(planePoseFromProperties());
    gizmo_server_->applyChanges();
    return;
  }

  gizmo_server_->setPose(gizmo_name_, planePoseFromProperties());
  gizmo_server_->applyChanges();
}

// ============================================================================
// Embedded interactive-marker client (renders the plane gizmo in this display)
// ============================================================================

void ReachabilityMapDisplay::gizmoInitializeCallback(
  visualization_msgs::srv::GetInteractiveMarkers::Response::SharedPtr msg)
{
  eraseAllGizmoMarkers();
  if (msg) {
    updateGizmoMarkers(msg->markers);
  }
}

void ReachabilityMapDisplay::gizmoUpdateCallback(
  visualization_msgs::msg::InteractiveMarkerUpdate::ConstSharedPtr msg)
{
  if (!msg) {
    return;
  }
  updateGizmoMarkers(msg->markers);
  updateGizmoPoses(msg->poses);
  for (const std::string & marker_name : msg->erases) {
    interactive_markers_map_.erase(marker_name);
  }
}

void ReachabilityMapDisplay::gizmoResetCallback()
{
  eraseAllGizmoMarkers();
}

void ReachabilityMapDisplay::gizmoStatusCallback(
  interactive_markers::InteractiveMarkerClient::Status status,
  const std::string & message)
{
  rviz_common::properties::StatusProperty::Level level =
    rviz_common::properties::StatusProperty::Ok;
  switch (status) {
    case interactive_markers::InteractiveMarkerClient::STATUS_WARN:
      level = rviz_common::properties::StatusProperty::Warn;
      break;
    case interactive_markers::InteractiveMarkerClient::STATUS_ERROR:
      level = rviz_common::properties::StatusProperty::Error;
      break;
    default:
      break;
  }
  setStatusStd(level, "Gizmo Visible", message);
}

void ReachabilityMapDisplay::updateGizmoMarkers(
  const std::vector<visualization_msgs::msg::InteractiveMarker> & markers)
{
  for (const visualization_msgs::msg::InteractiveMarker & marker : markers) {
    if (!validateFloats(marker)) {
      setStatusStd(
        rviz_common::properties::StatusProperty::Error, marker.name,
        "Marker contains invalid floats!");
      continue;
    }

    auto int_marker_entry = interactive_markers_map_.find(marker.name);
    if (int_marker_entry == interactive_markers_map_.end()) {
      auto scene_marker =
        std::make_shared<rviz_default_plugins::displays::InteractiveMarker>(
          getSceneNode(), context_);
      // Drag feedback -> server (which writes the plane properties).
      connect(
        scene_marker.get(),
        SIGNAL(userFeedback(visualization_msgs::msg::InteractiveMarkerFeedback&)),
        this,
        SLOT(publishGizmoFeedback(visualization_msgs::msg::InteractiveMarkerFeedback&)));
      connect(
        scene_marker.get(),
        SIGNAL(statusUpdate(
          rviz_common::properties::StatusProperty::Level,
          const std::string&,
          const std::string&)),
        this,
        SLOT(gizmoStatusUpdate(
          rviz_common::properties::StatusProperty::Level,
          const std::string&,
          const std::string&)));
      int_marker_entry =
        interactive_markers_map_.emplace(marker.name, scene_marker).first;
    }

    if (int_marker_entry->second->processMessage(marker)) {
      int_marker_entry->second->setShowDescription(false);
      // The plane gizmo is rotated by its ROTATE_AXIS rings; the white XYZ
      // triad helper just adds clutter, so keep axis + visual-aid overlays off.
      int_marker_entry->second->setShowAxes(false);
      int_marker_entry->second->setShowVisualAids(false);
    } else {
      setStatusStd(
        rviz_common::properties::StatusProperty::Error, marker.name,
        "Failed to process interactive marker.");
    }
  }
}

void ReachabilityMapDisplay::updateGizmoPoses(
  const std::vector<visualization_msgs::msg::InteractiveMarkerPose> & poses)
{
  for (const visualization_msgs::msg::InteractiveMarkerPose & pose : poses) {
    auto int_marker_entry = interactive_markers_map_.find(pose.name);
    if (int_marker_entry != interactive_markers_map_.end()) {
      int_marker_entry->second->processMessage(pose);
    }
  }
}

void ReachabilityMapDisplay::eraseAllGizmoMarkers()
{
  interactive_markers_map_.clear();
}

void ReachabilityMapDisplay::publishGizmoFeedback(
  visualization_msgs::msg::InteractiveMarkerFeedback & feedback)
{
  if (gizmo_client_) {
    gizmo_client_->publishFeedback(feedback);
  }
}

void ReachabilityMapDisplay::gizmoStatusUpdate(
  rviz_common::properties::StatusProperty::Level level,
  const std::string & name,
  const std::string & text)
{
  setStatusStd(level, name, text);
}

void ReachabilityMapDisplay::updateGizmoVisible()
{
  const bool want = gizmo_visible_property_ && gizmo_visible_property_->getBool();
  if (want && !gizmo_active_) {
    createGizmo();
    gizmo_active_ = true;
  } else if (!want && gizmo_active_) {
    destroyGizmo();
    gizmo_active_ = false;
  }
}

// ============================================================================
// Per-frame update
// ============================================================================


}  // namespace curobov2_ros_rviz

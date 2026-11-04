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

std::string CuroboPlanningDisplay::markerName(const std::string & tool) const
{
  return "EE:" + gizmo_name_ + "_" + (tool.empty() ? "target" : tool);
}

void CuroboPlanningDisplay::createGizmo()
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
      node->get_logger(), "Target markers on /%s/* (%zu tool frame(s)).",
      gizmo_name_.c_str(), tool_order_.size());
  }
  rebuildGizmoMarkers();
  gizmo_server_->applyChanges();
  connectGizmoClient();
}

void CuroboPlanningDisplay::destroyGizmo()
{
  disconnectGizmoClient();
  if (gizmo_server_) {
    gizmo_server_->clear();
    gizmo_server_->applyChanges();
  }
  gizmo_server_.reset();
}

void CuroboPlanningDisplay::rebuildGizmoMarkers()
{
  if (!gizmo_server_) {
    return;
  }
  gizmo_server_->clear();
  for (const auto & frame : tool_order_) {
    insertGizmoMarker(frame);
  }
  gizmo_server_->applyChanges();
}

void CuroboPlanningDisplay::insertGizmoMarker(const std::string & tool)
{
  if (!gizmo_server_) {
    return;
  }
  auto & target = tools_[tool];
  visualization_msgs::msg::InteractiveMarker marker;
  marker.header.frame_id = gizmo_frame_;
  marker.pose = target.pose;
  marker.scale = gizmo_scale_property_ ? gizmo_scale_property_->getFloat() : 0.2;
  marker.name = markerName(tool);
  marker.description = tool.empty() ? "Target" : tool;

  static const float palette[][3] = {
    {0.10f, 0.75f, 0.75f},
    {0.90f, 0.55f, 0.10f},
    {0.55f, 0.35f, 0.90f},
    {0.25f, 0.75f, 0.25f},
    {0.90f, 0.25f, 0.45f},
    {0.70f, 0.70f, 0.20f},
  };
  size_t index = 0;
  for (size_t i = 0; i < tool_order_.size(); ++i) {
    if (tool_order_[i] == tool) {
      index = i;
      break;
    }
  }
  const auto & colour = palette[index % (sizeof(palette) / sizeof(palette[0]))];

  visualization_msgs::msg::Marker sphere;
  sphere.type = visualization_msgs::msg::Marker::SPHERE;
  sphere.pose.orientation.w = 1.0;
  sphere.scale.x = 0.025;
  sphere.scale.y = 0.025;
  sphere.scale.z = 0.025;
  sphere.color.r = colour[0];
  sphere.color.g = colour[1];
  sphere.color.b = colour[2];
  sphere.color.a = 0.85f;

  visualization_msgs::msg::Marker axis;
  axis.type = visualization_msgs::msg::Marker::ARROW;
  axis.pose.orientation.w = 1.0;
  axis.scale.x = 0.08;
  axis.scale.y = 0.015;
  axis.scale.z = 0.015;
  axis.color.r = colour[0];
  axis.color.g = colour[1];
  axis.color.b = colour[2];
  axis.color.a = 1.0f;

  visualization_msgs::msg::InteractiveMarkerControl body;
  body.always_visible = true;
  body.markers.push_back(sphere);
  body.markers.push_back(axis);
  marker.controls.push_back(body);

  static constexpr double kSqrt2Inv = 0.7071067811865476;
  auto add_axis_control = [&marker](double x, double y, double z, uint8_t mode) {
      visualization_msgs::msg::InteractiveMarkerControl control;
      control.orientation.w = kSqrt2Inv;
      control.orientation.x = x * kSqrt2Inv;
      control.orientation.y = y * kSqrt2Inv;
      control.orientation.z = z * kSqrt2Inv;
      control.interaction_mode = mode;
      marker.controls.push_back(control);
    };
  using Control = visualization_msgs::msg::InteractiveMarkerControl;
  add_axis_control(1, 0, 0, Control::ROTATE_AXIS);
  add_axis_control(0, 1, 0, Control::ROTATE_AXIS);
  add_axis_control(0, 0, 1, Control::ROTATE_AXIS);
  add_axis_control(1, 0, 0, Control::MOVE_AXIS);
  add_axis_control(0, 1, 0, Control::MOVE_AXIS);
  add_axis_control(0, 0, 1, Control::MOVE_AXIS);

  gizmo_server_->insert(
    marker,
    std::bind(&CuroboPlanningDisplay::gizmoFeedback, this, tool,
      std::placeholders::_1));
}

void CuroboPlanningDisplay::gizmoFeedback(
  const std::string & tool,
  const visualization_msgs::msg::InteractiveMarkerFeedback::ConstSharedPtr & feedback)
{
  if (!feedback ||
    feedback->event_type != visualization_msgs::msg::InteractiveMarkerFeedback::POSE_UPDATE)
  {
    return;
  }
  std::lock_guard<std::mutex> lock(gizmo_mutex_);
  feedback_poses_[tool] = feedback->pose;
  feedback_dirty_.insert(tool);
}

void CuroboPlanningDisplay::syncGizmoToProperties()
{
  std::map<std::string, geometry_msgs::msg::Pose> poses;
  {
    std::lock_guard<std::mutex> lock(gizmo_mutex_);
    if (feedback_dirty_.empty()) {
      return;
    }
    for (const auto & tool : feedback_dirty_) {
      auto it = feedback_poses_.find(tool);
      if (it != feedback_poses_.end()) {
        poses.emplace(tool, it->second);
      }
    }
    feedback_dirty_.clear();
  }

  std::string last_dragged;
  for (const auto & entry : poses) {
    ensureTarget(entry.first);
    tools_[entry.first].pose = entry.second;
    tools_[entry.first].tf_seeded = true;
    last_dragged = entry.first;
  }
  if (!poses.empty()) {
    goal_refresh_all_ = true;
  }
  if (!last_dragged.empty()) {
    if (!isToolEnabled(last_dragged)) {
      setToolEnabled(last_dragged, true);
    }
    active_tool_ = last_dragged;
  }
  updatePoseProperties(activePose());
}

void CuroboPlanningDisplay::syncPropertiesToGizmo()
{
  if (!gizmo_server_ || sync_pending_.empty()) {
    sync_pending_.clear();
    return;
  }
  for (const auto & tool : sync_pending_) {
    const std::string name = markerName(tool);
    gizmo_server_->setPose(name, tools_[tool].pose);
  }
  sync_pending_.clear();
  gizmo_server_->applyChanges();
}

void CuroboPlanningDisplay::connectGizmoClient()
{
  if (!gizmo_client_ || gizmo_client_connected_) {
    return;
  }
  gizmo_client_->connect(gizmo_name_);
  gizmo_client_connected_ = true;
}

void CuroboPlanningDisplay::disconnectGizmoClient()
{
  if (gizmo_client_) {
    gizmo_client_->disconnect();
  }
  gizmo_client_connected_ = false;
  eraseAllGizmoMarkers();
}

void CuroboPlanningDisplay::gizmoInitializeCallback(
  visualization_msgs::srv::GetInteractiveMarkers::Response::SharedPtr msg)
{
  eraseAllGizmoMarkers();
  if (msg) {
    updateGizmoMarkers(msg->markers);
  }
}

void CuroboPlanningDisplay::gizmoUpdateCallback(
  visualization_msgs::msg::InteractiveMarkerUpdate::ConstSharedPtr msg)
{
  if (!msg) {
    return;
  }
  updateGizmoMarkers(msg->markers);
  updateGizmoPoses(msg->poses);
  for (const std::string & name : msg->erases) {
    interactive_markers_map_.erase(name);
  }
}

void CuroboPlanningDisplay::gizmoResetCallback()
{
  eraseAllGizmoMarkers();
}

void CuroboPlanningDisplay::gizmoStatusCallback(
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
  setStatusStd(level, "Gizmo", message);
}

void CuroboPlanningDisplay::updateGizmoMarkers(
  const std::vector<visualization_msgs::msg::InteractiveMarker> & markers)
{
  for (const auto & marker : markers) {
    if (!validateFloats(marker)) {
      setStatusStd(
        rviz_common::properties::StatusProperty::Error, marker.name,
        "Marker contains invalid floats!");
      continue;
    }

    auto entry = interactive_markers_map_.find(marker.name);
    if (entry == interactive_markers_map_.end()) {
      auto scene_marker =
        std::make_shared<rviz_default_plugins::displays::InteractiveMarker>(
          getSceneNode(), context_);
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
      entry = interactive_markers_map_.emplace(marker.name, scene_marker).first;
    }

    if (entry->second->processMessage(marker)) {
      entry->second->setShowDescription(false);
      entry->second->setShowAxes(false);
      entry->second->setShowVisualAids(false);
    } else {
      setStatusStd(
        rviz_common::properties::StatusProperty::Error, marker.name,
        "Failed to process interactive marker.");
    }
  }
}

void CuroboPlanningDisplay::updateGizmoPoses(
  const std::vector<visualization_msgs::msg::InteractiveMarkerPose> & poses)
{
  for (const auto & pose : poses) {
    auto entry = interactive_markers_map_.find(pose.name);
    if (entry != interactive_markers_map_.end()) {
      entry->second->processMessage(pose);
    }
  }
}

void CuroboPlanningDisplay::eraseAllGizmoMarkers()
{
  interactive_markers_map_.clear();
}

void CuroboPlanningDisplay::publishGizmoFeedback(
  visualization_msgs::msg::InteractiveMarkerFeedback & feedback)
{
  if (gizmo_client_) {
    gizmo_client_->publishFeedback(feedback);
  }
}

void CuroboPlanningDisplay::gizmoStatusUpdate(
  rviz_common::properties::StatusProperty::Level level,
  const std::string & name,
  const std::string & text)
{
  setStatusStd(level, name, text);
}

void CuroboPlanningDisplay::updateGizmoVisible()
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

void CuroboPlanningDisplay::updateGizmoScale()
{
  if (gizmo_active_) {
    rebuildGizmoMarkers();
  }
}

// ============================================================================
// Playback panel
// ============================================================================


}  // namespace curobov2_ros_rviz

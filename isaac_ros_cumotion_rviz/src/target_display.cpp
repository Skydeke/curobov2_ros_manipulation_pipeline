#include <cmath>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/time.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/logging.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <rviz_common/validate_floats.hpp>

#include <isaac_ros_cumotion_rviz/target_display.hpp>

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

namespace isaac_ros_cumotion_rviz
{

// ============================================================================
// Construction / destruction
// ============================================================================

TargetDisplay::TargetDisplay()
  : target_position_x_(nullptr)
  , target_position_y_(nullptr)
  , target_position_z_(nullptr)
  , target_orientation_x_(nullptr)
  , target_orientation_y_(nullptr)
  , target_orientation_z_(nullptr)
  , target_orientation_w_(nullptr)
  , gizmo_visible_property_(nullptr)
{
  const int instance = gizmo_instance_counter_++;
  gizmo_name_ = (instance == 0) ? "target" :
    "target_" + std::to_string(instance);
}

std::atomic<int> TargetDisplay::gizmo_instance_counter_{0};

TargetDisplay::~TargetDisplay()
{
  destroyGizmo();
}

// ============================================================================
// Display lifecycle
// ============================================================================

void TargetDisplay::onInitialize()
{
  Display::onInitialize();

  // --- Target pose (MoveIt-style: position + quaternion X, Y, Z, W) ---
  target_position_x_ = new rviz_common::properties::FloatProperty(
    "Target Position X", 0.4, "Target X (m).", this, SLOT(updateTargetPose()));
  target_position_y_ = new rviz_common::properties::FloatProperty(
    "Target Position Y", 0.0, "Target Y (m).", this, SLOT(updateTargetPose()));
  target_position_z_ = new rviz_common::properties::FloatProperty(
    "Target Position Z", 0.4, "Target Z (m).", this, SLOT(updateTargetPose()));

  target_orientation_x_ = new rviz_common::properties::FloatProperty(
    "Target Orientation X", 0.0, "Target quaternion X.", this, SLOT(updateTargetPose()));
  target_orientation_y_ = new rviz_common::properties::FloatProperty(
    "Target Orientation Y", 0.0, "Target quaternion Y.", this, SLOT(updateTargetPose()));
  target_orientation_z_ = new rviz_common::properties::FloatProperty(
    "Target Orientation Z", 0.0, "Target quaternion Z.", this, SLOT(updateTargetPose()));
  target_orientation_w_ = new rviz_common::properties::FloatProperty(
    "Target Orientation W", 1.0, "Target quaternion W.", this, SLOT(updateTargetPose()));

  gizmo_visible_property_ = new rviz_common::properties::BoolProperty(
    "Gizmo Visible", true,
    "Interactive 6-DOF marker for dragging the target pose. Rendered in-place "
    "by this display (no separate 'Interactive Markers' display needed).", this,
    SLOT(updateGizmoVisible()));

  // --- Embedded interactive-marker client (self-contained gizmo rendering) ---
  auto ros_node_abstraction = context_->getRosNodeAbstraction().lock();
  if (ros_node_abstraction) {
    auto node = ros_node_abstraction->get_raw_node();
    auto transformer = context_->getFrameManager()->getTransformer();
    gizmo_client_ = std::make_unique<interactive_markers::InteractiveMarkerClient>(
      node, transformer, fixed_frame_.toStdString());
    gizmo_client_->setInitializeCallback(
      std::bind(
        &TargetDisplay::gizmoInitializeCallback, this,
        std::placeholders::_1));
    gizmo_client_->setUpdateCallback(
      std::bind(
        &TargetDisplay::gizmoUpdateCallback, this,
        std::placeholders::_1));
    gizmo_client_->setResetCallback(
      std::bind(&TargetDisplay::gizmoResetCallback, this));
    gizmo_client_->setStatusCallback(
      std::bind(&TargetDisplay::gizmoStatusCallback, this,
        std::placeholders::_1, std::placeholders::_2));
  }

  // Initial pose state.
  target_pose_ = targetPoseFromProperties();

  // The base Display only toggles scene-node visibility when the Enabled
  // checkbox changes, and the box starts unchecked, so a display loaded with
  // "Enabled: false" short-circuits in Display::load() and the content built
  // above would render anyway. Match the box now; later enable/disable goes
  // through onEnableChanged()/onEnable()/onDisable() like any other display.
  if (!isEnabled()) {
    scene_node_->setVisible(false);
  }
}

void TargetDisplay::reset()
{
  Display::reset();
  eraseAllGizmoMarkers();
}

void TargetDisplay::onEnable()
{
  Display::onEnable();
  if (gizmo_visible_property_ && gizmo_visible_property_->getBool() && !gizmo_active_) {
    createGizmo();
    gizmo_active_ = true;
  }
}

void TargetDisplay::onDisable()
{
  if (gizmo_active_) {
    destroyGizmo();
    gizmo_active_ = false;
  }
  Display::onDisable();
}

void TargetDisplay::fixedFrameChanged()
{
  if (gizmo_client_) {
    gizmo_client_->setTargetFrame(fixed_frame_.toStdString());
  }
  Display::fixedFrameChanged();
}

// ============================================================================
// Panel interface
// ============================================================================

geometry_msgs::msg::Pose TargetDisplay::getPose() const
{
  return target_pose_;
}

void TargetDisplay::setPose(const geometry_msgs::msg::Pose & pose)
{
  updatePoseProperties(pose);
}

// ============================================================================
// Target pose properties
// ============================================================================

geometry_msgs::msg::Pose TargetDisplay::targetPoseFromProperties() const
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = target_position_x_->getFloat();
  pose.position.y = target_position_y_->getFloat();
  pose.position.z = target_position_z_->getFloat();
  pose.orientation.x = target_orientation_x_->getFloat();
  pose.orientation.y = target_orientation_y_->getFloat();
  pose.orientation.z = target_orientation_z_->getFloat();
  pose.orientation.w = target_orientation_w_->getFloat();
  return pose;
}

void TargetDisplay::updateTargetPose()
{
  if (applying_gizmo_) {
    return;
  }
  target_pose_ = targetPoseFromProperties();
  gizmo_sync_pending_ = true;
}

void TargetDisplay::updatePoseProperties(const geometry_msgs::msg::Pose & pose)
{
  target_pose_ = pose;
  // Suppress the property->pose echo for a moment: the next syncPropertiesToGizmo
  // only needs to run once per change.
  applying_gizmo_ = true;
  target_position_x_->setFloat(pose.position.x);
  target_position_y_->setFloat(pose.position.y);
  target_position_z_->setFloat(pose.position.z);
  target_orientation_x_->setFloat(pose.orientation.x);
  target_orientation_y_->setFloat(pose.orientation.y);
  target_orientation_z_->setFloat(pose.orientation.z);
  target_orientation_w_->setFloat(pose.orientation.w);
  applying_gizmo_ = false;
  gizmo_sync_pending_ = true;
}

// ============================================================================
// Interactive target gizmo (self-contained MoveIt-style 6-DOF marker)
// ============================================================================

void TargetDisplay::createGizmo()
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
      "Target gizmo on /%s/* (rendered via embedded "
      "interactive-marker client).", gizmo_name_.c_str());
  }
  makeGizmoMarker(target_pose_);
  gizmo_server_->applyChanges();
  connectGizmoClient();
}

void TargetDisplay::destroyGizmo()
{
  disconnectGizmoClient();
  if (gizmo_server_) {
    gizmo_server_->clear();
    gizmo_server_->applyChanges();
  }
  gizmo_server_.reset();
}

void TargetDisplay::connectGizmoClient()
{
  if (!gizmo_client_ || gizmo_client_connected_) {
    return;
  }
  gizmo_client_->connect(gizmo_name_);
  gizmo_client_connected_ = true;
}

void TargetDisplay::disconnectGizmoClient()
{
  if (gizmo_client_) {
    gizmo_client_->disconnect();
  }
  gizmo_client_connected_ = false;
  eraseAllGizmoMarkers();
}

void TargetDisplay::makeGizmoMarker(const geometry_msgs::msg::Pose & pose)
{
  if (!gizmo_server_) {
    return;
  }

  visualization_msgs::msg::InteractiveMarker int_marker;
  int_marker.header.frame_id = gizmo_frame_;
  auto ros_node_abstraction = context_->getRosNodeAbstraction().lock();
  if (ros_node_abstraction) {
    int_marker.header.stamp =
      ros_node_abstraction->get_raw_node()->get_clock()->now();
  }
  int_marker.pose = pose;
  int_marker.scale = 1.0;
  int_marker.name = gizmo_name_;
  int_marker.description = "Target (6-DOF)";

  // Grey arrow pointing along +X (the tool axis), plus a centre sphere.
  visualization_msgs::msg::Marker arrow;
  arrow.type = visualization_msgs::msg::Marker::ARROW;
  arrow.pose.orientation.w = 1.0;
  arrow.scale.x = 0.12f;
  arrow.scale.y = 0.03f;
  arrow.scale.z = 0.03f;
  arrow.color.r = 0.0f;
  arrow.color.g = 0.55f;
  arrow.color.b = 0.55f;
  arrow.color.a = 1.0f;

  visualization_msgs::msg::Marker sphere;
  sphere.type = visualization_msgs::msg::Marker::SPHERE;
  sphere.pose.orientation.w = 1.0;
  sphere.scale.x = 0.05f;
  sphere.scale.y = 0.05f;
  sphere.scale.z = 0.05f;
  sphere.color.r = 0.9f;
  sphere.color.g = 0.9f;
  sphere.color.b = 0.9f;
  sphere.color.a = 1.0f;

  visualization_msgs::msg::InteractiveMarkerControl body_control;
  body_control.always_visible = true;
  body_control.markers.push_back(arrow);
  body_control.markers.push_back(sphere);
  int_marker.controls.push_back(body_control);

  // 6-DOF controls (rotate / move per axis), same scheme as the reachability
  // plane gizmo.
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
      &TargetDisplay::gizmoFeedback, this, std::placeholders::_1));
}

void TargetDisplay::gizmoFeedback(
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

void TargetDisplay::syncGizmoToProperties()
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
  // trip). Suppress the property->gizmo echo in updatePoseProperties().
  applying_gizmo_ = true;
  target_position_x_->setFloat(pose.position.x);
  target_position_y_->setFloat(pose.position.y);
  target_position_z_->setFloat(pose.position.z);
  target_orientation_x_->setFloat(pose.orientation.x);
  target_orientation_y_->setFloat(pose.orientation.y);
  target_orientation_z_->setFloat(pose.orientation.z);
  target_orientation_w_->setFloat(pose.orientation.w);
  applying_gizmo_ = false;
  target_pose_ = pose;
}

void TargetDisplay::syncPropertiesToGizmo()
{
  gizmo_sync_pending_ = false;
  if (!gizmo_server_) {
    return;
  }
  gizmo_server_->setPose(gizmo_name_, targetPoseFromProperties());
  gizmo_server_->applyChanges();
}

// ============================================================================
// Embedded interactive-marker client (renders the gizmo in this display)
// ============================================================================

void TargetDisplay::gizmoInitializeCallback(
  visualization_msgs::srv::GetInteractiveMarkers::Response::SharedPtr msg)
{
  eraseAllGizmoMarkers();
  if (msg) {
    updateGizmoMarkers(msg->markers);
  }
}

void TargetDisplay::gizmoUpdateCallback(
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

void TargetDisplay::gizmoResetCallback()
{
  eraseAllGizmoMarkers();
}

void TargetDisplay::gizmoStatusCallback(
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

void TargetDisplay::updateGizmoMarkers(
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
      // Drag feedback -> server (which writes the target pose).
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
      // The axis triad helper just adds clutter to a 6-DOF target gizmo.
      int_marker_entry->second->setShowAxes(false);
      int_marker_entry->second->setShowVisualAids(false);
    } else {
      setStatusStd(
        rviz_common::properties::StatusProperty::Error, marker.name,
        "Failed to process interactive marker.");
    }
  }
}

void TargetDisplay::updateGizmoPoses(
  const std::vector<visualization_msgs::msg::InteractiveMarkerPose> & poses)
{
  for (const visualization_msgs::msg::InteractiveMarkerPose & pose : poses) {
    auto int_marker_entry = interactive_markers_map_.find(pose.name);
    if (int_marker_entry != interactive_markers_map_.end()) {
      int_marker_entry->second->processMessage(pose);
    }
  }
}

void TargetDisplay::eraseAllGizmoMarkers()
{
  interactive_markers_map_.clear();
}

void TargetDisplay::publishGizmoFeedback(
  visualization_msgs::msg::InteractiveMarkerFeedback & feedback)
{
  if (gizmo_client_) {
    gizmo_client_->publishFeedback(feedback);
  }
}

void TargetDisplay::gizmoStatusUpdate(
  rviz_common::properties::StatusProperty::Level level,
  const std::string & name,
  const std::string & text)
{
  setStatusStd(level, name, text);
}

void TargetDisplay::updateGizmoVisible()
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

void TargetDisplay::update(float wall_dt, float ros_dt)
{
  Display::update(wall_dt, ros_dt);

  // Drive the embedded interactive-marker client (service call, pending
  // updates) and its rendered scene objects.
  if (gizmo_client_) {
    gizmo_client_->update();
  }
  for (const auto & name_marker_pair : interactive_markers_map_) {
    name_marker_pair.second->update();
  }

  // Interactive gizmo: drag feedback -> properties, property edits -> marker.
  if (gizmo_active_) {
    if (gizmo_feedback_dirty_) {
      syncGizmoToProperties();
    }
    if (gizmo_sync_pending_) {
      syncPropertiesToGizmo();
    }
  }

  // "Gizmo Visible" is a pure visibility toggle: only create/destroy on the
  // actual transition, never every frame.
  const bool gizmo_want = gizmo_visible_property_ && gizmo_visible_property_->getBool();
  if (gizmo_want && !gizmo_active_) {
    createGizmo();
    gizmo_active_ = true;
  } else if (!gizmo_want && gizmo_active_) {
    destroyGizmo();
    gizmo_active_ = false;
  }

  // The fixed frame may only be set after the display loaded; rebind the
  // gizmo so it lives in the real frame rather than a startup fallback.
  if (gizmo_active_ && gizmo_server_) {
    const std::string fixed = context_->getFrameManager()->getFixedFrame();
    if (!fixed.empty() && fixed != gizmo_frame_) {
      gizmo_server_->clear();
      gizmo_server_->applyChanges();
      gizmo_server_.reset();
      gizmo_frame_.clear();
    }
  }
  if (gizmo_active_ && !gizmo_server_) {
    createGizmo();
  }
}

}  // namespace isaac_ros_cumotion_rviz

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(isaac_ros_cumotion_rviz::TargetDisplay, rviz_common::Display)
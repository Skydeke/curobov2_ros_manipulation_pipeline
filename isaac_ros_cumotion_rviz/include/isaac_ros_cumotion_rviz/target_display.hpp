#pragma once

#ifndef ISAAC_ROS_CUMOTION_RVIZ__TARGET_DISPLAY_HPP_
#define ISAAC_ROS_CUMOTION_RVIZ__TARGET_DISPLAY_HPP_

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <rviz_common/display.hpp>
#include <rviz_common/properties/bool_property.hpp>
#include <rviz_common/properties/float_property.hpp>

#include <geometry_msgs/msg/pose.hpp>
#include <interactive_markers/interactive_marker_server.hpp>
#include <visualization_msgs/msg/interactive_marker.hpp>
#include <visualization_msgs/msg/interactive_marker_control.hpp>
#include <visualization_msgs/msg/interactive_marker_feedback.hpp>
#include <visualization_msgs/msg/interactive_marker_pose.hpp>
#include <visualization_msgs/msg/interactive_marker_update.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/srv/get_interactive_markers.hpp>

#ifndef Q_MOC_RUN
#include <interactive_markers/interactive_marker_client.hpp>
#include <rviz_default_plugins/displays/interactive_markers/interactive_marker.hpp>
#endif

namespace isaac_ros_cumotion_rviz
{

/**
 * RViz Display showing a generic draggable 6-DOF target pose.
 *
 * This is the planner-agnostic replacement for the legacy MPCTargetDisplay: it
 * only renders and edits a target pose (position + quaternion) via an
 * in-place interactive 6-DOF gizmo and exposes it through getPose()/setPose().
 * It owns NO planner clients, services, actions or goal streams -- deciding
 * what the planner does with the pose (plan, execute, MPC track ...) is the
 * job of the RvizArgsPanel.
 *
 * Like ReachabilityMapDisplay it embeds BOTH halves of the interactive-marker
 * stack:
 *
 *  * an InteractiveMarkerServer publishing the target marker on the "target"
 *    namespace,
 *  * an embedded InteractiveMarkerClient (the same classes RViz's "Interactive
 *    Markers" display uses) that subscribes to /update, renders the marker as
 *    selectable Ogre objects and publishes drag feedback back to the server.
 *
 * No separate "Interactive Markers" display is needed in the RViz config.
 * Renders in the fixed frame; set the RViz fixed frame to the robot base frame.
 */
class TargetDisplay : public rviz_common::Display
{
  Q_OBJECT
public:
  TargetDisplay();
  ~TargetDisplay() override;

  // --- Panel interface (used by RvizArgsPanel) ---
  geometry_msgs::msg::Pose getPose() const;
  void setPose(const geometry_msgs::msg::Pose & pose);

protected:
  void onInitialize() override;
  void onEnable() override;
  void onDisable() override;
  void update(float wall_dt, float ros_dt) override;
  void reset() override;
  void fixedFrameChanged() override;

private Q_SLOTS:
  void updateTargetPose();
  void updateGizmoVisible();
  void publishGizmoFeedback(visualization_msgs::msg::InteractiveMarkerFeedback & feedback);
  void gizmoStatusUpdate(
    rviz_common::properties::StatusProperty::Level level,
    const std::string & name,
    const std::string & text);

private:
  void updatePoseProperties(const geometry_msgs::msg::Pose & pose);
  geometry_msgs::msg::Pose targetPoseFromProperties() const;

  // --- Interactive target gizmo (self-contained, ReachabilityMapDisplay-style) ---
  void createGizmo();
  void destroyGizmo();
  void makeGizmoMarker(const geometry_msgs::msg::Pose & pose);
  void gizmoFeedback(
    const visualization_msgs::msg::InteractiveMarkerFeedback::ConstSharedPtr & feedback);
  void syncGizmoToProperties();
  void syncPropertiesToGizmo();
  void connectGizmoClient();
  void disconnectGizmoClient();

  // Embedded interactive-marker client callbacks (mirror of
  // rviz_default_plugins::displays::InteractiveMarkerDisplay).
  void gizmoInitializeCallback(
    visualization_msgs::srv::GetInteractiveMarkers::Response::SharedPtr msg);
  void gizmoUpdateCallback(
    visualization_msgs::msg::InteractiveMarkerUpdate::ConstSharedPtr msg);
  void gizmoResetCallback();
  void gizmoStatusCallback(
    interactive_markers::InteractiveMarkerClient::Status status,
    const std::string & message);
  void updateGizmoMarkers(
    const std::vector<visualization_msgs::msg::InteractiveMarker> & markers);
  void updateGizmoPoses(
    const std::vector<visualization_msgs::msg::InteractiveMarkerPose> & poses);
  void eraseAllGizmoMarkers();

  std::shared_ptr<interactive_markers::InteractiveMarkerServer> gizmo_server_;
  std::unique_ptr<interactive_markers::InteractiveMarkerClient> gizmo_client_;
  bool gizmo_client_connected_ = false;
  std::map<std::string, rviz_default_plugins::displays::InteractiveMarker::SharedPtr>
    interactive_markers_map_;
  static std::atomic<int> gizmo_instance_counter_;
  std::string gizmo_name_;
  std::string gizmo_frame_;
  std::mutex gizmo_mutex_;
  geometry_msgs::msg::Pose gizmo_feedback_pose_;
  bool gizmo_feedback_dirty_ = false;
  bool gizmo_sync_pending_ = false;
  bool applying_gizmo_ = false;
  bool gizmo_active_ = false;

  // Current authoritative target pose.
  geometry_msgs::msg::Pose target_pose_;

  // --- Properties ---
  rviz_common::properties::FloatProperty * target_position_x_;
  rviz_common::properties::FloatProperty * target_position_y_;
  rviz_common::properties::FloatProperty * target_position_z_;
  rviz_common::properties::FloatProperty * target_orientation_x_;
  rviz_common::properties::FloatProperty * target_orientation_y_;
  rviz_common::properties::FloatProperty * target_orientation_z_;
  rviz_common::properties::FloatProperty * target_orientation_w_;
  rviz_common::properties::BoolProperty * gizmo_visible_property_;
};

}  // namespace isaac_ros_cumotion_rviz

#endif  // ISAAC_ROS_CUMOTION_RVIZ__TARGET_DISPLAY_HPP_
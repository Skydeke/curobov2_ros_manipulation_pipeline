#pragma once

#ifndef ISAAC_ROS_CUMOTION_RVIZ__MPC_TARGET_DISPLAY_HPP_
#define ISAAC_ROS_CUMOTION_RVIZ__MPC_TARGET_DISPLAY_HPP_

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rviz_common/display.hpp>
#include <rviz_common/properties/bool_property.hpp>
#include <rviz_common/properties/float_property.hpp>
#include <rviz_common/properties/string_property.hpp>

#include <geometry_msgs/msg/pose.hpp>
#include <interactive_markers/interactive_marker_server.hpp>
#include <visualization_msgs/msg/interactive_marker.hpp>
#include <visualization_msgs/msg/interactive_marker_control.hpp>
#include <visualization_msgs/msg/interactive_marker_feedback.hpp>
#include <visualization_msgs/msg/interactive_marker_pose.hpp>
#include <visualization_msgs/msg/interactive_marker_update.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/srv/get_interactive_markers.hpp>

#include <isaac_ros_cumotion_interfaces/action/send_trajectory.hpp>
#include <isaac_ros_cumotion_interfaces/srv/set_planner.hpp>

#ifndef Q_MOC_RUN
#include <interactive_markers/interactive_marker_client.hpp>
#include <rviz_default_plugins/displays/interactive_markers/interactive_marker.hpp>
#endif

namespace isaac_ros_cumotion_rviz
{

/**
 * RViz Display that drives the cuRoBo MPC (reactive) controller from a
 * draggable 6-DOF target.
 *
 * This is the self-contained replacement for the legacy ArrowInteractionDisplay
 * (which published an interactive marker that only rendered if the user also
 * added an external "Interactive Markers" display). Like ReachabilityMapDisplay
 * it embeds BOTH halves of the interactive-marker stack:
 *
 *  * an InteractiveMarkerServer publishing the target marker on the
 *    "mpc_target" namespace,
 *  * an embedded InteractiveMarkerClient (the same classes RViz's "Interactive
 *    Markers" display uses) that subscribes to /update, renders the marker as
 *    selectable Ogre objects and publishes drag feedback back to the server.
 *
 * No separate display is needed in the RViz configuration.
 *
 * The display owns the whole MPC interaction, mirroring the viser
 * `mpc_viser_node` flow:
 *
 *  - "Start MPC" calls `/<planner>/set_planner` (MPC=1) and then sends the
 *    `/<planner>/execute_trajectory` action with the target as the goalset
 *    (start_pose deliberately empty, so the server resolves the start state
 *    from the live robot pose -- raw /joint_states carries extra joints that
 *    cuRobo's MPC rejects),
 *  - while active the display streams the current target pose to
 *    `/<planner>/mpc_goal` as the gizmo is dragged (10 Hz),
 *  - "Stop MPC" (or disabling the display) cancels the action goal.
 *
 * The planner node name is configurable and defaults to "curobo_server" (the
 * kortex deployment's name). Renders in the fixed frame; set the RViz fixed
 * frame to the robot base frame.
 */
class MPCTargetDisplay : public rviz_common::Display
{
  Q_OBJECT
public:
  MPCTargetDisplay();
  ~MPCTargetDisplay() override;

  // --- Panel interface (used by RvizArgsPanel) ---
  geometry_msgs::msg::Pose getPose() const;
  void setPose(const geometry_msgs::msg::Pose & pose);
  void startMpc();
  void stopMpc();

protected:
  void onInitialize() override;
  void onEnable() override;
  void onDisable() override;
  void update(float wall_dt, float ros_dt) override;
  void reset() override;
  void fixedFrameChanged() override;

private Q_SLOTS:
  void updateServiceName();
  void updateTargetPose();
  void updateGizmoVisible();
  void updateStart();
  void updateStop();
  void publishGizmoFeedback(visualization_msgs::msg::InteractiveMarkerFeedback & feedback);
  void gizmoStatusUpdate(
    rviz_common::properties::StatusProperty::Level level,
    const std::string & name,
    const std::string & text);

private:
  using SendTrajectory = isaac_ros_cumotion_interfaces::action::SendTrajectory;
  using SetPlanner = isaac_ros_cumotion_interfaces::srv::SetPlanner;
  using GoalHandle = rclcpp_action::ClientGoalHandle<SendTrajectory>;

  // --- ROS plumbing (lazily created against the planner node name) ---
  void ensureClients();
  void updatePoseProperties(const geometry_msgs::msg::Pose & pose);
  geometry_msgs::msg::Pose targetPoseFromProperties() const;

  // --- MPC start / stop ---
  void onPlannerSwitch(SetPlanner::Response::SharedPtr response);
  void sendMpcGoal();
  void publishGoalPose();
  void onGoalResponse(GoalHandle::SharedPtr goal_handle);
  void onFeedback(GoalHandle::SharedPtr handle, const std::shared_ptr<const SendTrajectory::Feedback> feedback);
  void onResult(const GoalHandle::WrappedResult & result);

  rclcpp::Client<SetPlanner>::SharedPtr set_planner_client_;
  rclcpp_action::Client<SendTrajectory>::SharedPtr action_client_;
  rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr mpc_goal_pub_;
  std::string clients_ns_;
  GoalHandle::SharedPtr goal_handle_;
  bool mpc_active_ = false;
  bool mpc_starting_ = false;
  bool goal_publish_pending_ = false;
  double goal_publish_accum_ = 0.0;
  geometry_msgs::msg::Pose last_published_goal_;

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
  rviz_common::properties::StringProperty * service_name_property_;
  rviz_common::properties::FloatProperty * target_position_x_;
  rviz_common::properties::FloatProperty * target_position_y_;
  rviz_common::properties::FloatProperty * target_position_z_;
  rviz_common::properties::FloatProperty * target_orientation_x_;
  rviz_common::properties::FloatProperty * target_orientation_y_;
  rviz_common::properties::FloatProperty * target_orientation_z_;
  rviz_common::properties::FloatProperty * target_orientation_w_;
  rviz_common::properties::BoolProperty * gizmo_visible_property_;
  rviz_common::properties::BoolProperty * start_mpc_property_;
  rviz_common::properties::BoolProperty * stop_mpc_property_;
};

}  // namespace isaac_ros_cumotion_rviz

#endif  // ISAAC_ROS_CUMOTION_RVIZ__MPC_TARGET_DISPLAY_HPP_
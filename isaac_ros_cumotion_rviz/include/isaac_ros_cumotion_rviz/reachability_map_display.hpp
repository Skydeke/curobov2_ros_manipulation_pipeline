#pragma once

#ifndef ISAAC_ROS_CUMOTION_RVIZ__REACHABILITY_MAP_DISPLAY_HPP_
#define ISAAC_ROS_CUMOTION_RVIZ__REACHABILITY_MAP_DISPLAY_HPP_

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rviz_common/display.hpp>
#include <urdf_model/model.h>
#include <urdf_parser/urdf_parser.h>
#include <rviz_common/properties/bool_property.hpp>
#include <rviz_common/properties/color_property.hpp>
#include <rviz_common/properties/enum_property.hpp>
#include <rviz_common/properties/float_property.hpp>
#include <rviz_common/properties/int_property.hpp>
#include <rviz_common/properties/string_property.hpp>
#include <rviz_default_plugins/robot/robot.hpp>

#include <rviz_rendering/objects/arrow.hpp>
#include <rviz_rendering/objects/point_cloud.hpp>
#include <rviz_rendering/objects/shape.hpp>

#include <std_msgs/msg/string.hpp>

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

#include <isaac_ros_cumotion_interfaces/msg/reachability_metrics.hpp>
#include <isaac_ros_cumotion_interfaces/srv/generate_rm.hpp>

#include <isaac_ros_cumotion_rviz/curobo_fk.hpp>

namespace Ogre
{
class SceneNode;
}  // namespace Ogre

namespace isaac_ros_cumotion_rviz
{

/**
 * RViz Display for the curobo_server reachability map.
 *
 * Reaches into the ``/curobo_server/generate_rm`` service with a user-editable
 * plane (position, quaternion orientation, size) and grid resolution, and
 * renders the result:
 *
 *  - a thin plane cube showing the solved grid's position/orientation/extent,
 *  - one coloured sphere per grid cell (green = IK converged, red = not),
 *  - optionally, full-robot ghosts of the solved configurations, either at
 *    every solved cell or at a single user-selected cell (reusing the same
 *    URDF FK / LinkUpdater machinery as CuroboTrajectoryDisplay).
 *
 * The URDF is always read from the latched `/robot_description` topic. No
 * MoveIt dependency.
 *
 * Renders in the fixed frame directly: the goals are expressed in the robot's
 * base frame by the server, so set the RViz fixed frame to the robot base
 * (the default setup in kortex_sim.rviz).
 */
class ReachabilityMapDisplay : public rviz_common::Display
{
  Q_OBJECT
public:
  ReachabilityMapDisplay();
  ~ReachabilityMapDisplay() override;

protected:
  void onInitialize() override;
  void onEnable() override;
  void onDisable() override;
  void update(float wall_dt, float ros_dt) override;
  void reset() override;
  void fixedFrameChanged() override;

private Q_SLOTS:
  void updateServiceName();
  void updatePlane();
  void updateAutoRefresh();
  void updateRefresh();
  void updateCellStyle();
  void updateCellColors();
  void updateShowSolutions();
  void updateSelectedCell();
  void updateSolutionAlpha();
  void updateGizmoVisible();
  void publishGizmoFeedback(visualization_msgs::msg::InteractiveMarkerFeedback & feedback);
  void gizmoStatusUpdate(
    rviz_common::properties::StatusProperty::Level level,
    const std::string & name,
    const std::string & text);

private:
  using GenerateRM = isaac_ros_cumotion_interfaces::srv::GenerateRM;
  using ReachabilityMetrics = isaac_ros_cumotion_interfaces::msg::ReachabilityMetrics;

  // --- URDF / robot model (always from `/robot_description`) ---
  void loadURDF();
  void loadRobotModel();
  bool urdf_loaded_ = false;
  bool robot_loaded_ = false;
  std::string urdf_xml_;
  urdf::ModelInterfaceSharedPtr urdf_model_;
  CuroboFK fk_engine_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr robot_description_sub_;

  // --- Solve pumping ---
  void queueSolve();
  void drainSolve();
  void onSolveResponse(rclcpp::Client<GenerateRM>::SharedFuture future);
  void promoteMetrics();
  rclcpp::Client<GenerateRM>::SharedPtr client_;
  std::atomic<bool> solve_pending_{true};
  std::atomic<bool> solve_inflight_{false};

  // Plane changes are debounced (like the legacy viser reachability control):
  // re-solving only starts once the user lets go for kReachabilitySettleTime,
  // so dragging a slider never thrashes the GPU with a solve per tick.
  static constexpr float kReachabilitySettleTime = 0.5f;
  bool plane_dirty_ = false;
  float settle_timer_ = 0.0f;

  // --- Metrics state (cross-thread: service response vs render thread) ---
  std::mutex metrics_mutex_;
  std::shared_ptr<ReachabilityMetrics> metrics_msg_new_;
  std::shared_ptr<ReachabilityMetrics> metrics_active_;
  std::string last_solve_message_;
  bool last_solve_ok_ = false;

  // --- Visualization ---
  void rebuildVisualization();
  void updatePlaneShape();
  void updateMapPoints();
  void updateSolutionRobots();
  void clearSolutionRobots();
  // Destroy the ghost at ``index`` (hide, delete the Robot, prune its scene
  // node subtree). Shared by clearSolutionRobots() and the pool-shrink path.
  void destroySolutionRobot(size_t index);
  void updateSolutionAlphaInternal();
  std::map<std::string, double> jointMapForCell(size_t cell_index) const;
  // FK the cell's solved joint config and return the link pose that lands at
  // the requested goal. Returns false when the FK is unavailable or nothing
  // lands near the goal (bad joint-name match / unsolved cell).
  bool solvedLinkPose(
    size_t cell_index,
    const geometry_msgs::msg::Pose & goal,
    Eigen::Isometry3d & pose_out) const;

  Ogre::Quaternion currentPlaneOrientation() const;

  // --- Interactive plane control (self-contained MoveIt-style 6-DOF marker) ---
  // Both halves of the interactive-marker stack live in this display:
  //  * an InteractiveMarkerServer publishes the plane marker on the
  //    "reachability_plane" namespace (/reachability_plane/update ...).
  //  * an embedded InteractiveMarkerClient (the same client classes RViz's
  //    "Interactive Markers" display uses) subscribes to /update, renders the
  //    marker as selectable Ogre objects and publishes drag feedback back to
  //    the server, which writes back into the plane properties.
  // No separate display is needed in the RViz configuration.
  geometry_msgs::msg::Pose planePoseFromProperties() const;
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
  // Per-display unique namespace so several ReachabilityMapDisplays can live
  // in the same RViz without competing for the same /update / feedback topics.
  // The first instance keeps "reachability_plane"; later ones get a numeric
  // suffix ("reachability_plane_1", ...).
  static std::atomic<int> gizmo_instance_counter_;
  std::string gizmo_name_;
  std::string gizmo_frame_;
  std::mutex gizmo_mutex_;
  geometry_msgs::msg::Pose gizmo_feedback_pose_;
  bool gizmo_feedback_dirty_ = false;
  bool gizmo_sync_pending_ = false;
  bool applying_gizmo_ = false;
  // Whether the gizmo (server + client + rendered marker) is currently up;
  // the "Plane Control Marker" property is a pure visibility toggle on it.
  bool gizmo_active_ = false;
  double gizmo_body_size_x_ = 0.0;
  double gizmo_body_size_y_ = 0.0;

  std::unique_ptr<rviz_rendering::Shape> plane_shape_;
  std::unique_ptr<rviz_rendering::PointCloud> cloud_;
  std::vector<std::unique_ptr<rviz_rendering::Arrow>> cell_arrows_;

  // One ghost robot per solved cell, each under its own scene node so turning
  // "Show Solutions" off removes every Ogre object immediately, independent of
  // Robot's own teardown.
  struct SolutionRobot
  {
    Ogre::SceneNode * node = nullptr;
    std::unique_ptr<rviz_default_plugins::robot::Robot> robot;
  };
  std::vector<SolutionRobot> solution_robots_;
  static constexpr size_t kMaxSolutionRobots = 120;
  // Sentinel for the "Arrows" cell style (must not collide with the
  // rviz_rendering::PointCloud::RenderMode enum values).
  static constexpr int kCellStyleArrows = 100;

  // --- Properties ---
  rviz_common::properties::StringProperty * service_name_property_;
  rviz_common::properties::FloatProperty * plane_position_x_;
  rviz_common::properties::FloatProperty * plane_position_y_;
  rviz_common::properties::FloatProperty * plane_position_z_;
  rviz_common::properties::FloatProperty * plane_orientation_x_;
  rviz_common::properties::FloatProperty * plane_orientation_y_;
  rviz_common::properties::FloatProperty * plane_orientation_z_;
  rviz_common::properties::FloatProperty * plane_orientation_w_;
  rviz_common::properties::FloatProperty * plane_size_x_;
  rviz_common::properties::FloatProperty * plane_size_y_;
  rviz_common::properties::ColorProperty * plane_color_property_;
  rviz_common::properties::FloatProperty * plane_alpha_property_;
  rviz_common::properties::BoolProperty * gizmo_visible_property_;
  rviz_common::properties::IntProperty * grid_cells_x_;
  rviz_common::properties::IntProperty * grid_cells_y_;
  rviz_common::properties::BoolProperty * auto_refresh_property_;
  rviz_common::properties::BoolProperty * refresh_property_;
  rviz_common::properties::EnumProperty * cell_style_property_;
  rviz_common::properties::ColorProperty * solved_color_property_;
  rviz_common::properties::ColorProperty * failed_color_property_;
  rviz_common::properties::FloatProperty * cell_alpha_property_;
  rviz_common::properties::EnumProperty * show_solutions_property_;
  rviz_common::properties::IntProperty * selected_cell_property_;
  rviz_common::properties::FloatProperty * solution_alpha_property_;
};

}  // namespace isaac_ros_cumotion_rviz

#endif  // ISAAC_ROS_CUMOTION_RVIZ__REACHABILITY_MAP_DISPLAY_HPP_
#pragma once

#ifndef CUROBOV2_ROS_RVIZ__CUROBO_PLANNING_DISPLAY_HPP_
#define CUROBOV2_ROS_RVIZ__CUROBO_PLANNING_DISPLAY_HPP_

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <OgreQuaternion.h>
#include <OgreVector3.h>

#include <rclcpp/rclcpp.hpp>
#include <rviz_common/display.hpp>
#include <rviz_common/properties/bool_property.hpp>
#include <rviz_common/properties/editable_enum_property.hpp>
#include <rviz_common/properties/float_property.hpp>
#include <rviz_common/properties/int_property.hpp>
#include <rviz_common/properties/ros_topic_property.hpp>
#include <rviz_common/properties/string_property.hpp>
#include <rviz_rendering/objects/shape.hpp>
#include <rviz_default_plugins/robot/robot.hpp>
#include <urdf_model/model.h>
#include <urdf_parser/urdf_parser.h>

#include <geometry_msgs/msg/pose.hpp>
#include <interactive_markers/interactive_marker_server.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/string.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <visualization_msgs/msg/interactive_marker.hpp>
#include <visualization_msgs/msg/interactive_marker_control.hpp>
#include <visualization_msgs/msg/interactive_marker_feedback.hpp>
#include <visualization_msgs/msg/interactive_marker_pose.hpp>
#include <visualization_msgs/msg/interactive_marker_update.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/srv/get_interactive_markers.hpp>

#ifndef Q_MOC_RUN
#include <interactive_markers/interactive_marker_client.hpp>
#include <rviz_default_plugins/displays/interactive_markers/interactive_marker.hpp>
#endif

#include <curobov2_ros_interfaces/srv/fk_batch.hpp>
#include <curobov2_ros_interfaces/srv/ik_batch.hpp>
#include <curobov2_ros_rviz/trajectory_panel.hpp>

namespace Ogre
{
class SceneNode;
}  // namespace Ogre

namespace rviz_common
{
class PanelDockWidget;
}  // namespace rviz_common

namespace curobov2_ros_rviz
{

/**
 * MoveIt MotionPlanning-style display for cuRobo.
 *
 * One display owns:
 *  - the live robot model (driven by TF),
 *  - the planned trajectory ghost with trail and attached-object spheres,
 *  - one draggable 6-DOF target per tool frame,
 *  - the docked trajectory scrubber.
 */
class CuroboPlanningDisplay : public rviz_common::Display
{
  Q_OBJECT
public:
  CuroboPlanningDisplay();
  ~CuroboPlanningDisplay() override;

  // --- Target interface (used by PlanningTab) ---
  geometry_msgs::msg::Pose getPose() const;
  void setPose(const geometry_msgs::msg::Pose & pose);
  geometry_msgs::msg::Pose getPose(const std::string & tool) const;
  void setPose(const std::string & tool, const geometry_msgs::msg::Pose & pose);
  std::string getGizmoFrame() const;
  bool isDefaultPose() const;
  std::string getActiveTool() const;
  void setActiveTool(const std::string & tool);
  std::vector<std::string> getToolFrames() const;
  void setToolFrames(const std::vector<std::string> & frames);
  bool toolFramesUserSet() const {return tool_frames_user_set_;}
  std::vector<std::pair<std::string, geometry_msgs::msg::Pose>> getToolPoses() const;

protected:
  void onInitialize() override;
  void onEnable() override;
  void onDisable() override;
  void update(float wall_dt, float ros_dt) override;
  void reset() override;
  void fixedFrameChanged() override;

private Q_SLOTS:
  void updateTopic();
  void updatePlannerNode();
  void updateAlpha();
  void updateShowTrail();
  void updateTrailStepSize();
  void updateLoop();
  void updateSpeed();
  void updateCollisionTopic();
  void updateShowAttached();
  void updateShowCollisionSpheres();
  void updateSceneRobotVisibility();
  void updateSceneRobotAlpha();
  void updateGoalStateVisibility();
  void updateGoalStateAlpha();
  void updateTargetPose();
  void updateGizmoVisible();
  void updateGizmoScale();
  void updateToolFrames();
  void updateActiveTool();
  void publishGizmoFeedback(visualization_msgs::msg::InteractiveMarkerFeedback & feedback);
  void gizmoStatusUpdate(
    rviz_common::properties::StatusProperty::Level level,
    const std::string & name,
    const std::string & text);

private:
  // --- URDF / robots ---
  bool loadURDF();
  void loadRobotModel();
  void updateSceneRobot();
  void updateTrajectoryRoot();
  void updateTrailVisibility();
  void rebuildTrail();
  void clearTrailRobots();

  std::unique_ptr<rviz_default_plugins::robot::Robot> scene_robot_;
  std::unique_ptr<rviz_default_plugins::robot::Robot> goal_robot_;
  std::unique_ptr<rviz_default_plugins::robot::Robot> trajectory_robot_;
  urdf::ModelInterfaceSharedPtr urdf_model_;
  bool robot_loaded_ = false;
  std::string urdf_xml_;
  Ogre::SceneNode * scene_robot_root_ = nullptr;
  Ogre::SceneNode * goal_root_ = nullptr;
  Ogre::SceneNode * trajectory_root_ = nullptr;
  std::string robot_root_frame_;
  // Frame of the server's FK poses (cuRobo base link). Empty falls back to
  // the URDF root for older servers.
  std::string fk_base_link_;
  Ogre::Vector3 robot_root_offset_ = Ogre::Vector3::ZERO;
  Ogre::Quaternion robot_root_rotation_ = Ogre::Quaternion::IDENTITY;

  struct TrailRobot
  {
    Ogre::SceneNode * node = nullptr;
    std::unique_ptr<rviz_default_plugins::robot::Robot> robot;
    int waypoint_index = -1;
  };
  std::vector<TrailRobot> trail_robots_;

  // --- Trajectory playback ---
  struct WaypointData
  {
    std::map<std::string, Eigen::Isometry3d> link_transforms;
    double time_from_start = 0.0;
  };
  std::vector<WaypointData> waypoints_;
  std::map<std::string, Eigen::Isometry3d> interpolateWaypoints(
    size_t waypoint_a, size_t waypoint_b, double t) const;

  std::mutex trajectory_mutex_;
  trajectory_msgs::msg::JointTrajectory::ConstSharedPtr trajectory_msg_new_;

  bool animating_ = false;
  int current_state_ = 0;
  double current_frac_ = 0.0;
  double anim_time_ = 0.0;
  std::map<std::string, Eigen::Isometry3d> current_transforms_;

  // --- Server FK for the trajectory ---
  struct FkResponse
  {
    uint64_t seq = 0;
    bool ok = false;
    std::string error;
    std::string base_link;
    std::vector<std::string> link_names;
    std::vector<std::array<double, 7>> poses;
  };
  rclcpp::Client<curobov2_ros_interfaces::srv::FkBatch>::SharedPtr fk_client_;
  std::string planner_node_;
  uint64_t fk_seq_ = 0;
  uint64_t fk_inflight_seq_ = 0;
  bool fk_in_flight_ = false;
  std::chrono::steady_clock::time_point fk_sent_at_;
  trajectory_msgs::msg::JointTrajectory::ConstSharedPtr fk_pending_msg_;
  std::mutex fk_mutex_;
  std::shared_ptr<FkResponse> fk_response_;
  void createFkClient();
  void resetFkState();
  bool requestFk(const trajectory_msgs::msg::JointTrajectory::ConstSharedPtr & msg);
  bool consumeFk(uint64_t seq);

  // --- Goal-state ghost (IK the active target, then FK it) ---
  enum class GoalStage { Idle, WaitingIk, WaitingFk };
  struct GoalIkResult
  {
    uint64_t seq = 0;
    bool ok = false;
    std::string error;
    sensor_msgs::msg::JointState joint_state;
  };
  struct GoalFkResult
  {
    uint64_t seq = 0;
    bool ok = false;
    std::string error;
    std::string base_link;
    std::vector<std::string> link_names;
    std::vector<std::array<double, 7>> poses;
  };
  rclcpp::Client<curobov2_ros_interfaces::srv::IkBatch>::SharedPtr ik_client_;
  rclcpp::Client<curobov2_ros_interfaces::srv::FkBatch>::SharedPtr goal_fk_client_;
  GoalStage goal_stage_ = GoalStage::Idle;
  uint64_t goal_seq_ = 0;
  uint64_t goal_inflight_seq_ = 0;
  bool goal_valid_ = false;
  bool goal_attempted_ = false;
  geometry_msgs::msg::Pose last_goal_pose_;
  std::chrono::steady_clock::time_point last_goal_request_;
  std::chrono::steady_clock::time_point goal_stage_start_;
  std::mutex goal_mutex_;
  std::shared_ptr<GoalIkResult> goal_ik_result_;
  std::shared_ptr<GoalFkResult> goal_fk_result_;
  void requestGoalState();
  void consumeGoalIk();
  void consumeGoalFk();
  void resetGoalState();
  bool goalPoseInBase(
    const geometry_msgs::msg::Pose & pose, geometry_msgs::msg::Pose & base_pose) const;
  static bool posesEqual(
    const geometry_msgs::msg::Pose & a, const geometry_msgs::msg::Pose & b);

  // --- Attached-object spheres ---
  struct SphereVisual
  {
    Ogre::SceneNode * node = nullptr;
    std::shared_ptr<rviz_rendering::Shape> shape;
    bool attached = false;
  };
  std::vector<SphereVisual> sphere_pool_;
  size_t sphere_active_count_ = 0;
  std::mutex spheres_mutex_;
  visualization_msgs::msg::MarkerArray::ConstSharedPtr spheres_msg_new_;
  bool spheres_got_message_ = false;

  struct SphereDatum
  {
    Ogre::Vector3 local = Ogre::Vector3::ZERO;
    std::string frame;
    float sx = 0.0f, sy = 0.0f, sz = 0.0f;
    float r = 0.0f, g = 0.0f, b = 0.0f, a = 0.5f;
    bool attached = false;
  };
  std::vector<SphereDatum> live_spheres_;
  std::vector<SphereDatum> playback_spheres_;
  void syncSpheres();
  void updateSpheresVisibility();
  void applySpheres(const std::vector<SphereDatum> & data);
  void poseSpheres();

  // --- Targets ---
  struct ToolTarget
  {
    std::string frame;
    geometry_msgs::msg::Pose pose;
    bool tf_seeded = false;
  };
  static geometry_msgs::msg::Pose defaultPose();
  void ensureTarget(const std::string & frame);
  bool seedToolFromTf(ToolTarget & target);
  void seedPendingTargets();
  geometry_msgs::msg::Pose activePose() const;
  void updatePoseProperties(const geometry_msgs::msg::Pose & pose);
  geometry_msgs::msg::Pose targetPoseFromProperties() const;

  std::map<std::string, ToolTarget> tools_;
  std::vector<std::string> tool_order_;
  std::string active_tool_;
  bool tool_frames_user_set_ = false;

  // --- Interactive marker gizmo ---
  void createGizmo();
  void destroyGizmo();
  void rebuildGizmoMarkers();
  void insertGizmoMarker(const std::string & tool);
  void gizmoFeedback(
    const std::string & tool,
    const visualization_msgs::msg::InteractiveMarkerFeedback::ConstSharedPtr & feedback);
  void syncGizmoToProperties();
  void syncPropertiesToGizmo();
  void connectGizmoClient();
  void disconnectGizmoClient();
  std::string markerName(const std::string & tool) const;

  std::shared_ptr<interactive_markers::InteractiveMarkerServer> gizmo_server_;
  std::unique_ptr<interactive_markers::InteractiveMarkerClient> gizmo_client_;
  bool gizmo_client_connected_ = false;
  std::map<std::string, rviz_default_plugins::displays::InteractiveMarker::SharedPtr>
    interactive_markers_map_;
  static std::atomic<int> gizmo_instance_counter_;
  std::string gizmo_name_;
  std::string gizmo_frame_;
  mutable std::mutex gizmo_mutex_;
  std::map<std::string, geometry_msgs::msg::Pose> feedback_poses_;
  std::set<std::string> feedback_dirty_;
  std::set<std::string> sync_pending_;
  bool applying_gizmo_ = false;
  bool applying_tool_frames_ = false;
  bool applying_active_tool_ = false;
  bool gizmo_active_ = false;

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

  // --- Docked playback panel ---
  rviz_common::PanelDockWidget * trajectory_dock_panel_ = nullptr;
  TrajectoryPanel * trajectory_panel_ = nullptr;
  void createTrajectoryPanel();
  void destroyTrajectoryPanel();

  // --- Properties ---
  // Planning request / targets
  rviz_common::properties::StringProperty * tool_frames_property_;
  rviz_common::properties::EditableEnumProperty * active_tool_property_;
  rviz_common::properties::BoolProperty * seed_from_tf_property_;
  rviz_common::properties::BoolProperty * gizmo_visible_property_;
  rviz_common::properties::FloatProperty * gizmo_scale_property_;
  rviz_common::properties::FloatProperty * target_position_x_;
  rviz_common::properties::FloatProperty * target_position_y_;
  rviz_common::properties::FloatProperty * target_position_z_;
  rviz_common::properties::FloatProperty * target_orientation_x_;
  rviz_common::properties::FloatProperty * target_orientation_y_;
  rviz_common::properties::FloatProperty * target_orientation_z_;
  rviz_common::properties::FloatProperty * target_orientation_w_;

  // Scene robot
  rviz_common::properties::BoolProperty * show_scene_robot_property_;
  rviz_common::properties::FloatProperty * scene_robot_alpha_property_;

  // Goal-state ghost
  rviz_common::properties::BoolProperty * show_goal_state_property_;
  rviz_common::properties::FloatProperty * goal_state_alpha_property_;

  // Planned path
  rviz_common::properties::RosTopicProperty * topic_property_;
  rviz_common::properties::StringProperty * planner_node_property_;
  rviz_common::properties::FloatProperty * alpha_property_;
  rviz_common::properties::BoolProperty * show_trail_property_;
  rviz_common::properties::IntProperty * trail_step_size_property_;
  rviz_common::properties::BoolProperty * loop_property_;
  rviz_common::properties::FloatProperty * speed_property_;
  rviz_common::properties::RosTopicProperty * collision_topic_property_;
  rviz_common::properties::BoolProperty * show_attached_property_;
  rviz_common::properties::BoolProperty * show_collision_spheres_property_;

  // --- Subscribers ---
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr robot_description_sub_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr subscription_;
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr spheres_subscription_;
  void onTrajectoryMessage(trajectory_msgs::msg::JointTrajectory::ConstSharedPtr msg);
  void onCollisionSpheresMessage(
    visualization_msgs::msg::MarkerArray::ConstSharedPtr msg);
};

}  // namespace curobov2_ros_rviz

#endif  // CUROBOV2_ROS_RVIZ__CUROBO_PLANNING_DISPLAY_HPP_

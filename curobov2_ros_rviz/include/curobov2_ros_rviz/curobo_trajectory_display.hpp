#pragma once

#ifndef CUROBOV2_ROS_RVIZ__CUROBO_TRAJECTORY_DISPLAY_HPP_
#define CUROBOV2_ROS_RVIZ__CUROBO_TRAJECTORY_DISPLAY_HPP_

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <rclcpp/rclcpp.hpp>
#include <rviz_common/display.hpp>
#include <rviz_rendering/objects/shape.hpp>
#include <urdf_model/model.h>
#include <urdf_parser/urdf_parser.h>
#include <rviz_common/properties/bool_property.hpp>
#include <rviz_common/properties/float_property.hpp>
#include <rviz_common/properties/int_property.hpp>
#include <rviz_common/properties/ros_topic_property.hpp>
#include <rviz_default_plugins/robot/robot.hpp>

#include <std_msgs/msg/string.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <curobov2_ros_rviz/curobo_fk.hpp>

namespace Ogre
{
class SceneNode;
}  // namespace Ogre

namespace curobov2_ros_rviz
{

/**
 * RViz Display that visualises a cuRobo JointTrajectory as an animated
 * full-robot model.  Subscribes to trajectory_msgs/JointTrajectory, runs
 * URDF forward kinematics for each waypoint, and animates through them
 * using rviz_default_plugins::robot::Robot.
 *
 * The URDF is always read from the latched `/robot_description` topic
 * (std_msgs/String, transient_local — the convention used by
 * robot_state_publisher and kortex.rviz).  No source configuration is
 * exposed in the panel.
 *
 * The grasped object's spheres (the server's `attached_spheres` namespace
 * of its collision-sphere MarkerArray) render here, on by default, so the
 * object in the gripper stays visible without a hand-added MarkerArray
 * display. Robot collision spheres are deliberately NOT shown: one display
 * element owns the grasp visualisation (robot, trail, attached object).
 * Colours come from the message (blue attached, red colliding), so the
 * display never reinterprets what the server decided.
 *
 * No MoveIt dependency.
 */
class CuroboTrajectoryDisplay : public rviz_common::Display
{
  Q_OBJECT
public:
  CuroboTrajectoryDisplay();
  ~CuroboTrajectoryDisplay() override;

protected:
  void onInitialize() override;
  void update(float wall_dt, float ros_dt) override;
  void reset() override;

private Q_SLOTS:
  void updateTopic();
  void updateAlpha();
  void updateShowTrail();
  void updateTrailStepSize();
  void updateLoop();
  void updateSpeed();
  void updateCollisionTopic();
  void updateShowAttached();

private:
  // --- URDF loading (called every frame until robot_loaded_) ---
  bool loadURDF();

  // --- Robot model ---
  std::unique_ptr<rviz_default_plugins::robot::Robot> robot_;
  urdf::ModelInterfaceSharedPtr urdf_model_;
  bool robot_loaded_ = false;
  CuroboFK fk_engine_;
  std::string urdf_xml_;
  void loadRobotModel();

  // --- Trail (ghost copies) ---
  // Grow-only pool (see rebuildTrail): ghosts share the single parsed URDF
  // model and are re-posed, never rebuilt, in steady operation.
  struct TrailRobot
  {
    std::unique_ptr<rviz_default_plugins::robot::Robot> robot;
    int waypoint_index = -1;
  };
  std::vector<TrailRobot> trail_robots_;
  void rebuildTrail();
  void updateTrailVisibility();

  // --- Interpolated pose for the current frame ---
  std::map<std::string, Eigen::Isometry3d> interpolateWaypoints(
    size_t waypoint_a, size_t waypoint_b, double t) const;

  // --- Per-waypoint precomputed data ---
  struct WaypointData
  {
    std::map<std::string, Eigen::Isometry3d> link_transforms;
    double time_from_start = 0.0;
  };
  std::vector<WaypointData> waypoints_;

  // --- Trajectory message ---
  std::mutex trajectory_mutex_;
  trajectory_msgs::msg::JointTrajectory::ConstSharedPtr trajectory_msg_new_;
  trajectory_msgs::msg::JointTrajectory::ConstSharedPtr trajectory_msg_active_;

  // --- Animation state ---
  bool animating_ = false;
  int current_state_ = 0;            // base waypoint index (floor)
  double current_frac_ = 0.0;        // interpolation fraction within the segment
  double anim_time_ = 0.0;           // elapsed playback time (trajectory seconds)
  std::map<std::string, Eigen::Isometry3d> current_transforms_;

  // --- Properties (display) ---
  rviz_common::properties::RosTopicProperty * topic_property_;
  rviz_common::properties::FloatProperty * alpha_property_;
  rviz_common::properties::BoolProperty * show_trail_property_;
  rviz_common::properties::IntProperty * trail_step_size_property_;
  rviz_common::properties::BoolProperty * loop_property_;
  rviz_common::properties::FloatProperty * speed_property_;
  rviz_common::properties::RosTopicProperty * collision_topic_property_;
  rviz_common::properties::BoolProperty * show_attached_property_;

  // --- Collision spheres (attached-object overlay) ---
  // One pooled Shape per sphere marker, each parented to its own child scene
  // node so visibility toggles without touching shared Ogre resources. The
  // pool only ever grows (like the trail ghosts): steady operation re-poses
  // and re-colours, never allocates. `attached` records which namespace the
  // marker came from, which is what the two checkboxes filter on.
  struct SphereVisual
  {
    Ogre::SceneNode * node = nullptr;
    std::shared_ptr<rviz_rendering::Shape> shape;
    bool attached = false;
  };
  std::vector<SphereVisual> sphere_pool_;
  /// How many leading pool entries the last applySpheres() filled. The rest
  /// is surplus and stays hidden: visibility toggles must never resurrect
  /// stale entries — that is how detached objects lingered as balls.
  size_t sphere_active_count_ = 0;
  std::mutex spheres_mutex_;
  visualization_msgs::msg::MarkerArray::ConstSharedPtr spheres_msg_new_;
  /// Set on the first consumed spheres message; until then the display warns
  /// with the topic it is waiting on, so a wrong topic on ONE display (when
  /// several trajectory displays each configure their own) reads as that
  /// display's status instead of silent emptiness.
  bool spheres_got_message_ = false;
  void syncSpheres();
  void updateSpheresVisibility();
  /// Shared pool writer: positions/scales/colours a datum list, hides the
  /// surplus.
  struct SphereDatum
  {
    Ogre::Vector3 pos = Ogre::Vector3::ZERO;
    float sx = 0.0f, sy = 0.0f, sz = 0.0f;
    float r = 0.0f, g = 0.0f, b = 0.0f, a = 0.5f;
    bool attached = false;
  };
  void applySpheres(const std::vector<SphereDatum> & data);

  // --- Subscriber (URDF topic, always `/robot_description`) ---
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr robot_description_sub_;

  // --- Subscriber (JointTrajectory) ---
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr subscription_;
  void onTrajectoryMessage(trajectory_msgs::msg::JointTrajectory::ConstSharedPtr msg);

  // --- Subscriber (collision-sphere MarkerArray) ---
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr spheres_subscription_;
  void onCollisionSpheresMessage(
    visualization_msgs::msg::MarkerArray::ConstSharedPtr msg);
};

}  // namespace curobov2_ros_rviz

#endif  // CUROBOV2_ROS_RVIZ__CUROBO_TRAJECTORY_DISPLAY_HPP_

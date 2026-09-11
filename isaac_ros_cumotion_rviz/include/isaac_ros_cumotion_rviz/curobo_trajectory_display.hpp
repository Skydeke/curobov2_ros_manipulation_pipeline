#pragma once

#ifndef ISAAC_ROS_CUMOTION_RVIZ__CUROBO_TRAJECTORY_DISPLAY_HPP_
#define ISAAC_ROS_CUMOTION_RVIZ__CUROBO_TRAJECTORY_DISPLAY_HPP_

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <rclcpp/rclcpp.hpp>
#include <rviz_common/display.hpp>
#include <urdf_model/model.h>
#include <urdf_parser/urdf_parser.h>
#include <rviz_common/properties/bool_property.hpp>
#include <rviz_common/properties/float_property.hpp>
#include <rviz_common/properties/int_property.hpp>
#include <rviz_common/properties/ros_topic_property.hpp>
#include <rviz_default_plugins/robot/robot.hpp>

#include <std_msgs/msg/string.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include <isaac_ros_cumotion_rviz/curobo_fk.hpp>

namespace isaac_ros_cumotion_rviz
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

  // --- Subscriber (URDF topic, always `/robot_description`) ---
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr robot_description_sub_;

  // --- Subscriber (JointTrajectory) ---
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr subscription_;
  void onTrajectoryMessage(trajectory_msgs::msg::JointTrajectory::ConstSharedPtr msg);
};

}  // namespace isaac_ros_cumotion_rviz

#endif  // ISAAC_ROS_CUMOTION_RVIZ__CUROBO_TRAJECTORY_DISPLAY_HPP_

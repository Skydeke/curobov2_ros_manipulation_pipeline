// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/core/unified_planner_node.py
//
// At M2 this file holds the pure free functions of the node module and the
// UnifiedPlannerNode class wiring the M2 services (scene, FK, IK, collision,
// attach, joint info) over the shared bridge. The planner/action/timer parts of
// the Python node arrive with their milestones.
#ifndef CUROBO_CORE__CORE__UNIFIED_PLANNER_NODE_HPP_
#define CUROBO_CORE__CORE__UNIFIED_PLANNER_NODE_HPP_

#include <memory>
#include <string>
#include <vector>

#include <memory>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>

#include "curobo_core/core/attachment_services.hpp"
#include "curobo_core/core/camera_system_manager.hpp"
#include "curobo_core/core/collision_distance.hpp"
#include "curobo_core/core/fk_services.hpp"
#include "curobo_core/core/ik_services.hpp"
#include "curobo_core/core/laser_system_manager.hpp"
#include "curobo_core/core/obstacle_manager.hpp"
#include "curobo_core/core/robot_segmentation.hpp"
#include "curobo_core/core/config_services.hpp"
#include "curobo_core/core/planner_services.hpp"
#include "curobo_core/core/reachability_services.hpp"
#include "curobo_core/core/trajectory_action_server.hpp"
#include "curobo_core/robot/robot_context.hpp"
#include "curobo_core/core/ros_service_manager.hpp"
#include "curobo_runtime/bridge.hpp"

namespace curobo_core
{

// PORT: unified_planner_node.py:should_refresh_world
//
// Whether refresh_perception_world must rebuild + push (pure policy). Analytic
// scene content is pushed eagerly at mutation time, so this gate is purely
// about ESDF freshness.
inline bool should_refresh_world(
  bool force,
  int epoch,
  int last_pushed_epoch,
  double now,
  double last_push_time,
  double min_interval_s)
{
  if (last_pushed_epoch < 0) {
    return true;
  }
  if (force) {
    return epoch != last_pushed_epoch;
  }
  if (epoch == last_pushed_epoch) {
    return false;
  }
  return (now - last_push_time) >= min_interval_s;
}

// PORT: unified_planner_node.py:UnifiedPlannerNode (M2 subset)
class UnifiedPlannerNode : public rclcpp::Node
{
public:
  explicit UnifiedPlannerNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~UnifiedPlannerNode() override;

  const std::vector<std::string> & joint_names() const {return joint_names_;}

private:
  void declare_parameters();
  void setup();

  // PORT: the emulator's live joint pose. RobotContext lands at M3/M5; until
  // then the node has no robot and this reports the cspace zeros.
  std::vector<float> current_pose() const;

  // PORT: unified_planner_node.mpc_goal_callback
  void handle_mpc_goal(const geometry_msgs::msg::PoseStamped & msg);
  // PORT: unified_planner_node._publish_tool_pose_callback (1 Hz)
  void publish_tool_pose();

  curobo_runtime::IBridge * bridge_ = nullptr;
  curobo_runtime::RobotHandle handle_ = -1;
  std::string base_link_;
  std::vector<std::string> joint_names_;

  std::unique_ptr<ObstacleManager> obstacle_manager_;
  std::unique_ptr<FKServices> fk_services_;
  std::unique_ptr<IKServices> ik_services_;
  std::unique_ptr<CollisionDistance> collision_distance_;
  std::unique_ptr<AttachmentServices> attachment_services_;
  std::unique_ptr<RosServiceManager> ros_service_manager_;
  std::unique_ptr<ConfigServices> config_services_;
  std::unique_ptr<PlannerServices> planner_services_;
  std::unique_ptr<ReachabilityServices> reachability_services_;
  std::unique_ptr<RobotContext> robot_context_;
  std::unique_ptr<TrajectoryActionServer> action_server_;
  // Perception (M4): the mapper cameras, the lidar feeds and the depth-map
  // robot segmentation (mask services registered via RosServiceManager).
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr mpc_goal_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr tool_pose_pub_;
  rclcpp::TimerBase::SharedPtr tool_pose_timer_;
  // PORT: obstacle_manager._build_mapper - the mapper cfg JSON is built at
  // startup but COMPLETED (with decay_factor, which needs the camera/laser
  // frame rates) after their managers exist. The bridge builds the mapper
  // lazily, so configuring it late is safe.
  std::string mapper_cfg_base_;
  std::unique_ptr<CameraSystemManager> camera_system_manager_;
  std::unique_ptr<LaserSystemManager> laser_system_manager_;
  std::unique_ptr<RobotSegmentation> robot_segmentation_;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__CORE__UNIFIED_PLANNER_NODE_HPP_

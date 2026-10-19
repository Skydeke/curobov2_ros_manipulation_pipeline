// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/core/ros_service_manager.py
//
// The M2 subset of RosServiceManager: the scene services (add_object,
// remove_object, remove_all_objects, set_objects, get_obstacles,
// get_scene_objects) and get_joint_info. Further services are added with their
// milestones, keeping this file name.
#ifndef CUROBO_CORE__CORE__ROS_SERVICE_MANAGER_HPP_
#define CUROBO_CORE__CORE__ROS_SERVICE_MANAGER_HPP_

#include <functional>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>

#include "curobov2_ros_interfaces/msg/scene_object.hpp"
#include "curobov2_ros_interfaces/msg/sparse_voxel_grid.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "curobov2_ros_interfaces/srv/add_object.hpp"
#include "curobov2_ros_interfaces/srv/get_joint_info.hpp"
#include "curobov2_ros_interfaces/srv/get_scene_objects.hpp"
#include "curobov2_ros_interfaces/srv/get_voxel_grid.hpp"
#include "curobov2_ros_interfaces/srv/remove_object.hpp"
#include "curobov2_ros_interfaces/srv/set_mask.hpp"
#include "curobov2_ros_interfaces/srv/set_link_collision.hpp"
#include "curobov2_ros_interfaces/srv/set_objects.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "curobo_core/core/obstacle_manager.hpp"
#include "curobo_runtime/bridge.hpp"

namespace curobo_core
{

class RobotSegmentation;

class RosServiceManager
{
public:
  RosServiceManager(
    rclcpp::Node * node, curobo_runtime::IBridge * bridge,
    curobo_runtime::RobotHandle handle, ObstacleManager * obstacle_manager);

  // PORT: ros_service_manager.init_services (M2 subset)
  void init_services();
  // PORT: ros_service_manager.publish_mapper_workspace + _publish_sparse_voxel_grid
  void publish_mapper_workspace();
  void publish_sparse_voxel_grid();
  // PORT: ros_service_manager.publish_scene_obstacles
  void publish_scene_obstacles();
  // PORT: ros_service_manager.publish_collision_spheres (timer; pose provider
  // is the node's current joint pose).
  void set_pose_provider(std::function<std::vector<float>()> provider)
  {pose_provider_ = std::move(provider);}
  // Frame the Mapper is integrated in (map/odom when TF provides one, else the
  // robot base). The node owns the TF buffer, so it supplies the resolver; used
  // for the mapper_workspace marker so the box matches the integrated voxels.
  void set_integration_frame_provider(std::function<std::string()> provider)
  {integration_frame_provider_ = std::move(provider);}
  void publish_collision_spheres();

  // PORT: ros_service_manager.register_robot_segmentation (set_mask / remove_mask,
  // registered after the component is built).
  void register_robot_segmentation(RobotSegmentation * segmentation);

  // Message -> spec conversions (PORT: the request handling in each callback).
  static SceneObjectSpec spec_from_add_object(
    const curobov2_ros_interfaces::srv::AddObject::Request & request);
  static SceneObjectSpec spec_from_scene_object(
    const curobov2_ros_interfaces::msg::SceneObject & object);

private:
  void handle_add_object(
    const std::shared_ptr<curobov2_ros_interfaces::srv::AddObject::Request> request,
    std::shared_ptr<curobov2_ros_interfaces::srv::AddObject::Response> response);
  void handle_remove_object(
    const std::shared_ptr<curobov2_ros_interfaces::srv::RemoveObject::Request> request,
    std::shared_ptr<curobov2_ros_interfaces::srv::RemoveObject::Response> response);
  void handle_remove_all_objects(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response);
  void handle_set_objects(
    const std::shared_ptr<curobov2_ros_interfaces::srv::SetObjects::Request> request,
    std::shared_ptr<curobov2_ros_interfaces::srv::SetObjects::Response> response);
  void handle_get_obstacles(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response);
  void handle_get_scene_objects(
    const std::shared_ptr<curobov2_ros_interfaces::srv::GetSceneObjects::Request> request,
    std::shared_ptr<curobov2_ros_interfaces::srv::GetSceneObjects::Response> response);
  void handle_get_joint_info(
    const std::shared_ptr<curobov2_ros_interfaces::srv::GetJointInfo::Request> request,
    std::shared_ptr<curobov2_ros_interfaces::srv::GetJointInfo::Response> response);
  void handle_get_voxel_grid(
    const std::shared_ptr<curobov2_ros_interfaces::srv::GetVoxelGrid::Request> request,
    std::shared_ptr<curobov2_ros_interfaces::srv::GetVoxelGrid::Response> response);
  void handle_clear_voxel_map(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response);
  // PORT: robot_model_manager.set_link_collision_callback
  void handle_set_link_collision(
    const std::shared_ptr<curobov2_ros_interfaces::srv::SetLinkCollision::Request> request,
    std::shared_ptr<curobov2_ros_interfaces::srv::SetLinkCollision::Response> response);
  // PORT: ros_service_manager._callback_set_collision_spheres_enabled
  void handle_set_collision_spheres_enabled(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response);

  rclcpp::Node * node_;
  curobo_runtime::IBridge * bridge_;
  curobo_runtime::RobotHandle handle_;
  ObstacleManager * obstacle_manager_;

  rclcpp::Service<curobov2_ros_interfaces::srv::AddObject>::SharedPtr add_object_srv_;
  rclcpp::Service<curobov2_ros_interfaces::srv::RemoveObject>::SharedPtr remove_object_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr remove_all_srv_;
  rclcpp::Service<curobov2_ros_interfaces::srv::SetObjects>::SharedPtr set_objects_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr get_obstacles_srv_;
  rclcpp::Service<curobov2_ros_interfaces::srv::GetSceneObjects>::SharedPtr
    get_scene_objects_srv_;
  rclcpp::Service<curobov2_ros_interfaces::srv::GetJointInfo>::SharedPtr get_joint_info_srv_;
  rclcpp::Service<curobov2_ros_interfaces::srv::GetVoxelGrid>::SharedPtr get_voxel_grid_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr clear_voxel_map_srv_;
  rclcpp::Service<curobov2_ros_interfaces::srv::SetLinkCollision>::SharedPtr
    set_link_collision_srv_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr set_collision_spheres_srv_;
  bool collision_spheres_enabled_ = true;
  bool workspace_viz_enabled_ = true;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr workspace_viz_pub_;
  rclcpp::Publisher<curobov2_ros_interfaces::msg::SparseVoxelGrid>::SharedPtr sparse_voxel_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr scene_obstacle_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr collision_spheres_pub_;
  rclcpp::TimerBase::SharedPtr collision_spheres_timer_;
  std::function<std::vector<float>()> pose_provider_;
  std::function<std::string()> integration_frame_provider_;
  rclcpp::TimerBase::SharedPtr sparse_voxel_timer_;
  rclcpp::CallbackGroup::SharedPtr viz_callback_group_;
  rclcpp::Service<curobov2_ros_interfaces::srv::SetMask>::SharedPtr set_mask_srv_;
  rclcpp::Service<curobov2_ros_interfaces::srv::RemoveObject>::SharedPtr remove_mask_srv_;
  RobotSegmentation * segmentation_ = nullptr;
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__CORE__ROS_SERVICE_MANAGER_HPP_

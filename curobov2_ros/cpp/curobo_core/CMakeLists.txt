cmake_minimum_required(VERSION 3.16)
project(curobo_core)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(curobo_runtime REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(curobov2_ros_interfaces REQUIRED)
find_package(std_srvs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

# libcurobo_core.so: RobotContext, ObstacleManager, strategies, planners.
# Links to libcurobo_runtime.so dynamically (never static).
add_library(curobo_core SHARED
  src/core/unified_planner_node.cpp
  src/core/obstacle_manager.cpp
  src/core/fk_services.cpp
  src/core/ik_services.cpp
  src/core/attachment_services.cpp
  src/core/collision_distance.cpp
  src/core/ros_service_manager.cpp
  src/core/config_services.cpp
  src/core/planner_services.cpp
  src/core/trajectory_action_server.cpp
  src/planners/classic_planner.cpp
  src/planners/cartesian_planner.cpp
  src/planners/reactive_controller.cpp
  src/planners/mpc_planner.cpp
  src/planners/retarget_controller.cpp
  src/robot/joint_control_strategy.cpp
  src/robot/emulator_strategy.cpp
  src/robot/robot_registry.cpp
  src/robot/robot_context.cpp
  src/cameras/camera_cfg.cpp
  src/cameras/image_decode.cpp
  src/cameras/camera_strategy.cpp
  src/cameras/camera_context.cpp
  src/cameras/camera_depth_map_strategy.cpp
  src/lasers/range_image.cpp
  src/lasers/laser_cfg.cpp
  src/lasers/laser_context.cpp
  src/lasers/laser_strategy.cpp
  src/lasers/laser_pointcloud_strategy.cpp
  src/lasers/laser_scan_strategy.cpp
  src/core/laser_system_manager.cpp
  src/cameras/camera_robot_segmentation_strategy.cpp
  src/core/robot_segmentation.cpp
  src/core/camera_system_manager.cpp
  src/core/frame_utils.cpp
  src/planners/planner_factory.cpp
  src/convert.cpp
)

target_include_directories(curobo_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(curobo_core PUBLIC
  curobo_runtime::curobo_runtime
  Eigen3::Eigen
  ${geometry_msgs_TARGETS}
  rclcpp::rclcpp
  rclcpp_action::rclcpp_action
  tf2_ros::tf2_ros
  tf2::tf2
  ${curobov2_ros_interfaces_TARGETS}
  ${std_srvs_TARGETS}
  ${control_msgs_TARGETS}
  ${trajectory_msgs_TARGETS}
  ${visualization_msgs_TARGETS}
  ${builtin_interfaces_TARGETS}
)

set_target_properties(curobo_core PROPERTIES
  CXX_VISIBILITY_PRESET default
  VISIBILITY_INLINES_HIDDEN OFF
)

install(TARGETS curobo_core
  EXPORT export_curobo_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_world_refresh_policy test/test_world_refresh_policy.cpp)
  target_link_libraries(test_world_refresh_policy curobo_core)

  ament_add_gtest(test_convert test/test_convert.cpp)
  target_link_libraries(test_convert curobo_core)

  ament_add_gtest(test_obstacle_manager test/test_obstacle_manager.cpp)
  target_link_libraries(test_obstacle_manager curobo_core)

  ament_add_gtest(test_obstacle_perception test/test_obstacle_perception.cpp)
  target_link_libraries(test_obstacle_perception curobo_core)

  ament_add_gtest(test_camera_cfg test/test_camera_cfg.cpp)
  target_link_libraries(test_camera_cfg curobo_core rclcpp::rclcpp)

  ament_add_gtest(test_frame_utils test/test_frame_utils.cpp)
  target_link_libraries(test_frame_utils curobo_core)

  ament_add_gtest(test_image_decode test/test_image_decode.cpp)
  target_link_libraries(test_image_decode curobo_core)

  ament_add_gtest(test_camera_depth test/test_camera_depth.cpp)
  target_link_libraries(test_camera_depth curobo_core rclcpp::rclcpp ${curobov2_ros_interfaces_TARGETS})

  ament_add_gtest(test_range_image test/test_range_image.cpp)
  target_link_libraries(test_range_image curobo_core)

  ament_add_gtest(test_camera_system_manager test/test_camera_system_manager.cpp)
  target_link_libraries(test_camera_system_manager curobo_core rclcpp::rclcpp ${curobov2_ros_interfaces_TARGETS})

  ament_add_gtest(test_laser test/test_laser.cpp)
  target_link_libraries(test_laser curobo_core rclcpp::rclcpp ${curobov2_ros_interfaces_TARGETS})

  ament_add_gtest(test_robot_segmentation test/test_robot_segmentation.cpp)
  target_link_libraries(test_robot_segmentation curobo_core rclcpp::rclcpp ${curobov2_ros_interfaces_TARGETS})

  ament_add_gtest(test_voxel_grid test/test_voxel_grid.cpp)
  target_include_directories(test_voxel_grid PRIVATE test)
  target_link_libraries(test_voxel_grid curobo_core rclcpp::rclcpp)

  ament_add_gtest(test_fk_services test/test_fk_services.cpp)
  target_link_libraries(test_fk_services curobo_core rclcpp::rclcpp ${curobov2_ros_interfaces_TARGETS})

  ament_add_gtest(test_ik_services test/test_ik_services.cpp)
  target_link_libraries(test_ik_services curobo_core rclcpp::rclcpp ${curobov2_ros_interfaces_TARGETS})

  ament_add_gtest(test_attachment_services test/test_attachment_services.cpp)
  target_link_libraries(test_attachment_services curobo_core rclcpp::rclcpp ${curobov2_ros_interfaces_TARGETS} ${std_srvs_TARGETS})

  ament_add_gtest(test_collision_distance test/test_collision_distance.cpp)
  target_link_libraries(test_collision_distance curobo_core rclcpp::rclcpp ${curobov2_ros_interfaces_TARGETS})

  ament_add_gtest(test_unified_planner_node test/test_unified_planner_node.cpp)
  target_link_libraries(test_unified_planner_node curobo_core rclcpp::rclcpp ${curobov2_ros_interfaces_TARGETS})

  ament_add_gtest(test_planners test/test_planners.cpp)
  target_link_libraries(test_planners curobo_core)

  ament_add_gtest(test_reactive_controller test/test_reactive_controller.cpp)
  target_link_libraries(test_reactive_controller curobo_core rclcpp::rclcpp ${curobov2_ros_interfaces_TARGETS})

  ament_add_gtest(test_robot_strategy test/test_robot_strategy.cpp)
  target_link_libraries(test_robot_strategy curobo_core rclcpp::rclcpp ${curobov2_ros_interfaces_TARGETS})

  ament_add_gtest(test_trajectory_action test/test_trajectory_action.cpp)
  target_link_libraries(test_trajectory_action curobo_core rclcpp::rclcpp rclcpp_action::rclcpp_action ${curobov2_ros_interfaces_TARGETS})
endif()

ament_export_include_directories(include)
ament_export_libraries(curobo_core)
ament_export_targets(export_curobo_core NAMESPACE curobo_core::)
ament_export_dependencies(curobo_runtime)
ament_package()

cmake_minimum_required(VERSION 3.16)
project(curobov2_ros)

# ---------------------------------------------------------------------------
# curobov2_ros is the C++ server (UD-6 / Q5). The C++ implementation (the
# runtime + core libraries, the planner component and the standalone
# `curobo_trajectory_planner` executable) is folded in under cpp/; the package's
# config/, launch/, robots/, rviz/ and Python reference sources stay and are
# installed. The Python ROS node is kept in-tree as the reference until the C++
# has full feature parity, at which point it is deleted. The executable keeps
# the Python entry point's name so the launch files need no change.
# ---------------------------------------------------------------------------

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(curobov2_ros_interfaces REQUIRED)

# The interpreter that has torch and cuRobo: system Python 3.12 (decision Q2).
find_package(Python3 COMPONENTS Development.Embed REQUIRED)
find_package(pybind11 REQUIRED)

# ---------------------------------------------------------------------------
# libcurobo_runtime.so: Runtime singleton, Lanes, Barrier, GIL guards, Bridge.
# Defined in this library ONLY (design section 4 singleton rule): components and
# libcurobo_core.so link to it dynamically; never static-link it.
# ---------------------------------------------------------------------------
add_library(curobo_runtime SHARED
  cpp/curobo_runtime/src/gpu_lanes.cpp
  cpp/curobo_runtime/src/fake_bridge.cpp
  cpp/curobo_runtime/src/runtime.cpp
  cpp/curobo_runtime/src/py_bridge.cpp
)

target_include_directories(curobo_runtime PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/curobo_runtime/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(curobo_runtime PRIVATE
  Threads::Threads
  Python3::Python
  pybind11::embed
)

# The interpreter's libpython must be loadable with RTLD_GLOBAL or numpy/torch
# extension modules fail with undefined symbols (design section 5).
get_filename_component(_py_lib_dir "${Python3_LIBRARIES}" DIRECTORY)
get_filename_component(_py_lib_name "${Python3_LIBRARIES}" NAME)
target_compile_definitions(curobo_runtime PRIVATE
  CUROBO_PYTHON_LIB_NAME="${_py_lib_name}"
  CUROBO_PYTHON_LIB_DIR="${_py_lib_dir}"
)

set_target_properties(curobo_runtime PROPERTIES
  CXX_VISIBILITY_PRESET default
  VISIBILITY_INLINES_HIDDEN OFF
  BUILD_RPATH "${_py_lib_dir}"
  INSTALL_RPATH "${_py_lib_dir};$ORIGIN"
)

install(TARGETS curobo_runtime
  EXPORT export_curobo_runtime
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY cpp/curobo_runtime/include/ DESTINATION include)

# ---------------------------------------------------------------------------
# libcurobo_core.so: RobotContext, ObstacleManager, strategies, planners.
# ---------------------------------------------------------------------------
add_library(curobo_core SHARED
  cpp/curobo_core/src/core/unified_planner_node.cpp
  cpp/curobo_core/src/core/obstacle_manager.cpp
  cpp/curobo_core/src/core/fk_services.cpp
  cpp/curobo_core/src/core/ik_services.cpp
  cpp/curobo_core/src/core/attachment_services.cpp
  cpp/curobo_core/src/core/collision_distance.cpp
  cpp/curobo_core/src/core/collision_diagnostics.cpp
  cpp/curobo_core/src/core/ros_service_manager.cpp
  cpp/curobo_core/src/core/config_services.cpp
  cpp/curobo_core/src/core/planner_services.cpp
  cpp/curobo_core/src/core/trajectory_action_server.cpp
  cpp/curobo_core/src/core/frame_utils.cpp
  cpp/curobo_core/src/core/laser_system_manager.cpp
  cpp/curobo_core/src/core/robot_segmentation.cpp
  cpp/curobo_core/src/core/reachability_services.cpp
  cpp/curobo_core/src/core/camera_system_manager.cpp
  cpp/curobo_core/src/planners/classic_planner.cpp
  cpp/curobo_core/src/planners/cartesian_planner.cpp
  cpp/curobo_core/src/planners/reactive_controller.cpp
  cpp/curobo_core/src/planners/mpc_planner.cpp
  cpp/curobo_core/src/planners/retarget_controller.cpp
  cpp/curobo_core/src/planners/planner_factory.cpp
  cpp/curobo_core/src/robot/joint_control_strategy.cpp
  cpp/curobo_core/src/robot/emulator_strategy.cpp
  cpp/curobo_core/src/robot/robot_registry.cpp
  cpp/curobo_core/src/robot/robot_context.cpp
  cpp/curobo_core/src/robot/joint_speed_strategy.cpp
  cpp/curobo_core/src/robot/joint_pose_strategy.cpp
  cpp/curobo_core/src/robot/ghost_strategy.cpp
  cpp/curobo_core/src/core/diagnostics.cpp
  cpp/curobo_core/src/cameras/camera_cfg.cpp
  cpp/curobo_core/src/cameras/image_decode.cpp
  cpp/curobo_core/src/cameras/camera_strategy.cpp
  cpp/curobo_core/src/cameras/camera_context.cpp
  cpp/curobo_core/src/cameras/camera_depth_map_strategy.cpp
  cpp/curobo_core/src/cameras/camera_robot_segmentation_strategy.cpp
  cpp/curobo_core/src/lasers/range_image.cpp
  cpp/curobo_core/src/lasers/laser_cfg.cpp
  cpp/curobo_core/src/lasers/laser_context.cpp
  cpp/curobo_core/src/lasers/laser_strategy.cpp
  cpp/curobo_core/src/lasers/laser_pointcloud_strategy.cpp
  cpp/curobo_core/src/lasers/laser_scan_strategy.cpp
  cpp/curobo_core/src/convert.cpp
)

target_include_directories(curobo_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/curobo_core/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(curobo_core PUBLIC
  curobo_runtime
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
  ${nav2_msgs_TARGETS}
)

set_target_properties(curobo_core PROPERTIES
  CXX_VISIBILITY_PRESET default
  VISIBILITY_INLINES_HIDDEN OFF
  INSTALL_RPATH "$ORIGIN"
)

install(TARGETS curobo_core
  EXPORT export_curobo_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY cpp/curobo_core/include/ DESTINATION include)

# ---------------------------------------------------------------------------
# The planner component + the standalone server.
# ---------------------------------------------------------------------------
add_library(planner_component SHARED cpp/curobo_planner_component/src/planner_component.cpp)
target_link_libraries(planner_component
  curobo_core
  rclcpp::rclcpp
  rclcpp_components::component
)
rclcpp_components_register_nodes(planner_component "curobo_core::UnifiedPlannerNode")
set_target_properties(planner_component PROPERTIES INSTALL_RPATH "$ORIGIN")

# The standalone executable keeps the Python entry point's name so the launch
# files (package 'curobov2_ros', executable 'curobo_trajectory_planner') are
# unchanged; the node name is still 'unified_planner'.
add_executable(curobo_trajectory_planner
  cpp/curobo_planner_component/src/unified_planner_main.cpp)
target_link_libraries(curobo_trajectory_planner curobo_core rclcpp::rclcpp)
set_target_properties(curobo_trajectory_planner PROPERTIES INSTALL_RPATH "$ORIGIN/..")

install(TARGETS planner_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(TARGETS curobo_trajectory_planner RUNTIME DESTINATION lib/${PROJECT_NAME})

# ---------------------------------------------------------------------------
# Package assets (kept from the Python package; adapted launch to the C++ node).
# ---------------------------------------------------------------------------
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY robots DESTINATION share/${PROJECT_NAME})
install(DIRECTORY rviz DESTINATION share/${PROJECT_NAME})
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/scripts)
  install(DIRECTORY scripts DESTINATION share/${PROJECT_NAME})
endif()

ament_export_include_directories(include)
ament_export_libraries(curobo_runtime curobo_core)
ament_export_targets(export_curobo_runtime NAMESPACE curobo_runtime::)
ament_export_targets(export_curobo_core NAMESPACE curobo_core::)

# ---------------------------------------------------------------------------
# Tests: C++ gtest suites + the Python/launch parity suites.
# ---------------------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  # --- curobo_runtime ---
  ament_add_gtest(test_gpu_lanes cpp/curobo_runtime/test/test_gpu_lanes.cpp)
  target_link_libraries(test_gpu_lanes curobo_runtime)

  ament_add_gtest(test_bridge_import cpp/curobo_runtime/test/test_bridge_import.cpp)
  target_link_libraries(test_bridge_import curobo_runtime)

  ament_add_gtest(test_bridge_contract cpp/curobo_runtime/test/test_bridge_contract.cpp)
  target_link_libraries(test_bridge_contract curobo_runtime)

  add_library(runtime_probe_a SHARED cpp/curobo_runtime/test/runtime_probe.cpp)
  add_library(runtime_probe_b SHARED cpp/curobo_runtime/test/runtime_probe.cpp)
  target_link_libraries(runtime_probe_a PRIVATE curobo_runtime Python3::Python)
  target_link_libraries(runtime_probe_b PRIVATE curobo_runtime Python3::Python)

  ament_add_gtest(test_runtime_singleton cpp/curobo_runtime/test/test_runtime_singleton.cpp)
  target_link_libraries(test_runtime_singleton curobo_runtime ${CMAKE_DL_LIBS})
  target_compile_definitions(test_runtime_singleton PRIVATE
    PROBE_A_PATH="$<TARGET_FILE:runtime_probe_a>"
    PROBE_B_PATH="$<TARGET_FILE:runtime_probe_b>"
  )
  add_dependencies(test_runtime_singleton runtime_probe_a runtime_probe_b)

  # --- curobo_core ---
  ament_add_gtest(test_world_refresh_policy cpp/curobo_core/test/test_world_refresh_policy.cpp)
  target_link_libraries(test_world_refresh_policy curobo_core)

  ament_add_gtest(test_convert cpp/curobo_core/test/test_convert.cpp)
  target_link_libraries(test_convert curobo_core)

  ament_add_gtest(test_obstacle_manager cpp/curobo_core/test/test_obstacle_manager.cpp)
  target_link_libraries(test_obstacle_manager curobo_core)

  ament_add_gtest(test_obstacle_perception cpp/curobo_core/test/test_obstacle_perception.cpp)
  target_link_libraries(test_obstacle_perception curobo_core)

  ament_add_gtest(test_camera_cfg cpp/curobo_core/test/test_camera_cfg.cpp)
  target_link_libraries(test_camera_cfg curobo_core rclcpp::rclcpp)

  ament_add_gtest(test_frame_utils cpp/curobo_core/test/test_frame_utils.cpp)
  target_link_libraries(test_frame_utils curobo_core)

  ament_add_gtest(test_image_decode cpp/curobo_core/test/test_image_decode.cpp)
  target_link_libraries(test_image_decode curobo_core)

  ament_add_gtest(test_camera_depth cpp/curobo_core/test/test_camera_depth.cpp)
  target_link_libraries(test_camera_depth curobo_core rclcpp::rclcpp ${curobov2_ros_interfaces_TARGETS})

  ament_add_gtest(test_range_image cpp/curobo_core/test/test_range_image.cpp)
  target_link_libraries(test_range_image curobo_core)

  ament_add_gtest(test_camera_system_manager cpp/curobo_core/test/test_camera_system_manager.cpp)
  target_link_libraries(test_camera_system_manager curobo_core rclcpp::rclcpp ${curobov2_ros_interfaces_TARGETS})

  ament_add_gtest(test_laser cpp/curobo_core/test/test_laser.cpp)
  target_link_libraries(test_laser curobo_core rclcpp::rclcpp ${curobov2_ros_interfaces_TARGETS})

  ament_add_gtest(test_robot_segmentation cpp/curobo_core/test/test_robot_segmentation.cpp)
  target_link_libraries(test_robot_segmentation curobo_core rclcpp::rclcpp ${curobov2_ros_interfaces_TARGETS})

  ament_add_gtest(test_voxel_grid cpp/curobo_core/test/test_voxel_grid.cpp)
  target_include_directories(test_voxel_grid PRIVATE cpp/curobo_core/test)
  target_link_libraries(test_voxel_grid curobo_core rclcpp::rclcpp)

  ament_add_gtest(test_fk_services cpp/curobo_core/test/test_fk_services.cpp)
  target_link_libraries(test_fk_services curobo_core rclcpp::rclcpp ${curobov2_ros_interfaces_TARGETS})

  ament_add_gtest(test_ik_services cpp/curobo_core/test/test_ik_services.cpp)
  target_link_libraries(test_ik_services curobo_core rclcpp::rclcpp ${curobov2_ros_interfaces_TARGETS})

  ament_add_gtest(test_attachment_services cpp/curobo_core/test/test_attachment_services.cpp)
  target_link_libraries(test_attachment_services curobo_core rclcpp::rclcpp ${curobov2_ros_interfaces_TARGETS} ${std_srvs_TARGETS})

  ament_add_gtest(test_collision_distance cpp/curobo_core/test/test_collision_distance.cpp)
  target_link_libraries(test_collision_distance curobo_core rclcpp::rclcpp ${curobov2_ros_interfaces_TARGETS})

  ament_add_gtest(test_collision_diagnostics cpp/curobo_core/test/test_collision_diagnostics.cpp)
  target_link_libraries(test_collision_diagnostics curobo_core rclcpp::rclcpp ${curobov2_ros_interfaces_TARGETS})

  ament_add_gtest(test_unified_planner_node cpp/curobo_core/test/test_unified_planner_node.cpp)
  target_link_libraries(test_unified_planner_node curobo_core rclcpp::rclcpp ${curobov2_ros_interfaces_TARGETS})

  ament_add_gtest(test_planners cpp/curobo_core/test/test_planners.cpp)
  target_link_libraries(test_planners curobo_core)

  ament_add_gtest(test_reactive_controller cpp/curobo_core/test/test_reactive_controller.cpp)
  target_link_libraries(test_reactive_controller curobo_core rclcpp::rclcpp ${curobov2_ros_interfaces_TARGETS})

  ament_add_gtest(test_robot_strategy cpp/curobo_core/test/test_robot_strategy.cpp)
  target_link_libraries(test_robot_strategy curobo_core rclcpp::rclcpp ${curobov2_ros_interfaces_TARGETS})

  ament_add_gtest(test_trajectory_action cpp/curobo_core/test/test_trajectory_action.cpp)
  target_link_libraries(test_trajectory_action curobo_core rclcpp::rclcpp rclcpp_action::rclcpp_action ${curobov2_ros_interfaces_TARGETS})

endif()

ament_package()

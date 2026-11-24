cmake_minimum_required(VERSION 3.16)
project(graspgenx_ros)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(grasp_pose_detection_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

find_package(Python3 COMPONENTS Development.Embed REQUIRED)
find_package(pybind11 REQUIRED)

add_library(graspgenx_component SHARED src/graspgenx_component.cpp)
target_include_directories(graspgenx_component PRIVATE src)
target_link_libraries(graspgenx_component
  rclcpp::rclcpp
  rclcpp_components::component
  ${grasp_pose_detection_interfaces_TARGETS}
  ${sensor_msgs_TARGETS}
  ${geometry_msgs_TARGETS}
  ${std_msgs_TARGETS}
  ${visualization_msgs_TARGETS}
  Python3::Python
  pybind11::embed
)

# libpython is loaded by cuRobo's PyBridge; this library must resolve the same
# libpython at runtime (mirrors curobov2_ros/curobo_runtime).
get_filename_component(_py_lib_dir "${Python3_LIBRARIES}" DIRECTORY)
set_target_properties(graspgenx_component PROPERTIES
  CXX_VISIBILITY_PRESET default
  VISIBILITY_INLINES_HIDDEN OFF
  BUILD_RPATH "${_py_lib_dir}"
  INSTALL_RPATH "${_py_lib_dir};$ORIGIN"
)

rclcpp_components_register_nodes(
  graspgenx_component "graspgenx_ros::GraspGenXComponent")

install(TARGETS graspgenx_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_export_dependencies(
  rclcpp rclcpp_components grasp_pose_detection_interfaces sensor_msgs
  geometry_msgs std_msgs visualization_msgs)
ament_package()

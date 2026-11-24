cmake_minimum_required(VERSION 3.16)
project(cutamp_ros)

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
find_package(curobov2_ros_interfaces REQUIRED)
find_package(cutamp_ros_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)

find_package(Python3 COMPONENTS Development.Embed REQUIRED)
find_package(pybind11 REQUIRED)

add_library(cutamp_component SHARED src/cutamp_component.cpp)
target_include_directories(cutamp_component PRIVATE src)
target_link_libraries(cutamp_component
  rclcpp::rclcpp
  rclcpp_components::component
  ${curobov2_ros_interfaces_TARGETS}
  ${cutamp_ros_interfaces_TARGETS}
  ${geometry_msgs_TARGETS}
  Python3::Python
  pybind11::embed
)

# libpython is loaded by cuRobo's PyBridge; this library must resolve the same
# libpython at runtime (mirrors curobov2_ros/curobo_runtime).
get_filename_component(_py_lib_dir "${Python3_LIBRARIES}" DIRECTORY)
set_target_properties(cutamp_component PROPERTIES
  CXX_VISIBILITY_PRESET default
  VISIBILITY_INLINES_HIDDEN OFF
  BUILD_RPATH "${_py_lib_dir}"
  INSTALL_RPATH "${_py_lib_dir};$ORIGIN"
)

rclcpp_components_register_nodes(
  cutamp_component "cutamp_ros::CuTampComponent")

install(TARGETS cutamp_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# The Python helper must be importable by the embedded interpreter. Installing
# it into the colcon site-packages directory puts it on PYTHONPATH whenever the
# workspace's setup.bash is sourced.
set(PYTHON_INSTALL_DIR
  "lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")
install(DIRECTORY python/cutamp_ros DESTINATION ${PYTHON_INSTALL_DIR}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE)

install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_export_dependencies(
  rclcpp rclcpp_components curobov2_ros_interfaces cutamp_ros_interfaces
  geometry_msgs)
ament_package()

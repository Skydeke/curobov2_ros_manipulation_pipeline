cmake_minimum_required(VERSION 3.8)
project(isaac_ros_cumotion_rviz)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++20
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Qt5 boilerplate options from http://doc.qt.io/qt-5/cmake-manual.html
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rviz_common REQUIRED)

set(dependencies_pkgs
  rclcpp
  rclcpp_action
  rcl_interfaces
  std_msgs
  std_srvs
  isaac_ros_cumotion_interfaces
  rviz_common
  rviz_rendering
  rviz_default_plugins
  Qt5Core
  Qt5Gui
  Qt5Widgets
  geometry_msgs
  visualization_msgs
  interactive_markers
  nav2_msgs
  sensor_msgs
  trajectory_msgs
  urdf
  eigen3_cmake_module
  Eigen3
)

foreach(pkg IN LISTS dependencies_pkgs)
  find_package(${pkg} REQUIRED)
endforeach()

# build
include_directories(include)
set(library_name isaac_ros_cumotion_rviz)

# Define source file
set(${PROJECT_NAME}_SRCS
  src/add_objects_display.cpp
  src/add_objects_panel.cpp
  src/isaac_ros_cumotion_rviz.cpp
  src/arrow_interaction.cpp
  src/arrow_interaction_display.cpp
  src/sparse_voxel_grid_display.cpp
  src/curobo_trajectory_display.cpp
  src/reachability_map_display.cpp
  src/curobo_fk.cpp
  src/curobo_link_updater.cpp
)

# Define header file
set(${PROJECT_NAME}_HDRS
  include/${PROJECT_NAME}/isaac_ros_cumotion_rviz.hpp
  include/${PROJECT_NAME}/add_objects_display.hpp
  include/${PROJECT_NAME}/add_objects_panel.hpp
  include/${PROJECT_NAME}/arrow_interaction.hpp
  include/${PROJECT_NAME}/arrow_interaction_display.hpp
  include/${PROJECT_NAME}/sparse_voxel_grid_display.hpp
  include/${PROJECT_NAME}/node_spinner.hpp
  include/${PROJECT_NAME}/curobo_trajectory_display.hpp
  include/${PROJECT_NAME}/reachability_map_display.hpp
  include/${PROJECT_NAME}/curobo_fk.hpp
  include/${PROJECT_NAME}/curobo_link_updater.hpp
)

# Define ui file
set(${PROJECT_NAME}_UIS
  resource/isaac_ros_cumotion_rviz_panel.ui
  resource/add_object_panel.ui
)

# Generate header for ui
qt5_wrap_ui(${PROJECT_NAME}_UIS_H ${${PROJECT_NAME}_UIS})
qt5_wrap_cpp(${PROJECT_NAME}_MOCS ${${PROJECT_NAME}_HDRS})

add_library(${library_name} SHARED
  ${${PROJECT_NAME}_SRCS}
  ${${PROJECT_NAME}_UIS_H}
  ${${PROJECT_NAME}_MOCS} 
)
target_link_libraries(${library_name} Qt5::Widgets rviz_common::rviz_common rviz_default_plugins::rviz_default_plugins urdf::urdf
  Eigen3::Eigen)
ament_target_dependencies(${library_name} ${dependencies_pkgs})

# I prefer the Qt signals and slots to avoid defining "emit", "slots",
# etc because they can conflict with boost signals, so define QT_NO_KEYWORDS here
# e.g. http://muddyazian.blogspot.de/2012/04/getting-qt-app-working-with-boost-using.html
target_compile_definitions(${library_name} PRIVATE QT_NO_KEYWORDS)

# install
pluginlib_export_plugin_description_file(rviz_common rviz2_plugin.xml)

install(
  TARGETS ${library_name}
  EXPORT ${library_name}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(
  DIRECTORY include/
  DESTINATION include/
)

install(DIRECTORY
  launch
  rviz2
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)
ament_export_targets(${library_name} HAS_LIBRARY_TARGET)
ament_export_dependencies(${dependencies_pkgs})

ament_package()
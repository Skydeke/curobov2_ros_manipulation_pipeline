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
  # AddObjectsDisplay was removed: the objects it drew are still added server-side
  # and still listed (and removable) by SceneObjectsTab, which reads the server's
  # own get_scene_objects services rather than a display topic. See git log for the
  # removal if the 3D rendering of added obstacles is ever wanted back.
  #
  # AddObjectsPanel was removed too: it was the add form as a separate rviz panel,
  # stacked above the object list in a QSplitter, which is not a shape MoveIt's
  # Scene Objects tab has. Its form and its add_object call now live in MoveIt's
  # own "Add/Remove scene object(s)" group, inside scene_objects_tab.ui.
  src/target_display.cpp
  src/sparse_voxel_grid_display.cpp
  src/curobo_trajectory_display.cpp
  src/reachability_map_display.cpp
  src/curobo_fk.cpp
  src/curobo_link_updater.cpp
  # The five tabs of the composite "curobo" panel, which mirror MoveIt's Motion
  # Planning panel. planning_tab.cpp is the EXISTING planner-args panel (formerly
  # the standalone RvizArgsPanel), reused as the Planning tab; the other four are
  # panel-native widgets. CuroboPanel itself is the only registered panel --
  # PlanningTab deliberately has no PLUGINLIB_EXPORT_CLASS, so there is no way
  # to add it as a dock of its own.
  src/curobo_panel.cpp
  src/context_tab.cpp
  src/planning_tab.cpp
  src/joints_tab.cpp
  src/manipulation_tab.cpp
  src/scene_objects_tab.cpp
  # MoveIt's draggable joint-value bar, ported. Needs moc (it declares Q_OBJECT),
  # hence its presence in HDRS as well as SRCS.
  src/progress_bar_delegate.cpp
)

# Define header file
set(${PROJECT_NAME}_HDRS
  include/${PROJECT_NAME}/target_display.hpp
  include/${PROJECT_NAME}/sparse_voxel_grid_display.hpp
  include/${PROJECT_NAME}/node_spinner.hpp
  include/${PROJECT_NAME}/curobo_trajectory_display.hpp
  include/${PROJECT_NAME}/reachability_map_display.hpp
  include/${PROJECT_NAME}/curobo_fk.hpp
  include/${PROJECT_NAME}/curobo_link_updater.hpp
  # Listed here, not only in SRCS, so qt5_wrap_cpp generates a moc TU for each:
  # all six have Q_OBJECT, and the vtable is emitted in the moc file. Omitting
  # a header surfaces at link time as "undefined symbol: vtable for ...".
  include/${PROJECT_NAME}/curobo_panel.hpp
  include/${PROJECT_NAME}/context_tab.hpp
  include/${PROJECT_NAME}/planning_tab.hpp
  include/${PROJECT_NAME}/joints_tab.hpp
  include/${PROJECT_NAME}/manipulation_tab.hpp
  include/${PROJECT_NAME}/scene_objects_tab.hpp
  include/${PROJECT_NAME}/progress_bar_delegate.hpp
)

# Define ui file
# Three of the five tabs are a transcription of the corresponding MoveIt tab page
# (motion_planning_rviz_plugin_frame.ui) rather than a hand-built layout, so the
# shape is the upstream one by construction and the deviations live in the
# comment at the top of each file:
#   manipulation_tab.ui     MoveIt's Manipulation tab grid
#   scene_objects_tab.ui   MoveIt's Scene Objects tab two columns
#   planning_tab.ui        MoveIt's Planning tab groups
# Context and Joints build their widgets in code (the Joints table is a QTreeWidget
# whose value column is painted by ProgressBarDelegate, which a .ui cannot express).
set(${PROJECT_NAME}_UIS
  resource/planning_tab.ui
  resource/manipulation_tab.ui
  resource/scene_objects_tab.ui
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

# CuroboPanel embeds another rviz panel inside a tab widget, so an undefined
# vtable or a missing PLUGINLIB_EXPORT_CLASS symbol would otherwise surface only
# as an opaque "undefined symbol" when rviz dlopens this library. Let the linker
# name the symbol at build time instead.
target_link_options(${library_name} PRIVATE "-Wl,--no-undefined")

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
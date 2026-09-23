cmake_minimum_required(VERSION 3.16)
project(curobo_task_constructor_rviz)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(curobo_task_constructor_interfaces REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rviz_common REQUIRED)
find_package(visualization_msgs REQUIRED)

# Qt: this rviz stack is Qt5-based — link Qt5::Widgets explicitly, the same
# way isaac_ros_cumotion_rviz (sibling in this workspace) and upstream
# moveit_task_constructor_visualization do. There is no rviz_qt package.
find_package(Qt5 REQUIRED COMPONENTS Widgets)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

# The panel header carries Q_OBJECT, so the class's Qt meta-object symbols
# (staticMetaObject, metaObject(), qt_metacall, tr) — and critically the
# class VTABLE — only exist in the moc output: Q_OBJECT declares
# `virtual const QMetaObject * metaObject() const`, making it the class's key
# function, so the vtable is emitted in the TU that DEFINES it (the moc file),
# not in the .cpp. AUTOMOC alone only mocs a header that is listed in the
# target sources or sits beside a same-named source; this header lives at
# include/curobo_task_constructor_rviz/ (different path from src/), so moc it
# explicitly — exactly like the sibling isaac_ros_cumotion_rviz does for every
# one of its headers. WITHOUT this the library links (or build-links) without
# a vtable / staticMetaObject and rviz dlopen fails on
# "undefined symbol: vtable for ...::TaskConstructorPanel".
set(${PROJECT_NAME}_MOCS "")
set(${PROJECT_NAME}_HDRS
  include/curobo_task_constructor_rviz/task_constructor_panel.hpp
)
qt5_wrap_cpp(${PROJECT_NAME}_MOCS ${${PROJECT_NAME}_HDRS})

add_library(${PROJECT_NAME} SHARED
  src/task_constructor_panel.cpp
  ${${PROJECT_NAME}_MOCS}
)

target_include_directories(${PROJECT_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
ament_target_dependencies(${PROJECT_NAME}
  curobo_task_constructor_interfaces
  pluginlib
  rclcpp
  rclcpp_action
  rviz_common
  visualization_msgs
)
target_link_libraries(${PROJECT_NAME} Qt5::Widgets)

# A genuinely missing symbol must fail the BUILD, not ship a silently broken
# .so: an undefined vtable (e.g. from a stale/partial build) currently
# surfaces only at rviz dlopen time as "undefined symbol: vtable for
# curobo_task_constructor_rviz::TaskConstructorPanel". With --no-undefined the
# linker names the exact missing symbol plus the object that needs it.
target_link_options(${PROJECT_NAME} PRIVATE "-Wl,--no-undefined")

# NOTE: no VERSION property on the target. The sibling rviz package ships an
# unversioned libfoo.so, and pluginlib dlopens exactly that name. Setting a
# bare (empty-string) VERSION property makes CMake emit `libfoo.so.` instead,
# which class_loader cannot resolve as libfoo.so — the panel then fails to
# load with "Could not find library corresponding to plugin ...".

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY rviz
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

pluginlib_export_plugin_description_file(rviz_common plugin_description.xml)

ament_export_include_directories(include)
ament_export_dependencies(ament_cmake)
ament_export_dependencies(curobo_task_constructor_interfaces)
ament_export_dependencies(pluginlib)
ament_export_dependencies(rclcpp)
ament_export_dependencies(rclcpp_action)
ament_export_dependencies(rviz_common)
ament_export_dependencies(visualization_msgs)
ament_package()
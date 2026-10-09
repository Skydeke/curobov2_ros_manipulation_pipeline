cmake_minimum_required(VERSION 3.8)
project(curobo_task_constructor_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

set(msg_files
  "msg/Property.msg"
  "msg/StageDescription.msg"
  "msg/StageStatistics.msg"
  "msg/SolutionInfo.msg"
  "msg/SubSolution.msg"
  "msg/SubTrajectory.msg"
  "msg/Solution.msg"
  "msg/TaskDescription.msg"
  "msg/TaskStatistics.msg"
  "msg/TrajectoryExecutionInfo.msg"
)

set(srv_files
  "srv/GetSolution.srv"
)

set(action_files
  "action/ExecuteTaskSolution.action"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  ${action_files}
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs visualization_msgs trajectory_msgs moveit_msgs builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()

cmake_minimum_required(VERSION 3.8)
project(isaac_ros_cumotion_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/Ik.srv"
  "srv/IkBatch.srv"
  "srv/Fk.srv"
  "srv/FkBatch.srv"
  "srv/AddObject.srv"
  "srv/AttachObject.srv"
  "srv/RemoveObject.srv"
  "srv/SetMask.srv"
  "msg/ObjectParameters.msg"
  "msg/SparseVoxelGrid.msg"
  "action/SendTrajectory.action"
  "srv/TrajectoryGeneration.srv"
  "srv/GetVoxelGrid.srv"
  "srv/GetCollisionDistance.srv"
  "srv/SetPlanner.srv"
  "srv/SetLinkCollision.srv"
  "srv/GetPlanners.srv"
  "srv/SetCollisionCache.srv"
  "srv/SetRobotStrategy.srv"
  "srv/GetRobotStrategies.srv"
  "srv/WarmupIK.srv"
  "srv/WarmupFK.srv"
  "msg/MpcCosts.msg"
  DEPENDENCIES geometry_msgs nav2_msgs sensor_msgs std_msgs trajectory_msgs moveit_msgs
)

ament_package()

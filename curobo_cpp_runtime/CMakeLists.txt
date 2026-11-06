cmake_minimum_required(VERSION 3.16)
project(curobo_cpp_runtime)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# The pure selection/artifact layer has no ROS or CUDA dependency. It is a
# separate static library so its focused tests can build on a machine without
# GPU libraries (`-DCUROBO_CPP_SELECTION_ONLY=ON`).
option(
  CUROBO_CPP_SELECTION_ONLY
  "Build only the artifact/runtime-selection library and its tests" OFF)

add_library(curobo_cpp_selection STATIC
  src/artifact_bundle.cpp
  src/runtime_selection.cpp
)
target_include_directories(curobo_cpp_selection PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
set_target_properties(curobo_cpp_selection PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

if(NOT CUROBO_CPP_SELECTION_ONLY)
  find_package(rclcpp REQUIRED)
  find_package(rclcpp_action REQUIRED)
  find_package(tf2_ros REQUIRED)
  find_package(tf2 REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(std_srvs REQUIRED)
  find_package(control_msgs REQUIRED)
  find_package(trajectory_msgs REQUIRED)
  find_package(visualization_msgs REQUIRED)
  find_package(builtin_interfaces REQUIRED)
  find_package(nav2_msgs REQUIRED)
  find_package(Eigen3 REQUIRED)
  find_package(curobov2_ros_interfaces REQUIRED)
  find_package(curobov2_ros REQUIRED)

  set(CUROBO_ROS_LIBS ${curobov2_ros_LIBRARIES})
  if(NOT CUROBO_ROS_LIBS)
    set(CUROBO_ROS_LIBS curobo_runtime curobo_core)
  endif()

  # -------------------------------------------------------------------------
  # Prebuilt native cpp_runtime. It is built by standalone CMake in the image;
  # it is deliberately NOT a colcon package.
  # -------------------------------------------------------------------------
  set(CUROBO_CPP_ROOT "/opt/curobo/native" CACHE PATH
      "Install prefix of the prebuilt cpp_runtime libraries")
  set(CUROBO_CPP_INCLUDE_DIR "${CUROBO_CPP_ROOT}/include" CACHE PATH
      "Headers for curobo_robot_runtime / curobo_mapper_runtime")

  if(NOT EXISTS "${CUROBO_CPP_INCLUDE_DIR}/curobo_robot_runtime/motion_planner.h")
    # In-tree development: use the merged cuRobo submodule. Once the pipeline
    # image installs headers under /opt/curobo/native/include this branch is not
    # taken. This keeps the ROS package outside the curobo source tree while
    # still compiling in a checkout.
    set(_repo_native_include
        "${CMAKE_CURRENT_SOURCE_DIR}/../curobo/curobo/cpp_runtime/robot/include")
    if(EXISTS "${_repo_native_include}/curobo_robot_runtime/motion_planner.h")
      set(CUROBO_CPP_INCLUDE_DIR "${_repo_native_include}")
    endif()
  endif()
  if(NOT EXISTS "${CUROBO_CPP_INCLUDE_DIR}/curobo_robot_runtime/motion_planner.h")
    message(FATAL_ERROR
      "curobo_robot_runtime headers were not found. Set "
      "CUROBO_CPP_INCLUDE_DIR (default ${CUROBO_CPP_ROOT}/include).")
  endif()

  find_library(CUROBO_ROBOT_RUNTIME curobo_robot_runtime
               PATHS ${CUROBO_CPP_ROOT}/lib)
  find_library(CUROBO_MAPPER_RUNTIME curobo_mapper_runtime
               PATHS ${CUROBO_CPP_ROOT}/lib)
  if(NOT CUROBO_ROBOT_RUNTIME)
    message(FATAL_ERROR
      "libcurobo_robot_runtime was not found under ${CUROBO_CPP_ROOT}/lib")
  endif()
  if(NOT CUROBO_MAPPER_RUNTIME)
    message(FATAL_ERROR
      "libcurobo_mapper_runtime was not found under ${CUROBO_CPP_ROOT}/lib")
  endif()

  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    set(_cudart_target CUDA::cudart)
    set(_cuda_include ${CUDAToolkit_INCLUDE_DIRS})
    set(_cuda_lib_root ${CUDAToolkit_LIBRARY_ROOT})
  else()
    find_path(CUDA_INCLUDE_DIR cuda_runtime_api.h PATHS
              /usr/local/cuda/include /opt/cuda/include)
    find_library(CUDART_LIBRARY cudart PATHS
                 /usr/local/cuda/lib64 /opt/cuda/lib64)
    if(NOT CUDA_INCLUDE_DIR OR NOT CUDART_LIBRARY)
      message(FATAL_ERROR
        "CUDA runtime headers/library were not found (needed by cpp_runtime).")
    endif()
    set(_cudart_target ${CUDART_LIBRARY})
    set(_cuda_include ${CUDA_INCLUDE_DIR})
    get_filename_component(_cuda_lib_root "${CUDART_LIBRARY}" DIRECTORY)
  endif()

  # cpp_runtime loads CUDA driver entry points (cuInit/cuLaunchKernel/...).
  # Docker builds have the toolkit stubs but no runtime driver, so link the
  # stub explicitly. At runtime the deployed driver provides libcuda.so.1.
  find_library(CUDA_DRIVER_LIBRARY cuda
    PATHS
      "${_cuda_lib_root}/stubs"
      /usr/local/cuda/lib64/stubs
      /usr/local/cuda/lib/stubs
    NO_DEFAULT_PATH)
  if(NOT CUDA_DRIVER_LIBRARY)
    find_library(CUDA_DRIVER_LIBRARY cuda)
  endif()
  if(NOT CUDA_DRIVER_LIBRARY)
    message(FATAL_ERROR
      "libcuda (driver API) was not found; install the CUDA toolkit stubs "
      "(libcuda.so under <cuda>/lib64/stubs).")
  endif()
  message(STATUS "curobo_cpp_runtime: CUDA driver library ${CUDA_DRIVER_LIBRARY}")

  target_compile_definitions(curobo_cpp_selection PUBLIC
    CUROBO_CPP_RUNTIME_HAVE_CUDA=1)
  target_include_directories(curobo_cpp_selection PUBLIC ${_cuda_include})
  target_link_libraries(curobo_cpp_selection PUBLIC ${_cudart_target})

  add_library(curobo_cpp_runtime SHARED
    src/cpp_bridge.cpp
    src/native_scene.cpp
    src/runtime_selector.cpp
  )
  target_include_directories(curobo_cpp_runtime PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CUROBO_CPP_INCLUDE_DIR}>
    ${curobov2_ros_INCLUDE_DIRS}
  )
  target_link_libraries(curobo_cpp_runtime PUBLIC
    curobo_cpp_selection
    ${CUROBO_ROS_LIBS}
    rclcpp::rclcpp
    rclcpp_action::rclcpp_action
    tf2_ros::tf2_ros
    tf2::tf2
    Eigen3::Eigen
    ${geometry_msgs_TARGETS}
    ${sensor_msgs_TARGETS}
    ${std_srvs_TARGETS}
    ${control_msgs_TARGETS}
    ${trajectory_msgs_TARGETS}
    ${visualization_msgs_TARGETS}
    ${builtin_interfaces_TARGETS}
    ${nav2_msgs_TARGETS}
    ${curobov2_ros_interfaces_TARGETS}
    ${CUROBO_ROBOT_RUNTIME}
    ${CUROBO_MAPPER_RUNTIME}
    ${_cudart_target}
    ${CUDA_DRIVER_LIBRARY}
  )
  set_target_properties(curobo_cpp_runtime PROPERTIES
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
    INSTALL_RPATH "$ORIGIN;${CUROBO_CPP_ROOT}/lib"
    BUILD_RPATH "${CUROBO_CPP_ROOT}/lib"
  )

  # The server executable. It installs the selected bridge before constructing
  # curobo_core::UnifiedPlannerNode; the ROS node name remains curobo_server.
  add_executable(curobo_trajectory_planner_cpp src/curobo_server_main.cpp)
  target_link_libraries(curobo_trajectory_planner_cpp
    curobo_cpp_runtime
    ${CUROBO_ROS_LIBS}
    rclcpp::rclcpp
  )
  set_target_properties(curobo_trajectory_planner_cpp PROPERTIES
    INSTALL_RPATH "$ORIGIN/..;${CUROBO_CPP_ROOT}/lib")

  install(TARGETS curobo_cpp_runtime
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )
  install(TARGETS curobo_trajectory_planner_cpp
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )
  install(DIRECTORY include/ DESTINATION include)
  ament_export_include_directories(include)
  ament_export_libraries(curobo_cpp_runtime)
  ament_export_dependencies(rclcpp curobov2_ros curobov2_ros_interfaces)
endif()

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_artifact_selection test/test_artifact_selection.cpp)
  target_link_libraries(test_artifact_selection curobo_cpp_selection)

  if(NOT CUROBO_CPP_SELECTION_ONLY)
    ament_add_gtest(test_cpp_bridge_contract test/test_cpp_bridge_contract.cpp)
    target_link_libraries(test_cpp_bridge_contract curobo_cpp_runtime)
  endif()
endif()

ament_package()

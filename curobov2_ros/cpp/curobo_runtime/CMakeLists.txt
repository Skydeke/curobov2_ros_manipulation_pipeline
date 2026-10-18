cmake_minimum_required(VERSION 3.16)
project(curobo_runtime)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(Threads REQUIRED)

# The interpreter that has torch and cuRobo: system Python 3.12 (decision Q2).
find_package(Python3 COMPONENTS Development.Embed REQUIRED)
find_package(pybind11 REQUIRED)

# ---------------------------------------------------------------------------
# libcurobo_runtime.so: Runtime singleton, Lanes, Barrier, GIL guards, Bridge.
# Defined in this library ONLY (design section 4 singleton rule): components and
# libcurobo_core.so link to it dynamically; never static-link it.
# ---------------------------------------------------------------------------
add_library(curobo_runtime SHARED
  src/gpu_lanes.cpp
  src/fake_bridge.cpp
  src/runtime.cpp
  src/py_bridge.cpp
)

target_include_directories(curobo_runtime PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(curobo_runtime PRIVATE
  Threads::Threads
  Python3::Python
  pybind11::embed
)

# The interpreter's libpython must be loadable with RTLD_GLOBAL or numpy/torch
# extension modules fail with undefined symbols (design section 5). The library
# name is passed to py_bridge.cpp, which dlopens it before creating the
# interpreter.
get_filename_component(_py_lib_dir "${Python3_LIBRARIES}" DIRECTORY)
get_filename_component(_py_lib_name "${Python3_LIBRARIES}" NAME)
target_compile_definitions(curobo_runtime PRIVATE
  CUROBO_PYTHON_LIB_NAME="${_py_lib_name}"
  CUROBO_PYTHON_LIB_DIR="${_py_lib_dir}"
)

# Default visibility: the runtime's public API must be globally reachable so the
# dlopen singleton test can find it.
set_target_properties(curobo_runtime PROPERTIES
  CXX_VISIBILITY_PRESET default
  VISIBILITY_INLINES_HIDDEN OFF
  BUILD_RPATH "${_py_lib_dir}"
  INSTALL_RPATH "${_py_lib_dir}"
)

install(TARGETS curobo_runtime
  EXPORT export_curobo_runtime
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_gpu_lanes test/test_gpu_lanes.cpp)
  target_link_libraries(test_gpu_lanes curobo_runtime)

  ament_add_gtest(test_bridge_import test/test_bridge_import.cpp)
  target_link_libraries(test_bridge_import curobo_runtime)

  ament_add_gtest(test_bridge_contract test/test_bridge_contract.cpp)
  target_link_libraries(test_bridge_contract curobo_runtime)

  # Two probe libraries built from one source, for the dlopen singleton test.
  add_library(runtime_probe_a SHARED test/runtime_probe.cpp)
  add_library(runtime_probe_b SHARED test/runtime_probe.cpp)
  target_link_libraries(runtime_probe_a PRIVATE curobo_runtime Python3::Python)
  target_link_libraries(runtime_probe_b PRIVATE curobo_runtime Python3::Python)

  ament_add_gtest(test_runtime_singleton test/test_runtime_singleton.cpp)
  target_link_libraries(test_runtime_singleton curobo_runtime ${CMAKE_DL_LIBS})
  target_compile_definitions(test_runtime_singleton PRIVATE
    PROBE_A_PATH="$<TARGET_FILE:runtime_probe_a>"
    PROBE_B_PATH="$<TARGET_FILE:runtime_probe_b>"
  )
  add_dependencies(test_runtime_singleton runtime_probe_a runtime_probe_b)
endif()

ament_export_include_directories(include)
ament_export_libraries(curobo_runtime)
ament_export_targets(export_curobo_runtime NAMESPACE curobo_runtime::)
ament_package()

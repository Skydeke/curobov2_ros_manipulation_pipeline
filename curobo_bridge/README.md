# curobo_bridge

ROS-free bridge between the C++ `curobov2_ros` port and the vendored Python
cuRobo package. It serializes robot config, scene and planning requests/results
across the pybind boundary (design section 6). No ROS nodes.

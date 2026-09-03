"""Command-line tools shipped with the package.

Unlike the nodes in ``isaac_ros_cumotion/core``, these are one-shot utilities run by hand
against an already-running planner (``ros2 run isaac_ros_cumotion <tool>``). They depend
only on rclpy and the isaac_ros_cumotion_interfaces interfaces.

Offline analysis tools (matplotlib/pandas, no ROS) stay in ``scripts/`` and are
not installed -- see ``scripts/plot_mpc_diag.py``.
"""

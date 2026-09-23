"""Core ROS-free framework (migration step 1 of the plan).

No ROS imports here on purpose: ``state`` / ``stage`` / ``container`` /
``robot`` / ``robot_config`` / ``registry`` are pure Python and unit-testable
against a mocked ``curobo_server``. The rclpy deployment adapter lives in
``curobo_task_constructor.robot``.
"""
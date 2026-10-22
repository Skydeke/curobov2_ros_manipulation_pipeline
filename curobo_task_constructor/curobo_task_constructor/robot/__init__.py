"""ROS 2 deployment adapter: the real curobo_server behind ``RobotInterface``. ``CuroboServerInterface`` converts the core's ROS-free
``PlanRequest``/``PlanResult``/``ObjectSpec`` types to/from
``curobov2_ros_interfaces`` messages and drives the curobo_server
services synchronously. It is imported by the action server
(``curobo_task_constructor.node``) — the pure-Python core in
``curobo_task_constructor.core`` never imports it, which is what keeps the
framework unit-testable against a mocked curobo_server."""

from curobo_task_constructor.robot.curobo import CuroboServerInterface

__all__ = ["CuroboServerInterface"]

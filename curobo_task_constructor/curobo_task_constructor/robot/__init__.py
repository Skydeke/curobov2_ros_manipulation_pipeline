"""ROS 2 deployment adapter: the real curobo_server behind ``RobotInterface``.

``CuroboServerInterface`` converts the core's ROS-free
``PlanRequest``/``PlanResult``/``ObjectSpec`` types to/from
``isaac_ros_cumotion_interfaces`` messages and drives the curobo_server
services synchronously. It is imported by the action server
(``curobo_task_constructor.node``) — the pure-Python core in
``curobo_task_constructor.core`` never imports it, which is what keeps the
framework unit-testable against a mocked curobo_server.

Deployment contract: this adapter blocks the calling thread on each
round-trip. The node must therefore run a MultiThreadedExecutor with at
least two threads — the worker thread executes the task solve while other
threads deliver service/action responses and /joint_states.
"""

from curobo_task_constructor.robot.curobo import CuroboServerInterface

__all__ = ["CuroboServerInterface"]

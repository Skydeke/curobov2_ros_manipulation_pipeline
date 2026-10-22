"""curobo_task_constructor — an open MTC-equivalent for cuRobo. Package layout
--------------
``curobo_task_constructor.core``
    Pure-python, ROS-free framework: InterfaceState / Solution / Stage /
    ContainerStage / StageRegistry plus the abstract RobotInterface the
    stages talk to (a mocked ``curobo_server`` in unit tests, a real rclpy
    client in deployment). This is migration step 1 of the plan: no ROS,
    unit-testable against a mocked curobo_server."""

from curobo_task_constructor.core.registry import STAGE_REGISTRY, register_stage

__version__ = "0.1.0"

__all__ = ["STAGE_REGISTRY", "register_stage", "__version__"]
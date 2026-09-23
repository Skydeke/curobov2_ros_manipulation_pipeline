"""curobo_task_constructor — an open MTC-equivalent for cuRobo.

Package layout
--------------
``curobo_task_constructor.core``
    Pure-python, ROS-free framework: InterfaceState / Solution / Stage /
    ContainerStage / StageRegistry plus the abstract RobotInterface the
    stages talk to (a mocked ``curobo_server`` in unit tests, a real rclpy
    client in deployment). This is migration step 1 of the plan: no ROS,
    unit-testable against a mocked curobo_server.

``curobo_task_constructor.stages``
    The built-in Stage subclasses (open catalog): CurrentState, FixedState,
    GenerateGraspPose, ComputeIK, MoveTo, MoveRelative, ModifyScene, Connect.

``curobo_task_constructor.graph``
    Declarative task description: StageSpec (mirror of
    curobo_task_constructor_interfaces/msg/StageSpec.msg), the registry-based
    spec->Stage tree builder and interface-adjacency validation.

``curobo_task_constructor.executor`` / ``node``
    The MTC ``Task::init()/plan()/execute()`` equivalent and the thin ROS 2
    action server that fronts it. ``node`` runs at
    ``/curobo_task_constructor/task`` and publishes the Sec. 7 introspection
    topics (TaskDescription / SolutionInfo / StageStatistics).

``curobo_task_constructor.robot``
    Deployment adapter (rclpy): ``CuroboServerInterface`` converts the core's
    ROS-free request/result types to/from ``isaac_ros_cumotion_interfaces``
    messages. The core never imports this module — that is what keeps it
    unit-testable against a mocked ``curobo_server``.
"""

from curobo_task_constructor.core.registry import STAGE_REGISTRY, register_stage

__version__ = "0.1.0"

__all__ = ["STAGE_REGISTRY", "register_stage", "__version__"]
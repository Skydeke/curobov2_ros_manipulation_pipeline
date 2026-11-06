"""Built-in stages (Sec. 2 of the plan; the catalog is open, not closed here).

Importing this package registers every builtin in ``STAGE_REGISTRY``. New
capabilities are added by writing a ``Stage`` subclass elsewhere and
decorating it with ``@register_stage`` — the executor and the wire format
never change.
"""

from curobo_task_constructor.stages import (  # noqa: F401  (side-effect: registry)
    cartesian_path,
    cartesian_waypoints,
    compute_ik,
    connect,
    current_state,
    fixed_state,
    generate_grasp_pose,
    generate_pose,
    gripper_command,
    modify_scene,
    move_relative,
    move_to,
    multi_tool_cartesian_path,
    multi_tool_move_to,
    multi_tool_waypoints,
)

from curobo_task_constructor.stages.cartesian_waypoints import (  # noqa: F401
    CartesianWaypoints,
)
from curobo_task_constructor.stages.multi_tool_cartesian_path import (  # noqa: F401
    MultiToolCartesianPath,
)
from curobo_task_constructor.stages.multi_tool_move_to import (  # noqa: F401
    MultiToolMoveTo,
)
from curobo_task_constructor.stages.multi_tool_waypoints import (  # noqa: F401
    MultiToolWaypoints,
)
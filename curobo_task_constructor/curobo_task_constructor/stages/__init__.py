"""Built-in stages (Sec. 2 of the plan; the catalog is open, not closed here).

Importing this package registers every builtin in ``STAGE_REGISTRY``. New
capabilities are added by writing a ``Stage`` subclass elsewhere and
decorating it with ``@register_stage`` — the executor and the wire format
never change.
"""

from curobo_task_constructor.stages import (  # noqa: F401  (side-effect: registry)
    compute_ik,
    connect,
    current_state,
    fixed_state,
    generate_grasp_pose,
    modify_scene,
    move_relative,
    move_to,
)
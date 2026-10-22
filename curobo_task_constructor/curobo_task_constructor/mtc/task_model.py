
from __future__ import annotations

from typing import Any, Optional

#: Introspection namespace default (MTC task ns; absolute default kept for compat).
#: MTC uses RELATIVE topics ``description/statistics/solution`` in the task's
#: namespace plus service ``get_solution_<task_id>``. This package keeps the
#: absolute default ``/curobo_task_constructor/...`` but honours ``Task(ns)``:
#: ``Task(ns="my_ns")`` publishes ``my_ns/description`` etc. and serves
#: ``my_ns/get_solution_<task_id>`` — see ``Task._ns_base``.

from curobo_task_constructor.mtc.constants import *
from curobo_task_constructor.mtc.ros_runtime import *
from curobo_task_constructor.mtc.planners import *
from curobo_task_constructor.mtc.containers import *
from curobo_task_constructor.mtc.introspection_support import *
from curobo_task_constructor.mtc.introspection import *

class _TaskModelMixin:
    def introspection(self) -> Any:
        """Access introspection object (MTC ``Task.introspection``).

        MTC's ``Task::introspection()`` (task.cpp:176-180) calls
        ``enableIntrospection(true)`` first, so reaching for it both creates
        and enables it — which is exactly what the ``Task.publish`` Python
        binding relies on (``core/python/bindings/src/core.cpp:481``). Created
        on first use and owned by the Task.
        """
        if self._introspection is None:
            self.enableIntrospection(True)
        return self._introspection

    def loadRobotModel(self, robot_description: str = "robot_description") -> None:
        """Point this task at the ROBOT DESCRIPTOR (curobo's loadRobotModel). MTC reads the ``robot_description`` parameter into a
        ``RobotModelLoader`` and throws when that fails
        (``task.cpp:138-144``), because without the model there are no
        kinematics to solve with. cuRobo has no client-side kinematics to
        load — but the core still reads two things from the robot
        DESCRIPTOR itself:

        * ``kinematics.cspace.joint_names`` — the canonical cspace order
          that ``CuroboServerInterface.normalize_joint_state`` reorders
          ``/joint_states`` into before anything reaches the server."""
        path = self._resolve_robot_descriptor(robot_description)
        if path is None:
            return
        if self.robot is None:
            # MTC's loadRobotModel builds the loader; here the robot is a
            # RobotInterface, and it needs a node (which the introspection
            # owns). Without one there is nothing to point at.
            node = self._introspection_node
            if node is None:
                try:
                    import rclpy

                    node = rclpy.create_node(f"introspection_{self.task_id}")
                    _IntrospectionExecutor.add_node(node)
                except Exception:
                    node = None
            if node is None:
                return
            from curobo_task_constructor.robot.curobo import (
                CuroboServerInterface,
            )
            self.attach_robot(CuroboServerInterface(node,
                                                   robot_config_path=path))
            return
        # Interface already attached: give it the descriptor it was built
        # without, and re-derive the joint order (once).
        if getattr(self.robot, "_robot_config_path", None):
            return
        self.robot._robot_config_path = str(path)
        try:
            from curobo_task_constructor.core.robot_config import (
                canonical_joint_order,
            )
            self.robot._joint_order = canonical_joint_order(path)
        except Exception:
            self.robot._joint_order = None

    def _resolve_robot_descriptor(self, robot_description: Any) -> Any:
        """Descriptor path for ``loadRobotModel``, or None when unknown.

        Order: an explicit filesystem path in the argument wins; else the
        robot parameter on our introspection node (or the caller's, if one
        is still attached); else None (a caller with no descriptor anywhere
        gets the server default, exactly as before).
        """
        if isinstance(robot_description, str) and (
                "/" in robot_description or robot_description.endswith((".yml", ".yaml"))):
            return robot_description
        for node in (self._introspection_node, getattr(self, "_node", None)):
            if node is None:
                continue
            for name in ("robot_config_path", "robot"):
                try:
                    if not node.has_parameter(name):
                        continue
                    value = node.get_parameter(name).value
                except Exception:
                    continue
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    def setCostTerm(self, *args, **kwargs) -> None:
        """Specify a CostTerm for calculation of stage costs (MTC ``Task.setCostTerm``).

        cuRobo stages carry their own cost terms via ``setCost``; this
        stores the task-level term for inspection.
        """
        if args:
            self._properties["cost_term"] = args[0]

    # -- introspection plumbing (MTC Introspection-backed) --------------

__all__ = [
    '_TaskModelMixin',
]

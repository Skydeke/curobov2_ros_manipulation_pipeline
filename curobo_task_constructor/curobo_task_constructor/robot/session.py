"""RobotInterface adapter for the real curobov2_ros curobo_server (ROS 2). Converts the core's ROS-free ``PlanRequest``/``PlanResult``/``ObjectSpec``
types to/from ``curobov2_ros_interfaces`` messages and drives the
curobo_server services synchronously. ``joint_state_cls``/``pose_cls`` are
overridden so the stages emit real message types on the wire."""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Any, Optional

from geometry_msgs.msg import Point, Pose as RosPose, Quaternion, Vector3
from control_msgs.action import GripperCommand as GripperCommandAction
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from curobov2_ros_interfaces.action import SendTrajectory
from curobov2_ros_interfaces.msg import Goalset, TrajectoryGoal
from curobov2_ros_interfaces.srv import (
    AddObject,
    AttachObject,
    Fk,
    GetSceneObjects,
    Ik,
    RemoveObject,
    SetLinkCollision,
    SetPlanner,
    TrajectoryGeneration,
    TrajectoryGenerationBatch,
)

from curobo_task_constructor.core.geom import Pose3, pose_to_any
from curobo_task_constructor.core.state import ObjectSpec
from curobo_task_constructor.core.robot import (
    GoalsetSpec,
    PlanRequest,
    PlanResult,
    RobotInterface,
    ServiceError,
)
from curobo_task_constructor.core.robot_config import (
    NamedJointConfig,
    canonical_joint_order,
    load_robot_descriptor,
    reorder_joint_vectors,
)



class PlannerSessionState:
    """Sticky planner/collision session: skip identical setup round trips. Consecutive motion legs usually re-send the same configuration — three
    best-of-3 attempts of one request, back-to-back same-planner legs — so
    remembering the last sent state removes those round trips. The server
    mutates both states on its own during execution replay (segments run
    under their cached plans' planners) and any failed call leaves the
    state unknown, so ``invalidate()`` forgets everything: ``execute()``
    and every ``ServiceError`` path call it, and the next request re-sends
    unconditionally."""

    def __init__(self):
        self._planner_key = None
        self._planner_known = False
        self._collision_key = None
        self._collision_known = False

    def planner_request_needed(self, key) -> bool:
        """True when a set_planner(key) round trip must go out."""
        return not self._planner_known or key != self._planner_key

    def note_planner_sent(self, key) -> None:
        self._planner_key = key
        self._planner_known = True

    def collision_transition_needed(self, links, enabled: bool) -> bool:
        """True when a set_link_collision(links, enabled) must go out.

        An empty link set never needs a call (the adapter skips it) and
        never changes the remembered state.
        """
        if not links:
            return False
        key = (tuple(sorted(links)), bool(enabled))
        return not self._collision_known or key != self._collision_key

    def note_collision_sent(self, links, enabled: bool) -> None:
        if not links:
            return
        self._collision_key = (tuple(sorted(links)), bool(enabled))
        self._collision_known = True

    def invalidate(self) -> None:
        """Forget everything (execute drove, or a call failed)."""
        self._planner_known = False
        self._collision_known = False

#: ObjectSpec.shape -> AddObject type constant (mirror of the srv).
_ADD_OBJECT_SHAPES = {
    "cuboid": AddObject.Request.CUBOID,
    "box": AddObject.Request.CUBOID,
    "sphere": AddObject.Request.SPHERE,
    "cylinder": AddObject.Request.CYLINDER,
    "capsule": AddObject.Request.CAPSULE,
    "mesh": AddObject.Request.MESH,
}
#: Default service round-trip budget (s). The FIRST solve after a planner
#: switch / solver rebuild re-records the ESDF CUDA graph and compiles kernels
#: on top of the planning itself, so a tight cap races legitimate cold-start
#: latency and kills otherwise-good plans at the call boundary. 60s is generous
#: yet still bounded — parity with the C++ client's kDefaultServiceTimeoutSeconds
#: (60s). Overridable per node via the `service_timeout` ROS parameter.
_SERVICE_TIMEOUT = 60.0  # s per call

#: How long a `get_scene_objects` reply is reused before asking again. The
#: executor's reverse sync walks every object in the scene, so an uncached query
#: per object would be one service call per object per stage. Short enough that a
#: stage's own `add_object` is visible to its next `get_object_pose`; the add path
#: invalidates the cache outright, so this bound only governs changes made by
#: OTHER clients.
_SCENE_CACHE_SECONDS = 1.0

#: Per-joint tolerance (rad) under which a joint-space goal counts as already
#: reached (see _trivial_goal). 1e-6 was tighter than encoder reality: live
#: joint readings routinely sit ~1e-3..1e-4 off a constant goal (the finger
#: at 0.001 vs a requested 0.000), so no-op legs — ready/open/close issued
#: while already there — missed the fast path and each burned a full
#: ~300 ms trajopt solve. 1e-3 rad (~0.06 deg) is far above float and
#: encoder noise yet an order of magnitude below MoveIt's 0.01 rad
#: allowed_start_tolerance: anything skipped here is physically no motion.
_TRIVIAL_GOAL_TOLERANCE = 1e-3

#: GetSceneObjects type constant -> the `ObjectSpec.shape` spelling the rest of
#: the task constructor uses. The constants are AddObject's; rosidl cannot import
#: constants across .srv files, so the reply carries them and this maps them.
#: An unknown value falls back to "cuboid", which is AddObject's own default and
#: the most likely wrong answer to be harmless for (a wrong shape only changes how
#: a re-added object is sized).
_SCENE_SHAPES = {
    0: "cuboid",
    1: "sphere",
    2: "capsule",
    3: "cylinder",
    4: "mesh",
}

__all__ = [
    'PlannerSessionState',
    '_ADD_OBJECT_SHAPES',
    '_SERVICE_TIMEOUT',
    '_SCENE_CACHE_SECONDS',
    '_TRIVIAL_GOAL_TOLERANCE',
    '_SCENE_SHAPES',
]

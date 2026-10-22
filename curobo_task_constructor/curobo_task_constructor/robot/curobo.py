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
    get_gripper_controllers,
    reorder_joint_vectors,
)



from curobo_task_constructor.robot.session import *
from curobo_task_constructor.robot.msg import *
from curobo_task_constructor.robot.plan_client import *
from curobo_task_constructor.robot.scene_client import *

class CuroboServerInterface(_MsgMixin, _PlanClientMixin, _SceneClientMixin, RobotInterface):
    """Blocking rclpy client for the curobo_server services + action."""

    joint_state_cls = JointState
    pose_cls = RosPose

    def __init__(
        self,
        node,
        planner: Optional[int] = None,
        planner_service: Optional[str] = None,
        robot_config_path: Optional[str] = None,
        service_timeout: Optional[float] = None,
    ):
        #: the owning rclpy Node (clients are created on it)
        self._node = node
        self._planner = planner
        self._planner_service = planner_service or "/curobo_server/set_planner"
        #: robot descriptor YAML (named_joint_configs section), if any
        self._robot_config_path = robot_config_path
        #: canonical cspace joint order derived from the descriptor (None = no
        #: joint-order knowledge; joint-state readings pass through verbatim)
        self._joint_order = (
            canonical_joint_order(robot_config_path) if robot_config_path else None
        )
        #: dedicated gripper controllers (MoveIt-style schema from the robot
        #: descriptor): ``{name: normalized cfg}``. The task names one per
        #: command; blank names fall back to the first entry.
        self._grippers = (
            get_gripper_controllers(robot_config_path) if robot_config_path else {}
        )
        self._default_gripper = next(iter(self._grippers), "")
        self._gripper_clients: dict = {}
        #: once-flag for the reorder log in ``normalize_joint_state``
        self._joint_order_logged = False
        #: per-call service/action budget (see _SERVICE_TIMEOUT)
        self._service_timeout = (
            service_timeout if service_timeout is not None else _SERVICE_TIMEOUT
        )
        self._planner_active = False
        #: SetPlanner enum of the planner currently active on the server (None
        #: until the first explicit switch). ``_ensure_planner`` re-issues a
        #: switch when a different planner is requested, so a task may use
        #: per-stage planners (joint moves -> JOINT_SPACE, lift pose -> CLASSIC).
        self._active_planner = None
        #: Sticky setup session (see PlannerSessionState): identical
        #: set_planner / set_link_collision repeats are skipped; execute()
        #: and every failed call invalidate it.
        self._session = PlannerSessionState()
        #: Re-entrant allowance depth, per planning thread (see
        #: allowance_scope): nested OFF/ON pairs around one request's
        #: attempts compose into a single OFF...ON instead of flapping.
        self._allowance_tls = threading.local()

        self._traj_client = self._client(
            TrajectoryGeneration, "/curobo_server/generate_trajectory"
        )
        self._batch_client = self._client(
            TrajectoryGenerationBatch, "/curobo_server/trajectory_generation_batch"
        )
        self._ik_client = self._client(Ik, "/curobo_server/ik")
        self._fk_client = self._client(Fk, "/curobo_server/fk")
        self._add_client = self._client(AddObject, "/curobo_server/add_object")
        self._remove_client = self._client(RemoveObject, "/curobo_server/remove_object")
        self._remove_all_client = self._client(
            Trigger, "/curobo_server/remove_all_objects"
        )
        self._scene_objects_client = self._client(
            GetSceneObjects, "/curobo_server/get_scene_objects"
        )
        self._attach_client = self._client(AttachObject, "/curobo_server/attach_object")
        self._detach_client = self._client(Trigger, "/curobo_server/detach_object")
        self._planner_client = self._client(SetPlanner, self._planner_service)
        self._link_collision_client = self._client(
            SetLinkCollision, "/curobo_server/set_link_collision"
        )
        self._exec_client = ActionClient(
            node, SendTrajectory, "/curobo_server/execute_trajectory"
        )

        #: (label, client) pairs polled by ``start_ready_poll`` — everything
        #: the interface touches, minus the warm-up (see that method's doc):
        #: the warm-up batch size is owned by the client that configures it.
        self._ready_services = [
            ("set_planner", self._planner_client),
            ("set_link_collision", self._link_collision_client),
            ("generate_trajectory", self._traj_client),
            ("trajectory_generation_batch", self._batch_client),
            ("ik", self._ik_client),
            ("fk", self._fk_client),
            ("add_object", self._add_client),
            ("remove_object", self._remove_client),
            ("remove_all_objects", self._remove_all_client),
            ("attach_object", self._attach_client),
            ("detach_object", self._detach_client),
            ("get_scene_objects", self._scene_objects_client),
        ]

        # Last successful `get_scene_objects` reply, as {name: object record}. A
        # CACHE of the server, not a record of this interface's own writes: it is
        # replaced wholesale on every successful query, so it describes the whole
        # scene including objects another client added. Empty until the first
        # query succeeds -- deliberately, because an empty dict here must never
        # be mistaken for "the scene is empty"; see `_query_scene_objects`.
        self._scene_objects: dict = {}
        self._scene_objects_at: float = 0.0
        self._scene_objects_queried: bool = False

        #: Plan-time scene preview (see `plan_scene`): the attach state the
        #: current plan burst has previewed on the shared server, and the state
        #: it must be restored to when the burst ends. ``_plan_scene_active``
        #: is True only between the first `plan_scene` and `end_plan_scene`.
        self._plan_scene_active: bool = False
        self._plan_scene_original: list = []
        self._plan_scene_attached = None

    def _client(self, srv_cls, name):
        # Creation is non-blocking — the node starts even when the curobo
        # server is down, and per-call `wait_for_service` in `_call` reports
        # unavailability when a task actually needs the service.
        return self._node.create_client(srv_cls, name)

    # ------------------------------------------------------------------
    # startup readiness
    # ------------------------------------------------------------------
    def start_ready_poll(self, on_ready, period=0.5):
        """Non-blocking startup watchdog (the viser nodes' ``start_service_poll``
        pattern): poll the curobo_server services + the SendTrajectory action
        on a wall timer instead of blocking during construction. As soon as
        every client is reachable the timer is cancelled and ``on_ready()`` is
        invoked exactly once. Discovery progresses normally while the node's executor is spinning,
        so there is no hand-rolled ``rclpy.spin_once`` and no blocking wait in
        ``__init__``."""
        state = {"done": False, "timer": None}

        def _tick():
            if state["done"]:
                return
            for what, client in self._ready_services:
                if not client.service_is_ready():
                    return
            if not self._exec_client.server_is_ready():
                return
            state["done"] = True
            state["timer"].cancel()
            self._node.get_logger().info(
                "curobo_server services + execute_trajectory action reachable"
            )
            try:
                on_ready()
            except Exception as exc:  # noqa: BLE001 - see _await repr notes
                self._node.get_logger().error(f"readiness handler failed: {exc}")

        state["timer"] = self._node.create_timer(period, _tick)

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    def normalize_joint_state(self, joint_state):
        """Reorder a joint-state reading into the server's cspace order.

        ``/joint_states`` order is publisher-defined (the kortex sim emits the
        finger joint FIRST); the server resolves ``start_pose`` position lists
        VERBATIM in cspace order (``kinematics.cspace.joint_names``), so an
        un-reordered reading misaligns every joint of the start state. Readings
        are reordered by name into the canonical order derived from the robot
        descriptor; when the descriptor carries no joint order — or the reading
        is already in canonical-prefix order — it passes through unchanged.
        """
        order = self._joint_order
        names = list(getattr(joint_state, "name", []) or [])
        if not order or not names or names == order[: len(names)]:
            return joint_state
        position = list(getattr(joint_state, "position", []) or [])
        if len(position) != len(names):
            # malformed reading — leave it for the server to reject as-is
            return joint_state
        new_names, new_position, new_velocity, new_effort = reorder_joint_vectors(
            order,
            names,
            position,
            velocity=getattr(joint_state, "velocity", None),
            effort=getattr(joint_state, "effort", None),
        )
        js = type(joint_state)()
        js.header = joint_state.header
        js.name = new_names
        js.position = new_position
        js.velocity = new_velocity
        js.effort = new_effort
        if not self._joint_order_logged:
            self._joint_order_logged = True
            self._node.get_logger().info(
                f"reordered /joint_states {dict(zip(names, position))} "
                f"into cspace order {dict(zip(new_names, new_position))}"
            )
        return js

    def get_current_joint_state(self):
        # The task constructor starts from the newest /joint_states reading.
        # The driver node caches it on a timer subscription; this interface
        # hands that snapshot out (a synchronous subscriber would not deliver
        # without spinning the executor, which would deadlock a service
        # callback).
        snap = getattr(self._node, "_joint_state_cache", None)
        if snap is not None:
            return snap
        raise ServiceError("no cached /joint_states reading available")


    def _call(self, client, request, timeout: Optional[float] = None):
        timeout = self._service_timeout if timeout is None else timeout
        what = f"service call to '{client.srv_name}'"
        try:
            ready = client.wait_for_service(timeout_sec=timeout)
        except Exception as exc:  # noqa: BLE001 - preflight flakiness
            # The entity preflight runs on the executor thread while the task
            # solve blocks a worker thread. Jazzy rclpy's graph handling can
            # raise here with an EMPTY str (StopIteration/CancelledError from
            # the graph event machinery) even though the server is fully up —
            # killing the call with "plan call failed: " (no message). The
            # preflight is only a fast availability check; the authoritative
            # budget is the _await deadline below, so log and attempt the
            # call anyway.
            self._node.get_logger().debug(
                f"{what}: wait_for_service preflight raised {exc!r} — "
                "attempting the call; the _await deadline is authoritative"
            )
            ready = True
        if not ready:
            self._session.invalidate()
            raise ServiceError(f"service '{client.srv_name}' unavailable")
        try:
            future = client.call_async(request)
        except Exception as exc:  # noqa: BLE001 - see _await repr notes
            # rclpy's call_async raises a BARE TypeError() (no message) when
            # request is not an instance of the srv's Request type — e.g. a
            # caller handed it a plain message instead of the srv Request
            # wrapper. Label it here so it reads as a protocol bug, not
            # "plan call failed: TypeError()".
            self._session.invalidate()
            raise ServiceError(f"{what} rejected client-side: {exc!r}") from None
        try:
            return self._await(future, what, timeout)
        except Exception:
            # Timeout, cancellation or server-side failure: the server may be
            # left in any planner/collision state, so the sticky session is
            # forgotten and the next request re-sends unconditionally.
            self._session.invalidate()
            raise

    @staticmethod
    def _await(future, what: str, timeout: float = _SERVICE_TIMEOUT):
        """Block on an rclpy future without nested spinning and without polling. The task constructor executes inside an action-server callback on a
        worker thread of a MultiThreadedExecutor. ``rclpy.spin_until_future
        _complete`` must not be called there (the executor is already
        spinning in another thread), so the worker thread blocks on a
        ``threading.Event`` instead: the future's done-callback sets it, and
        the executor's remaining threads deliver the response."""
        import threading

        if future.done():
            settled = True
        else:
            event = threading.Event()
            future.add_done_callback(lambda _f: event.set())
            settled = event.wait(timeout)
        if not settled:
            # Cancel the stale in-flight request so a later call_async on
            # the same client isn't (a) blocked by it, or (b) implicitly
            # canceled by rclpy's single-pending-request reuse.
            future.cancel()
            raise ServiceError(f"{what} timed out after {timeout}s") from None
        try:
            return future.result()
        except Exception as exc:  # rclpy client errors etc.
            # repr, not str: Canceled/other rclpy futures can carry an EMPTY
            # string message, and str(CancelledError()) == "" would surface as
            # "failed: " with no diagnostic value (see _call's preflight note).
            raise ServiceError(f"{what} failed: {exc!r}") from None

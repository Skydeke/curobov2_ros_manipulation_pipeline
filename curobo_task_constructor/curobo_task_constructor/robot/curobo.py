"""RobotInterface adapter for the real curobov2_ros curobo_server (ROS 2).

Converts the core's ROS-free ``PlanRequest``/``PlanResult``/``ObjectSpec``
types to/from ``curobov2_ros_interfaces`` messages and drives the
curobo_server services synchronously. ``joint_state_cls``/``pose_cls`` are
overridden so the stages emit real message types on the wire.

Known wire limitations (mirror of the framework contract):

- ``Ik.srv`` carries only the pose — the server solves from its own solver
  seeds, it cannot seed a solve from a joint state. The ``seed`` argument is
  accepted for ``RobotInterface`` parity; when a seed is supplied and the
  unseeded solve is invalid, the call retries a few times (the server's
  random solver seeds provide the variety).
- ``Fk.srv`` solves the tool-tip pose only; a ``link`` argument is accepted
  but the server-side FK does not target arbitrary links, so a non-tool
  ``link`` degrades to the tool pose (documented per-call).
- Object state is read back from the server, not mirrored locally:
  ``get_scene_objects`` reports every object's type, pose, size and attach
  state from the scene the solver holds. It replaces a local mirror of this
  interface's own ``add_object`` calls, which could not see an object another
  client added and reported a stale pose for a name that had been removed and
  re-added.
"""

from __future__ import annotations

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
    reorder_joint_vectors,
)

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


class CuroboServerInterface(RobotInterface):
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
        invoked exactly once.

        Discovery progresses normally while the node's executor is spinning,
        so there is no hand-rolled ``rclpy.spin_once`` and no blocking wait in
        ``__init__``. It keeps polling until the server appears (self-healing:
        a late curobo_server still triggers ``on_ready``), so the caller
        should advertise anything that must wait for the server from
        ``on_ready`` — clients that gate on that advertisement observe the
        full readiness chain.

        Neither the IK solver nor the FK model is warmed here: the server
        builds both during its own startup (``warmup_ik`` / ``warmup_fk``, both
        on by default), so a client that also called ``WarmupIK`` would
        re-initialize the solver and clobber the batch size the server chose.
        Call ``/warmup_ik`` only to change that batch size — the service is
        still the way to ask for a different one.
        """
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

    def get_object_pose(self, name: str):
        rec = self._query_scene_objects().get(name)
        if rec is None:
            return None
        return pose_to_any(rec["pose"], self.pose_cls)

    def get_named_joint_config(self, name: str) -> Optional[NamedJointConfig]:
        from curobo_task_constructor.core.robot_config import (
            resolve_named_config,
        )

        path = self._robot_config_path
        if not path:
            return None
        return resolve_named_config(path, name)

    def get_attached_objects(self) -> list:
        """Names of objects attached to the robot, per the server.

        The attach link holds one payload, so this is a 0- or 1-element list.
        """
        return sorted(
            name for name, rec in self._query_scene_objects().items()
            if rec.get("attached")
        )

    def get_object_names(self) -> list:
        """Names of every object in the server's scene.

        Includes objects this interface never touched — the task constructor's
        own stage, the grasp orchestrator, a ``ros2 service call`` — which a
        local mirror of this interface's own ``add_object`` calls could not.
        """
        return sorted(self._query_scene_objects())

    def get_object_spec(self, name: str) -> Optional[ObjectSpec]:
        """A server-reported object as an ``ObjectSpec``, or None if unknown.

        This is what ``build_base_scene`` wants: a name alone cannot rebuild an
        object, and hardcoding ``shape="mesh""`` for every entry made the
        reverse sync describe the world wrong for anything that was not a mesh.
        """
        rec = self._query_scene_objects().get(name)
        if rec is None:
            return None
        return ObjectSpec(
            name=name,
            shape=rec["shape"],
            pose=pose_to_any(rec["pose"], self.pose_cls),
            dimensions=list(rec["dimensions"]),
            # Empty for an inline mesh: the server does not echo vertices, so a
            # re-add from this spec would be wrong for one. Left None rather than
            # "" so a caller can tell "no path" from "path unknown".
            mesh_path=rec["mesh_path"] or None,
        )

    def _query_scene_objects(self) -> dict:
        """Ask the server for the whole scene, and return {name: record}.

        Cached, and re-queried at most once per ``_SCENE_CACHE_SECONDS``, because
        the executor's reverse sync walks every object and would otherwise make
        one service call per object per stage.

        On failure the PREVIOUS reply is returned rather than an empty dict. That
        distinction is the whole reason ``_scene_objects_queried`` exists: an
        unreachable server and an empty scene must not look the same, because
        ``build_base_scene`` would then clear the task's base scene and plan
        against a world it merely failed to read. A first query that fails
        returns empty, which is the honest answer given nothing is known yet, and
        ``_scene_objects_queried`` stays False so callers can tell.
        """
        now = time.monotonic()
        if self._scene_objects_queried and (
            now - self._scene_objects_at < _SCENE_CACHE_SECONDS
        ):
            return self._scene_objects

        try:
            res = self._call(self._scene_objects_client, GetSceneObjects.Request())
        except ServiceError as exc:
            self._node.get_logger().warning(
                f"get_scene_objects failed ({exc}); reusing the last known scene"
            )
            return self._scene_objects

        if not res.success:
            self._node.get_logger().warning(
                f"get_scene_objects refused: {res.message}; "
                "reusing the last known scene"
            )
            return self._scene_objects

        # Parallel arrays, so index i of each describes the same object. Every
        # access is bounds-checked: a short array must not silently misalign the
        # rest of the scene, which would attribute one object's pose to another.
        n = len(res.names)
        scene: dict = {}
        for i in range(n):
            name = res.names[i]
            if not name:
                continue
            if (i >= len(res.poses) or i >= len(res.dimensions)
                    or i >= len(res.types)):
                self._node.get_logger().warning(
                    f"get_scene_objects returned a short array for {name!r}; "
                    "omitting it rather than reporting misaligned values"
                )
                continue

            pose = res.poses[i]
            rec = {
                "shape": _SCENE_SHAPES.get(int(res.types[i]), "cuboid"),
                "pose": Pose3(
                    [float(pose.position.x), float(pose.position.y),
                     float(pose.position.z)],
                    [float(pose.orientation.x), float(pose.orientation.y),
                     float(pose.orientation.z), float(pose.orientation.w)],
                ),
                "dimensions": [
                    float(res.dimensions[i].x),
                    float(res.dimensions[i].y),
                    float(res.dimensions[i].z),
                ],
                "mesh_path": (
                    res.mesh_file_paths[i] if i < len(res.mesh_file_paths) else ""
                ),
                "attached": (
                    bool(res.attached[i]) if i < len(res.attached) else False
                ),
            }
            scene[name] = rec

        self._scene_objects = scene
        self._scene_objects_at = now
        self._scene_objects_queried = True
        return scene

    # ------------------------------------------------------------------
    # kinematics
    # ------------------------------------------------------------------
    def fk(self, joint_state, link: Optional[str] = None):
        req = Fk.Request()
        req.joint_states.append(joint_state)
        res = self._call(self._fk_client, req)
        if not res.poses:
            raise ServiceError(
                f"fk failed: {getattr(res, 'error_msg', '') or 'no pose returned'}"
            )
        if link is not None and link != "tool_link":
            self._node.get_logger().debug(
                f"fk(link={link!r}) degrades to tool-pose FK "
                "(Fk.srv does not target arbitrary links)"
            )
        return res.poses[0]

    def fk_batch(self, joint_states: list, link: Optional[str] = None) -> list:
        """Whole-trajectory FK in ONE round trip.

        ``Fk.srv`` already takes a ``JointState[]`` and returns a parallel
        ``Pose[]``, so measuring a 100-waypoint trajectory's straightness costs
        a single service call instead of 100.
        """
        states = list(joint_states or [])
        if not states:
            return []
        req = Fk.Request()
        for js in states:
            req.joint_states.append(js)
        res = self._call(self._fk_client, req)
        if not res.poses:
            raise ServiceError(
                f"fk_batch failed: {getattr(res, 'error_msg', '') or 'no pose returned'}"
            )
        return list(res.poses)

    def ik(self, pose, seed: Optional[Any] = None):
        req = Ik.Request()
        req.pose = parse_pose_msg(pose, self.pose_cls)
        attempts = 3 if seed is not None else 1
        for _ in range(attempts):
            res = self._call(self._ik_client, req)
            if res.success and res.joint_states_valid.data:
                js = res.joint_states
                return self._merge_seed(js, seed)
        return None

    @staticmethod
    def _merge_seed(result: JointState, seed) -> JointState:
        """Overlay a seed's trailing joints onto the solver's flat result.

        Ik.srv cannot be seeded (server-side solver seeds only), so a seed is
        honoured as far as the wire allows: the server returns the active-DOF
        position list without names; joints the solver did not cover (e.g.
        finger_joint) keep their seed values, positional overlay style.
        """
        if seed is None:
            return result
        seed_pos = list(getattr(seed, "position", []) or [])
        out_pos = list(result.position)
        if len(out_pos) < len(seed_pos):
            out_pos = out_pos + seed_pos[len(out_pos) :]
        js = JointState()
        js.name = list(getattr(seed, "name", []))
        js.position = [float(v) for v in out_pos]
        return js

    def ik_batch(self, poses: list) -> list:
        return [self.ik(p) for p in poses]

    # ------------------------------------------------------------------
    # planning
    # ------------------------------------------------------------------
    def set_planner(self, planner) -> None:
        if planner is None:
            return
        req = SetPlanner.Request()
        req.planner_type = int(planner)
        res = self._call(self._planner_client, req)
        if not res.success:
            raise ServiceError(f"set_planner failed: {res.message}")
        self._planner_active = True
        self._active_planner = int(planner)

    def _ensure_planner(self, planner) -> None:
        """Make sure the requested planner is the server's current one.

        The switch is sticky per planner: once set, a repeat request for the
        SAME planner is skipped (no extra service round-trip), but a DIFFERENT
        planner is switched to on the spot. Without this, a task mixing planner
        types (e.g. joint-space reach + classic pose lift) would silently run
        every stage under whichever planner was switched first.
        """
        key = planner if planner is not None else self._planner
        if key is None:
            return
        if self._planner_active and self._active_planner == int(key):
            return
        self.set_planner(key)

    def _set_links_collision(self, requests: list, enabled: bool) -> None:
        """Enable/disable collision spheres for the goalsets' allowed links.

        Contact/collision allowance lives ENTIRELY in the task constructor:
        stages derive it from the scene (``GoalsetSpec.allowed_collisions``)
        and this adapter applies it via the server's ``set_link_collision``
        service for the duration of each solve — spheres OFF
        (``enabled=False``) allows contact, spheres ON (``enabled=True``)
        restores the check. The wire ``Goalset`` carries no collision concept;
        the planning interfaces never see it.
        """
        links = sorted(
            {
                link
                for r in (requests or [])
                for gs in (r.goalsets or [])
                for link in (gs.allowed_collisions or [])
            }
        )
        if not links:
            return
        req = SetLinkCollision.Request()
        req.link_names = links
        req.enabled = enabled
        res = self._call(self._link_collision_client, req)
        if not res.success:
            raise ServiceError(f"set_link_collision failed: {res.message}")

    def plan(self, request: PlanRequest) -> PlanResult:
        self._ensure_planner(request.planner)
        # Apply the goalset's allowed links (spheres OFF) for the duration of
        # the solve and restore (spheres ON) right after — exception-safe.
        self._set_links_collision([request], False)
        try:
            # TrajectoryGeneration.srv embeds the goal as `TrajectoryGoal request`
            # (§5 DRY). The bare goal is NOT the srv Request type, and rclpy's
            # Client.call_async isinstance-checks it and raises a BARE TypeError()
            # (empty str, repr exactly "TypeError()") when handed one — the
            # "plan call failed: TypeError()" everyone saw. Wrap it in the srv
            # Request like every other client call builds one.
            req = TrajectoryGeneration.Request()
            req.request = self._to_goal(request)
            res = self._call(self._traj_client, req)
            # TrajectoryGeneration.srv embeds the result as `TrajectoryResult
            # response`.
            return self._from_result(res.response)
        finally:
            self._set_links_collision([request], True)

    def plan_batch(self, requests: list) -> list:
        planners = {r.planner for r in requests if r.planner is not None}
        for p in planners:
            self.set_planner(p)
        # Apply the goalsets' allowed links (spheres OFF) for the duration of
        # the solve and restore (spheres ON) right after — exception-safe.
        self._set_links_collision(requests, False)
        try:
            batch = TrajectoryGenerationBatch.Request()
            for r in requests:
                batch.requests.append(self._to_goal(r))
            res = self._call(self._batch_client, batch)
            if not res.success and not res.responses:
                return [
                    PlanResult(False, res.error_msg or "plan_batch failed")
                    for _ in requests
                ]
            out = [self._from_result(r) for r in res.responses]
            if len(out) < len(requests):
                out += [
                    PlanResult(False, "missing batch response")
                    for _ in range(len(requests) - len(out))
                ]
            return out
        finally:
            self._set_links_collision(requests, True)

    def execute(self, request: PlanRequest) -> PlanResult:
        """Drive one segment: SendTrajectory, which replays the matching
        cached (pre-planned) trajectory when it can and re-solves on a miss.

        The goalset's ``allowed_collisions`` is applied here exactly as in
        ``plan`` / ``plan_batch``. It has to be: the wire ``Goalset`` carries
        no collision concept, so this adapter's ``set_link_collision`` call is
        the ONLY way the server ever learns that e.g. the finger spheres may
        touch the grasped object. The server's execute path replays the
        cached plan whenever the request matches one — with the multi-entry
        trajectory cache, EVERY segment of a chain hits, so the driven
        trajectory is the validated plan rather than a fresh solve (a fresh
        solve of a multi-solution pose goal is free to pick the OTHER IK
        branch, which is exactly the divergence the chain-continuity check
        exists to catch). Only a genuine miss (cache cleared, TTL expired,
        world mutated) falls back to re-solving, against whatever sphere state
        is current — with the spheres restored, a grasp close that planned
        cleanly (spheres off) fails its goal state with "finger -> object"
        contacts that the plan itself proved were acceptable.

        ``force_cached`` stays OFF: the task-constructor prefers the graceful
        re-solve fallback on a genuine miss. Callers that demand "replay or
        fail" can set it on the action goal directly.

        The window here is wider than in ``plan``: the allowance is held off for
        the whole physical motion, not just the solve. That is correct for a
        grasp — the fingers legitimately stay against the object from the
        descent until the retreat ends — and inert everywhere else, because
        ``_set_links_collision`` returns early when a request allows no links,
        so stages like ``return`` / ``open`` do not touch the spheres at all.
        """
        self._ensure_planner(request.planner)
        if not self._exec_client.wait_for_server(timeout_sec=self._service_timeout):
            raise ServiceError("execute_trajectory action unavailable")
        goal = SendTrajectory.Goal()
        goal.goal = self._to_goal(request)
        goal.allow_cached = True
        goal.force_cached = False  # replay-or-fail knob for explicit callers
        self._set_links_collision([request], False)
        try:
            gh = self._await(
                self._exec_client.send_goal_async(goal),
                "execute_trajectory goal handshake",
                self._service_timeout,
            )
            if gh is None or not gh.accepted:
                return PlanResult(False, "execute_trajectory goal rejected")
            res = self._await(
                gh.get_result_async(),
                "execute_trajectory result",
                self._service_timeout,
            )
            if res is None:
                return PlanResult(False, "execute_trajectory result timeout")
            return self._from_result(res.result.result)
        finally:
            self._set_links_collision([request], True)

    # ------------------------------------------------------------------
    # scene
    # ------------------------------------------------------------------
    def add_object(self, spec) -> bool:
        req = AddObject.Request()
        req.type = _ADD_OBJECT_SHAPES.get(spec.shape, AddObject.Request.MESH)
        req.name = spec.name
        if spec.pose is not None:
            req.pose = parse_pose_msg(spec.pose, self.pose_cls)
        dims = list(spec.dimensions or [0.0, 0.0, 0.0])
        req.dimensions = Vector3(
            x=float(dims[0]) if len(dims) > 0 else 0.0,
            y=float(dims[1]) if len(dims) > 1 else 0.0,
            z=float(dims[2]) if len(dims) > 2 else 0.0,
        )
        if spec.shape == "mesh":
            if spec.vertices:
                for v in spec.vertices:
                    p = Point()
                    p.x, p.y, p.z = (float(v[0]), float(v[1]), float(v[2]))
                    req.vertices.append(p)
                req.triangles = [int(t) for t in (spec.triangles or [])]
            elif spec.mesh_path:
                req.mesh_file_path = spec.mesh_path
        res = self._call(self._add_client, req)
        if res.success:
            # Drop the cached reply rather than patching it: the next query asks
            # the server what it actually stored, which is not always what was
            # sent (AddObject rejects non-positive dimensions, and for a capsule
            # or cylinder the third component is ignored entirely).
            self._invalidate_scene_cache()
        return res.success

    def remove_object(self, name: str) -> bool:
        req = RemoveObject.Request()
        req.name = name
        res = self._call(self._remove_client, req)
        if res.success:
            self._invalidate_scene_cache()
        return res.success

    def remove_all_objects(self) -> None:
        self._call(self._remove_all_client, Trigger.Request())
        self._invalidate_scene_cache()

    def attach_object(self, name: str) -> bool:
        req = AttachObject.Request()
        req.object_name = name
        res = self._call(self._attach_client, req)
        if res.success:
            # attach keeps the object LISTED (it is disabled by name, not
            # removed) and changes only the flag, so the name set is unchanged —
            # but `get_attached_objects` reads that flag, so the cached reply is
            # stale either way.
            self._invalidate_scene_cache()
        return res.success

    def detach_object(self, name: Optional[str] = None) -> bool:
        # `name` is accepted for interface parity and ignored: the server's
        # detach is a bare Trigger over the single attach link and takes no
        # argument. Which object it released comes back from get_scene_objects.
        res = self._call(self._detach_client, Trigger.Request())
        if res.success:
            self._invalidate_scene_cache()
        return res.success

    def _invalidate_scene_cache(self) -> None:
        """Force the next `_query_scene_objects` to actually ask the server.

        Called after every mutating scene call THIS interface makes. Without it a
        stage that adds an object and then reads its pose back could be served the
        pre-add reply, and would see its own write missing.
        """
        self._scene_objects_queried = False

    # ------------------------------------------------------------------
    # conversions
    # ------------------------------------------------------------------
    def _to_goal(self, request: PlanRequest) -> TrajectoryGoal:
        goal = TrajectoryGoal()
        if request.start_pose is not None:
            goal.start_pose = self._to_joint_msg(request.start_pose)
        for gs in request.goalsets or []:
            goal.goalsets.append(self._to_goalset(gs))
        return goal

    @staticmethod
    def _to_joint_msg(joint_state) -> JointState:
        js = JointState()
        js.name = list(getattr(joint_state, "name", []))
        js.position = [float(p) for p in (getattr(joint_state, "position", []) or [])]
        return js

    def _to_goalset(self, gs: GoalsetSpec) -> Goalset:
        g = Goalset()
        for pose in gs.poses or []:
            g.poses.append(parse_pose_msg(pose, self.pose_cls))
        g.target_joint_positions = [float(v) for v in (gs.target_joint_positions or [])]
        # Whole-path Cartesian axis holds, int8[6] = [theta_x, theta_y,
        # theta_z, x, y, z] (1 = hold that axis). This is the only way to ask
        # for a straight-line (rather than free-space) approach / descent /
        # retreat: the server reads it into ToolPoseCriteria
        # (SinglePlanner._apply_pose_constraints) and scores every non-terminal
        # waypoint against the goal on the held axes. The wire is strict about
        # the length — a malformed vector is dropped with a warning — so emit
        # either [] (unconstrained) or exactly 6 ints.
        hold = [int(c) for c in (gs.trajectory_constraints or [])]
        g.trajectory_constraints = hold if len(hold) == 6 else []
        return g

    @staticmethod
    def _from_result(res) -> PlanResult:
        return PlanResult(
            success=bool(res.success),
            message=str(res.message or ""),
            trajectory=list(res.trajectory or []),
            dt=float(getattr(res, "dt", 0.0) or 0.0),
            cost=float("inf"),  # framework falls back to trajectory length
            raw=res,
            selected_goal_index=list(getattr(res, "selected_goal_index", []) or []),
            waypoint_status=list(getattr(res, "waypoint_status", []) or []),
            stats=getattr(res, "stats", None),
        )

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
            raise ServiceError(f"service '{client.srv_name}' unavailable")
        try:
            future = client.call_async(request)
        except Exception as exc:  # noqa: BLE001 - see _await repr notes
            # rclpy's call_async raises a BARE TypeError() (no message) when
            # request is not an instance of the srv's Request type — e.g. a
            # caller handed it a plain message instead of the srv Request
            # wrapper. Label it here so it reads as a protocol bug, not
            # "plan call failed: TypeError()".
            raise ServiceError(f"{what} rejected client-side: {exc!r}") from None
        return self._await(future, what, timeout)

    @staticmethod
    def _await(future, what: str, timeout: float = _SERVICE_TIMEOUT):
        """Block on an rclpy future without nested spinning.

        The task constructor executes inside an action-server callback on a
        worker thread of a MultiThreadedExecutor. ``rclpy.spin_until_future
        _complete`` must not be called there (the executor is already
        spinning in another thread), so we block the worker thread instead:
        the executor's remaining threads deliver the response and resolve the
        future.

        Jazzy's rclpy client/action futures do NOT accept a ``timeout``
        kwarg on ``Future.result()`` — calling it that way raises
        ``TypeError: Future.result() got an unexpected keyword argument
        'timeout'`` and killed every plan at the call boundary. The budget is
        therefore enforced by polling ``done()`` against a monotonic
        deadline; ``result()`` is called without arguments once the future is
        resolved.
        """
        deadline = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() >= deadline:
                # Cancel the stale in-flight request so a later call_async on
                # the same client isn't (a) blocked by it, or (b) implicitly
                # canceled by rclpy's single-pending-request reuse.
                future.cancel()
                raise ServiceError(f"{what} timed out after {timeout}s") from None
            time.sleep(0.02)
        try:
            return future.result()
        except Exception as exc:  # rclpy client errors etc.
            # repr, not str: Canceled/other rclpy futures can carry an EMPTY
            # string message, and str(CancelledError()) == "" would surface as
            # "failed: " with no diagnostic value (see _call's preflight note).
            raise ServiceError(f"{what} failed: {exc!r}") from None


def parse_pose_msg(pose, pose_cls):
    """Pose / Pose3 / list -> geometry_msgs/Pose (or pose_cls)."""
    if pose_cls is not None and isinstance(pose, pose_cls):
        return pose
    p = Pose3.from_any(pose)
    out = pose_cls() if pose_cls is not None else RosPose()
    out.position = Point(x=p.position[0], y=p.position[1], z=p.position[2])
    q = p.orientation
    out.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
    return out

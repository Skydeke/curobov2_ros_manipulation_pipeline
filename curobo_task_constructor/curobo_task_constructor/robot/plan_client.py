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
from control_msgs.action import (
    GripperCommand as GripperCommandAction,
    ParallelGripperCommand as ParallelGripperCommandAction,
)
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



from curobo_task_constructor.robot.session import *

class _PlanClientMixin:
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

    def invalidate_planner_session(self) -> None:
        """Forget the sticky "planner already active" belief (see robot.py).

        The ``_ensure_planner`` skip is only sound WITHIN one planning burst:
        a remote ``task.execute()`` drives the solution through the execute
        server's OWN interface, which mutates the shared ``/curobo_server``
        planner, and a sibling node may switch it too. Dropping the belief
        makes the next request re-assert its planner unconditionally.
        """
        self._session.invalidate()

    def _ensure_planner(self, planner) -> None:
        """Make sure the requested planner is the server's current one. Identical repeats are skipped through the session (see
        PlannerSessionState): three best-of-3 attempts of one request and
        back-to-back same-planner legs send once. The skip used to be wrong
        on its own — the server switches planners by itself during execution
        replay, so a cached "already active" belief goes stale (a joint-space
        request then plans under Classic and fails validation, with no client
        error pointing at the cause)."""
        key = planner if planner is not None else self._planner
        if key is None:
            return
        if not self._session.planner_request_needed(key):
            return
        try:
            self.set_planner(key)
        except Exception:
            self._session.invalidate()
            raise
        self._session.note_planner_sent(key)

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
        tls = self._allowance_tls
        depth = getattr(tls, "depth", 0)
        held = getattr(tls, "links", None)
        if depth > 0 and held != links:
            # Mixed nesting (not produced by our graphs — every scope wraps
            # one request's attempts): fall back to a stateless extra call
            # so the outer scope still balances. Session and depth untouched.
            self._send_collision_update(links, enabled)
            return
        if not enabled:
            if depth == 0 and self._session.collision_transition_needed(
                    links, False):
                self._send_collision_update(links, False)
                self._session.note_collision_sent(links, False)
            tls.links = links
            tls.depth = depth + 1
            return
        if depth <= 1:
            if self._session.collision_transition_needed(links, True):
                self._send_collision_update(links, True)
                self._session.note_collision_sent(links, True)
            tls.links = None
            tls.depth = 0
            return
        tls.depth = depth - 1

    def _send_collision_update(self, links, enabled: bool) -> None:
        """One set_link_collision round trip (refuse/fail invalidates)."""
        req = SetLinkCollision.Request()
        req.link_names = list(links)
        req.enabled = bool(enabled)
        try:
            res = self._call(self._link_collision_client, req)
        except Exception:
            self._session.invalidate()
            raise
        if not res.success:
            self._session.invalidate()
            raise ServiceError(f"set_link_collision failed: {res.message}")

    @contextlib.contextmanager
    def allowance_scope(self, requests):
        """Hold this request's allowance OFF across nested planning calls. A stage planning N attempts of one request enters once around the
        loop: the first OFF sends (outermost), nested OFF/ON pairs from each
        ``plan()`` compose silently, and the exit restores ON. Standalone
        ``plan()``/``execute()`` calls keep their own balanced OFF...ON
        without any scope."""
        self._set_links_collision(requests, False)
        try:
            yield
        finally:
            self._set_links_collision(requests, True)

    def plan(self, request: PlanRequest) -> PlanResult:
        # Identity fast path: a joint-space goal equal to the start needs no
        # motion and, more importantly, no server round trip. Beyond saving
        # the solve, this avoids sending degenerate zero-distance problems
        # (e.g. a "ready"/"home" leg issued while already there) into trajopt,
        # where they have preceded CUDA faults. Tight tolerance, joint goals
        # only, single goalset: anything else plans normally.
        trivial = self._trivial_goal(request)
        if trivial is not None:
            return trivial
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

    @staticmethod
    def _trivial_goal(request: PlanRequest):
        """Trivial success for a no-motion joint goal, or None.

        Single goalset with joint targets matching the start pose within
        ``_TRIVIAL_GOAL_TOLERANCE`` on every joint: returns a single-waypoint
        success without touching the server. Returning the start state as the
        whole trajectory is safe precisely because nothing moves — there is
        no path to collide-check.
        """
        goalsets = list(getattr(request, "goalsets", None) or [])
        if len(goalsets) != 1:
            return None
        want = [float(v) for v in (
            getattr(goalsets[0], "target_joint_positions", None) or [])]
        if not want:
            return None
        start = getattr(request, "start_pose", None)
        have = [float(v) for v in (getattr(start, "position", None) or [])]
        if len(have) != len(want):
            return None
        if any(abs(a - b) > _TRIVIAL_GOAL_TOLERANCE for a, b in zip(have, want)):
            return None
        return PlanResult(True, "start equals goal (no motion needed)",
                          trajectory=[start], cost=0.0)

    def plan_batch(self, requests: list) -> list:
        planners = {r.planner for r in requests if r.planner is not None}
        for p in planners:
            try:
                self.set_planner(p)
            except Exception:
                self._session.invalidate()
                raise
            self._session.note_planner_sent(p)
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

    def execute(self, request: PlanRequest, trajectory) -> PlanResult:
        """Drive one segment (SendTrajectory action). MTC ``Task::execute`` parity: the caller passes the segment's VALIDATED
        joint waypoints and the server stages+streams them verbatim
        (``replay``) — no cache lookup, no re-solve, so a multi-solution pose
        goal cannot jump IK branch mid-chain. ``request`` is still sent for the
        planner (a SetPlanner switch selects the executor) and its
        ``allowed_collisions`` (the wire Goalset has no collision concept, so
        this adapter's ``set_link_collision`` is the only way the server learns
        the finger spheres may touch the grasped object)."""
        # Driving may leave the server in any planner/collision state, so the
        # sticky session is forgotten here and rebuilt by the next requests.
        self._session.invalidate()
        self._ensure_planner(request.planner)
        if not self._exec_client.wait_for_server(timeout_sec=self._service_timeout):
            raise ServiceError("execute_trajectory action unavailable")
        goal = SendTrajectory.Goal()
        goal.goal = self._to_goal(request)
        goal.replay = True
        goal.replay_trajectory = self._to_joint_trajectory_msg(trajectory)
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

    def gripper_command(self, gripper: str = "", joint: str = "",
                        position: float = 0.0, effort: float = 0.0,
                        controller: str = "", action_ns: str = "",
                        max_effort: float = 0.0, **_) -> PlanResult:
        """Send a gripper command to a named end effector, without planning.

        Blank fields resolve against the descriptor's ``gripper_controllers``
        entry (selected by ``gripper``; blank falls back to the first entry).
        ``GripperCommand`` sends a single position; ``ParallelGripperCommand``
        sends one position per joint.
        """
        cfg = (self._grippers.get(gripper)
               or self._grippers.get(self._default_gripper) or {})
        if not cfg:
            return PlanResult(
                False, f"no gripper controller {gripper!r} in the descriptor")
        gtype = str(cfg.get("type") or "GripperCommand")
        # Position-commanded grippers complete only on reach-or-stall, and a
        # close on a payload parks the finger short of the target: the goal then
        # hinges entirely on the controller reporting a stall. Bound the wait so
        # the jaws can never stall the whole task; the controller's own
        # cancel_callback holds the finger where it is, which is exactly the
        # gripped position.
        timeout = float(cfg.get("timeout", 5.0) or 5.0)
        name = controller or cfg.get("name") or "gripper"
        ns = action_ns or cfg.get("action_ns") or "gripper_cmd"
        action = f"/{name}/{ns}" if name else ns
        effort = float(effort) or float(max_effort) or float(
            cfg.get("max_effort") or 0.0)
        client = self._gripper_clients.get(action)
        if client is None:
            action_type = (ParallelGripperCommandAction
                           if gtype == "ParallelGripperCommand"
                           else GripperCommandAction)
            client = ActionClient(self._node, action_type, action)
            self._gripper_clients[action] = client
        if not client.wait_for_server(timeout_sec=timeout):
            return PlanResult(False, f"gripper action {action} unavailable")
        if gtype == "ParallelGripperCommand":
            joints = list(cfg.get("joints") or ([joint] if joint else []))
            if not joints:
                return PlanResult(False, f"gripper {gripper!r} lists no joints")
            goal = ParallelGripperCommandAction.Goal()
            goal.command.name = [str(j) for j in joints]
            goal.command.position = [float(position)] * len(joints)
            goal.command.effort = [float(effort)] * len(joints)
            label = f"{'/'.join(joints)}={float(position)}"
        else:
            joint = joint or cfg.get("command_joint", "")
            goal = GripperCommandAction.Goal()
            goal.command.position = float(position)
            goal.command.max_effort = float(effort)
            label = f"{joint}={float(position)}"
        try:
            gh = self._await(client.send_goal_async(goal),
                             "gripper goal handshake", timeout)
        except ServiceError as exc:
            return PlanResult(False, str(exc))
        if gh is None or not gh.accepted:
            return PlanResult(False, "gripper goal rejected")
        try:
            res = self._await(gh.get_result_async(), "gripper result", timeout)
            if res is None:
                return PlanResult(False, "gripper result timeout")
        except ServiceError:
            # Accepted but never reported: it parked on the payload. Cancel so
            # the controller stops holding the goal (its cancel_callback keeps
            # the finger where it is) and treat the command as delivered.
            try:
                gh.cancel_goal_async()
            except Exception:
                pass
            return PlanResult(
                True, f"gripper {label} (no result in {timeout:g}s; canceled)")
        return PlanResult(True, f"gripper {label}")

    # ------------------------------------------------------------------
    # scene
    # ------------------------------------------------------------------

__all__ = [
    '_PlanClientMixin',
]

from .msg import parse_pose_msg

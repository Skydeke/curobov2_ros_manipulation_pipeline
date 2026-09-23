"""ROS 2 action server fronting the ``TaskExecutor`` (migration step 4).

Runs at ``/curobo_task_constructor/task`` (``Task.action``) and mirrors the
MTC ``Task::init()/plan()/execute()`` lifecycle: it builds the ``TaskExecutor``
from the goal's ``StageSpec`` tree, validates against the live curobo world,
solves, ranks, and (per ``goal.execute``) drives the winner back through the
curobo_server SendTrajectory action.

Introspection (Sec. 7 of the plan):

- ``/curobo_task_constructor/task_description`` — the built StageSpec tree +
  validity, published once before the solve (transient_local so a panel that
  joins later can render structure anyway).
- ``/curobo_task_constructor/solution_info`` — one message per stage attempt,
  successes AND failures, streamed as ``compute()`` runs (the stage-level
  ``on_solution``/``on_failure`` hooks).
- ``/curobo_task_constructor/stage_statistics`` — per-stage rollups after the
  solve.

Execution model: the task solve runs inside the action-server execute
callback on a worker thread of a ``MultiThreadedExecutor``. The adapter
(``robot.CuroboServerInterface``) blocks that worker thread per service
round-trip; the executor's *other* threads deliver the responses and the
/joint_states readings. A second goal is rejected while one solve is in
flight so two tasks never interleave on the same server.
"""

from __future__ import annotations

import threading
from functools import partial

import rclpy
from rclpy.action import ActionServer, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import JointState
from visualization_msgs.msg import MarkerArray

from curobo_task_constructor_interfaces.action import Task
from curobo_task_constructor_interfaces.msg import (
    SolutionInfo,
    StageSpec as StageSpecMsg,
    StageStatistics,
    TaskDescription,
)

#: Populate STAGE_REGISTRY with the builtin stages (registry is import-driven).
import curobo_task_constructor.stages  # noqa: F401
from curobo_task_constructor.core.state import SceneDiff
from curobo_task_constructor.core.stage import chain_hook
from curobo_task_constructor.executor import TaskExecutor
from curobo_task_constructor.graph.spec import StageSpec
from curobo_task_constructor.robot import CuroboServerInterface
from curobo_task_constructor.robot.curobo import _SERVICE_TIMEOUT

#: Default wall-clock budget for the planning phase of one task (see the
#: ``plan_timeout`` parameter). 180 s is ample for a full pick task (the
#: observed plan phase is ~15 s) while still bounding a wedged solve.
_PLAN_TIMEOUT = 180.0

#: solution_id carried by failed attempts (uint32 field; 0xFFFFFFFF = none).
NO_SOLUTION_ID = 0xFFFFFFFF

_ACTION_TOPIC = "/curobo_task_constructor/task"
_INTROSPECT_QOS_TOPICS = {
    "task_description": "/curobo_task_constructor/task_description",
    "solution_info": "/curobo_task_constructor/solution_info",
    "stage_statistics": "/curobo_task_constructor/stage_statistics",
}


class TaskConstructorNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("curobo_task_constructor")
        self.declare_parameter("robot_config_path", "")
        self.declare_parameter("planner", -1)
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("service_timeout", _SERVICE_TIMEOUT)
        #: Wall-clock budget (s) for the whole task PLANNING phase. Every
        #: stage's robot.plan() call already has the per-call service_timeout
        #: budget, but the executor's compute loop can outlive those (reconnect
        #: re-plans, stall before the first root solution): while it runs, the
        #: action server holds _solve_in_progress and rejects every new task
        #: goal ("another solve is in progress"). This bounds that window so a
        #: wedged solve cannot park the node for minutes.
        self.declare_parameter("plan_timeout", _PLAN_TIMEOUT)

        self._robot_config_path = str(
            self.get_parameter("robot_config_path").value)
        self._joint_states_topic = str(
            self.get_parameter("joint_states_topic").value)
        planner_param = int(self.get_parameter("planner").value)
        service_timeout = float(self.get_parameter("service_timeout").value)
        self._plan_timeout = float(self.get_parameter("plan_timeout").value)

        #: newest /joint_states reading, consumed by the adapter's
        #: get_current_joint_state (CurrentState seeds the task from here).
        self._joint_state_cache = None
        self._js_warned = False
        self._joint_state_sub = self.create_subscription(
            JointState, self._joint_states_topic, self._on_joint_state, 10,
            callback_group=ReentrantCallbackGroup())
        self.create_timer(10.0, self._joint_state_watchdog)

        self._solver = CuroboServerInterface(
            self, planner=(planner_param if planner_param >= 0 else None),
            planner_service="/curobo_server/set_planner",
            robot_config_path=(self._robot_config_path or None),
            service_timeout=service_timeout)

        # Introspection publishers (transient_local: a panel may join after
        # the task started and still see the structure / last attempts).
        qos = QoSProfile(
            depth=10, history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._pub_desc = self.create_publisher(
            TaskDescription, _INTROSPECT_QOS_TOPICS["task_description"], qos)
        self._pub_sol = self.create_publisher(
            SolutionInfo, _INTROSPECT_QOS_TOPICS["solution_info"], qos)
        self._pub_stat = self.create_publisher(
            StageStatistics, _INTROSPECT_QOS_TOPICS["stage_statistics"], qos)

        # Solve serialization: one task at a time on the shared curobo server.
        self._solve_lock = threading.Lock()
        self._solve_in_progress = False

        # Active-solve context for the introspection hooks.
        self._active_goal_handle = None
        self._active_task_id = ""
        self._attempt_count = 0
        self._solutions_total = 0

        self._planner_label = (
            planner_param if planner_param >= 0 else "server default")

        # Startup readiness gate (the viser nodes' start_service_poll pattern):
        # a wall timer polls service_is_ready()/server_is_ready() until every
        # curobo_server service the interface touches AND the SendTrajectory
        # action are reachable; only then is the task ActionServer advertised.
        # No blocking wait in __init__ and no hand-rolled rclpy.spin_once —
        # discovery progresses normally while the executor is spinning.
        # Clients that gate on _ACTION_TOPIC (the grasp orchestrator) therefore
        # see the full readiness chain.
        #
        # The IK warm-up is deliberately NOT triggered here: WarmupIK builds
        # the solver for its requested batch size, so a second warmer would
        # re-initialize the solver and clobber the batch size configured by
        # the node that owns the warm-up (the grasp orchestrator primes
        # batch 2 and gates its first task on that warm-up completing).
        self._solver.start_ready_poll(
            self._advertise_action_server, period=0.5)

    def _advertise_action_server(self):
        """Advertise the task action server once the curobo_server is up.

        Called exactly once from the readiness poll after every curobo service
        and the SendTrajectory action are reachable. Creating the server only
        now keeps ``server_is_ready()`` False for any client until the
        dependency chain has passed — which is precisely what the grasp
        orchestrator gates its first task on.
        """
        if getattr(self, "_action", None) is not None:
            return
        self._action = ActionServer(
            self, Task, _ACTION_TOPIC,
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            callback_group=ReentrantCallbackGroup())
        self.get_logger().info(
            f"task constructor ready at {_ACTION_TOPIC} "
            f"(planner={self._planner_label})")

    # ------------------------------------------------------------------
    # joint state snapshot
    # ------------------------------------------------------------------
    def _on_joint_state(self, msg):
        # Normalize at the ingest point so EVERY consumer of the snapshot
        # (CurrentState seeding, FK/IK seeds, and the plan start_pose, which
        # the server resolves VERBATIM in cspace order) sees the canonical
        # joint order. The adapter reorders name[]/position[] by name into
        # cspace order (the kortex sim publishes the finger joint FIRST); a
        # descriptor without joint order passes readings through unchanged.
        self._joint_state_cache = self._solver.normalize_joint_state(msg)

    def _joint_state_watchdog(self):
        if self._joint_state_cache is None:
            if not self._js_warned:
                self._js_warned = True
                self.get_logger().warn(
                    f"no reading on '{self._joint_states_topic}' yet — tasks "
                    "that start from CurrentState will fail with "
                    "'no cached /joint_states reading available'")
        else:
            self._js_warned = False

    # ------------------------------------------------------------------
    # action server
    # ------------------------------------------------------------------
    def _goal_callback(self, goal_request):
        with self._solve_lock:
            if self._solve_in_progress:
                self.get_logger().warn(
                    "rejecting task goal: another solve is in progress")
                return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _execute_callback(self, goal_handle):
        with self._solve_lock:
            self._solve_in_progress = True
        self._active_goal_handle = goal_handle
        self._active_task_id = goal_handle.request.task_name
        self._attempt_count = 0
        self._solutions_total = 0
        try:
            return self._run_task(goal_handle)
        finally:
            self._active_goal_handle = None
            self._active_task_id = ""
            with self._solve_lock:
                self._solve_in_progress = False

    def _run_task(self, goal_handle) -> Task.Result:
        goal = goal_handle.request

        try:
            spec = StageSpec.from_msg_list(goal.stages)
            spec.validate()
        except Exception as exc:
            self.get_logger().error(
                f"task '{goal.task_name}' rejected: {exc}")
            goal_handle.abort()
            return Task.Result(
                success=False, error=f"invalid task spec: {exc}",
                failed_stage_name="<spec>")

        executor = TaskExecutor(spec, self._solver, task_id=goal.task_name)
        try:
            executor.base_scene = executor.build_base_scene()
        except Exception as exc:  # server unreachable / names-only mirrors
            self.get_logger().warn(
                f"base scene unavailable ({exc}); planning against an empty "
                "diff — scene objects must come from the task itself")
            executor.base_scene = SceneDiff()

        if not executor.init():
            error = executor._init_error or "task failed init"
            self.get_logger().error(f"task '{goal.task_name}' init failed: {error}")
            goal_handle.abort()
            return Task.Result(
                success=False, error=f"init failed: {error}",
                failed_stage_name="<init>")

        # Stream per-attempt introspection as compute() runs (Sec. 7): every
        # stage's hooks hand the publishers Solution/StageFailure objects.
        # Chain onto the handlers containers installed at init() — the serial
        # container wraps each child's on_solution with its chain-lifting
        # handler (ContainerStage.init). Replacing it severed lifting: leaves
        # planned fine and published, but root.solutions stayed empty, so the
        # task always failed with "produced no complete solution".
        for stg in executor.root.subtree_stages():
            stg.on_solution = chain_hook(
                stg.on_solution, partial(self._publish_solution, stg))
            stg.on_failure = chain_hook(
                stg.on_failure, partial(self._publish_failure, stg))

        self._publish_task_description(executor)
        self._publish_feedback("planning", "", hint="task accepted")

        ok = executor.plan(deadline=self._plan_timeout)
        self._publish_stage_statistics(executor)

        if not ok:
            failed = self._last_failed_stage(executor)
            reason = ""
            self.get_logger().error(
                f"task '{goal.task_name}' produced no complete solution"
                + (f" (failed at '{failed}')" if failed else ""))
            # Surface every stage's last failure right here: the per-attempt
            # SolutionInfo comments are only visible in the rviz panel, which
            # may not be attached during bring-up.
            for stg in executor.root.subtree_stages():
                if stg.failures:
                    msg = stg.failures[-1].message
                    self.get_logger().error(f"  stage '{stg.name}': {msg}")
                    if not reason and msg:
                        reason = msg[:400]
            goal_handle.abort()
            if reason:
                err = (f"no complete solution (failed at '{failed}'): {reason}"
                       if failed else f"no complete solution: {reason}")
            else:
                err = "no complete solution"
            return Task.Result(success=False, error=err,
                                failed_stage_name=failed)

        sol = executor.best()
        self._publish_feedback("solved", sol.stage.name if sol else "",
                               hint=f"{self._solutions_total} solution(s)")

        if goal.execute and sol is not None:
            try:
                # Map each drive failure back to its leaf stage so clients can
                # tell a variant reach failure (approach_*/grasp_*) from a
                # post-reach failure (close/lift/return/...). Only motion leaves
                # yield results; scene-only leaves (modify_scene) contribute
                # none, so filter them before zipping.
                motion_leaves = [
                    leaf for leaf in executor.flatten_leaves(sol)
                    if leaf.plan_request is not None]
                results = executor.execute(sol)
            except Exception as exc:
                self.get_logger().error(f"task '{goal.task_name}' execution failed: {exc}")
                goal_handle.abort()
                return Task.Result(
                    success=False, error=f"execution failed: {exc}",
                    failed_stage_name="<execute>")
            bad = [(leaf.stage.name, r) for leaf, r
                   in zip(motion_leaves, results) if not r.success]
            if bad:
                name, first = bad[0]
                self.get_logger().error(
                    f"task '{goal.task_name}' execution failed at "
                    f"'{name}': {first.message or 'SendTrajectory failed'}")
                goal_handle.abort()
                return Task.Result(
                    success=False,
                    error=first.message or "execution failed",
                    failed_stage_name=name)
            self._publish_feedback("executed", "", hint="winner executed")

        goal_handle.succeed()
        return Task.Result(success=True, error="", failed_stage_name="")

    # ------------------------------------------------------------------
    # introspection publishing (Sec. 7)
    # ------------------------------------------------------------------
    def _publish_task_description(self, executor) -> None:
        desc = executor.describe()
        msg = TaskDescription()
        msg.task_id = desc["task_id"]
        msg.stages = StageSpec.from_dict(desc["root"]).to_msg_list(StageSpecMsg)
        msg.stage_count = int(desc["stage_count"])
        msg.valid = bool(desc["valid"])
        msg.comment = desc["comment"] or ""
        self._pub_desc.publish(msg)

    def _publish_solution(self, stg, sol) -> None:
        self._attempt_count += 1
        self._solutions_total += 1
        msg = SolutionInfo()
        msg.task_id = self._active_task_id
        msg.stage_id = stg.stage_id
        msg.stage_name = stg.name
        msg.solution_id = sol.solution_id
        msg.cost = float(sol.cost)
        msg.success = True
        msg.comment = sol.comment or ""
        msg.planner_id = self._planner_id(stg, sol)
        msg.markers = MarkerArray()
        self._pub_sol.publish(msg)
        self._publish_feedback("computing", stg.name)

    def _publish_failure(self, stg, failure) -> None:
        self._attempt_count += 1
        msg = SolutionInfo()
        msg.task_id = self._active_task_id
        msg.stage_id = stg.stage_id
        msg.stage_name = stg.name
        msg.solution_id = NO_SOLUTION_ID
        msg.cost = float("inf")
        msg.success = False
        msg.comment = failure.message or ""
        msg.planner_id = self._planner_id(stg, None)
        msg.markers = MarkerArray()
        self._pub_sol.publish(msg)
        self._publish_feedback("computing", stg.name)

    def _publish_stage_statistics(self, executor) -> None:
        for stg in executor.root.subtree_stages():
            msg = StageStatistics()
            msg.task_id = executor.task_id
            msg.stage_id = stg.stage_id
            msg.stage_name = stg.name
            msg.stage_type = stg.stage_type()
            msg.attempt_count = stg.attempt_count
            msg.success_count = len(stg.solutions)
            msg.last_cost = (stg.solutions[-1].cost
                             if stg.solutions else float("inf"))
            msg.total_compute_time = stg.compute_time
            self._pub_stat.publish(msg)

    def _publish_feedback(self, state: str, current_stage: str,
                          hint: str = "") -> None:
        gh = self._active_goal_handle
        if gh is None:
            return
        fb = Task.Feedback()
        fb.feedback = hint
        fb.current_stage_name = current_stage
        fb.attempts = self._attempt_count
        fb.solutions = self._solutions_total
        gh.publish_feedback(fb)

    @staticmethod
    def _planner_id(stg, sol) -> str:
        req = getattr(sol, "plan_request", None)
        if req is not None and getattr(req, "planner", None) is not None:
            return str(req.planner)
        planner = getattr(stg, "planner", None)
        return str(planner) if planner is not None else ""

    @staticmethod
    def _last_failed_stage(executor) -> str:
        for stg in reversed(executor.root.subtree_stages()):
            if stg.failures:
                return stg.name
        return ""


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TaskConstructorNode()
    # >= 2 threads required: the task solve blocks a worker thread while
    # others deliver service/action responses and /joint_states.
    executor = MultiThreadedExecutor(num_threads=4)
    try:
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
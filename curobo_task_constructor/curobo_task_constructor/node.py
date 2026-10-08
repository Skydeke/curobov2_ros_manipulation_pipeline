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
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import JointState

from curobo_task_constructor_interfaces.action import ExecuteTaskSolution, Task
from curobo_task_constructor_interfaces.msg import (
    SolutionInfo,
    StageSpec as StageSpecMsg,
    StageStatistics,
    TaskDescription,
    TaskSolution,
    TaskStatistics,
)

#: Populate STAGE_REGISTRY with the builtin stages (registry is import-driven).
import curobo_task_constructor.stages  # noqa: F401
from curobo_task_constructor.core.state import SceneDiff
from curobo_task_constructor.core.stage import chain_hook
from curobo_task_constructor.executor import TaskExecutor
from curobo_task_constructor.graph.spec import StageSpec
from curobo_task_constructor.robot import CuroboServerInterface
from curobo_task_constructor.robot.curobo import _SERVICE_TIMEOUT

#: id carried by rows with no stored record (uint32 field; 0xFFFFFFFF = none):
#: non-ranked seeds share it (they are display-only, never looked up).
NO_SOLUTION_ID = 0xFFFFFFFF

#: stage_id addressing a whole stored chain rather than one segment.
NO_STAGE = 0xFFFFFFFF

#: Depth of the introspection publishers/subscribers. The node publishes one
#: StageStatistics per stage per compute pass, one per planner try, and one
#: SolutionInfo per emit/failure in tight bursts at solve time; the pick task
#: alone has 27 stages, each motion stage planning up to 3 tries. A depth of
#: 10 (the rviz default) silently DROPPED the messages for the first ~17 stages,
#: so the panel showed compute time / attempts only for the last 10 — which are
#: mostly never-run fallback stages. The multi-attempt planning happened, but
#: its statistics were invisible for exactly the stages that matter. Sized to
#: comfortably exceed the largest burst with margin.
INTROSPECTION_QOS_DEPTH = 500

_ACTION_TOPIC = "/curobo_task_constructor/task"
_EXECUTE_ACTION_TOPIC = "/curobo_task_constructor/execute_task_solution"
_TASK_SOLUTION_TOPIC = "/curobo_task_constructor/task_solutions"
#: Solved tasks kept for ExecuteTaskSolution (task_id -> ranked chains, MTC's
#: stored task solutions). Bounded: trajectories are small, but unbounded
#: growth across pick cycles is not.
_MAX_STORED_TASKS = 3
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
        self.declare_parameter("live_stats_period", 0.5)

        self._robot_config_path = str(
            self.get_parameter("robot_config_path").value)
        self._joint_states_topic = str(
            self.get_parameter("joint_states_topic").value)
        planner_param = int(self.get_parameter("planner").value)
        service_timeout = float(self.get_parameter("service_timeout").value)

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
            depth=INTROSPECTION_QOS_DEPTH, history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._pub_desc = self.create_publisher(
            TaskDescription, _INTROSPECT_QOS_TOPICS["task_description"], qos)
        self._pub_sol = self.create_publisher(
            SolutionInfo, _INTROSPECT_QOS_TOPICS["solution_info"], qos)
        self._pub_stat = self.create_publisher(
            StageStatistics, _INTROSPECT_QOS_TOPICS["stage_statistics"], qos)
        self._pub_task_stat = self.create_publisher(
            TaskStatistics, "/curobo_task_constructor/task_statistics", qos)
        self._pub_chains = self.create_publisher(
            TaskSolution, _TASK_SOLUTION_TOPIC, qos)

        # Solve serialization: one task at a time on the shared curobo server.
        self._solve_lock = threading.Lock()
        self._solve_in_progress = False

        # Stored complete solutions for ExecuteTaskSolution (task_id -> executor
        # holding the ranked chains, MTC's stored task solutions). Driving a
        # stored chain replays its own segment requests (force_cached) with
        # no replanning. Bounded: oldest task evicted past _MAX_STORED_TASKS.
        self._stored = {}
        # One execution at a time on the shared server (same reason as solves)
        # plus its cancel flag (MTC preempt stops between segments).
        self._exec_lock = threading.Lock()
        self._exec_in_progress = False
        self._exec_cancel = threading.Event()

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
        self._exec_action = ActionServer(
            self, ExecuteTaskSolution, _EXECUTE_ACTION_TOPIC,
            execute_callback=self._execute_solution_callback,
            goal_callback=self._execute_goal_callback,
            cancel_callback=self._execute_cancel_callback,
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
            # Clear the panel LAST, after the result is on its way, so the
            # operator can still read the stage tree for a finished task. The
            # publish is the panel's only "nothing is running" signal: without
            # it the tree keeps showing the last task with its Re-run button
            # live, which reads as a pending/idle plan rather than a finished
            # one. The pick pipeline fires a dozen tasks through this one
            # action, so the stale tree is the normal case, not the exception.
            self._publish_task_cleared()
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
            # Non-ranked seeds (best-of-N discards): rows without storing
            # or lifting, so attempts always sum to tries. Containers never
            # see this hook (only on_solution lifts).
            stg.on_considered = chain_hook(
                stg.on_considered, partial(self._publish_considered, stg))
            # Per-planner-try progress: best-of-N stages call the planner
            # several times but emit once, so without this the panel would
            # sit still until the Nth try finished. Each try publishes one
            # stage rollup; attempts/compute-time walk 1, 2, 3 live.
            stg.on_progress = chain_hook(
                stg.on_progress, self._publish_stage_progress)

        self._publish_task_description(executor)
        self._publish_feedback("planning", "", hint="task accepted")

        # Live statistics while planning: attempts and compute time would
        # otherwise only appear after the solve (see
        # _publish_stage_statistics), leaving the panel static for the whole
        # plan. A wall timer snapshots the per-stage rollups during compute
        # and is destroyed right after plan() returns, before the final
        # burst below. Period 0 disables the live snapshots.
        live_stats_timer = None
        try:
            period = float(self.get_parameter("live_stats_period").value)
        except Exception:
            period = 0.5
        if period > 0.0:
            live_stats_timer = self.create_timer(
                period, lambda: self._publish_stage_statistics(executor))
        try:
            # Per-pass rollups (MTC publishTaskState): statistics stream
            # after every compute pass on top of the per-try progress above.
            ok = executor.plan(progress_callback=lambda:
                               self._publish_stage_statistics(executor))
        finally:
            if live_stats_timer is not None:
                live_stats_timer.cancel()
                try:
                    self.destroy_timer(live_stats_timer)
                except Exception:
                    pass
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
        self._publish_winner_trajectory(sol)
        self._store_and_publish_solutions(executor)

        if goal.execute and sol is not None:
            ok_exec, error, failed_stage = self._drive_solution(
                executor, sol,
                lambda stage_name: self._publish_feedback(
                    "executing", stage_name))
            if not ok_exec:
                if failed_stage == "<execute>":
                    self.get_logger().error(
                        f"task '{goal.task_name}' execution failed: {error}")
                    goal_handle.abort()
                    return Task.Result(
                        success=False, error=f"execution failed: {error}",
                        failed_stage_name="<execute>")
                self.get_logger().error(
                    f"task '{goal.task_name}' execution failed at "
                    f"'{failed_stage}': {error or 'SendTrajectory failed'}")
                goal_handle.abort()
                return Task.Result(
                    success=False,
                    error=error or "execution failed",
                    failed_stage_name=failed_stage)
            self._publish_feedback("executed", "", hint="winner executed")

        goal_handle.succeed()
        return Task.Result(success=True, error="", failed_stage_name="")

    def _drive_solution(self, executor, sol, feedback_fn=None,
                        cancel_event=None):
        """Drive one ranked solution segment by segment (shared by the Task
        execute flag and ExecuteTaskSolution).

        Map each drive failure back to its leaf stage so clients can tell a
        variant reach failure (approach_*/grasp_*) from a post-reach failure
        (close/lift/return/...). Only motion leaves yield results; scene-only
        leaves (modify_scene) contribute none, so filter them before zipping.
        Returns (ok, error, failed_stage_name); cancelled counts as not-ok
        with failed_stage_name "<cancelled>".
        """
        try:
            motion_leaves = [
                leaf for leaf in executor.flatten_leaves(sol)
                if leaf.plan_request is not None]

            def _progress(stage_name):
                if feedback_fn is not None:
                    feedback_fn(stage_name)

            results = executor.execute(
                sol, progress_callback=_progress, cancel_event=cancel_event)
        except Exception as exc:
            return False, f"{exc}", "<execute>"
        if cancel_event is not None and cancel_event.is_set():
            return False, "cancelled", "<cancelled>"
        bad = [(leaf.stage.name, r) for leaf, r
               in zip(motion_leaves, results) if not r.success]
        if bad:
            name, first = bad[0]
            return False, first.message or "execution failed", name
        if len(results) < len(motion_leaves):
            # Truncated without a failing result (e.g. divergence guard):
            # the chain stopped early, report where.
            idx = len(results)
            name = motion_leaves[idx].stage.name if idx < len(motion_leaves) else ""
            msg = results[-1].message if results else "chain stopped early"
            return False, msg, name
        return True, "", ""

    def _store_and_publish_solutions(self, executor) -> None:
        """Keep the solved executor for ExecuteTaskSolution and list its chains.

        MTC keeps all task solutions executable; the panel lists them and
        drives any one of them with no replanning. Stored per task_id,
        oldest task evicted past _MAX_STORED_TASKS.
        """
        self._stored[executor.task_id] = executor
        while len(self._stored) > _MAX_STORED_TASKS:
            self._stored.pop(next(iter(self._stored)))
        for index, sol in enumerate(executor.rank()):
            leaves = executor.flatten_leaves(sol)
            msg = TaskSolution()
            msg.task_id = executor.task_id
            msg.solution_index = index
            msg.cost = float(sol.cost)
            for leaf in leaves:
                msg.stage_names.append(getattr(leaf.stage, "name", ""))
                msg.stage_ids.append(int(getattr(leaf.stage, "stage_id", -1)))
                msg.solution_ids.append(int(getattr(leaf, "solution_id", -1)))
            self._pub_chains.publish(msg)

    # ------------------------------------------------------------------
    # execute-solution server (MTC ExecuteTaskSolution equivalent)
    # ------------------------------------------------------------------
    @staticmethod
    def _lookup_solution(executor, solution_index):
        chains = executor.rank()
        if 0 <= solution_index < len(chains):
            return chains[solution_index]
        return None

    @staticmethod
    def _lookup_segment(executor, stage_id, attempt_id):
        """One stored successful attempt (single-segment execute)."""
        for stg in executor.root.subtree_stages():
            if stg.stage_id != stage_id:
                continue
            for sol in stg.solutions:
                if sol.solution_id == attempt_id:
                    return sol
        return None

    def _execute_goal_callback(self, goal_request):
        with self._exec_lock:
            if self._exec_in_progress:
                self.get_logger().warn(
                    "rejecting execute goal: another execution is in progress")
                return GoalResponse.REJECT
        executor = self._stored.get(goal_request.task_id)
        if executor is None:
            self.get_logger().warn(
                f"rejecting execute goal: unknown task {goal_request.task_id} "
                "(send the task first)")
            return GoalResponse.REJECT
        if int(goal_request.stage_id) == NO_STAGE:
            ok = self._lookup_solution(
                executor, goal_request.solution_index) is not None
        else:
            ok = self._lookup_segment(
                executor, int(goal_request.stage_id),
                int(goal_request.attempt_id)) is not None
        if not ok:
            self.get_logger().warn(
                f"rejecting execute goal: unknown solution "
                f"{goal_request.task_id}#{goal_request.solution_index} "
                "(send the task first)")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _execute_cancel_callback(self, goal_handle):
        self._exec_cancel.set()
        return CancelResponse.ACCEPT

    def _execute_solution_callback(self, goal_handle):
        with self._exec_lock:
            self._exec_in_progress = True
        self._exec_cancel.clear()
        try:
            return self._run_execute_solution(goal_handle)
        finally:
            with self._exec_lock:
                self._exec_in_progress = False

    def _run_execute_solution(self, goal_handle) -> ExecuteTaskSolution.Result:
        def _unknown(detail: str) -> ExecuteTaskSolution.Result:
            goal_handle.abort()
            return ExecuteTaskSolution.Result(
                success=False,
                error=f"unknown solution {goal.task_id}#{goal.solution_index} "
                      f"({detail} — send the task first)",
                failed_stage_name="<lookup>")

        goal = goal_handle.request
        executor = self._stored.get(goal.task_id)
        if executor is None:
            return _unknown("task evicted or never solved")
        if int(goal.stage_id) == NO_STAGE:
            sol = self._lookup_solution(executor, goal.solution_index)
            if sol is None:
                return _unknown("solution index out of range")
        else:
            # One stored segment, driven standalone (MTC executes any
            # solution, whole or part). Only successful attempts carry a
            # plannable trajectory; failures have nothing to drive.
            sol = self._lookup_segment(
                executor, int(goal.stage_id), int(goal.attempt_id))
            if sol is None or sol.plan_request is None:
                return _unknown("no successful attempt there")
            from curobo_task_constructor.core.stage import Solution
            sol = Solution(
                start=sol.start, end=sol.end, trajectory=sol.trajectory,
                cost=sol.cost, comment=sol.comment, response=sol.response,
                children=[sol], plan_request=sol.plan_request,
                scene_ops=list(sol.scene_ops or []))
        motion_total = sum(
            1 for leaf in executor.flatten_leaves(sol)
            if leaf.plan_request is not None)
        fb = ExecuteTaskSolution.Feedback()
        fb.segments_total = motion_total
        ok, error, failed = self._drive_solution(
            executor, sol, lambda name: self._execute_feedback(
                goal_handle, fb, name),
            cancel_event=self._exec_cancel)
        if not ok and failed == "<cancelled>":
            goal_handle.canceled()
            return ExecuteTaskSolution.Result(
                success=False, error="cancelled", failed_stage_name="")
        if not ok:
            goal_handle.abort()
            return ExecuteTaskSolution.Result(
                success=False, error=error, failed_stage_name=failed)
        goal_handle.succeed()
        return ExecuteTaskSolution.Result(
            success=True, error="", failed_stage_name="")

    def _execute_feedback(self, goal_handle, fb, stage_name: str) -> None:
        fb.current_stage_name = stage_name
        fb.segments_done += 1
        goal_handle.publish_feedback(fb)

    # ------------------------------------------------------------------
    # introspection publishing (Sec. 7)
    # ------------------------------------------------------------------
    @staticmethod
    def _interface_flags(stage) -> int:
        """Resolved interface as MTC StageDescription.flags bits (0x01 reads
        start, 0x02 reads end, 0x04 writes next start, 0x08 writes prev end).
        Our InterfaceFlag(read, write) pair maps exactly onto MTC's
        READS_START/READS_END/WRITES_NEXT_START/WRITES_PREV_END.
        """
        try:
            start, end = stage.interface_flags()
        except Exception:
            return 0
        flags = 0
        if start.read:
            flags |= 0x01
        if end.read:
            flags |= 0x02
        if end.write:
            flags |= 0x04
        if start.write:
            flags |= 0x08
        return flags

    def _publish_task_description(self, executor) -> None:
        desc = executor.describe()
        msg = TaskDescription()
        msg.task_id = desc["task_id"]
        msg.stages = StageSpec.from_dict(desc["root"]).to_msg_list(StageSpecMsg)
        live = {stg.stage_id: stg
                for stg in executor.root.subtree_stages()}
        for stage_msg in msg.stages:
            stg = live.get(int(stage_msg.id))
            if stg is not None:
                stage_msg.flags = self._interface_flags(stg)
        msg.stage_count = int(desc["stage_count"])
        msg.valid = bool(desc["valid"])
        msg.comment = desc["comment"] or ""
        self._pub_desc.publish(msg)

    def _publish_task_cleared(self) -> None:
        """Publish an EMPTY task description: the panel's "task finished" signal.

        An empty ``stages`` list (and ``stage_count`` 0) is unambiguous — the
        executor's real description is never stage-less, since ``init()``
        rejects a spec with no root stage, and a spec is rejected before
        ``_publish_task_description`` ever runs. So an empty message cannot be
        confused with a real one, and any client (the rviz panel, a log tap, a
        test) can treat it as "there is no task in flight".

        Transient-local durability means a panel that opens LATER still sees
        the cleared state instead of replaying the previous task's structure.
        """
        msg = TaskDescription()
        self._pub_desc.publish(msg)

    def _publish_solution(self, stg, sol) -> None:
        self._attempt_count += 1
        self._solutions_total += 1
        msg = SolutionInfo()
        msg.id = int(sol.solution_id)
        msg.cost = float(sol.cost)
        msg.comment = sol.comment or ""
        msg.stage_id = stg.stage_id
        msg.planner_id = self._planner_id(stg, sol)
        msg.task_id = self._active_task_id
        msg.stage_name = stg.name
        msg.success = True
        msg.ranked = True
        msg.trajectory = self._solution_trajectory(sol)
        msg.tool_poses = self._solution_tool_poses(sol)
        msg.markers = self._attempt_markers(sol, msg.tool_poses, success=True)
        self._pub_sol.publish(msg)
        self._publish_feedback("computing", stg.name)

    def _publish_considered(self, stg, sol) -> None:
        """One non-ranked success row per discarded best-of-N seed: every
        planner call is visible exactly once, so successful + failed rows
        always sum to planner attempts. No counters, no feedback here — the
        try already ticked those when it ran."""
        msg = SolutionInfo()
        msg.id = NO_SOLUTION_ID
        msg.cost = float(sol.cost)
        msg.comment = sol.comment or ""
        msg.stage_id = stg.stage_id
        msg.planner_id = self._planner_id(stg, sol)
        msg.task_id = self._active_task_id
        msg.stage_name = stg.name
        msg.success = True
        msg.ranked = False
        msg.trajectory = self._solution_trajectory(sol)
        msg.tool_poses = []
        msg.markers = self._attempt_markers(sol, [], success=True)
        self._pub_sol.publish(msg)

    def _publish_failure(self, stg, failure) -> None:
        self._attempt_count += 1
        msg = SolutionInfo()
        msg.id = int(failure.failure_id)
        msg.cost = float("inf")
        msg.comment = failure.message or ""
        msg.stage_id = stg.stage_id
        msg.planner_id = self._planner_id(stg, None)
        msg.task_id = self._active_task_id
        msg.stage_name = stg.name
        msg.success = False
        msg.ranked = False
        msg.markers = self._failure_markers(failure)
        msg.trajectory = []
        msg.tool_poses = []
        self._pub_sol.publish(msg)
        self._publish_feedback("computing", stg.name)

    def _failure_markers(self, failure) -> list:
        """Red marker at the state a failed attempt started from (best
        effort): MTC marks failed goals red, and the start state is the
        only pose a failure always carries."""
        from visualization_msgs.msg import Marker
        out = []
        try:
            state = getattr(failure, "from_state", None)
            js = getattr(state, "joint_state", None) if state else None
            if js is None:
                return out
            pose = self._solver.fk(js)
            mark = Marker()
            mark.header.frame_id = "world"
            mark.ns = "failed_at"
            mark.id = 0
            mark.type = Marker.SPHERE
            mark.action = Marker.ADD
            mark.pose = pose
            mark.scale.x = mark.scale.y = mark.scale.z = 0.025
            mark.color.r, mark.color.a = 1.0, 1.0
            out.append(mark)
        except Exception as exc:
            self.get_logger().debug(f"failure markers failed: {exc!r}")
        return out

    def _publish_stage_statistics(self, executor) -> None:
        for stg in executor.root.subtree_stages():
            self._pub_stat.publish(
                self._stage_statistics(stg, executor.task_id))
        self._publish_task_statistics(executor)

    def _publish_task_statistics(self, executor) -> None:
        """Whole-task rollup, MTC TaskStatistics shape (one entry per
        stage). Published alongside the per-stage stream after every
        compute pass and at the end, so MTC-shaped tooling can read one
        message instead of joining the stream."""
        msg = TaskStatistics()
        msg.task_id = executor.task_id
        for stg in executor.root.subtree_stages():
            msg.stages.append(self._stage_statistics(stg, executor.task_id))
        self._pub_task_stat.publish(msg)

    def _publish_stage_progress(self, stg) -> None:
        """One stage's rollup after every planner attempt (MTC per-pass
        TaskStatistics, at try granularity).

        ``on_progress`` fires inside a stage's multi-attempt loop, so each
        try streams attempts/compute-time immediately instead of only when
        the stage emits after its Nth try.
        """
        self._pub_stat.publish(
            self._stage_statistics(stg, self._active_task_id))

    @staticmethod
    def _stage_statistics(stg, task_id: str) -> StageStatistics:
        msg = StageStatistics()
        msg.id = stg.stage_id
        msg.solved = [int(s.solution_id) for s in stg.solutions]
        msg.failed = [int(f.failure_id) for f in stg.failures]
        msg.num_failed = len(stg.failures)
        msg.total_compute_time = stg.compute_time  # seconds
        msg.task_id = task_id
        msg.stage_name = stg.name
        msg.stage_type = stg.stage_type()
        msg.attempt_count = stg.attempt_count
        msg.success_count = len(stg.solutions)
        msg.last_cost = min((s.cost for s in stg.solutions),
                            default=float("inf"))
        return msg

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

    def _publish_winner_trajectory(self, sol) -> None:
        """Show the winning solution in RViz like ``Task.publish`` does.

        The panel republishes whatever solution is selected, but a solve
        without an open panel should still land on the trajectory display:
        transient_local keeps it for a display that subscribes later.
        """
        if sol is None:
            return
        try:
            from curobo_task_constructor.viz import publish_solution_trajectory
            publish_solution_trajectory(self, sol)
        except Exception as exc:
            self.get_logger().debug(f"winner trajectory publish failed: {exc!r}")

    def _attempt_markers(self, sol, tool_poses, success: bool) -> list:
        """Debug markers for one attempt (MTC stage start/goal frames).

        MTC's stages mark the start frame and the goal frame(s) green on
        success and red on failure; the panel shows the selected attempt's
        markers verbatim. Here: one sphere per goal candidate (steel blue,
        so every alternative the planner chose from stays visible), the
        start tool pose green, and the reached tool pose green (success) or
        red (failure).
        """
        from visualization_msgs.msg import Marker
        out = []
        try:
            req = getattr(sol, "plan_request", None)
            goalsets = list(getattr(req, "goalsets", None) or [])
            positions = [p.position for gs in goalsets
                         for p in (getattr(gs, "poses", None) or [])]
            if positions:
                spheres = Marker()
                spheres.header.frame_id = "world"
                spheres.ns = "candidates"
                spheres.id = 0
                spheres.type = Marker.SPHERE_LIST
                spheres.action = Marker.ADD
                spheres.scale.x = spheres.scale.y = spheres.scale.z = 0.02
                spheres.color.r, spheres.color.g = 0.3, 0.6
                spheres.color.b, spheres.color.a = 1.0, 0.8
                spheres.points.extend(positions)
                out.append(spheres)
            poses = list(tool_poses or [])
            if poses:
                for i, (label, pose) in enumerate(
                        (("start", poses[0]), ("end", poses[-1]))):
                    mark = Marker()
                    mark.header.frame_id = "world"
                    mark.ns = label
                    mark.id = 1 + i
                    mark.type = Marker.SPHERE
                    mark.action = Marker.ADD
                    mark.pose = pose
                    mark.scale.x = mark.scale.y = mark.scale.z = 0.025
                    if success or i == 0:
                        mark.color.g, mark.color.a = 1.0, 1.0
                    else:
                        mark.color.r, mark.color.a = 1.0, 1.0
                    out.append(mark)
        except Exception as exc:
            self.get_logger().debug(f"attempt markers failed: {exc!r}")
        return out

    @staticmethod
    def _solution_trajectory(sol) -> list:
        """Solution waypoints as sensor_msgs/JointState (empty on failure).

        Full-clone trajectory playback (MTC TaskDisplay equivalent): the
        panel scrubs these waypoints with a slider instead of only coloring
        the stage row. Waypoints may be ROS messages already or ROS-free
        stubs (tests) carrying name/position attrs.
        """
        out = []
        for wp in (getattr(sol, "trajectory", None) or []):
            if isinstance(wp, JointState):
                out.append(wp)
                continue
            js = JointState()
            js.name = [str(n) for n in (getattr(wp, "name", []) or [])]
            js.position = [float(v) for v in (getattr(wp, "position", []) or [])]
            out.append(js)
        return out

    def _solution_tool_poses(self, sol) -> list:
        """FK tool pose per solution waypoint (best-effort, may be empty)."""
        traj = list(getattr(sol, "trajectory", None) or [])
        if not traj:
            return []
        try:
            fk = getattr(self._solver, "fk_batch", None)
            if fk is None:
                return []
            poses = fk(traj)
            return list(poses or [])
        except Exception as exc:
            self.get_logger().debug(f"tool_poses FK failed: {exc!r}")
            return []

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
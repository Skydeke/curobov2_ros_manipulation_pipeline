"""ExecuteTaskSolution action server (MTC ExecuteTaskSolutionCapability equivalent).

Planning is local (in-process, like MoveIt: ``core.Task.plan()`` publishes
introspection itself); this node only executes. It hosts the
``execute_task_solution`` action at ``/curobo_task_constructor/``: the goal
carries a full ``Solution`` message, whose sub-trajectories drive in order
with no replanning — each segment's ``replay_goal`` replays the validated
plan from the curobo_server trajectory cache (a miss fails instead of
re-solving), and each segment's scene ops apply in chain order first.

Result is a ``moveit_msgs/MoveItErrorCodes`` (SUCCESS, INVALID_MOTION_PLAN
for an unusable goal, CONTROL_FAILED for a failed drive, PREEMPTED on
cancel); feedback reports ``sub_id``/``sub_no`` per driven sub-trajectory,
exactly like MoveIt.
"""

from __future__ import annotations

import threading

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState

from curobo_task_constructor_interfaces.action import ExecuteTaskSolution

from curobo_task_constructor import msg_convert
from curobo_task_constructor.executor import drive_solution_dict
from curobo_task_constructor.robot import CuroboServerInterface
from curobo_task_constructor.robot.curobo import _SERVICE_TIMEOUT

_EXECUTE_ACTION_TOPIC = "/curobo_task_constructor/execute_task_solution"


def _error_code(value: int):
    """moveit_msgs/MoveItErrorCodes value without importing moveit at module scope."""
    from moveit_msgs.msg import MoveItErrorCodes

    code = MoveItErrorCodes()
    code.val = int(value)
    return code


class TaskConstructorNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("curobo_task_constructor")
        self.declare_parameter("robot_config_path", "")
        self.declare_parameter("planner", -1)
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("service_timeout", _SERVICE_TIMEOUT)

        self._robot_config_path = str(
            self.get_parameter("robot_config_path").value)
        self._joint_states_topic = str(
            self.get_parameter("joint_states_topic").value)
        planner_param = int(self.get_parameter("planner").value)
        service_timeout = float(self.get_parameter("service_timeout").value)

        #: newest /joint_states reading, consumed by the adapter.
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

        # One execution at a time on the shared server, plus its cancel
        # flag (MTC preempt stops between segments).
        self._exec_lock = threading.Lock()
        self._exec_in_progress = False
        self._exec_cancel = threading.Event()

        self._planner_label = (
            planner_param if planner_param >= 0 else "server default")

        # Startup readiness gate: a wall timer polls until every
        # curobo_server service the interface touches AND the SendTrajectory
        # action are reachable; only then is the execute ActionServer
        # advertised. No blocking wait in __init__.
        self._solver.start_ready_poll(
            self._advertise_action_server, period=0.5)

    def _advertise_action_server(self):
        """Advertise the execute action server once the curobo_server is up."""
        if getattr(self, "_exec_action", None) is not None:
            return
        self._exec_action = ActionServer(
            self, ExecuteTaskSolution, _EXECUTE_ACTION_TOPIC,
            execute_callback=self._execute_solution_callback,
            goal_callback=self._execute_goal_callback,
            cancel_callback=self._execute_cancel_callback,
            callback_group=ReentrantCallbackGroup())
        self.get_logger().info(
            f"task constructor execute server ready at {_EXECUTE_ACTION_TOPIC} "
            f"(planner={self._planner_label})")

    # ------------------------------------------------------------------
    # joint state snapshot
    # ------------------------------------------------------------------
    def _on_joint_state(self, msg):
        self._joint_state_cache = self._solver.normalize_joint_state(msg)

    def _joint_state_watchdog(self):
        if self._joint_state_cache is None:
            if not self._js_warned:
                self._js_warned = True
                self.get_logger().warn(
                    f"no reading on '{self._joint_states_topic}' yet")
        else:
            self._js_warned = False

    # ------------------------------------------------------------------
    # execute-solution server (MTC ExecuteTaskSolutionCapability equivalent)
    # ------------------------------------------------------------------
    def _execute_goal_callback(self, goal_request):
        with self._exec_lock:
            if self._exec_in_progress:
                self.get_logger().warn(
                    "rejecting execute goal: another execution is in progress")
                return GoalResponse.REJECT
        try:
            d = msg_convert.msg_to_solution_dict(goal_request.solution)
        except Exception as exc:
            self.get_logger().warn(f"rejecting execute goal: {exc!r}")
            return GoalResponse.REJECT
        if not (d.get("sub_trajectories") or d.get("sub_solution")):
            self.get_logger().warn(
                "rejecting execute goal: solution carries nothing to drive")
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
        from moveit_msgs.msg import MoveItErrorCodes
        goal = goal_handle.request
        try:
            d = msg_convert.msg_to_solution_dict(goal.solution)
        except Exception as exc:
            goal_handle.abort()
            return ExecuteTaskSolution.Result(
                error_code=_error_code(
                    MoveItErrorCodes.INVALID_MOTION_PLAN))
        segments = list(d.get("sub_trajectories", None) or [])
        if not segments and not d.get("sub_solution"):
            goal_handle.abort()
            return ExecuteTaskSolution.Result(
                error_code=_error_code(
                    MoveItErrorCodes.INVALID_MOTION_PLAN))
        # MTC constructMotionPlan warns (not fails) when a non-empty segment
        # carries no controller names; free-space replay has no controllers.
        for seg in segments:
            info = (seg.get("execution_info", None) or {}) if isinstance(seg, dict) else {}
            names = (info.get("controller_names", None) or []) if isinstance(info, dict) else []
            traj = (seg.get("trajectory", None) or []) if isinstance(seg, dict) else []
            if traj and not names:
                self.get_logger().warn(
                    "execute: sub-trajectory carries no controller_names; "
                    "replaying without a controller claim (MTC warns here)")
                break

        applied_ops: list = []

        def _progress(index, total, info):
            fb = ExecuteTaskSolution.Feedback()
            fb.sub_id = int(index)
            fb.sub_no = int(total)
            goal_handle.publish_feedback(fb)

        try:
            results, failed_index, cancelled = drive_solution_dict(
                self._solver, d, applied_ops,
                progress_callback=_progress,
                cancel_event=self._exec_cancel)
        except Exception as exc:
            self.get_logger().error(f"execution failed: {exc!r}")
            goal_handle.abort()
            return ExecuteTaskSolution.Result(
                error_code=_error_code(MoveItErrorCodes.CONTROL_FAILED))
        if cancelled:
            goal_handle.canceled()
            return ExecuteTaskSolution.Result(
                error_code=_error_code(MoveItErrorCodes.PREEMPTED))
        if failed_index is not None:
            # Never swallow the reason: the drive returns the failed
            # segment's message (server error or endpoint-continuity
            # divergence vs the plan); without it the caller only sees
            # "execution failed" with no failing segment.
            reason = ""
            try:
                reason = str(
                    getattr(results[failed_index], "message", "") or "")
            except Exception:
                reason = ""
            info = (segments[failed_index].get("info", {})
                    if isinstance(segments[failed_index], dict) else {})
            self.get_logger().error(
                f"segment {failed_index + 1}/{len(segments)} failed "
                f"(planner={info.get('planner_id', '')!r} "
                f"comment={info.get('comment', '')!r}): {reason}")
            goal_handle.abort()
            return ExecuteTaskSolution.Result(
                error_code=_error_code(MoveItErrorCodes.CONTROL_FAILED))
        goal_handle.succeed()
        return ExecuteTaskSolution.Result(
            error_code=_error_code(MoveItErrorCodes.SUCCESS))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TaskConstructorNode()
    # >= 2 threads required: execution blocks a worker thread while
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

"""RViz publishing for task solutions (MTC ``Task.publish`` surface).

``task.publish(task.solutions[0])`` makes RViz show the solution, exactly
like MoveIt Task Constructor: the solution trajectory rides
``trajectory_msgs/JointTrajectory`` on :data:`SOLUTION_TRAJECTORY_TOPIC`,
which the ``CuroboTrajectoryDisplay`` (full-robot animation + trail)
renders. Both the local ``mtc.Task.publish`` and the action server use this
module, so local and remote solves display identically.
"""

from __future__ import annotations

#: Topic the trajectory display listens on by default.
SOLUTION_TRAJECTORY_TOPIC = "/curobo_task_constructor/solution_trajectory"

#: Waypoint spacing (s) stamped on the published points. The display
#: interpolates between points, so this only sets the playback rate.
DEFAULT_POINT_DT = 0.05


def waypoint_names_positions(waypoint):
    """(names, positions) out of a JointState-like waypoint."""
    names = [str(n) for n in (getattr(waypoint, "name", None) or [])]
    positions = [float(v) for v in (getattr(waypoint, "position", None) or [])]
    return names, positions


def to_joint_trajectory(waypoints, msg_cls=None, point_cls=None,
                        dt: float = DEFAULT_POINT_DT, fallback_state=None):
    """Waypoints -> ``trajectory_msgs/JointTrajectory`` (or stub equivalent).

    ``fallback_state`` supplies joints a planned waypoint omits. The planner's
    cspace has no gripper joints, so a pose leg's waypoints can carry only the
    arm; the task state still knows the commanded jaw. Without this the ghost
    rendered the finger at 0.0 (closed) even though the task had opened it.
    """
    points = list(waypoints or [])
    if not points:
        return None
    names, _ = waypoint_names_positions(points[0])
    if not names:
        return None
    fallback = {}
    if fallback_state is not None:
        fb_names, fb_positions = waypoint_names_positions(fallback_state)
        fallback = dict(zip(fb_names, fb_positions))
        for name in fb_names:
            if name not in names:
                names.append(name)
    if msg_cls is None or point_cls is None:
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        msg_cls = JointTrajectory
        point_cls = JointTrajectoryPoint
    msg = msg_cls()
    msg.joint_names = list(names)
    for i, wp in enumerate(points):
        wp_names, wp_positions = waypoint_names_positions(wp)
        values = dict(zip(wp_names, wp_positions))
        pt = point_cls()
        pt.positions = [
            float(values[name]) if name in values else float(fallback.get(name, 0.0))
            for name in names
        ]
        total = float(i) * float(dt)
        sec = int(total)
        pt.time_from_start.sec = sec
        pt.time_from_start.nanosec = int((total - sec) * 1e9)
        msg.points.append(pt)
    if not msg.points:
        return None
    return msg


def solution_waypoints(solution) -> list:
    """Trajectory waypoints out of a ``Solution`` (or a waypoint list)."""
    if isinstance(solution, (list, tuple)):
        return list(solution)
    return list(getattr(solution, "trajectory", None) or [])


def publish_solution_trajectory(node, solution, topic: str = SOLUTION_TRAJECTORY_TOPIC,
                                dt: float = DEFAULT_POINT_DT,
                                qos=None) -> bool:
    """Publish a solution's trajectory for the RViz trajectory display.

    The publisher is created once and cached on the node, transient_local
    so a display that subscribes later still picks up the last solution.
    Returns True when a message was published.
    """
    from trajectory_msgs.msg import JointTrajectory
    fallback = getattr(getattr(solution, "end", None), "joint_state", None)
    if fallback is None:
        fallback = getattr(getattr(solution, "start", None), "joint_state", None)
    msg = to_joint_trajectory(
        solution_waypoints(solution), dt=dt, fallback_state=fallback)
    if msg is None:
        return False
    cache = getattr(node, "_curobo_traj_pubs", None)
    if cache is None:
        cache = {}
        node._curobo_traj_pubs = cache
    pub = cache.get(topic)
    if pub is None:
        if qos is None:
            qos = _transient_local_qos()
        pub = node.create_publisher(JointTrajectory, topic, qos)
        cache[topic] = pub
    pub.publish(msg)
    return True


def _transient_local_qos():
    """Depth-1 transient_local QoS, or None without rclpy (duck-typed node)."""
    try:
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
        return QoSProfile(
            depth=1, history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
    except Exception:
        return None

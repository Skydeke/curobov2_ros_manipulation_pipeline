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
                        dt: float = DEFAULT_POINT_DT):
    """Waypoints -> ``trajectory_msgs/JointTrajectory`` (or stub equivalent).

    ``msg_cls``/``point_cls`` default to the ROS messages; tests pass
    duck-typed stand-ins. Returns ``None`` when there is nothing plottable
    (no waypoints, or waypoints without joint names).
    """
    points = list(waypoints or [])
    if not points:
        return None
    names, _ = waypoint_names_positions(points[0])
    if not names:
        return None
    if msg_cls is None or point_cls is None:
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        msg_cls = JointTrajectory
        point_cls = JointTrajectoryPoint
    msg = msg_cls()
    msg.joint_names = list(names)
    for i, wp in enumerate(points):
        _, positions = waypoint_names_positions(wp)
        if len(positions) < len(names):
            continue
        pt = point_cls()
        pt.positions = [float(v) for v in positions[:len(names)]]
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
    msg = to_joint_trajectory(solution_waypoints(solution), dt=dt)
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

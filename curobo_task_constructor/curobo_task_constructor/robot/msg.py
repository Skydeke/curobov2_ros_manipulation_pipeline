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



from curobo_task_constructor.robot.session import *

class _MsgMixin:
    def _to_goal(self, request: PlanRequest) -> TrajectoryGoal:
        goal = TrajectoryGoal()
        if request.start_pose is not None:
            goal.start_pose = self._to_joint_msg(request.start_pose)
        for gs in request.goalsets or []:
            goal.goalsets.append(self._to_goalset(gs))
        # MTC CartesianPath interpolation options (consumed by the CARTESIAN
        # planner; ignored by every other planner). Sentinels are passed
        # through verbatim — the server fills in its own defaults.
        goal.cartesian_step_size = float(
            getattr(request, "cartesian_step_size", 0.0) or 0.0
        )
        goal.cartesian_precision = float(
            getattr(request, "cartesian_precision", 0.0) or 0.0
        )
        return goal

    @staticmethod
    def _to_joint_msg(joint_state) -> JointState:
        js = JointState()
        js.name = list(getattr(joint_state, "name", []))
        js.position = [float(p) for p in (getattr(joint_state, "position", []) or [])]
        return js

    @staticmethod
    def _to_joint_trajectory_msg(trajectory):
        """A validated solution segment -> ``trajectory_msgs/JointTrajectory``. MTC parity: the segment's waypoints travel with the execute request so
        the server can stream them verbatim. ``trajectory`` is the solution
        dict form ``{"joint_names", "points", "velocities"}``; ``points`` are
        the joint positions per waypoint, ``velocities`` (optional) the matching
        joint velocities."""
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        msg = JointTrajectory()
        msg.joint_names = [str(n) for n in (trajectory.get("joint_names") or [])]
        points = list(trajectory.get("points") or [])
        velocities = list(trajectory.get("velocities") or [])
        for i, pt in enumerate(points):
            p = JointTrajectoryPoint()
            p.positions = [float(v) for v in (pt or [])]
            if i < len(velocities) and velocities[i]:
                p.velocities = [float(v) for v in velocities[i]]
            msg.points.append(p)
        return msg

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

__all__ = [
    '_MsgMixin',
    'parse_pose_msg',
]

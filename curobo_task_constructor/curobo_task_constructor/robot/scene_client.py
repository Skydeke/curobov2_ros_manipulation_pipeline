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

class _SceneClientMixin:
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
        """Ask the server for the whole scene, and return {name: record}. Cached, and re-queried at most once per ``_SCENE_CACHE_SECONDS``, because
        the executor's reverse sync walks every object and would otherwise make
        one service call per object per stage. On failure the PREVIOUS reply is returned rather than an empty dict."""
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

    def attach_object(self, name: str, joint_state=None) -> bool:
        req = AttachObject.Request()
        req.object_name = name
        if joint_state is not None:
            # The grasp pose: the payload's spheres are fitted at this
            # configuration, so they land in the flange frame where the object
            # is actually picked (see AttachObject.srv).
            req.joint_state = self._to_joint_msg(joint_state)
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

    def invalidate_scene_cache(self) -> None:
        """Public form of ``_invalidate_scene_cache`` (``RobotInterface``).

        A task's scene ops are materialized by the EXECUTE SERVER's process, so
        none of this interface's own mutation-invalidation fires when they
        land; a caller that must read the post-execution world (a scene
        verification) has to drop the cache itself.
        """
        self._invalidate_scene_cache()

    def plan_scene(self, scene, joint_state=None) -> None:
        """Reconcile the server's attach state to ``scene`` for planning. See ``RobotInterface.plan_scene``. The shared ``/curobo_server`` holds
        one attach state; a plan burst previews the scene each leg must be
        solved in (MTC applies ``ModifyPlanningScene`` to the planning scene at
        compute time)."""
        if scene is None:
            return
        if not self._plan_scene_active:
            self._plan_scene_active = True
            self._plan_scene_original = list(self.get_attached_objects() or [])
            self._plan_scene_attached = None
        desired = getattr(scene, "attached_object", None)
        if desired == self._plan_scene_attached:
            return
        if self._plan_scene_attached is not None:
            self.detach_object(self._plan_scene_attached)
        if desired is not None:
            self.attach_object(desired, joint_state=joint_state)
        self._plan_scene_attached = desired

    def end_plan_scene(self) -> None:
        """Restore the attach state captured by ``plan_scene``. Called at the end of a plan burst (``TaskExecutor.plan``). Detaches the
        previewed object and re-attaches whatever was attached before, so
        execution starts from the task's base scene and materializes its scene
        ops in chain order."""
        if not self._plan_scene_active:
            return
        if self._plan_scene_attached is not None:
            self.detach_object(self._plan_scene_attached)
            self._plan_scene_attached = None
        for name in self._plan_scene_original:
            self.attach_object(name)
        self._plan_scene_active = False
        self._plan_scene_original = []

    # ------------------------------------------------------------------
    # conversions
    # ------------------------------------------------------------------

__all__ = [
    '_SceneClientMixin',
]

from .msg import parse_pose_msg

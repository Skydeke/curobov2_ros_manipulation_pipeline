"""Centralised ROS ↔ cuRobo type converters.

Every function in this module is pure (no ``rclpy`` dependency) so that
handler modules can be unit-tested without spinning up ROS.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from curobo.types import JointState as CuJointState
from curobo.types import Pose as CuPose
from curobo.scene import Cuboid, Cylinder, Mesh, Sphere
from geometry_msgs.msg import Point, Pose as RosPose
from moveit_msgs.msg import CollisionObject
from sensor_msgs.msg import JointState as RosJointState
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import rclpy.time


# ---------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------

def ros_pose_to_cu_pose(pose: RosPose) -> CuPose:
    return CuPose.from_list([
        pose.position.x, pose.position.y, pose.position.z,
        pose.orientation.w, pose.orientation.x, pose.orientation.y,
        pose.orientation.z,
    ])


def cu_pose_to_ros_pose(pose: CuPose) -> RosPose:
    r = RosPose()
    r.position.x = float(pose.position[0].item())
    r.position.y = float(pose.position[1].item())
    r.position.z = float(pose.position[2].item())
    r.orientation.w = float(pose.quaternion[0].item())
    r.orientation.x = float(pose.quaternion[1].item())
    r.orientation.y = float(pose.quaternion[2].item())
    r.orientation.z = float(pose.quaternion[3].item())
    return r


# ---------------------------------------------------------------------------
# JointState
# ---------------------------------------------------------------------------

def ros_joint_state_to_cu(state: RosJointState, device: str = "cuda:0") -> CuJointState:
    return CuJointState.from_position(
        position=torch.tensor(state.position, dtype=torch.float32, device=device).unsqueeze(0),
        joint_names=list(state.name),
    )


def cu_joint_state_to_ros(
    js: CuJointState,
    joint_names: Optional[List[str]] = None,
    stamp=None,
) -> RosJointState:
    q = js.position[0].cpu().numpy() if js.position is not None else []
    v = js.velocity[0].cpu().numpy() if js.velocity is not None else []
    n = joint_names or (js.joint_names if hasattr(js, "joint_names") and js.joint_names else [])
    r = RosJointState()
    if stamp is not None:
        r.header.stamp = stamp
    r.name = list(n)
    r.position = q.tolist() if hasattr(q, "tolist") else list(q)
    r.velocity = v.tolist() if hasattr(v, "tolist") else list(v)
    return r


# ---------------------------------------------------------------------------
# JointTrajectory
# ---------------------------------------------------------------------------

def cu_solution_to_joint_trajectory(
    js: CuJointState,
    dt: float,
    joint_names: Optional[List[str]] = None,
) -> JointTrajectory:
    traj = JointTrajectory()
    q_traj = js.position.view(-1, js.position.shape[-1]).cpu().numpy()
    vel = js.velocity.view(-1, js.position.shape[-1]).cpu().numpy() if js.velocity is not None else None
    acc = js.acceleration.view(-1, js.position.shape[-1]).cpu().numpy() if js.acceleration is not None else None
    for i in range(len(q_traj)):
        pt = JointTrajectoryPoint()
        pt.positions = q_traj[i].tolist()
        if vel is not None and i < len(vel):
            pt.velocities = vel[i].tolist()
        if acc is not None and i < len(acc):
            pt.accelerations = acc[i].tolist()
        pt.time_from_start.sec = int(i * dt)
        pt.time_from_start.nanosec = int((i * dt - pt.time_from_start.sec) * 1e9)
        traj.points.append(pt)
    traj.joint_names = joint_names if joint_names else (
        list(js.joint_names) if hasattr(js, "joint_names") and js.joint_names else []
    )
    return traj


# ---------------------------------------------------------------------------
# CollisionObject → cuRobo scene primitives
# ---------------------------------------------------------------------------

def collision_object_to_scene_objects(
    mv_object: CollisionObject,
) -> Tuple[List, bool]:
    objs: list = []
    pose = mv_object.pose
    world_pose = ros_pose_to_cu_pose(pose)
    ok = True

    for k in range(len(mv_object.primitives)):
        prim_pose = ros_pose_to_cu_pose(mv_object.primitive_poses[k])
        object_pose = world_pose.multiply(prim_pose).tolist()
        prim = mv_object.primitives[k]

        if prim.type == SolidPrimitive.BOX:
            objs.append(Cuboid(
                name=f"{mv_object.id}_{k}_cuboid",
                pose=object_pose,
                dims=[prim.dimensions[SolidPrimitive.BOX_X],
                      prim.dimensions[SolidPrimitive.BOX_Y],
                      prim.dimensions[SolidPrimitive.BOX_Z]],
            ))
        elif prim.type == SolidPrimitive.SPHERE:
            objs.append(Sphere(
                name=f"{mv_object.id}_{k}_sphere",
                pose=object_pose,
                radius=prim.dimensions[SolidPrimitive.SPHERE_RADIUS],
            ))
        elif prim.type == SolidPrimitive.CYLINDER:
            objs.append(Cylinder(
                name=f"{mv_object.id}_{k}_cylinder",
                pose=object_pose,
                height=prim.dimensions[SolidPrimitive.CYLINDER_HEIGHT],
                radius=prim.dimensions[SolidPrimitive.CYLINDER_RADIUS],
            ))
        elif prim.type == SolidPrimitive.CONE:
            ok = False
        else:
            ok = False

    for k in range(len(mv_object.meshes)):
        mesh_pose = ros_pose_to_cu_pose(mv_object.mesh_poses[k])
        object_pose = world_pose.multiply(mesh_pose).tolist()
        verts = [[v.x, v.y, v.z] for v in mv_object.meshes[k].vertices]
        tris = [[v.vertex_indices[0], v.vertex_indices[1], v.vertex_indices[2]]
                for v in mv_object.meshes[k].triangles]
        objs.append(Mesh(
            name=f"{mv_object.id}_{k}_mesh",
            pose=object_pose,
            vertices=verts,
            faces=tris,
        ))

    return objs, ok


# Import torch only where needed (prevents import-order issues)
import torch  # noqa: E402

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ROS leg: replay the IK benchmark goals through the running planner server.

Talks to the ``unified_planner`` node started by ``gen_traj.launch.py``:

- the parity world (``WORLD_IK`` — the upstream ``collision_table.yml`` table)
  is pushed via ``/unified_planner/add_object`` after
  ``/unified_planner/remove_all_objects``
- the server's collision cache is sized to the world (cuboid=1, mesh=0,
  voxel off) so the IK solver's Warp collision kernels match the native leg's
  ``{cuboid: 1}`` cache (``--no-size-cache`` leaves the server defaults for
  A/B)
- the IK solver is materialized with ``/unified_planner/warmup_ik`` (the
  server's lazy ``IKServices._init``; it picks up the sized cache and the
  registered world at construction — same construction order as the native
  leg's solver)
- each goal batch is solved through ``/unified_planner/ik_batch``
- per-goal success comes from ``joint_states_valid[i]`` (the server's
  best-seed convergence flag — the same ``result.success[i, 0]`` the native
  leg reads); the FK-verified pose errors are computed client-side from the
  returned first-seed joint states with the SAME formula as the native leg.

Only the ``cfree`` variant runs here: the server has no collision-free-off IK
mode. Timing is informational (client wall around each batch call, ms).
"""

# Standard Library
import time
from typing import Any, Dict, List, Optional

from geometry_msgs.msg import Pose as RosPose
from rclpy.node import Node

from isaac_ros_cumotion_interfaces.srv import (
    AddObject,
    IkBatch,
    SetCollisionCache,
    WarmupIK,
)
from std_srvs.srv import Trigger

from .compare import orientation_error_deg, position_error_mm
from .synthetic import (
    DEFAULT_ROBOT_CONFIG,
    IK_WORLD_CUBOIDS,
    WORLD_IK,
    fk_tool_pose,
    load_ik_goals,
    load_kinematics_cfg,
    order_ros_parity_setup,
)

SERVER_NODE = "unified_planner"
WARMUP_IK_SRV = f"{SERVER_NODE}/warmup_ik"
IK_BATCH_SRV = f"{SERVER_NODE}/ik_batch"
ADD_OBJECT_SRV = f"{SERVER_NODE}/add_object"
REMOVE_ALL_OBJECTS_SRV = f"{SERVER_NODE}/remove_all_objects"
SET_COLLISION_CACHE_SRV = f"{SERVER_NODE}/set_collision_cache"


class RosIkRunner(Node):
    """Service clients for the IK parity leg (mirrors ``RosBenchmarkRunner``)."""

    def __init__(self, service_timeout: float = 30.0):
        super().__init__("curobo_benchmark_ik_ros_runner")
        self._warmup_client = self.create_client(WarmupIK, WARMUP_IK_SRV)
        self._ik_client = self.create_client(IkBatch, IK_BATCH_SRV)
        self._add_client = self.create_client(AddObject, ADD_OBJECT_SRV)
        self._clear_client = self.create_client(Trigger, REMOVE_ALL_OBJECTS_SRV)
        self._cache_client = self.create_client(
            SetCollisionCache, SET_COLLISION_CACHE_SRV
        )

        for label, client in (
            ("warmup_ik", self._warmup_client),
            ("ik_batch", self._ik_client),
            ("add_object", self._add_client),
            ("remove_all_objects", self._clear_client),
            ("set_collision_cache", self._cache_client),
        ):
            if not client.wait_for_service(timeout_sec=service_timeout):
                self.get_logger().error(f"{label} service not available")
                raise RuntimeError(f"{label} service not available")
            self.get_logger().info(f"Connected to {client.srv_name}")

    def _call(self, client, request, timeout: float):
        import rclpy

        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if future.result() is None:
            raise RuntimeError(
                f"{client.srv_name}: service call failed (timeout={timeout}s)"
            )
        return future.result()

    def size_collision_cache(self, timeout: float = 120.0) -> None:
        """Size the server's collision cache to the IK world (native parity).

        Same rationale as ``ros_runner.size_collision_cache``: the IK solver's
        Warp collision kernels launch one thread per (sphere, padded cache
        slot), so an over-padded server cache (the 32/4 launch defaults)
        inflates the kernel grid vs the native leg's ``{cuboid: 1}`` cache.
        The one synchronous solver rebuild (~25-37 s) is paid inside the call.
        """
        request = SetCollisionCache.Request()
        request.obb = IK_WORLD_CUBOIDS
        request.mesh = 0
        request.blox = 0  # no cameras in the benchmark: disable the voxel layer
        self.get_logger().info(
            f"Sizing solver collision cache to IK world "
            f"(obb={request.obb}, mesh={request.mesh}, blox={request.blox})"
        )
        response = self._call(self._cache_client, request, timeout)
        if not response.success:
            raise RuntimeError(
                f"set_collision_cache failed: {getattr(response, 'message', '')}"
            )
        self.get_logger().info(
            f"Collision cache set: obb={response.obb_cache}, "
            f"mesh={response.mesh_cache}, blox={response.blox_cache}"
        )

    def clear_world(self, timeout: float = 30.0) -> None:
        response = self._call(self._clear_client, Trigger.Request(), timeout)
        if not response.success:
            raise RuntimeError(
                f"remove_all_objects failed: {getattr(response, 'message', '')}"
            )

    def add_world(self, timeout: float = 60.0) -> None:
        """Add the IK parity world (table cuboid) via add_object."""
        from .obstacle_convert import obstacles_dict_to_add_requests

        payloads = obstacles_dict_to_add_requests(WORLD_IK)
        self._add_payloads(payloads, timeout)

    def _add_payloads(self, payloads, timeout: float) -> None:
        from geometry_msgs.msg import Point as RosPoint

        for payload in payloads:
            request = AddObject.Request()
            request.type = int(payload["type"])
            request.name = payload["name"]
            px, py, pz, qw, qx, qy, qz = payload["pose"]
            request.pose.position.x = px
            request.pose.position.y = py
            request.pose.position.z = pz
            request.pose.orientation.w = qw
            request.pose.orientation.x = qx
            request.pose.orientation.y = qy
            request.pose.orientation.z = qz
            dx, dy, dz = payload["dims"]
            request.dimensions.x = dx
            request.dimensions.y = dy
            request.dimensions.z = dz
            cr, cg, cb, ca = payload["color"]
            request.color.r = cr
            request.color.g = cg
            request.color.b = cb
            request.color.a = ca
            if payload.get("mesh_file_path"):
                request.mesh_file_path = payload["mesh_file_path"]
            for v in payload.get("vertices", []):
                request.vertices.append(RosPoint(x=v[0], y=v[1], z=v[2]))
            request.triangles.extend(payload.get("triangles", []))
            response = self._call(self._add_client, request, timeout)
            if not response.success:
                raise RuntimeError(
                    f"add_object({request.name}) failed: "
                    f"{getattr(response, 'message', '')}"
                )

    def warmup_ik(self, batch: int, timeout: float = 120.0) -> None:
        request = WarmupIK.Request()
        request.batch_size = batch
        response = self._call(self._warmup_client, request, timeout)
        if not response.success:
            raise RuntimeError(f"warmup_ik failed: {getattr(response, 'message', '')}")

    def solve_batch(
        self,
        goals: List[Dict[str, Any]],
        batch_idx: int,
        kin,
        timeout: float = 120.0,
    ) -> List[Dict[str, Any]]:
        """Solve one goal batch; returns per-goal entries (schema as core leg)."""
        import torch

        request = IkBatch.Request()
        for g in goals:
            pose = RosPose()
            pose.position.x = float(g["position_xyz"][0])
            pose.position.y = float(g["position_xyz"][1])
            pose.position.z = float(g["position_xyz"][2])
            # geometry_msgs orientation is xyzw; the server converts to wxyz.
            pose.orientation.w = float(g["quaternion_wxyz"][0])
            pose.orientation.x = float(g["quaternion_wxyz"][1])
            pose.orientation.y = float(g["quaternion_wxyz"][2])
            pose.orientation.z = float(g["quaternion_wxyz"][3])
            request.poses.append(pose)

        t0 = time.perf_counter()
        response = self._call(self._ik_client, request, timeout)
        time_ms = (time.perf_counter() - t0) * 1000.0
        if not response.success:
            raise RuntimeError(
                f"ik_batch failed: {getattr(response, 'error_msg', '')}"
            )

        entries: List[Dict[str, Any]] = []
        solved = [js.position for js in response.joint_states]
        if solved and kin is not None:
            q = torch.tensor([list(s) for s in solved], dtype=torch.float32)
            pos, quat = fk_tool_pose(kin, q)
        for i, (g, js, valid) in enumerate(
            zip(goals, response.joint_states, response.joint_states_valid),
            start=1,
        ):
            entry: Dict[str, Any] = {
                "problem_name": f"ik_cfree_b{batch_idx:02d}_g{i:03d}",
                "scene_key": "ik_cfree",
                "capability": "ik",
                "variant": "cfree",
                "batch": batch_idx,
                "index": i,
                "n_goals": len(goals),
                "success": bool(valid.data),
                "time_ms": time_ms,
                "position_error_mm": None,
                "orientation_error_deg": None,
            }
            if entry["success"] and solved:
                entry["position_error_mm"] = position_error_mm(
                    [float(v) for v in pos[i - 1].tolist()], g["position_xyz"]
                )
                entry["orientation_error_deg"] = orientation_error_deg(
                    [float(v) for v in quat[i - 1].tolist()], g["quaternion_wxyz"]
                )
            entries.append(entry)
        return entries


def run_ik_ros(
    batch: int = 100,
    n_batches: int = 5,
    robot_config: str = DEFAULT_ROBOT_CONFIG,
    seed: int = 2,
    service_timeout: float = 30.0,
    call_timeout: float = 120.0,
    size_collision_cache: bool = True,
) -> List[Dict[str, Any]]:
    """Run the ROS-wrapped IK leg (``cfree`` variant) over the shared goals."""
    import rclpy

    from curobo.kinematics import Kinematics

    rclpy.init()
    node: Optional[RosIkRunner] = None
    try:
        node = RosIkRunner(service_timeout=service_timeout)
        goals_batches = load_ik_goals(
            variant="cfree",
            batch=batch,
            n_batches=n_batches,
            robot_config=robot_config,
            seed=seed,
        )
        kin = Kinematics(load_kinematics_cfg(robot_config))

        node.get_logger().info(
            f"ROS IK leg: {n_batches} batches of {batch} cfree goals "
            f"(robot_config={robot_config})"
        )

        # Cache-safe world registration (see order_ros_parity_setup): the
        # scene must never hold more cuboids than the active cache capacity,
        # so the order is clear -> size -> add -> warmup.
        order_ros_parity_setup(
            node,
            warmup=lambda *, timeout: node.warmup_ik(batch, timeout=timeout),
            size_collision_cache=size_collision_cache,
            timeout=call_timeout,
        )

        results: List[Dict[str, Any]] = []
        for b_idx, goals in enumerate(goals_batches, start=1):
            node.get_logger().info(f"Solving IK batch {b_idx} ...")
            results.extend(
                node.solve_batch(goals, b_idx, kin, timeout=call_timeout)
            )

        node.get_logger().info(
            f"ROS IK leg done: {len(results)} goals, "
            f"{sum(1 for r in results if r['success'])} succeeded"
        )
        return results
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
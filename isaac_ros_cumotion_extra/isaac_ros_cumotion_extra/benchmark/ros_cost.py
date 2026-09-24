# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ROS leg: replay the kinematics & collision configs through the FK service.

Talks to the ``unified_planner`` node started by ``gen_traj.launch.py`` and
mirrors the native ``core_cost`` leg:

- the parity world (``WORLD_COST`` — table + tall cuboid, the upstream
  ``cost_gradient_benchmark`` scene) is pushed via ``/add_object`` after
  ``/remove_all_objects``
- the server's collision cache is sized to the world (cuboid=2, mesh=0, voxel
  off) so the collision validator's Warp kernels match the native leg
  (``--no-size-cache`` leaves the server defaults)
- the FK model + validator are materialized with ``/warmup_fk`` (the server's
  lazy ``FKServices._init`` picks up the sized cache and registered world at
  construction — same construction order as the native leg)
- each config batch is evaluated through ``/fk_batch``: per-config tool pose
  (``poses``) and validity (``poses_valid``, the server's
  ``RobotCollisionChecker.validate`` — joint limits + self + scene collision)

Only the parity variant runs here (there is no server-side "plain FK" mode).
Timing is informational (client wall around each batch call, ms).
"""

# Standard Library
import time
from typing import Any, Dict, List, Optional

from rclpy.node import Node

from isaac_ros_cumotion_interfaces.srv import (
    AddObject,
    FkBatch,
    SetCollisionCache,
    WarmupFK,
)
from std_srvs.srv import Trigger

from .synthetic import (
    COST_WORLD_CUBOIDS,
    DEFAULT_ROBOT_CONFIG,
    WORLD_COST,
    load_cost_configs,
    order_ros_parity_setup,
)

SERVER_NODE = "unified_planner"
WARMUP_FK_SRV = f"{SERVER_NODE}/warmup_fk"
FK_BATCH_SRV = f"{SERVER_NODE}/fk_batch"
ADD_OBJECT_SRV = f"{SERVER_NODE}/add_object"
REMOVE_ALL_OBJECTS_SRV = f"{SERVER_NODE}/remove_all_objects"
SET_COLLISION_CACHE_SRV = f"{SERVER_NODE}/set_collision_cache"


class RosCostRunner(Node):
    """Service clients for the kinematics & collision parity leg."""

    def __init__(self, service_timeout: float = 30.0):
        super().__init__("curobo_benchmark_cost_ros_runner")
        self._warmup_client = self.create_client(WarmupFK, WARMUP_FK_SRV)
        self._fk_client = self.create_client(FkBatch, FK_BATCH_SRV)
        self._add_client = self.create_client(AddObject, ADD_OBJECT_SRV)
        self._clear_client = self.create_client(Trigger, REMOVE_ALL_OBJECTS_SRV)
        self._cache_client = self.create_client(
            SetCollisionCache, SET_COLLISION_CACHE_SRV
        )

        for label, client in (
            ("warmup_fk", self._warmup_client),
            ("fk_batch", self._fk_client),
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
        """Size the server's collision cache to the cost world (native parity)."""
        request = SetCollisionCache.Request()
        request.obb = COST_WORLD_CUBOIDS
        request.mesh = 0
        request.blox = 0  # no cameras in the benchmark: disable the voxel layer
        self.get_logger().info(
            f"Sizing solver collision cache to cost world "
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
        """Add the cost parity world (table + cube6 cuboids) via add_object."""
        from .obstacle_convert import obstacles_dict_to_add_requests

        self._add_payloads(obstacles_dict_to_add_requests(WORLD_COST), timeout)

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

    def warmup_fk(self, batch: int, timeout: float = 120.0) -> None:
        request = WarmupFK.Request()
        request.batch_size = batch
        response = self._call(self._warmup_client, request, timeout)
        if not response.success:
            raise RuntimeError(f"warmup_fk failed: {getattr(response, 'message', '')}")

    def evaluate_batch(
        self,
        configs: List[List[float]],
        batch_idx: int,
        timeout: float = 120.0,
    ) -> List[Dict[str, Any]]:
        """Evaluate one config batch through /fk_batch; returns per-config entries."""
        from sensor_msgs.msg import JointState as RosJointState

        request = FkBatch.Request()
        for q in configs:
            js = RosJointState()
            js.position = [float(v) for v in q]
            request.joint_states.append(js)

        t0 = time.perf_counter()
        response = self._call(self._fk_client, request, timeout)
        time_ms = (time.perf_counter() - t0) * 1000.0
        if not response.success:
            raise RuntimeError(
                f"fk_batch failed: {getattr(response, 'error_msg', '')}"
            )

        entries: List[Dict[str, Any]] = []
        for i, (config, pose, valid) in enumerate(
            zip(configs, response.poses, response.poses_valid),
            start=1,
        ):
            # orientation fields already expose curobo wxyz order (the server
            # maps quaternion[w,x,y,z] onto msg.orientation.{w,x,y,z}).
            entries.append(
                {
                    "problem_name": f"cost_b{batch_idx:02d}_g{i:03d}",
                    "scene_key": "cost",
                    "capability": "cost",
                    "batch": batch_idx,
                    "index": i,
                    "n_configs": len(configs),
                    "valid": bool(valid.data),
                    "position_xyz": [
                        float(pose.position.x),
                        float(pose.position.y),
                        float(pose.position.z),
                    ],
                    "quaternion_wxyz": [
                        float(pose.orientation.w),
                        float(pose.orientation.x),
                        float(pose.orientation.y),
                        float(pose.orientation.z),
                    ],
                    "time_ms": time_ms,
                }
            )
        return entries


def run_cost_ros(
    batch: int = 100,
    n_batches: int = 5,
    robot_config: str = DEFAULT_ROBOT_CONFIG,
    seed: int = 2,
    service_timeout: float = 30.0,
    call_timeout: float = 120.0,
    size_collision_cache: bool = True,
) -> List[Dict[str, Any]]:
    """Run the ROS-wrapped kinematics & collision leg over the shared configs."""
    import rclpy

    rclpy.init()
    node: Optional[RosCostRunner] = None
    try:
        node = RosCostRunner(service_timeout=service_timeout)
        config_batches = load_cost_configs(
            batch=batch,
            n_batches=n_batches,
            robot_config=robot_config,
            seed=seed,
        )
        node.get_logger().info(
            f"ROS cost leg: {n_batches} batches of {batch} configs "
            f"(robot_config={robot_config})"
        )

        # Cache-safe world registration (see order_ros_parity_setup): the
        # scene must never hold more cuboids than the active cache capacity,
        # so the order is clear -> size -> add -> warmup.
        order_ros_parity_setup(
            node,
            warmup=lambda *, timeout: node.warmup_fk(batch, timeout=timeout),
            size_collision_cache=size_collision_cache,
            timeout=call_timeout,
        )

        results: List[Dict[str, Any]] = []
        for b_idx, configs in enumerate(config_batches, start=1):
            node.get_logger().info(f"Evaluating FK batch {b_idx} ...")
            results.extend(
                node.evaluate_batch(configs, b_idx, timeout=call_timeout)
            )

        node.get_logger().info(
            f"ROS cost leg done: {len(results)} configs, "
            f"{sum(1 for r in results if r['valid'])} valid"
        )
        return results
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
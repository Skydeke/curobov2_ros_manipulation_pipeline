"""ROS example: compute forward kinematics via /cumotion/compute_fk.

Mirrors ``curobo.examples.getting_started.forward_kinematics`` using the
unified ``curobo_server_node``'s ``ComputeFK`` service.

Usage:

.. code-block:: bash

    ros2 run isaac_ros_cumotion ros_example_forward_kinematics

Expects ``curobo_server_node`` to be running with a Kinova Gen3 robot loaded.
"""

import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState as RosJointState

from isaac_ros_cumotion_interfaces.srv import ComputeFK


class ForwardKinematicsExample(Node):

    def __init__(self):
        super().__init__("ros_example_forward_kinematics")
        self.cli = self.create_client(ComputeFK, "cumotion/compute_fk")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for cumotion/compute_fk...")
        self.get_logger().info("Connected to cumotion/compute_fk")

    def run(self):
        req = ComputeFK.Request()

        # Single FK: one zero-angle configuration
        js = RosJointState()
        js.name = [
            "joint_1", "joint_2", "joint_3", "joint_4",
            "joint_5", "joint_6", "joint_7",
        ]
        js.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        req.joint_states = [js]

        # Also add a random configuration to demonstrate batched FK
        import random
        js2 = RosJointState()
        js2.name = js.name
        js2.position = [random.uniform(-1.0, 1.0) for _ in range(7)]
        req.joint_states = [js, js2]

        req.tool_frames = []
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if not future.result() or not future.result().success:
            self.get_logger().error(f"FK failed: {future.result().message if future.result() else 'no response'}")
            return False

        resp = future.result()
        n_configs = resp.num_configs
        n_frames = resp.num_frames

        self.get_logger().info(f"FK solved: {n_configs} configs x {n_frames} frames in {resp.solve_time_s:.4f}s")
        self.get_logger().info(f"Frame names: {resp.resolved_frame_names}")

        for i in range(n_configs):
            self.get_logger().info(f"  Config {i}:")
            for j in range(n_frames):
                idx = i * n_frames + j
                p = resp.tool_poses[idx]
                self.get_logger().info(
                    f"    {resp.resolved_frame_names[j]}: "
                    f"pos=({p.position.x:.4f}, {p.position.y:.4f}, {p.position.z:.4f})  "
                    f"quat=({p.orientation.w:.4f}, {p.orientation.x:.4f}, "
                    f"{p.orientation.y:.4f}, {p.orientation.z:.4f})"
                )

        return True


def main():
    rclpy.init()
    node = ForwardKinematicsExample()
    ok = node.run()
    node.destroy_node()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

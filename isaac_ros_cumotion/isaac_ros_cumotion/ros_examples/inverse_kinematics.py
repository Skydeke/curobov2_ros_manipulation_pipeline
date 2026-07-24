"""ROS example: solve inverse kinematics via /cumotion/compute_ik.

Mirrors ``curobo.examples.getting_started.inverse_kinematics`` using the
unified ``curobo_server_node``'s ``ComputeIK`` service.

Usage:

.. code-block:: bash

    ros2 run isaac_ros_cumotion ros_example_inverse_kinematics
"""

import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState as RosJointState
from geometry_msgs.msg import Pose as RosPose

from isaac_ros_cumotion_interfaces.srv import ComputeIK


class InverseKinematicsExample(Node):

    def __init__(self):
        super().__init__("ros_example_inverse_kinematics")
        self.cli = self.create_client(ComputeIK, "cumotion/compute_ik")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for cumotion/compute_ik...")
        self.get_logger().info("Connected to cumotion/compute_ik")

    def run_single_ik(self):
        req = ComputeIK.Request()
        p = RosPose()
        p.position.x, p.position.y, p.position.z = 0.4, 0.0, 0.4
        p.orientation.w = 1.0
        req.goal_poses = [p]
        req.tool_frame = ""

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()

        if resp and resp.success and resp.success[0]:
            self.get_logger().info(f"Single IK solved in {resp.solve_time_s:.4f}s")
            self.get_logger().info(f"  Position error: {resp.position_error[0] * 1000:.3f} mm")
            sol = resp.solutions[0]
            self.get_logger().info(f"  Joint angles: {[f'{v:.4f}' for v in sol.position]}")
            return True
        else:
            self.get_logger().error("Single IK failed")
            return False

    def run_batched_ik(self):
        n_poses = 10
        req = ComputeIK.Request()
        for i in range(n_poses):
            p = RosPose()
            p.position.x = 0.2 + i * 0.06
            p.position.y = 0.0
            p.position.z = 0.4
            p.orientation.w = 1.0
            req.goal_poses.append(p)
        req.tool_frame = ""

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()

        if resp:
            n_success = sum(1 for s in resp.success if s)
            self.get_logger().info(f"Batched IK: {n_success}/{n_poses} solved in {resp.solve_time_s:.4f}s")
            if n_success > 0:
                valid_errors = [resp.position_error[i] for i in range(n_poses) if resp.success[i]]
                avg_err = sum(valid_errors) / len(valid_errors)
                self.get_logger().info(f"  Mean position error: {avg_err * 1000:.3f} mm")
            return n_success > 0

        return False

    def run(self):
        ok = True
        self.get_logger().info("=== Single IK ===")
        ok &= self.run_single_ik()

        self.get_logger().info("=== Batched IK ===")
        ok &= self.run_batched_ik()

        return ok


def main():
    rclpy.init()
    node = InverseKinematicsExample()
    ok = node.run()
    node.destroy_node()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

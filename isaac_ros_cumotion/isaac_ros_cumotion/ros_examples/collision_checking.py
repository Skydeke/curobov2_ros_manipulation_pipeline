"""ROS example: check collision via /cumotion/check_collision.

Mirrors ``curobo.collision_checking`` examples using the unified
``curobo_server_node``'s ``CheckCollision`` service.

Usage:

.. code-block:: bash

    ros2 run isaac_ros_cumotion ros_example_collision_checking
"""

import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState as RosJointState

from isaac_ros_cumotion_interfaces.srv import CheckCollision


class CollisionCheckingExample(Node):

    def __init__(self):
        super().__init__("ros_example_collision_checking")
        self.cli = self.create_client(CheckCollision, "cumotion/check_collision")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for cumotion/check_collision...")
        self.get_logger().info("Connected to cumotion/check_collision")

    def run(self):
        req = CheckCollision.Request()
        joint_names = [
            "joint_1", "joint_2", "joint_3", "joint_4",
            "joint_5", "joint_6", "joint_7",
        ]

        js1 = RosJointState()
        js1.name = joint_names
        js1.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        req.joint_states = [js1]

        import random
        js2 = RosJointState()
        js2.name = joint_names
        js2.position = [random.uniform(-2.0, 2.0) for _ in range(7)]
        req.joint_states = [js1, js2]

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()

        if resp:
            for i, (js, coll, wcd, scd) in enumerate(
                zip(req.joint_states, resp.in_collision, resp.world_collision_distance, resp.self_collision_distance)
            ):
                self.get_logger().info(
                    f"Config {i}: in_collision={coll}, "
                    f"world_dist={wcd:.4f}m, self_dist={scd:.4f}m"
                )
            return True
        else:
            self.get_logger().error("CheckCollision failed")
            return False


def main():
    rclpy.init()
    node = CollisionCheckingExample()
    ok = node.run()
    node.destroy_node()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

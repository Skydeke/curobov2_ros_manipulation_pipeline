"""ROS example: plan a motion via /cumotion/plan_motion.

Mirrors ``curobo.examples.getting_started.motion_planning.pose_planning_example``
using the unified ``curobo_server_node``'s ``PlanMotion`` action.

Usage:

.. code-block:: bash

    ros2 run isaac_ros_cumotion ros_example_motion_planning
"""

import sys
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState as RosJointState
from geometry_msgs.msg import Pose as RosPose

from isaac_ros_cumotion_interfaces.action import PlanMotion


class MotionPlanningExample(Node):

    def __init__(self):
        super().__init__("ros_example_motion_planning")
        self._action_client = ActionClient(self, PlanMotion, "cumotion/plan_motion")
        while not self._action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info("Waiting for cumotion/plan_motion...")
        self.get_logger().info("Connected to cumotion/plan_motion")

    def run(self):
        goal = PlanMotion.Goal()

        js = RosJointState()
        js.name = [
            "joint_1", "joint_2", "joint_3", "joint_4",
            "joint_5", "joint_6", "joint_7",
        ]
        js.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        goal.start_state = js

        p = RosPose()
        p.position.x, p.position.y, p.position.z = 0.4, 0.0, 0.4
        p.orientation.w = 1.0
        goal.goal_poses = [p]
        goal.plan_goal_set = False

        send_goal_future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error("PlanMotion goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result

        if result.success:
            n_wp = len(result.trajectory.points)
            self.get_logger().info(f"Motion plan succeeded: {n_wp} waypoints in {result.planning_time_s:.3f}s")
            self.get_logger().info(f"  Matched goal index: {result.matched_goal_index}")
            return True
        else:
            self.get_logger().error(f"Motion plan failed: {result.message}")
            return False


def main():
    rclpy.init()
    node = MotionPlanningExample()
    ok = node.run()
    node.destroy_node()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

"""ROS example: plan a grasp motion via /cumotion/plan_grasp.

Mirrors ``curobo.examples.getting_started.motion_planning.grasp_planning_example``
using the unified ``curobo_server_node``'s ``PlanGrasp`` action.

Usage:

.. code-block:: bash

    ros2 run isaac_ros_cumotion ros_example_grasp_planning
"""

import sys

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState as RosJointState
from geometry_msgs.msg import Pose as RosPose

from isaac_ros_cumotion_interfaces.action import PlanGrasp


class GraspPlanningExample(Node):

    def __init__(self):
        super().__init__("ros_example_grasp_planning")
        self._action_client = ActionClient(self, PlanGrasp, "cumotion/plan_grasp")
        while not self._action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info("Waiting for cumotion/plan_grasp...")
        self.get_logger().info("Connected to cumotion/plan_grasp")

    def run(self):
        goal = PlanGrasp.Goal()

        js = RosJointState()
        js.name = [
            "joint_1", "joint_2", "joint_3", "joint_4",
            "joint_5", "joint_6", "joint_7",
        ]
        js.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        goal.start_state = js

        p = RosPose()
        p.position.x, p.position.y, p.position.z = 0.5, 0.0, 0.3
        p.orientation.w = 1.0
        goal.grasp_poses = [p]

        goal.grasp_approach_offset = 0.1
        goal.grasp_lift_offset = 0.1
        goal.plan_approach_to_grasp = True
        goal.plan_grasp_to_lift = True
        goal.grasp_lift_in_tool_frame = True

        send_goal_future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error("PlanGrasp goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result

        if result.success:
            self.get_logger().info(f"Grasp plan succeeded in {result.planning_time_s:.3f}s")
            if result.approach_trajectory.points:
                self.get_logger().info(f"  Approach: {len(result.approach_trajectory.points)} waypoints")
            if result.grasp_trajectory.points:
                self.get_logger().info(f"  Grasp: {len(result.grasp_trajectory.points)} waypoints")
            if result.lift_trajectory.points:
                self.get_logger().info(f"  Lift: {len(result.lift_trajectory.points)} waypoints")
            self.get_logger().info(f"  Matched goal index: {result.matched_goal_index}")
            return True
        else:
            self.get_logger().error(f"Grasp plan failed: {result.message}")
            return False


def main():
    rclpy.init()
    node = GraspPlanningExample()
    ok = node.run()
    node.destroy_node()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

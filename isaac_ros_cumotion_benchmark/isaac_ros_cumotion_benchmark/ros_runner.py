import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import Pose as RosPose
from sensor_msgs.msg import JointState as RosJointState

from isaac_ros_cumotion_interfaces.action import PlanMotion
from isaac_ros_cumotion_interfaces.srv import UpdateWorld

from isaac_ros_cumotion_benchmark.obstacle_convert import obstacles_dict_to_collision_objects


FRANKA_JOINT_NAMES = [
    'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
    'panda_joint5', 'panda_joint6', 'panda_joint7',
]


class RosBenchmarkRunner(Node):

    def __init__(self, time_dilation_factor=1.0):
        super().__init__('curobo_benchmark_ros_runner')
        self._time_dilation_factor = time_dilation_factor

        self._action_client = ActionClient(self, PlanMotion, 'cumotion/plan_motion')
        self._update_world_client = self.create_client(UpdateWorld, 'cumotion/update_world')

        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('cumotion/plan_motion action server not available')
            raise RuntimeError('cumotion/plan_motion not available')
        self.get_logger().info('Connected to cumotion/plan_motion')

        if not self._update_world_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('cumotion/update_world service not available')
            raise RuntimeError('cumotion/update_world not available')
        self.get_logger().info('Connected to cumotion/update_world')

    def _clear_world(self):
        req = UpdateWorld.Request()
        req.operation = UpdateWorld.Request.CLEAR_ALL
        future = self._update_world_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if future.result() is None or not future.result().success:
            raise RuntimeError('Failed to clear world')

    def _add_obstacles(self, obstacles_dict):
        collision_objects = obstacles_dict_to_collision_objects(obstacles_dict)
        req = UpdateWorld.Request()
        req.operation = UpdateWorld.Request.REPLACE
        req.objects = collision_objects
        future = self._update_world_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if future.result() is None or not future.result().success:
            raise RuntimeError('Failed to update world')

    def _plan_motion(self, start_q, goal_pose_list):
        goal = PlanMotion.Goal()

        js = RosJointState()
        js.name = FRANKA_JOINT_NAMES
        js.position = start_q
        goal.start_state = js

        ros_pose = RosPose()
        ros_pose.position.x = float(goal_pose_list[0])
        ros_pose.position.y = float(goal_pose_list[1])
        ros_pose.position.z = float(goal_pose_list[2])
        ros_pose.orientation.w = float(goal_pose_list[3])
        ros_pose.orientation.x = float(goal_pose_list[4])
        ros_pose.orientation.y = float(goal_pose_list[5])
        ros_pose.orientation.z = float(goal_pose_list[6])
        goal.goal_poses = [ros_pose]

        goal.plan_goal_set = False
        goal.tool_frame = ''
        goal.enable_graph_search = True
        goal.time_dilation_factor = float(self._time_dilation_factor)

        send_goal_future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=300.0)
        goal_handle = send_goal_future.result()

        if goal_handle is None or not goal_handle.accepted:
            return {'success': False, 'time_s': 0.0, 'n_waypoints': 0}

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=300.0)
        result = result_future.result().result

        if result.success:
            n_wp = len(result.trajectory.points)
            return {
                'success': True,
                'time_s': float(result.planning_time_s),
                'n_waypoints': n_wp,
            }
        else:
            return {'success': False, 'time_s': 0.0, 'n_waypoints': 0}


def run_ros(dataset, time_dilation_factor=1.0):
    from isaac_ros_cumotion_benchmark.problems import load_problems

    problems = load_problems(dataset)

    rclpy.init()
    runner = RosBenchmarkRunner(time_dilation_factor=time_dilation_factor)

    all_results = []

    for scene_key, scene_problems in problems.items():
        for i, problem in enumerate(scene_problems, start=1):
            if problem['collision_buffer_ik'] < 0.0:
                continue

            problem_name = f'{scene_key}_{i}'

            q_start = problem['start']
            pose = (
                problem['goal_pose']['position_xyz']
                + problem['goal_pose']['quaternion_wxyz']
            )

            runner._clear_world()
            runner._add_obstacles(problem['obstacles'])

            t_start = time.perf_counter()
            plan_result = runner._plan_motion(q_start, pose)
            t_end = time.perf_counter()

            plan_result['problem_name'] = problem_name
            plan_result['scene_key'] = scene_key
            if plan_result['time_s'] == 0.0:
                plan_result['time_s'] = t_end - t_start

            all_results.append(plan_result)

    runner.destroy_node()
    rclpy.shutdown()

    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='demo',
                        choices=['demo', 'motion_benchmaker', 'mpinets'])
    parser.add_argument('--time_dilation_factor', type=float, default=1.0)
    args = parser.parse_args()

    results = run_ros(args.dataset, args.time_dilation_factor)

    successes = sum(1 for r in results if r['success'])
    total = len(results)
    total_time = sum(r['time_s'] for r in results)
    print(f'Dataset: {args.dataset}')
    print(f'Problems: {total}')
    print(f'Successes: {successes}/{total} ({100*successes/total:.1f}%)')
    print(f'Total planning time: {total_time:.3f}s')
    if successes:
        avg_time = sum(r['time_s'] for r in results if r['success']) / successes
        print(f'Avg success time: {avg_time:.3f}s')
    for r in results:
        status = 'OK' if r['success'] else 'FAIL'
        print(f'  {r["problem_name"]}: {status} '
              f'({r["time_s"]:.3f}s, {r["n_waypoints"]} waypoints)')


if __name__ == '__main__':
    main()
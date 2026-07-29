import argparse
import time
from copy import deepcopy

from curobo._src.geom.types import SceneCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose


def _find_benchmark_dir():
    import os
    pkg_dir = os.path.dirname(os.path.realpath(__file__))
    ws = pkg_dir
    for _ in range(10):
        for sub in ('src', 'build', ''):
            candidate = os.path.join(ws, sub, 'curobo_core', 'curobo', 'benchmark')
            if os.path.isdir(candidate):
                return os.path.realpath(candidate)
        parent = os.path.dirname(ws)
        if parent == ws:
            break
        ws = parent
    ws = os.environ.get('ROS_WS', '/root/ros2_ws')
    for sub in ('src', 'build'):
        d = os.path.join(ws, sub, 'curobo_core', 'curobo', 'benchmark')
        if os.path.isdir(d):
            return os.path.realpath(d)
    ros_ws = os.environ.get('ROS_WS', 'unset')
    raise ImportError(
        'Cannot find curobo_core benchmark directory. '
        f'Searched from {pkg_dir} up through parents and $ROS_WS={ros_ws}'
    )


_benchmark_dir = _find_benchmark_dir()
import sys as _sys
_sys.path.insert(0, _benchmark_dir)
from motion_plan_benchmark import check_problems, load_curobo


def run_core(dataset, warmup_iters=3, max_attempts=100, enable_graph_attempt=1):
    from isaac_ros_cumotion_benchmark.problems import load_problems

    problems = load_problems(dataset)

    all_results = []

    for scene_key, scene_problems in problems.items():
        n_cubes = check_problems(scene_problems)

        args = argparse.Namespace(
            disable_cuda_graph=False,
            use_dynamics=False,
            mass=3.0,
            mesh=False,
        )

        mg, robot_cfg = load_curobo(
            n_cubes, ik_seeds=32, trajopt_seeds=4,
            mpinets=False, collision_buffer=0.0, args=args,
        )
        mg.warmup(enable_graph=True)

        for i, problem in enumerate(scene_problems, start=1):
            if problem['collision_buffer_ik'] < 0.0:
                continue

            problem_name = f'{scene_key}_{i}'

            q_start = problem['start']
            pose = (
                problem['goal_pose']['position_xyz']
                + problem['goal_pose']['quaternion_wxyz']
            )

            world = SceneCfg.create(
                deepcopy(problem['obstacles'])
            ).get_obb_world()
            mg.scene_collision_checker.clear_cache()
            mg.update_world(world)
            mg.reset_seed()

            start_state = JointState.from_position(
                mg.device_cfg.to_device([q_start])
            )
            goal_pose = Pose.from_list(pose)
            goal_tool_poses = GoalToolPose.from_poses(
                {mg.tool_frames[0]: goal_pose},
                ordered_tool_frames=mg.tool_frames,
            )

            if i == 1:
                for _ in range(warmup_iters):
                    mg.reset_seed()
                    mg.plan_pose(
                        goal_tool_poses, start_state,
                        max_attempts=1, enable_graph_attempt=1,
                    )

            mg.reset_seed()
            t_start = time.perf_counter()
            result = mg.plan_pose(
                goal_tool_poses, start_state,
                max_attempts=max_attempts,
                enable_graph_attempt=enable_graph_attempt,
            )
            t_end = time.perf_counter()

            if result is not None and result.success is not None and result.success.any().item():
                success = True
                planning_time_s = float(result.total_time)
                n_waypoints = int(
                    result.js_solution.position.shape[-2]
                ) if result.js_solution is not None else 0
            else:
                success = False
                planning_time_s = t_end - t_start
                n_waypoints = 0

            all_results.append({
                'problem_name': problem_name,
                'scene_key': scene_key,
                'success': success,
                'time_s': planning_time_s,
                'n_waypoints': n_waypoints,
            })

        mg.destroy()

    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='demo',
                        choices=['demo', 'motion_benchmaker', 'mpinets'])
    args = parser.parse_args()

    results = run_core(args.dataset)

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
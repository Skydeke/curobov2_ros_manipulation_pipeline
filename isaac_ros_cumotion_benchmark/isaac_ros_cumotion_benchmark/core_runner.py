import argparse
import time
from copy import deepcopy

from curobo._src.geom.collision.buffer_collision import CollisionBuffer
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
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg


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


def run_core_ik(dataset='demo'):
    results, _ = run_core_ik_single(dataset)
    return results


def run_core_ik_single(dataset='demo', max_problems=None):
    """Direct IK solve: compute FK on a start config, then IK back to it.
    
    Solves one problem at a time using max_batch_size=1 (SolveMode.SINGLE).
    Returns (results, total_time_s).
    """
    from isaac_ros_cumotion_benchmark.problems import load_problems
    problems = load_problems(dataset)
    results = []
    total_time = 0.0

    for scene_key, scene_problems in problems.items():
        valid_problems = [
            p for p in scene_problems
            if p['collision_buffer_ik'] >= 0.0
        ]
        if max_problems is not None:
            valid_problems = valid_problems[:max_problems]
        if not valid_problems:
            continue

        n_cubes = check_problems(scene_problems)
        args = argparse.Namespace(
            disable_cuda_graph=False, use_dynamics=False, mass=3.0, mesh=False,
        )
        mg, _ = load_curobo(
            n_cubes, ik_seeds=32, trajopt_seeds=4,
            mpinets=False, collision_buffer=0.0, args=args,
        )
        mg.ik_solver.config.max_batch_size = 1
        mg.trajopt_solver.config.max_batch_size = 1
        mg.warmup(enable_graph=True)

        for i, problem in enumerate(valid_problems, start=1):
            q_start = problem['start']
            problem_name = f'{scene_key}_ik_single_{i}'

            start_state = JointState.from_position(
                mg.device_cfg.to_device([q_start])
            )
            kin_state = mg.kinematics.compute_kinematics(start_state)
            goal_poses = kin_state.tool_poses.as_goal()

            t0 = time.perf_counter()
            ik_result = mg.ik_solver.solve_pose(goal_poses)
            dt = time.perf_counter() - t0
            total_time += dt

            success = (ik_result.success is not None and ik_result.success.any().item())
            results.append({
                'problem_name': problem_name,
                'capability': 'ik_single',
                'success': success,
                'time_s': float(ik_result.solve_time) if success else dt,
                'position_error': float(ik_result.position_error.max().item() * 1000.0) if success else -1.0,
                'rotation_error': float(ik_result.rotation_error.max().item() * 180.0 / 3.14159) if success else -1.0,
            })

        mg.destroy()

    return results, total_time


def _create_batch_planner(n_problems, n_cubes):
    """Build a MotionPlanner configured for max_batch_size=N.

    Replicates the setup from ``motion_plan_benchmark.load_curobo`` but
    passes ``max_batch_size=N`` into ``MotionPlannerCfg.create()`` so the
    solver is compiled from the start with the right batch dimension.
    """
    from curobo._src.types.robot import RobotCfg
    from curobo._src.types.device_cfg import DeviceCfg
    from curobo._src.util_file import (
        get_robot_configs_path, get_scene_configs_path, join_path, load_yaml,
    )
    from curobo._src.geom.types import SceneCfg
    import copy

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    if "robot_cfg" in robot_cfg:
        robot_cfg = robot_cfg["robot_cfg"]
    robot_cfg["kinematics"]["collision_sphere_buffer"] = 0.0
    robot_cfg["kinematics"]["tool_frames"] = ["panda_hand"]
    robot_cfg_instance = RobotCfg.create(
        copy.deepcopy(robot_cfg), device_cfg=DeviceCfg(),
    )

    scene_cfg = SceneCfg.create(
        load_yaml(join_path(get_scene_configs_path(), "collision_table.yml"))
    ).get_obb_world()

    cfg = MotionPlannerCfg.create(
        robot=robot_cfg_instance,
        scene_model=scene_cfg,
        collision_cache={"obb": n_cubes},
        num_ik_seeds=32,
        num_trajopt_seeds=4,
        max_batch_size=n_problems,
        max_goalset=n_problems,
    )
    mg = MotionPlanner(cfg)
    # NOTE: mg.warmup() initializes the IK solver's goal buffer (via
    # _plan_pose_goalset → ik_solver.solve_pose) with the default joint
    # state, which has jerk=[1,7].  _pad_batch_inputs in solver_ik.py pads
    # position/velocity/acceleration from 1→N but NOT jerk, so the cached
    # buffer ends up with jerk=[1,N] mismatched against any [4,7] input.
    # Skip the IK warmup here and let the dummy solve below create the
    # buffer at the right batch size from scratch.
    mg.graph_planner.warmup(
        num_warmup_iterations=10,
    )
    return mg


def run_core_ik_batched(dataset='demo', max_problems=None):
    """Direct IK solve: all problems in one batched call.

    Packs N problems into a single solve_pose call with max_batch_size=N
    (SolveMode.BATCH). The GPU solves all N in parallel.
    Returns (results, total_time_s).
    """
    import torch
    from isaac_ros_cumotion_benchmark.problems import load_problems
    problems = load_problems(dataset)
    results = []
    total_time = 0.0

    for scene_key, scene_problems in problems.items():
        valid = [
            (i, p) for i, p in enumerate(scene_problems, start=1)
            if p['collision_buffer_ik'] >= 0.0
        ]
        if max_problems is not None:
            valid = valid[:max_problems]
        n_problems = len(valid)
        if n_problems < 2:
            continue

        n_cubes = check_problems(scene_problems)
        mg = _create_batch_planner(n_problems, n_cubes)

        q_starts = [p['start'] for _, p in valid]
        start_positions = torch.stack([
            mg.device_cfg.to_device([q]).view(-1) for q in q_starts
        ])
        start_state_batch = JointState.from_position(start_positions)

        kin_state = mg.kinematics.compute_kinematics(start_state_batch)
        goal_poses_batch = kin_state.tool_poses.as_goal()

        # Dummy solve: creates the goal buffer at the right batch size
        # (jerk=[4,7] instead of the warmup's unpadded [1,7]).
        # Force exit_early=False so the CUDA graph for the optimizer path
        # is compiled during this (untimed) call, not the timed one.
        mg.ik_solver.config.exit_early = False
        mg.ik_solver.solve_pose(goal_poses_batch, current_state=start_state_batch)
        mg.ik_solver.config.exit_early = True

        t0 = time.perf_counter()
        ik_result = mg.ik_solver.solve_pose(goal_poses_batch, current_state=start_state_batch)
        total_time = time.perf_counter() - t0

        for idx, (i, problem) in enumerate(valid):
            problem_name = f'{scene_key}_ik_batch_{i}'
            success = (ik_result.success is not None and ik_result.success[idx].item())
            results.append({
                'problem_name': problem_name,
                'capability': 'ik_batch',
                'success': success,
                'time_s': float(ik_result.solve_time) if success else total_time,
                'position_error': float(ik_result.position_error[idx].max().item() * 1000.0) if success else -1.0,
                'rotation_error': float(ik_result.rotation_error[idx].max().item() * 180.0 / 3.14159) if success else -1.0,
            })

        mg.destroy()

    return results, total_time


def run_core_fk(dataset='demo'):
    """Direct FK: compute tool poses for start configs from problems."""
    from isaac_ros_cumotion_benchmark.problems import load_problems
    problems = load_problems(dataset)
    results = []

    for scene_key, scene_problems in problems.items():
        n_cubes = check_problems(scene_problems)
        args = argparse.Namespace(
            disable_cuda_graph=False, use_dynamics=False, mass=3.0, mesh=False,
        )
        mg, _ = load_curobo(
            n_cubes, ik_seeds=32, trajopt_seeds=4,
            mpinets=False, collision_buffer=0.0, args=args,
        )
        mg.warmup(enable_graph=True)

        for i, problem in enumerate(scene_problems[:3], start=1):
            if problem['collision_buffer_ik'] < 0.0:
                continue
            q_start = problem['start']
            problem_name = f'{scene_key}_fk_{i}'

            start_state = JointState.from_position(
                mg.device_cfg.to_device([q_start])
            )
            t0 = time.perf_counter()
            kin_state = mg.kinematics.compute_kinematics(start_state)
            dt = time.perf_counter() - t0

            tp = kin_state.tool_poses
            results.append({
                'problem_name': problem_name,
                'capability': 'fk',
                'success': True,
                'time_s': dt,
                'num_frames': len(tp.tool_frames),
                'frame_names': list(tp.tool_frames),
            })

        mg.destroy()

    return results


def run_core_collision(dataset='demo'):
    """Direct collision check: test start configs from problems."""
    from isaac_ros_cumotion_benchmark.problems import load_problems
    problems = load_problems(dataset)
    results = []

    for scene_key, scene_problems in problems.items():
        n_cubes = check_problems(scene_problems)
        args = argparse.Namespace(
            disable_cuda_graph=False, use_dynamics=False, mass=3.0, mesh=False,
        )
        mg, _ = load_curobo(
            n_cubes, ik_seeds=32, trajopt_seeds=4,
            mpinets=False, collision_buffer=0.0, args=args,
        )
        mg.warmup(enable_graph=True)

        for i, problem in enumerate(scene_problems[:3], start=1):
            if problem['collision_buffer_ik'] < 0.0:
                continue
            q_start = problem['start']
            problem_name = f'{scene_key}_collision_{i}'

            world = SceneCfg.create(
                deepcopy(problem['obstacles'])
            ).get_obb_world()
            mg.scene_collision_checker.clear_cache()
            mg.update_world(world)

            start_state = JointState.from_position(
                mg.device_cfg.to_device([q_start])
            )
            t0 = time.perf_counter()

            kin_state = mg.kinematics.compute_kinematics(start_state)
            num_spheres = kin_state.robot_spheres.shape[2]
            collision_buffer = CollisionBuffer.from_shape(
                (1, 1, num_spheres, 4), mg.device_cfg
            )
            weight = mg.device_cfg.to_device([1.0])
            activation_distance = mg.device_cfg.to_device([0.0])

            dist = mg.scene_collision_checker.get_sphere_collision(
                kin_state, collision_buffer, weight, activation_distance
            )
            dt = time.perf_counter() - t0

            in_collision = bool((dist > 0).any().item())
            world_dist = float(dist.max().item())

            results.append({
                'problem_name': problem_name,
                'capability': 'collision',
                'success': True,
                'time_s': dt,
                'in_collision': in_collision,
                'world_distance': world_dist,
                'self_distance': 0.0,
            })

        mg.destroy()

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='demo',
                        choices=['demo', 'motion_benchmaker', 'mpinets'])
    parser.add_argument('--capability', default='planning',
                        choices=['planning', 'ik', 'fk', 'collision', 'all'])
    args = parser.parse_args()

    if args.capability == 'planning':
        results = run_core(args.dataset)
    elif args.capability == 'ik':
        results = run_core_ik(args.dataset)
    elif args.capability == 'fk':
        results = run_core_fk(args.dataset)
    elif args.capability == 'collision':
        results = run_core_collision(args.dataset)
    elif args.capability == 'all':
        results = (run_core(args.dataset) + run_core_ik(args.dataset)
                   + run_core_fk(args.dataset) + run_core_collision(args.dataset))

    successes = sum(1 for r in results if r['success'])
    total = len(results)
    total_time = sum(r['time_s'] for r in results)
    print(f'Dataset: {args.dataset}')
    print(f'Capability: {args.capability}')
    print(f'Problems: {total}')
    print(f'Successes: {successes}/{total} ({100*successes/total:.1f}%)')
    print(f'Total planning time: {total_time:.3f}s')
    if successes:
        avg_time = sum(r['time_s'] for r in results if r['success']) / successes
        print(f'Avg success time: {avg_time:.3f}s')
    for r in results:
        status = 'OK' if r['success'] else 'FAIL'
        print(f'  {r["problem_name"]}: {status} '
              f'({r["time_s"]:.3f}s)')


if __name__ == '__main__':
    main()